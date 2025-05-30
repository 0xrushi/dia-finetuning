"""
LoRA Adapter Manager for Dia Model
Allows swapping between different LoRA adapters on the same base model.
"""

import torch
from pathlib import Path
from peft import PeftModel
import json
import shutil
from typing import Dict, List

class LoRAAdapterManager:
    """
    Manages multiple LoRA adapters for a single base Dia model.
    Allows quick switching between different voice styles/speakers.
    """
    
    def __init__(self, base_model_path, config_path, device="cuda"):
        """
        Initialize the adapter manager with a base model.
        
        Args:
            base_model_path: Path to base Dia model
            config_path: Path to model configuration
            device: Device to use
        """
        from .config import DiaConfig
        from .layers import DiaModel
        from .model import Dia
        import dac
        
        self.device = device
        self.base_model_path = base_model_path
        self.config_path = config_path
        
        # Load configuration and DAC model
        self.dia_cfg = DiaConfig.load(config_path)
        self.dac_model = dac.DAC.load(dac.utils.download()).to(device)
        
        # Load base model
        self.base_model = DiaModel(self.dia_cfg)
        if isinstance(base_model_path, str) and not Path(base_model_path).exists():
            from huggingface_hub import hf_hub_download
            ckpt_file = hf_hub_download(base_model_path, filename="dia-v0_1.pth")
        else:
            ckpt_file = base_model_path
        
        state_dict = torch.load(ckpt_file, map_location="cpu")
        self.base_model.load_state_dict(state_dict)
        self.base_model = self.base_model.to(device)
        
        # Track loaded adapters
        self.adapters: Dict[str, str] = {}  # name -> path
        self.current_adapter = None
        self.current_model = None
        
        print(f"✅ Base Dia model loaded on {device}")
    
    def register_adapter(self, name: str, adapter_path: str):
        """
        Register a LoRA adapter by name.
        
        Args:
            name: Friendly name for the adapter (e.g., "female_voice", "british_accent")
            adapter_path: Path to the LoRA adapter directory
        """
        if not Path(adapter_path).exists():
            raise FileNotFoundError(f"Adapter path not found: {adapter_path}")
        
        self.adapters[name] = adapter_path
        print(f"📝 Registered adapter '{name}' from {adapter_path}")
    
    def load_adapter(self, name: str, merge_adapters: bool = True):
        """
        Load a specific LoRA adapter.
        
        Args:
            name: Name of the registered adapter
            merge_adapters: Whether to merge for faster inference
        """
        if name not in self.adapters:
            raise ValueError(f"Adapter '{name}' not registered. Available: {list(self.adapters.keys())}")
        
        adapter_path = self.adapters[name]
        
        # Load fresh base model to avoid conflicts
        model = DiaModel(self.dia_cfg)
        state_dict = torch.load(
            self.base_model_path if Path(self.base_model_path).exists() 
            else hf_hub_download(self.base_model_path, filename="dia-v0_1.pth"), 
            map_location="cpu"
        )
        model.load_state_dict(state_dict)
        
        # Apply LoRA adapter
        model = PeftModel.from_pretrained(model, adapter_path)
        
        if merge_adapters:
            print(f"🔄 Merging adapter '{name}'...")
            model = model.merge_and_unload()
        
        model = model.to(self.device)
        model.eval()
        
        self.current_model = model
        self.current_adapter = name
        
        print(f"✅ Loaded adapter: '{name}'")
    
    def generate(self, text: str, adapter_name: str = None, **generation_kwargs):
        """
        Generate audio with a specific adapter.
        
        Args:
            text: Input text to synthesize
            adapter_name: Name of adapter to use (if None, uses current)
            **generation_kwargs: Additional generation parameters
        """
        if adapter_name and adapter_name != self.current_adapter:
            self.load_adapter(adapter_name)
        
        if self.current_model is None:
            raise ValueError("No adapter loaded. Call load_adapter() first.")
        
        # Create Dia generator
        from .model import Dia
        dia_gen = Dia(self.dia_cfg, self.device)
        dia_gen.model = self.current_model
        dia_gen.dac_model = self.dac_model
        
        with torch.inference_mode():
            return dia_gen.generate(text=text, **generation_kwargs)
    
    def list_adapters(self) -> List[str]:
        """List all registered adapters."""
        return list(self.adapters.keys())
    
    def get_adapter_info(self, name: str) -> Dict:
        """Get information about a specific adapter."""
        if name not in self.adapters:
            raise ValueError(f"Adapter '{name}' not registered")
        
        adapter_path = Path(self.adapters[name])
        config_file = adapter_path / "adapter_config.json"
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
            return {
                "name": name,
                "path": str(adapter_path),
                "config": config,
                "size_mb": sum(f.stat().st_size for f in adapter_path.rglob('*') if f.is_file()) / 1024 / 1024
            }
        else:
            return {"name": name, "path": str(adapter_path), "config": None}
    
    def download_adapter_from_hf(self, repo_id: str, adapter_name: str, local_dir: str = "./adapters"):
        """
        Download a LoRA adapter from Hugging Face Hub.
        
        Args:
            repo_id: Hugging Face repository ID (e.g., "username/dia-voice-adapter")
            adapter_name: Local name for the adapter
            local_dir: Local directory to save the adapter
        """
        from huggingface_hub import snapshot_download
        
        local_path = Path(local_dir) / adapter_name
        local_path.mkdir(parents=True, exist_ok=True)
        
        # Download adapter files
        snapshot_download(repo_id=repo_id, local_dir=str(local_path))
        
        # Register the downloaded adapter
        self.register_adapter(adapter_name, str(local_path))
        print(f"📥 Downloaded and registered adapter '{adapter_name}' from {repo_id}")
    
    def share_adapter(self, adapter_name: str, output_dir: str, include_readme: bool = True):
        """
        Package an adapter for sharing.
        
        Args:
            adapter_name: Name of the adapter to package
            output_dir: Directory to save the packaged adapter
            include_readme: Whether to include a README file
        """
        if adapter_name not in self.adapters:
            raise ValueError(f"Adapter '{adapter_name}' not registered")
        
        source_path = Path(self.adapters[adapter_name])
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Copy adapter files
        for file in source_path.rglob('*'):
            if file.is_file():
                rel_path = file.relative_to(source_path)
                dest_file = output_path / rel_path
                dest_file.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file, dest_file)
        
        # Create README if requested
        if include_readme:
            readme_content = f"""# {adapter_name} - Dia LoRA Adapter

## Usage

```python
from lora_inference import LoRADiaGenerator

# Load the adapter
generator = LoRADiaGenerator(
    base_model_path="nari-labs/Dia-1.6B",
    lora_adapter_path="./{adapter_name}",
    config_path="dia/config.json"
)

# Generate audio
audio = generator.generate("[en]Hello with custom voice!")
```

## Adapter Info
- **Name**: {adapter_name}
- **Compatible Base Model**: nari-labs/Dia-1.6B
- **Training Details**: Add your training details here

## Requirements
- torch
- peft
- transformers
- dac (for audio processing)
"""
            with open(output_path / "README.md", 'w') as f:
                f.write(readme_content)
        
        print(f"📦 Packaged adapter '{adapter_name}' in {output_path}")


# Example usage script
if __name__ == "__main__":
    import argparse
    import torchaudio
    
    parser = argparse.ArgumentParser(description="Manage and use multiple LoRA adapters")
    parser.add_argument("--base_model", default="nari-labs/Dia-1.6B")
    parser.add_argument("--config", default="dia/config.json")
    parser.add_argument("--text", required=True, help="Text to synthesize")
    parser.add_argument("--adapter", required=True, help="Adapter name to use")
    parser.add_argument("--output", default="output.wav")
    
    args = parser.parse_args()
    
    # Initialize manager
    manager = LoRAAdapterManager(args.base_model, args.config)
    
    # Register some example adapters (you'd have these from training)
    manager.register_adapter("my_voice", "./adapters/my_voice")
    manager.register_adapter("female_voice", "./adapters/female_voice")
    manager.register_adapter("british_accent", "./adapters/british_accent")
    
    # List available adapters
    print("Available adapters:", manager.list_adapters())
    
    # Generate with specific adapter
    audio = manager.generate(args.text, args.adapter)
    
    # Save result
    torchaudio.save(args.output, audio.unsqueeze(0), 44100)
    print(f"Generated audio with '{args.adapter}' adapter: {args.output}")