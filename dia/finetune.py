import argparse
import logging
import random
from dataclasses import dataclass
from pathlib import Path

import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, random_split

from torch.cuda.amp import autocast
from torch.utils.tensorboard import SummaryWriter
from torch.nn.utils import clip_grad_norm_
from transformers import get_scheduler
import torch.nn.functional as F
import bitsandbytes as bnb
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset, interleave_datasets
from huggingface_hub import hf_hub_download
import math
import gc

# Add LoRA imports
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType

import dac
from .config import DiaConfig
from .layers import DiaModel, DenseGeneral
from .model import Dia
from .audio import build_delay_indices, apply_audio_delay

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

# Language byte mappings (keep existing)
LANG2BYTE = {
    "en": 3,
    "de": 4,
    "fr": 5,
    "es": 6,
    "it": 7,
    "nl": 14,
    "pl": 15,
    "pt": 16,
    "tr": 17,
    "hu": 18,
}

test_sentences = {
    "en": "In order to fully assess performance and the accuracy of language tags, this test sentence contains multiple subordinate clauses, varied punctuation, and a sufficient word count.",
    "de": "Um Leistung und die Korrektheit der Sprach-Tags umfassend zu prüfen, enthält dieser Testsatz mehrere Nebensätze, unterschiedliche Zeichensetzung und eine ausreichende Wortzahl.",
}

@dataclass
class TrainConfig:
    epochs: int = 1
    batch_size: int = 2
    grad_accum_steps: int = 2
    learning_rate: float = 1e-4  # Usually higher for LoRA
    warmup_steps: int = 500
    unconditional_frac: float = 0.15
    eval_step: int = 200
    save_step: int = 2000
    split_ratio: float = 0.997
    shuffle_buffer_size: int = None
    seed: int = 42
    runs_dir: Path = Path("runs")
    run_name: str = "dia_lora_finetune"
    output_dir: Path = Path(".cpkts/dia_lora_finetune")
    
    # LoRA specific parameters
    use_lora: bool = True  # Re-enabled now that we have conversion
    lora_r: int = 16          # LoRA rank
    lora_alpha: int = 32      # LoRA alpha (scaling factor)
    lora_dropout: float = 0.1 # LoRA dropout
    lora_target_modules: list = None  # Will be set based on model architecture

def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Dia audio model with LoRA")
    parser.add_argument("--config", type=Path, default=Path("dia/config.json"))
    parser.add_argument("--dataset", type=str, default="Paradoxia/opendata-iisys-hui")
    parser.add_argument("--dataset2", type=str, default=None)
    parser.add_argument("--streaming", action="store_true")
    parser.add_argument("--hub_model", type=str, default="nari-labs/Dia-1.6B")
    parser.add_argument("--local_ckpt", type=str, default=None)
    parser.add_argument("--csv_path", type=Path, default=None)
    parser.add_argument("--audio_root", type=Path, default=None)
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--output_dir", type=Path, default=None)
    parser.add_argument("--shuffle_buffer_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--half", action="store_true", help="load model in fp16")
    parser.add_argument("--compile", action="store_true", help="torch compile model")
    
    # LoRA specific arguments
    parser.add_argument("--use_lora", action="store_true", default=True, 
                       help="Use LoRA for efficient finetuning")
    parser.add_argument("--lora_r", type=int, default=16,
                       help="LoRA rank (lower = more efficient, higher = more capacity)")
    parser.add_argument("--lora_alpha", type=int, default=32,
                       help="LoRA alpha scaling factor")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                       help="LoRA dropout rate")
    parser.add_argument("--lora_target_modules", type=str, nargs="+", default=None,
                       help="Target modules for LoRA")
    
    return parser.parse_args()

# FIXED: LoRA setup functions
# Create a wrapper class that handles reshaping for converted DenseGeneral->Linear
class LinearWrapper(torch.nn.Module):
    """Wrapper that makes Linear layer behave like DenseGeneral for tensor operations."""
    
    def __init__(self, linear_layer, original_input_shape, original_output_shape, axis):
        super().__init__()
        self.linear = linear_layer
        self.original_input_shape = original_input_shape
        self.original_output_shape = original_output_shape
        self.axis = axis
    
    def forward(self, x):
        # Store original shape for reshaping output
        orig_shape = x.shape
        
        # Ensure consistent dtype
        original_dtype = x.dtype
        x = x.float()
        
        # Calculate input features based on contraction axes
        if self.axis == (-1,):
            # Most common case: contract last dimension
            batch_dims = orig_shape[:-1]
            input_features = orig_shape[-1]
            x_flat = x.view(-1, input_features)
        else:
            # Handle more complex axis patterns
            # For now, assume we can flatten appropriately
            total_elements = 1
            for ax in self.axis:
                ax = ax if ax >= 0 else len(orig_shape) + ax
                total_elements *= orig_shape[ax]
            x_flat = x.view(-1, total_elements)
        
        # Apply linear transformation
        out_flat = self.linear(x_flat)
        
        # Reshape output back to expected dimensions
        if self.axis == (-1,):
            new_shape = batch_dims + self.original_output_shape
            output = out_flat.view(new_shape)
        else:
            # Reconstruct shape based on original output pattern
            output = out_flat.view(orig_shape[:-len(self.original_input_shape)] + self.original_output_shape)
        
        return output.to(original_dtype)

def convert_dense_general_to_linear(model):
    """
    Convert all DenseGeneral modules to equivalent Linear modules with proper reshaping.
    """
    conversions = 0
    
    def replace_dense_general(module):
        nonlocal conversions
        for name, child in module.named_children():
            if isinstance(child, DenseGeneral):
                # Convert DenseGeneral to Linear with wrapper
                weight = child.weight
                
                # Get original shapes and axis from DenseGeneral
                in_shapes = child.in_shapes
                out_features = child.out_features
                axis = child.axis
                
                # Flatten weight to 2D for Linear layer
                in_features = int(torch.prod(torch.tensor(in_shapes)))
                out_features_flat = int(torch.prod(torch.tensor(out_features)))
                
                # Create Linear layer
                linear = torch.nn.Linear(in_features, out_features_flat, bias=False, 
                                       device=weight.device, dtype=weight.dtype)
                linear.weight.data = weight.view(out_features_flat, in_features)
                
                # Wrap Linear layer to handle reshaping
                wrapper = LinearWrapper(linear, in_shapes, out_features, axis)
                
                # Replace the module
                setattr(module, name, wrapper)
                conversions += 1
            else:
                replace_dense_general(child)
    
    replace_dense_general(model)
    logger.info(f"Converted {conversions} DenseGeneral modules to wrapped Linear modules")
    return model

def get_dia_lora_target_modules(model):
    """Get target modules for LoRA specific to Dia model."""
    target_modules = []
    
    # After conversion, we should have LinearWrapper modules containing Linear layers
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            # Check if this Linear is inside a wrapper for attention/MLP modules
            parent_name = name.split('.')[-2] if '.' in name else ''
            if any(pattern in parent_name for pattern in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'wi_fused', 'wo']):
                target_modules.append(name)
        elif isinstance(module, LinearWrapper):
            # Target the wrapper if it corresponds to attention/MLP modules  
            if any(pattern in name for pattern in ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'wi_fused', 'wo']):
                # Target the inner linear layer
                target_modules.append(f"{name}.linear")
    
    if not target_modules:
        # Fallback: target ALL Linear modules
        logger.warning("No specific Linear modules found, targeting all Linear modules")
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                target_modules.append(name)
    
    logger.info(f"Found {len(target_modules)} Linear modules for LoRA")
    logger.info(f"Sample target modules: {target_modules[:3]}...")
    return target_modules

def setup_lora_model(model, train_cfg: TrainConfig):
    """Setup LoRA configuration and wrap the model."""
    
    # First, convert DenseGeneral modules to Linear modules
    logger.info("Converting DenseGeneral modules to Linear for LoRA compatibility...")
    model = convert_dense_general_to_linear(model)
    
    # Handle config compatibility with PEFT by temporarily removing it
    original_config = None
    if hasattr(model, 'config'):
        original_config = model.config
        delattr(model, 'config')  # Temporarily remove config to avoid PEFT issues
    
    if train_cfg.lora_target_modules is None:
        target_modules = get_dia_lora_target_modules(model)
    else:
        target_modules = train_cfg.lora_target_modules
        logger.info(f"Using specified LoRA target modules: {target_modules}")
    
    lora_config = LoraConfig(
        r=train_cfg.lora_r,
        lora_alpha=train_cfg.lora_alpha,
        target_modules=target_modules,
        lora_dropout=train_cfg.lora_dropout,
        bias="none",
        task_type=TaskType.FEATURE_EXTRACTION,
        modules_to_save=None,
    )
    
    model = get_peft_model(model, lora_config)
    
    # Restore the original config
    if original_config is not None:
        model.config = original_config
    
    model.print_trainable_parameters()
    return model

class LocalDiaDataset(Dataset):
    """Load from a local CSV (sep='|') + an audio folder."""
    def __init__(self, csv_path: Path, audio_root: Path, config: DiaConfig, dac_model: dac.DAC):
        self.df = pd.read_csv(csv_path, sep=r'\s*\|\s*', engine='python',
                              names=['audio','text'])
        self.audio_root = audio_root
        self.config = config
        self.dac_model = dac_model

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        try:
            row = self.df.iloc[idx]
            text = row['text']
            audio_path = self.audio_root / row['audio']
            waveform, sr = torchaudio.load(audio_path)
            if sr != 44100:
                waveform = torchaudio.functional.resample(waveform, sr, 44100)
            waveform = waveform.unsqueeze(0)
            with torch.no_grad():
                # preprocess + encode exactly as before
                audio_tensor = self.dac_model.preprocess(
                    waveform, 44100
                ).to(next(self.dac_model.parameters()).device)
                _, encoded, *_ = self.dac_model.encode(audio_tensor, n_quantizers=None)
                encoded = encoded.squeeze(0).transpose(0, 1)
            return text, encoded, waveform
        except Exception as e:
            row = self.df.iloc[idx]
            text = row['text']
            audio_path = self.audio_root / row['audio']
            logger.error(f"Error processing audio path {audio_path}: {e}")
            return text, None, None


class HFDiaDataset(Dataset):
    """Wrap a HuggingFace `datasets.Dataset` that has `audio.array` & `text`."""
    def __init__(self, hf_dataset, config: DiaConfig, dac_model: dac.DAC):
        self.dataset = hf_dataset
        self.config = config
        self.dac_model = dac_model

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        sample = self.dataset[idx]
        text = sample['text']
        audio_info = sample['audio']
        waveform = torch.tensor(audio_info['array'], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sr = audio_info['sampling_rate']
        if sr != 44100:
            waveform = torchaudio.functional.resample(waveform, sr, 44100)
        with torch.no_grad():
            audio_tensor = (
                self.dac_model.preprocess(waveform, 44100)
                .to(next(self.dac_model.parameters()).device)
            )
            _, encoded, *_ = self.dac_model.encode(audio_tensor, n_quantizers=None)
            encoded = encoded.squeeze(0).transpose(0, 1)
        return text, encoded, waveform


def collate_fn(batch, config: DiaConfig, device: torch.device):
    from torch.nn.functional import pad

    texts, encodings, waveforms = zip(*batch)
    
    # Filter out None encodings (failed audio loads)
    valid_samples = [(t, e, w) for t, e, w in zip(texts, encodings, waveforms) if e is not None]
    
    if not valid_samples:
        raise RuntimeError("All samples in batch failed to load!")
    
    if len(valid_samples) < len(batch):
        logger.warning(f"Filtered out {len(batch) - len(valid_samples)} failed samples from batch")
    
    texts, encodings, waveforms = zip(*valid_samples)

    # -- Text inputs ---------------------------------------------------------
    max_text = config.data.text_length
    pad_tok = config.data.text_pad_value
    text_ids = []
    for txt in texts:
        b_full = txt.encode('utf-8')
        # replace leading "[lang]" prefix
        for code, val in LANG2BYTE.items():
            prefix = f"[{code}]".encode('utf-8')
            if b_full.startswith(prefix):
                b_full = bytes([val]) + b_full[len(prefix):]
                break
        bts = b_full[:max_text]
        arr = list(bts) + [pad_tok] * (max_text - len(bts))
        text_ids.append(torch.tensor(arr, dtype=torch.long))
    src = torch.stack(text_ids).to(device)
    src_pos = torch.arange(max_text, device=device).unsqueeze(0).expand(src.size(0), -1)
    src_pad = src.ne(pad_tok)
    enc_self_attn_mask = (src_pad.unsqueeze(2) & src_pad.unsqueeze(1)).unsqueeze(1)

    # -- Audio codes --------------------------------------------------------
    max_audio = config.data.audio_length
    seq_lens = [min(e.size(0), max_audio) for e in encodings]
    batch_max = max(seq_lens)

    padded = [pad(e, (0, 0, 0, batch_max - e.size(0))) if e.size(0) < batch_max else e[:batch_max]
              for e in encodings]
    codes = torch.stack(padded).to(device)

    B, T, C = codes.shape
    t_idx, idxs = build_delay_indices(B, T, C, config.data.delay_pattern)
    delayed = apply_audio_delay(
        codes,
        config.data.audio_pad_value,
        config.data.audio_bos_value,
        (t_idx, idxs)
    )
    delayed = delayed[:, :max_audio, :]

    # -- Targets with per-sample EOS ----------------------------------------
    max_tgt_len = max_audio + 2
    pad_val = config.data.audio_pad_value
    bos_val = config.data.audio_bos_value
    eos_val = config.data.audio_eos_value

    tgt = torch.full((B, max_tgt_len, C), pad_val, dtype=torch.long, device=device)
    tgt[:, 0, :] = bos_val
    tgt_lens = []
    for i, L in enumerate(seq_lens):
        tgt[i, 1:1 + L, :] = delayed[i, :L, :]
        tgt[i, 1 + L, :] = eos_val
        tgt_lens.append(1 + L + 1)

    tgt_pos = torch.arange(max_tgt_len, device=device).unsqueeze(0).expand(B, -1)
    tgt_pad = tgt.ne(pad_val).any(-1)

    causal = torch.tril(torch.ones((max_tgt_len, max_tgt_len),
                                    dtype=torch.bool,
                                    device=device))
    dec_self_attn_mask = (tgt_pad.unsqueeze(2) & tgt_pad.unsqueeze(1) & causal).unsqueeze(1)
    dec_cross_attn_mask = (tgt_pad.unsqueeze(2) & src_pad.unsqueeze(1)).unsqueeze(1)

    return {
        'src_tokens': src,
        'src_positions': src_pos,
        'enc_self_attn_mask': enc_self_attn_mask,
        'tgt_tokens': tgt,
        'tgt_positions': tgt_pos,
        'dec_self_attn_mask': dec_self_attn_mask,
        'dec_cross_attn_mask': dec_cross_attn_mask,
        'waveforms': waveforms,
        'raw_text': texts[0],
        'tgt_lens': torch.tensor(tgt_lens, dtype=torch.long, device=device),
    }

def setup_loaders(dataset, dia_cfg: DiaConfig, train_cfg: TrainConfig, device):
    collate = lambda b: collate_fn(b, dia_cfg, device)
    # Handle missing HFDiaIterDataset class
    try:
        if isinstance(dataset, HFDiaIterDataset):
            total = getattr(dataset, "total_examples", None)
            if total is None:
                total = dataset.dataset.info.splits["train"].num_examples
            n_train = int(train_cfg.split_ratio * total)
            n_val = total - n_train
            if n_val <= 0:
                raise RuntimeError(f"No validation samples: total={total}, split_ratio={train_cfg.split_ratio}")
            base = dataset.dataset.shuffle(buffer_size=train_cfg.shuffle_buffer_size, seed=train_cfg.seed) if train_cfg.shuffle_buffer_size else dataset.dataset
            val_stream = base.take(n_val)
            train_stream = base.skip(n_val)
            train_ds = HFDiaIterDataset(train_stream, dia_cfg, dataset.dac_model)
            val_ds = HFDiaIterDataset(val_stream, dia_cfg, dataset.dac_model)
            train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=False, collate_fn=collate)
            train_loader.steps_per_epoch = math.ceil(n_train / train_cfg.batch_size)
            val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate)
            return train_loader, val_loader
    except NameError:
        # HFDiaIterDataset not defined, continue to regular dataset handling
        pass
        
    ds_len = len(dataset)
    n_train = int(train_cfg.split_ratio * ds_len)
    train_ds, val_ds = random_split(dataset, [n_train, ds_len - n_train])
    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, collate_fn=collate)
    return train_loader, val_loader

def setup_optimizer_and_scheduler(model, train_loader, train_cfg):
    # MODIFY: Only optimize LoRA parameters if LoRA is enabled
    if train_cfg.use_lora:
        optimizer_params = [p for p in model.parameters() if p.requires_grad]
        logger.info(f"Training {len(optimizer_params)} LoRA parameters")
    else:
        optimizer_params = model.parameters()
    
    opt = bnb.optim.AdamW8bit(optimizer_params, lr=train_cfg.learning_rate)
    
    try:
        steps_per_epoch = len(train_loader)
    except TypeError:
        if hasattr(train_loader, 'steps_per_epoch'):
            steps_per_epoch = train_loader.steps_per_epoch
        else:
            raise RuntimeError("Cannot determine steps_per_epoch for streaming loader")
    
    total_training_steps = steps_per_epoch * train_cfg.epochs
    sched = get_scheduler(
        'cosine', opt,
        num_warmup_steps=train_cfg.warmup_steps / train_cfg.grad_accum_steps,
        num_training_steps=total_training_steps / train_cfg.grad_accum_steps
    )
    return opt, sched

# ADD: LoRA checkpoint saving function
def save_lora_checkpoint(model, output_path):
    """Save LoRA adapter weights."""
    if hasattr(model, 'save_pretrained'):
        model.save_pretrained(output_path)
        logger.info(f"Saved LoRA checkpoint: {output_path}")
    else:
        torch.save(model.state_dict(), output_path)
        logger.info(f"Saved model checkpoint: {output_path}")

def train_step(model, batch, dia_cfg, train_cfg, opt, sched, writer, step, global_step):
    """Perform a single training step with LoRA support."""
    if random.random() < train_cfg.unconditional_frac:
        pad_tok = dia_cfg.data.text_pad_value
        batch['src_tokens'] = torch.zeros_like(batch['src_tokens'])
        batch['enc_self_attn_mask'] = torch.zeros_like(batch['enc_self_attn_mask'])
        batch['dec_cross_attn_mask'] = torch.zeros_like(batch['dec_cross_attn_mask'])

    with autocast():
        # For LoRA models, we need to call the base model directly with correct parameter names
        if hasattr(model, 'base_model'):
            # PEFT wrapped model - call base model directly
            logits = model.base_model.model(
                src_BxS=batch['src_tokens'],
                tgt_BxTxC=batch['tgt_tokens'],
                src_positions=batch['src_positions'],
                tgt_positions=batch['tgt_positions'],
                enc_self_attn_mask=batch['enc_self_attn_mask'],
                dec_self_attn_mask=batch['dec_self_attn_mask'],
                dec_cross_attn_mask=batch['dec_cross_attn_mask'],
                enable_dropout=True,
            )
        else:
            # Regular model
            logits = model(
                src_BxS=batch['src_tokens'],
                tgt_BxTxC=batch['tgt_tokens'],
                src_positions=batch['src_positions'],
                tgt_positions=batch['tgt_positions'],
                enc_self_attn_mask=batch['enc_self_attn_mask'],
                dec_self_attn_mask=batch['dec_self_attn_mask'],
                dec_cross_attn_mask=batch['dec_cross_attn_mask'],
                enable_dropout=True,
            )
        
        lens = batch['tgt_lens']
        max_L = int(lens.max().item())
        logits = logits[:, : max_L - 1]
        target = batch['tgt_tokens'][:, 1:max_L, :]

        B, Tm1, C = target.shape
        pad_val = dia_cfg.data.audio_pad_value

        time_idx = torch.arange(Tm1, device=lens.device).unsqueeze(0)
        valid_time = time_idx < (lens.unsqueeze(1) - 1)
        mask = valid_time.unsqueeze(-1).expand(-1, -1, C)

        channel_weights = [4.0] + [1.0] * (C - 1)
        loss_c = 0.0
        _, _, _, V = logits.size()

        for c, w in enumerate(channel_weights):
            lc = logits[:, :, c, :].reshape(-1, V)
            tc = target[:, :, c].reshape(-1)
            mc = mask[:, :, c].reshape(-1)

            lc_valid = lc[mc]
            tc_valid = tc[mc]
            loss_c += w * F.cross_entropy(
                lc_valid, tc_valid,
                ignore_index=pad_val
            )

        loss = loss_c / sum(channel_weights)

    loss = loss / train_cfg.grad_accum_steps
    loss.backward()

    grad_norm = clip_grad_norm_(model.parameters(), max_norm=1e9)
    writer.add_scalar('GradNorm/global', grad_norm, global_step)
    if (step + 1) % train_cfg.grad_accum_steps == 0:
        opt.step()
        sched.step()
        opt.zero_grad()
        true_loss = loss.item() * train_cfg.grad_accum_steps
        current_lr = sched.get_last_lr()[0]
        writer.add_scalar('LR', current_lr, global_step)
        writer.add_scalar('Loss/train', true_loss, global_step)

    return loss.item() * train_cfg.grad_accum_steps

def eval_step(model, val_loader, dia_cfg, dac_model, writer, global_step):
    """Run evaluation with LoRA model."""
    eval_losses = []
    last_batch = None
    with torch.inference_mode():
        for eb in tqdm(val_loader, desc="eval"):
            last_batch = eb

            with autocast():
                # Handle LoRA vs regular model
                if hasattr(model, 'base_model'):
                    # PEFT wrapped model - call base model directly
                    logits16 = model.base_model.model(
                        src_BxS=eb['src_tokens'],
                        tgt_BxTxC=eb['tgt_tokens'],
                        src_positions=eb['src_positions'],
                        tgt_positions=eb['tgt_positions'],
                        enc_self_attn_mask=eb['enc_self_attn_mask'],
                        dec_self_attn_mask=eb['dec_self_attn_mask'],
                        dec_cross_attn_mask=eb['dec_cross_attn_mask'],
                        enable_dropout=False,
                    )[:, :-1]
                else:
                    # Regular model
                    logits16 = model(
                        src_BxS=eb['src_tokens'],
                        tgt_BxTxC=eb['tgt_tokens'],
                        src_positions=eb['src_positions'],
                        tgt_positions=eb['tgt_positions'],
                        enc_self_attn_mask=eb['enc_self_attn_mask'],
                        dec_self_attn_mask=eb['dec_self_attn_mask'],
                        dec_cross_attn_mask=eb['dec_cross_attn_mask'],
                        enable_dropout=False,
                    )[:, :-1]

            logits = logits16.float()
            target = eb['tgt_tokens'][:, 1:]
            B_e, T_e, C_e = target.shape
            V_e = logits.size(-1)

            loss_e = 0.0
            weights_e = [4.0] + [1.0] * (C_e - 1)
            for c, w in enumerate(weights_e):
                lc = logits[:, :, c, :].reshape(-1, V_e)
                tc = target[:, :, c].reshape(-1)
                loss_e += w * F.cross_entropy(
                    lc, tc, ignore_index=dia_cfg.data.audio_pad_value
                )
            loss_e = loss_e / sum(weights_e)
            eval_losses.append(loss_e)

    avg_eval_loss = sum(eval_losses) / len(eval_losses)
    writer.add_scalar('Loss/eval', avg_eval_loss.item(), global_step)

    try:
        orig_dtype = next(model.parameters()).dtype
        
        # MODIFY: Handle LoRA model for evaluation
        if hasattr(model, 'base_model'):
            base_model = model.base_model.model  # PEFT wrapping
        else:
            base_model = model
            
        base_model = base_model.float()
        dia_gen = Dia(dia_cfg, device)
        dia_gen.model, dia_gen.dac_model = base_model, dac_model
        
        with torch.inference_mode():
            for lang_code, sentence in test_sentences.items():
                text = f"[{lang_code}]{sentence}"
                audio = None  # Initialize to prevent UnboundLocalError
                try:
                    audio = dia_gen.generate(text=text)
                    writer.add_audio(f"Eval/{lang_code}", audio, global_step, 44100)
                except Exception as e:
                    logger.exception(f"Error synthesizing test sentence in {lang_code}.")
                finally:
                    if audio is not None:
                        del audio
        gc.collect()
        torch.cuda.empty_cache()
        
    except Exception:
        logger.exception("Eval error")
    
    finally:
        if orig_dtype == torch.float16:
            if hasattr(model, 'base_model'):
                model.base_model.model = model.base_model.model.half()
            else:
                model = model.half()

def train(model, dia_cfg: DiaConfig, dac_model: dac.DAC, dataset, train_cfg: TrainConfig):
    """Run the full training loop with LoRA support."""
    # ADD: Setup LoRA if enabled
    if train_cfg.use_lora:
        model = setup_lora_model(model, train_cfg)
    
    train_cfg.output_dir.mkdir(parents=True, exist_ok=True)
    (train_cfg.runs_dir / train_cfg.run_name).mkdir(parents=True, exist_ok=True)
    model = model.to(device)

    train_loader, val_loader = setup_loaders(dataset, dia_cfg, train_cfg, device)
    opt, sched = setup_optimizer_and_scheduler(model, train_loader, train_cfg)

    writer = SummaryWriter(train_cfg.runs_dir / train_cfg.run_name)
    model.train()

    steps_per_epoch = getattr(train_loader, 'steps_per_epoch', None)
    if steps_per_epoch is None:
        try:
            steps_per_epoch = len(train_loader)
        except Exception:
            steps_per_epoch = None

    for epoch in range(train_cfg.epochs):
        loader_iter = tqdm(
            train_loader,
            desc=f"E{epoch+1}",
            total=steps_per_epoch
        )
        
        for step, batch in enumerate(loader_iter):
            global_step = epoch * (steps_per_epoch or 0) + step
            
            loss = train_step(model, batch, dia_cfg, train_cfg, opt, sched, writer, step, global_step)

            cur_alloc = torch.cuda.memory_allocated()
            peak_alloc = torch.cuda.max_memory_allocated()
            cur_gb = cur_alloc / 1024**3
            peak_gb = peak_alloc / 1024**3

            loader_iter.set_postfix({
                'loss': f"{loss:.4f}",
                'VRAM (GB)': f"{cur_gb:.2f}/{peak_gb:.2f}"
            })
            torch.cuda.reset_peak_memory_stats()

            if step % train_cfg.eval_step == 0:
                model.eval()
                with torch.no_grad():
                    eval_step(model, val_loader, dia_cfg, dac_model, writer, global_step)
                model.train()

            # MODIFY: LoRA checkpoint saving
            if step and step % train_cfg.save_step == 0:
                if train_cfg.use_lora:
                    ckpt_dir = train_cfg.output_dir / f"lora_step{global_step}"
                    save_lora_checkpoint(model, ckpt_dir)
                else:
                    ckpt = train_cfg.output_dir / f"ckpt_step{global_step}.pth"
                    torch.save(model.state_dict(), ckpt)
                    logger.info(f"Saved checkpoint: {ckpt}")

        # MODIFY: End of epoch LoRA checkpoint
        if train_cfg.use_lora:
            ckpt_dir = train_cfg.output_dir / f"lora_epoch{epoch+1}"
            save_lora_checkpoint(model, ckpt_dir)
        else:
            ckpt_e = train_cfg.output_dir / f"ckpt_epoch{epoch+1}.pth"
            torch.save(model.state_dict(), ckpt_e)
            logger.info(f"Saved end-of-epoch checkpoint: {ckpt_e}")

def main():
    args = get_args()
    dia_cfg   = DiaConfig.load(args.config)
    dac_model = dac.DAC.load(dac.utils.download()).to(device)

    dataset = None
    if not dataset:
        if args.csv_path:
            if not args.audio_root:
                raise ValueError("`--audio_root` must be set when using `--csv_path`")
            dataset = LocalDiaDataset(args.csv_path, args.audio_root, dia_cfg, dac_model)
        else:
            ds1 = load_dataset(args.dataset, split="train", streaming=args.streaming)
            
            if args.streaming:
                if args.dataset2:
                    ds2 = load_dataset(args.dataset2, split="train", streaming=True)
                    total1 = ds1.info.splits['train'].num_examples
                    total2 = ds2.info.splits['train'].num_examples
                    total = total1 + total2
                    hf_ds = interleave_datasets([ds1, ds2])
                    # Note: HFDiaIterDataset would need to be imported/defined for streaming
                    # For now, fall back to regular dataset
                    dataset = HFDiaDataset(hf_ds, dia_cfg, dac_model)
                else:
                    hf_ds = ds1
                    # Note: HFDiaIterDataset would need to be imported/defined for streaming
                    # For now, fall back to regular dataset
                    dataset = HFDiaDataset(hf_ds, dia_cfg, dac_model)
            else:
                dataset = HFDiaDataset(ds1, dia_cfg, dac_model)

    # MODIFY: Add LoRA config to train_cfg
    train_cfg = TrainConfig(
        run_name=args.run_name or TrainConfig.run_name,
        output_dir=args.output_dir or TrainConfig.output_dir,
        shuffle_buffer_size=args.shuffle_buffer_size,
        seed=args.seed,
        use_lora=args.use_lora,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=args.lora_target_modules,
    )

    if args.local_ckpt:
        ckpt_file = args.local_ckpt
    else:
        ckpt_file = hf_hub_download(args.hub_model, filename="dia-v0_1.pth")
    
    model = DiaModel(dia_cfg)
    if args.half:
        model = model.half()
    
    model.load_state_dict(torch.load(ckpt_file, map_location="cpu"))
    
    # MODIFY: Skip compile with LoRA
    if args.compile:
        if not train_cfg.use_lora:
            model = torch.compile(model, backend="inductor")
        else:
            logger.warning("Skipping torch.compile with LoRA - may cause issues")

    train(model, dia_cfg, dac_model, dataset, train_cfg)

if __name__ == "__main__":
    main()