import torch
import torchaudio
from dia.config import DiaConfig
from dia.layers import DiaModel
from dia.model import Dia
import dac

cfg = DiaConfig.load("/path/to/config.json")
model = DiaModel(cfg)
state = torch.load("checkpoints/123.pth", map_location="cpu")
model.load_state_dict(state)

device = torch.device("cpu" if torch.cuda.is_available() else "cpu")
model = model.to(device).eval()

dac_model = dac.DAC.load(dac.utils.download())
dac_model = dac_model.to(device)

inference = Dia(cfg, device)
inference.model     = model
inference.dac_model = dac_model

text_prompt = "Hello how are you"

with torch.no_grad():
    with torch.cuda.amp.autocast(enabled=True):
        audio_out = inference.generate(text=text_prompt)

torchaudio.save("output.wav", torch.from_numpy(audio_out).unsqueeze(0), 44100)
