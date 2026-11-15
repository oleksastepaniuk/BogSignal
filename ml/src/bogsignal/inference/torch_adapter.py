"""Frozen six-class U-Net adapter for the existing tiled inference workflow."""
import json
from pathlib import Path
import numpy as np
import torch
from bogsignal.training.core import make_model,sha256

def load_predictor(checkpoint):
    checkpoint=Path(checkpoint)
    manifest=json.loads((checkpoint.parent/'manifest.json').read_text())
    complete=json.loads((checkpoint.parent/'complete.json').read_text())
    if sha256(checkpoint)!=complete['checkpoint_sha256']:
        raise ValueError('Frozen checkpoint digest mismatch')
    config=manifest['config']
    if len(config['channels'])!=14:
        raise ValueError('Regional workflow requires all 14 frozen physical inputs')
    device='cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_num_threads(4)
    model=make_model(14,config['decoder_channels'],pretrained=False)
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=False)['model'])
    model=model.to(device).to(memory_format=torch.channels_last).eval()
    bf16=device=='cuda' and torch.cuda.is_bf16_supported()
    def predict(batch):
        x=np.asarray(batch,dtype='float32')
        if x.ndim!=4 or x.shape[1:]!=(28,256,256) or not np.isfinite(x).all():
            raise ValueError('Expected finite normalized [B,28,256,256] inputs')
        with torch.inference_mode(),torch.autocast(device_type=device,dtype=torch.bfloat16,enabled=bf16):
            result=model(torch.from_numpy(x).to(device).to(memory_format=torch.channels_last))
        return result.float().cpu().numpy()
    return predict
