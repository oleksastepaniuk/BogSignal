"""Compare adapter on one retained validation input, never test data."""
from pathlib import Path
import json
import numpy as np
from bogsignal.inference.torch_adapter import load_predictor
from bogsignal.training.temporal_d_compare import load_run
ROOT=Path(__file__).resolve().parents[1]
run=load_run(ROOT/'data/experiments/D_s2s1nisar_resnet34_seed17')
images=np.load(run['cache']/'image.npy',mmap_mode='r')
valid=np.load(run['cache']/'validity.npy',mmap_mode='r')
batch=np.concatenate([images[0],valid[0].astype('float32')],axis=0)[None]
logits=load_predictor(Path(run['run'])/'best.pt')(batch)[0]
assert logits.shape==(6,256,256) and np.isfinite(logits).all()
p=np.exp(logits-logits.max(axis=0,keepdims=True));p/=p.sum(axis=0,keepdims=True)
combined=np.stack([p[0],p[1]+p[2],p[3],p[4],p[5]]).argmax(axis=0)
keep=run['valid'][0].astype(bool)&(run['target'][0]!=255)
drift=float((combined[keep]!=run['predictions'][1][0][keep]).mean())
assert drift<0.005, f'Unexpected adapter prediction drift: {drift}'
print(json.dumps(dict(patch_id=run['ids'][0],scored_pixels=int(keep.sum()),combined_prediction_disagreement_fraction=drift,test_evaluated=False)))
