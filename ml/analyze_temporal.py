"""Evaluate the frozen monthly pilot checkpoints and write matched reports."""
from pathlib import Path
from bogsignal.training.finalize import finalize_run
from bogsignal.training.combined import evaluate_combined_run
from bogsignal.training.evaluate import evaluate_run
from bogsignal.training.temporal_compare import compare

runs = [Path('../data/experiments') / name for name in [
    'T0_provisional_resnet34_seed17', 'T1_provisional_resnet34_seed17',
    'T2_provisional_resnet34_seed17', 'TB_matched_static_resnet34_seed17']]
for run in runs:
    print(f'Evaluating {run.name}', flush=True)
    finalize_run(run)
    evaluate_combined_run(run)
    evaluate_run(run)
print(compare(runs, '../docs/experiments/temporal_provisional_seed17.md'), flush=True)
