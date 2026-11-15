"""Analyze the completed all-sensor temporal-mean run against frozen D."""
from pathlib import Path
from bogsignal.training.finalize import finalize_run
from bogsignal.training.combined import evaluate_combined_run
from bogsignal.training.report import write_report
from bogsignal.training.temporal_d_compare import compare


def main():
    run = Path('../data/experiments/TD_s2s1_mean_nisar_resnet34_seed17')
    finalize_run(run)
    evaluate_combined_run(run)
    write_report(run, '../docs/experiments/TD_s2s1_mean_nisar_resnet34_seed17.md')
    compare('../data/experiments/D_s2s1nisar_resnet34_seed17', run,
            '../docs/experiments/temporal_d_comparison_seed17')


if __name__ == '__main__':
    main()
