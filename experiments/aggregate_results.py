"""
Aggregate per-run JSON results into a summary table.

Usage:
    python3 experiments/aggregate_results.py experiments/results
"""
import os
import sys
import json
import numpy as np
import pandas as pd

METRICS = ['mAP', 'mINP', 'BAKS', 'AUCROC', 'top1', 'top3', 'top5', 'top10']

# Paper's Table 1 AAS row (macro-averaged over 13 datasets)
PAPER_TARGETS = {
    'mAP':    0.5614,
    'mINP':   0.3817,
    'BAKS':   0.6715,
    'AUCROC': 0.7521,
    'top1':   0.6771,
    'top3':   0.7908,
    'top5':   0.8504,
    'top10':  0.9118,
}


def aggregate(results_dir: str) -> pd.DataFrame:
    """
    Read all *_run*.json files, compute mean ± std over runs per dataset,
    and add a MACRO_AVG row.
    """
    records: dict = {}

    for fname in sorted(os.listdir(results_dir)):
        if not fname.endswith('.json'):
            continue
        fpath = os.path.join(results_dir, fname)
        with open(fpath) as f:
            data = json.load(f)

        dataset = data['dataset']
        metrics = data['metrics']
        records.setdefault(dataset, {k: [] for k in METRICS})

        for k in METRICS:
            if k in metrics:
                records[dataset][k].append(metrics[k])

    rows = []
    for dataset, runs in records.items():
        row = {'dataset': dataset}
        for k in METRICS:
            vals = runs[k]
            row[f'{k}_mean'] = np.mean(vals) if vals else float('nan')
            row[f'{k}_std']  = np.std(vals)  if vals else float('nan')
            row[f'{k}_n']    = len(vals)
        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index('dataset')

    # Macro-average across datasets
    mean_cols = [c for c in df.columns if c.endswith('_mean')]
    macro = {col: df[col].mean() for col in mean_cols}
    macro_row = pd.DataFrame([macro], index=['MACRO_AVG'])
    df = pd.concat([df, macro_row])

    return df


def print_comparison(df: pd.DataFrame) -> None:
    """Print our macro-average vs. paper targets."""
    if 'MACRO_AVG' not in df.index:
        print("No results to compare yet.")
        return

    macro = df.loc['MACRO_AVG']
    print("\n┌─────────┬──────────┬──────────┬──────────┐")
    print("│ Metric  │   Ours   │  Paper   │  Delta   │")
    print("├─────────┼──────────┼──────────┼──────────┤")
    for k, target in PAPER_TARGETS.items():
        col = f'{k}_mean'
        if col not in macro:
            continue
        ours = macro[col] * 100
        delta = ours - target * 100
        sign = '+' if delta >= 0 else ''
        print(f"│ {k:<7s} │ {ours:7.2f}% │ {target*100:7.2f}% │ {sign}{delta:6.2f}% │")
    print("└─────────┴──────────┴──────────┴──────────┘")


if __name__ == '__main__':
    results_dir = sys.argv[1] if len(sys.argv) > 1 else 'experiments/results'

    if not os.path.isdir(results_dir):
        print(f"Results directory not found: {results_dir}")
        sys.exit(1)

    df = aggregate(results_dir)

    if df.empty:
        print("No result JSON files found.")
        sys.exit(0)

    print("\n=== Per-dataset mAP (mean ± std) ===")
    display_cols = ['mAP_mean', 'mAP_std', 'mINP_mean', 'BAKS_mean',
                    'AUCROC_mean', 'top1_mean', 'top5_mean']
    display_cols = [c for c in display_cols if c in df.columns]
    print((df[display_cols] * 100).round(2).to_string())

    print_comparison(df)

    # Save summary CSV
    csv_path = os.path.join(results_dir, 'summary.csv')
    df.to_csv(csv_path)
    print(f"\nSummary saved → {csv_path}")
