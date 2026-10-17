"""
=============================================================================
   UNIFIED CYBER ATTACK DETECTION BENCHMARK SYSTEM
   Deep Learning & Machine Learning Pipeline Framework
   Supports:
     1. UNSW-NB15 (Network Anomaly & Cyber Attack Dataset)
     2. CSE-CIC-IDS2018 (Botnet & Intrusion Flow Capture Dataset)
=============================================================================
"""

import sys
import argparse
import time

# Ensure UTF-8 console output
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except Exception:
        pass

from pipeline_unsw_nb15 import run_unsw_pipeline
from pipeline_cicids2018 import run_cicids_pipeline

def parse_args():
    parser = argparse.ArgumentParser(
        description="Unified Cyber Attack Detection Benchmark Pipeline (UNSW-NB15 & CSE-CIC-IDS2018)"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["unsw", "cicids", "all"],
        default="all",
        help="Select dataset to evaluate: 'unsw', 'cicids', or 'all' (default: all)"
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=250000,
        help="Max flow samples to process for large datasets (default: 250,000)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./outputs",
        help="Directory to save generated charts and artifacts (default: ./outputs)"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    print("=" * 80)
    print("      UNIFIED CYBER ATTACK & INTRUSION DETECTION RESEARCH FRAMEWORK      ")
    print("=" * 80)
    print(f"[*] Target Dataset Mode : {args.dataset.upper()}")
    print(f"[*] Output Directory    : {args.output_dir}")
    print("=" * 80)

    start_time = time.time()
    results = {}

    if args.dataset in ["unsw", "all"]:
        print("\n" + "#" * 80)
        print(">>> EXECUTING DATASET 1: UNSW-NB15 BENCHMARK")
        print("#" * 80)
        out_unsw = f"{args.output_dir}/unsw_nb15"
        results["UNSW-NB15"] = run_unsw_pipeline(output_dir=out_unsw)

    if args.dataset in ["cicids", "all"]:
        print("\n" + "#" * 80)
        print(">>> EXECUTING DATASET 2: CSE-CIC-IDS2018 BENCHMARK")
        print("#" * 80)
        out_cicids = f"{args.output_dir}/cicids2018"
        results["CSE-CIC-IDS2018"] = run_cicids_pipeline(
            max_samples=args.max_samples, output_dir=out_cicids
        )

    total_time = time.time() - start_time
    print("\n" + "=" * 80)
    print("                     FINAL COMPARATIVE BENCHMARK SUMMARY                 ")
    print("=" * 80)
    print(f"{'Dataset':<20} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10} | {'ROC-AUC':<10}")
    print("-" * 80)
    for ds_name, m in results.items():
        print(f"{ds_name:<20} | {m['acc']*100:>8.2f}% | {m['prec']*100:>8.2f}% | {m['rec']*100:>8.2f}% | {m['f1']:>10.4f} | {m['roc_auc']:>10.4f}")
    print("=" * 80)
    print(f"[+] All requested benchmarks completed in {total_time:.2f} seconds!")
    print(f"[+] High-Resolution (300 DPI) Visualizations Saved to '{args.output_dir}'")
    print("=" * 80)

if __name__ == "__main__":
    main()
