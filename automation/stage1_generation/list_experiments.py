#!/usr/bin/env python3
"""
List all experiments

View generated synthetic data experiments, grouped by experiment purpose
"""

import argparse
from pathlib import Path
from experiment_manager import ExperimentManager

PROJECT_ROOT = Path("/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO")


def format_table(experiments):
    """Format as table"""
    if not experiments:
        return "No experiments found"

    # Header
    header = f"{'Experiment ID':<40} | {'Model':<20} | {'Temp':<6} | {'Fingerprint':<12} | {'Created':<20}"
    separator = "-" * 110

    lines = [header, separator]

    for exp in experiments:
        exp_id = exp.get('experiment_id', 'N/A')[:38]
        model = exp.get('generation', {}).get('model', 'N/A')[:18]
        temp = str(exp.get('generation', {}).get('temperature', 'N/A'))[:4]
        fingerprint = exp.get('parameter_fingerprint', 'N/A')[:10]
        created = exp.get('created_at', 'N/A')[:18]

        line = f"{exp_id:<40} | {model:<20} | {temp:<6} | {fingerprint:<12} | {created:<20}"
        lines.append(line)

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="List all synthetic data experiments")
    parser.add_argument(
        "--purpose",
        help="Only show experiments with specified purpose"
    )
    parser.add_argument(
        "--detail",
        action="store_true",
        help="Show detailed information"
    )
    args = parser.parse_args()

    # Initialize experiment manager
    output_base = PROJECT_ROOT / "Data_v2" / "synthetic"
    manager = ExperimentManager(output_base)

    # List experiments
    experiments = manager.list_experiments(experiment_purpose=args.purpose)

    if not experiments:
        print("No experiments found")
        return

    # Group by experiment purpose
    grouped = {}
    for exp in experiments:
        purpose = exp.get('experiment_purpose', 'general')
        if purpose not in grouped:
            grouped[purpose] = []
        grouped[purpose].append(exp)

    # Display
    print("\n" + "=" * 110)
    print("Synthetic Data Experiment List")
    print("=" * 110)

    for purpose, exps in sorted(grouped.items()):
        print(f"\n[{purpose}] ({len(exps)} experiments)")
        print(format_table(exps))

        if args.detail:
            for exp in exps:
                print(f"\n  Path: {exp.get('path', 'N/A')}")
                print(f"  Description: {exp.get('experiment_description', 'N/A')}")
                print(f"  Task: {exp.get('task_name', 'N/A')}")
                print(f"  Method: {exp.get('training_method', 'N/A')}")
                print("-" * 110)

    print(f"\nTotal: {len(experiments)} experiments")
    print("=" * 110 + "\n")


if __name__ == "__main__":
    main()
