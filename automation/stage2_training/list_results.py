#!/usr/bin/env python3
"""
Training Results Management Tool
List and organize all training results in the Results_v2 directory
"""

import os
import yaml
from pathlib import Path
from typing import Dict, List
from datetime import datetime
from collections import defaultdict

# Import unified configuration
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PROJECT_ROOT


class ResultsManager:
    def __init__(self):
        self.project_root = Path(PROJECT_ROOT)
        self.results_base = self.project_root / "Results_v2"

    def scan_results(self) -> Dict[str, List[Dict]]:
        """
        Scan Results_v2 directory, return results grouped by experiment purpose

        Returns:
            {
                experiment_purpose: [
                    {
                        'model': str,
                        'experiment_name': str,
                        'timestamp': str,
                        'path': Path,
                        'config': Dict
                    },
                    ...
                ],
                ...
            }
        """
        if not self.results_base.exists():
            return {}

        results_by_purpose = defaultdict(list)

        # Iterate through experiment purpose directories
        for purpose_dir in self.results_base.iterdir():
            if not purpose_dir.is_dir():
                continue

            purpose = purpose_dir.name

            # Iterate through model directories
            for model_dir in purpose_dir.iterdir():
                if not model_dir.is_dir():
                    continue

                model = model_dir.name

                # Iterate through experiment directories ({Task}_{Method}_{DataType}_{LR})
                for exp_dir in model_dir.iterdir():
                    if not exp_dir.is_dir():
                        continue

                    exp_name = exp_dir.name

                    # Iterate through timestamp directories
                    for timestamp_dir in exp_dir.iterdir():
                        if not timestamp_dir.is_dir():
                            continue

                        timestamp = timestamp_dir.name

                        # Read configuration file
                        config_path = timestamp_dir / "experiment_config.yaml"
                        config = None
                        if config_path.exists():
                            try:
                                with open(config_path, 'r', encoding='utf-8') as f:
                                    config = yaml.safe_load(f)
                            except Exception as e:
                                print(f"⚠️  Unable to read config: {config_path}: {e}")

                        results_by_purpose[purpose].append({
                            'model': model,
                            'experiment_name': exp_name,
                            'timestamp': timestamp,
                            'path': timestamp_dir,
                            'config': config
                        })

        # Sort results under each purpose by timestamp
        for purpose in results_by_purpose:
            results_by_purpose[purpose].sort(key=lambda x: x['timestamp'], reverse=True)

        return dict(results_by_purpose)

    def print_results_summary(self):
        """Print results summary"""
        results = self.scan_results()

        if not results:
            print("📂 Results_v2 directory is empty")
            return

        print("\n" + "="*80)
        print("Training Results Summary - Results_v2")
        print("="*80)

        total_experiments = 0
        for purpose, experiments in sorted(results.items()):
            total_experiments += len(experiments)

            print(f"\n📁 Experiment purpose: {purpose}")
            print(f"   Number of experiments: {len(experiments)}")

            # Group statistics by model
            by_model = defaultdict(int)
            for exp in experiments:
                by_model[exp['model']] += 1

            for model, count in sorted(by_model.items()):
                print(f"   └─ {model}: {count} experiments")

        print(f"\n{'='*80}")
        print(f"Total: {len(results)} experiment purposes, {total_experiments} training experiments")
        print("="*80)

    def print_results_detail(self, purpose: str = None):
        """
        Print detailed results list

        Args:
            purpose: Specify experiment purpose, if None display all
        """
        results = self.scan_results()

        if not results:
            print("📂 Results_v2 directory is empty")
            return

        # Filter specified purpose
        if purpose:
            if purpose not in results:
                print(f"❌ Experiment purpose not found: {purpose}")
                print(f"Available experiment purposes: {', '.join(sorted(results.keys()))}")
                return
            results = {purpose: results[purpose]}

        print("\n" + "="*80)
        print("Training Results Details")
        print("="*80)

        for purpose_name, experiments in sorted(results.items()):
            print(f"\n📁 Experiment purpose: {purpose_name}")
            print("-" * 80)

            for i, exp in enumerate(experiments, 1):
                print(f"\n  [{i}] {exp['experiment_name']}")
                print(f"      Model: {exp['model']}")
                print(f"      Time: {exp['timestamp']}")
                print(f"      Path: {exp['path'].relative_to(self.project_root)}")

                if exp['config']:
                    config = exp['config']
                    print(f"      Task: {config.get('task', 'N/A')}")
                    print(f"      Method: {config.get('method', 'N/A')}")

                    # Display hyperparameters
                    hp = config.get('hyperparameters', {})
                    if hp:
                        print(f"      Hyperparameters:")
                        print(f"        - LR: {hp.get('learning_rate', 'N/A')}")
                        print(f"        - BS: {hp.get('batch_size', 'N/A')}")
                        print(f"        - Steps: {hp.get('steps', 'N/A')}")
                        print(f"        - Seed: {hp.get('seed', 'N/A')}")

                    # Display data information
                    data_config = config.get('data', {})
                    if 'path' in data_config:
                        print(f"      Data: {data_config['path']}")
                    elif 'type' in data_config:
                        print(f"      Data: {data_config['type']}")

        print("\n" + "="*80)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Training Results Management Tool")
    parser.add_argument(
        "--purpose",
        help="Specify experiment purpose (used when viewing details)"
    )
    parser.add_argument(
        "--detail",
        action="store_true",
        help="Display detailed information"
    )

    args = parser.parse_args()

    manager = ResultsManager()

    if args.detail:
        manager.print_results_detail(purpose=args.purpose)
    else:
        manager.print_results_summary()


if __name__ == "__main__":
    main()
