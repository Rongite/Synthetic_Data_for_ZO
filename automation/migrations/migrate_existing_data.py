#!/usr/bin/env python3
"""
Migrate existing data to new organization structure
Archive existing results and synthetic data to pending directory
"""

import os
import shutil
import json
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = "/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO"

class DataMigrator:
    def __init__(self, dry_run=True):
        self.dry_run = dry_run
        self.project_root = Path(PROJECT_ROOT)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # New directory structure
        self.new_structure = {
            "data": self.project_root / "Data_v2",
            "results": self.project_root / "Results_v2",
            "pending": self.project_root / "Pending_Manual_Classification"
        }

    def create_new_directories(self):
        """Create new directory structure"""
        print("\n" + "="*80)
        print("Creating new directory structure")
        print("="*80)

        dirs_to_create = [
            # Data directories
            self.new_structure["data"] / "original",
            self.new_structure["data"] / "synthetic",

            # Results directory (standardized naming)
            self.new_structure["results"],

            # Pending directory
            self.new_structure["pending"] / "data" / "synthetic_legacy",
            self.new_structure["pending"] / "results" / "legacy_results",
        ]

        for dir_path in dirs_to_create:
            if self.dry_run:
                print(f"[DRY RUN] Will create: {dir_path}")
            else:
                dir_path.mkdir(parents=True, exist_ok=True)
                print(f"✓ Created: {dir_path}")

    def migrate_original_data(self):
        """Migrate original data (direct copy, as it's already validated)"""
        print("\n" + "="*80)
        print("Migrating original data")
        print("="*80)

        src = self.project_root / "Data" / "original"
        dst = self.new_structure["data"] / "original"

        if not src.exists():
            print(f"⚠ Source directory does not exist: {src}")
            return

        if self.dry_run:
            print(f"[DRY RUN] Will copy: {src} -> {dst}")
            # Show what will be copied
            for task_dir in src.iterdir():
                if task_dir.is_dir():
                    file_count = len(list(task_dir.glob("*.jsonl")))
                    print(f"  - {task_dir.name}: {file_count} files")
        else:
            shutil.copytree(src, dst, dirs_exist_ok=True)
            print(f"✓ Copy completed: {dst}")

    def migrate_synthetic_data_to_pending(self):
        """Move existing synthetic data to pending area"""
        print("\n" + "="*80)
        print("Archiving existing synthetic data to Pending")
        print("="*80)

        synthetic_dirs = [
            self.project_root / "Data" / "synthetic",
            self.project_root / "Data" / "rejection_sampling",
            self.project_root / "Data" / "original_n_synthetic",
        ]

        dst_base = self.new_structure["pending"] / "data" / "synthetic_legacy"

        for src in synthetic_dirs:
            if not src.exists():
                continue

            dst = dst_base / src.name

            if self.dry_run:
                print(f"[DRY RUN] Will move: {src} -> {dst}")
                # Count data
                jsonl_files = list(src.rglob("*.jsonl"))
                py_files = list(src.rglob("*.py"))
                print(f"  - {len(jsonl_files)} .jsonl files")
                print(f"  - {len(py_files)} .py scripts")
            else:
                shutil.move(str(src), str(dst))
                print(f"✓ Move completed: {dst}")

        # Create README file
        readme_path = dst_base / "README.md"
        readme_content = f"""# Synthetic Data Pending Classification

Migration time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Directory Description

- `synthetic/`: Original synthetic data generation scripts and outputs
- `rejection_sampling/`: Rejection sampling validation scripts and outputs
- `original_n_synthetic/`: Mixed data

## Suggested Actions

Please manually classify this data based on:
1. Review the prompt in generation scripts to determine training method type
2. Check the model used for data generation (GPT-4o)
3. Reorganize according to new naming convention

New naming convention:
```
Data_v2/synthetic/{{task}}_{{method}}_{{model}}_{{version}}/
```

Examples:
- Copa_mezo_gpt4o_v1/
- RTE_lora_gpt4o_v1/
"""

        if self.dry_run:
            print(f"[DRY RUN] Will create README file: {readme_path}")
        else:
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(readme_content)
            print(f"✓ Created README file: {readme_path}")

    def migrate_results_to_pending(self):
        """Move existing results to pending, classify as much as possible by rules"""
        print("\n" + "="*80)
        print("Archiving existing training results")
        print("="*80)

        src = self.project_root / "results"

        if not src.exists():
            print(f"⚠ Source directory does not exist: {src}")
            return

        # Analyze result structure
        classified = {
            "standard": [],  # Standard format: Model/Task/Method/DataType/
            "special": [],   # Special experiments: seed_2, 80000_step, large_model, no_method
            "unknown": []    # Cannot classify
        }

        for model_dir in src.iterdir():
            if not model_dir.is_dir() or model_dir.name.startswith('.'):
                continue

            model_name = model_dir.name

            for task_dir in model_dir.iterdir():
                if not task_dir.is_dir() or task_dir.name.startswith('.'):
                    continue

                task_name = task_dir.name

                # Check if this is a special experiment
                if task_name in ['seed_2', '80000_step', 'large_model', 'no_method']:
                    classified["special"].append({
                        "model": model_name,
                        "type": task_name,
                        "path": task_dir
                    })
                    continue

                # Check for standard structure
                for method_dir in task_dir.iterdir():
                    if not method_dir.is_dir():
                        continue

                    method_name = method_dir.name

                    # Check if DataType level exists
                    has_data_type = False
                    for data_type_dir in method_dir.iterdir():
                        if data_type_dir.is_dir():
                            has_data_type = True
                            result_files = list(data_type_dir.glob("*.out")) + list(data_type_dir.glob("*.err"))
                            if result_files:
                                classified["standard"].append({
                                    "model": model_name,
                                    "task": task_name,
                                    "method": method_name,
                                    "data_type": data_type_dir.name,
                                    "path": data_type_dir,
                                    "file_count": len(result_files)
                                })

                    if not has_data_type:
                        # Files directly in method level
                        result_files = list(method_dir.glob("*.out")) + list(method_dir.glob("*.err"))
                        if result_files:
                            classified["unknown"].append({
                                "model": model_name,
                                "task": task_name,
                                "method": method_name,
                                "path": method_dir
                            })

        print(f"\nClassification results:")
        print(f"  Standard structure: {len(classified['standard'])} experiments")
        print(f"  Special experiments: {len(classified['special'])} experiments")
        print(f"  Unknown structure: {len(classified['unknown'])} experiments")

        # Process standard structure - can migrate directly to new location
        print("\nProcessing standard structure experiments...")
        for exp in classified["standard"]:
            # New path convention
            # Results_v2/{model}/{task}_{method}_{data_type}/{timestamp}/
            new_dir_name = f"{exp['task']}_{exp['method']}_{exp['data_type']}"
            dst = self.new_structure["results"] / exp["model"] / new_dir_name / self.timestamp

            if self.dry_run:
                print(f"[DRY RUN] Migrate: {exp['path'].relative_to(src)} -> {dst.relative_to(self.new_structure['results'])}")
            else:
                dst.mkdir(parents=True, exist_ok=True)
                # Copy all result files
                for f in exp['path'].iterdir():
                    if f.is_file():
                        shutil.copy2(f, dst / f.name)
                print(f"✓ Migrated: {exp['file_count']} files -> {dst}")

        # Process special experiments and unknown structure - move to pending
        print("\nProcessing special experiments and unknown structure...")
        pending_results = self.new_structure["pending"] / "results" / "legacy_results"

        all_special = classified["special"] + classified["unknown"]
        for exp in all_special:
            relative_path = exp['path'].relative_to(src)
            dst = pending_results / relative_path

            if self.dry_run:
                print(f"[DRY RUN] Archive to pending: {relative_path}")
            else:
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copytree(exp['path'], dst, dirs_exist_ok=True)
                print(f"✓ Archived: {dst}")

        # Create README file
        readme_path = pending_results / "README.md"
        readme_content = f"""# Training Results Pending Classification

Migration time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## Statistics

- Auto-classified (standard structure): {len(classified['standard'])} experiments
- Pending manual classification (special experiments): {len(classified['special'])} experiments
- Pending manual classification (unknown structure): {len(classified['unknown'])} experiments

## Special Experiment Description

- `seed_2/`: Reproduction experiments with SEED=2
- `80000_step/`: Experiments with extended training steps
- `large_model/`: Large model experiments
- `no_method/`: Experiments without method annotation

## New Naming Convention

Results_v2/{{model}}/{{task}}_{{method}}_{{data_type}}/{{timestamp}}/

Examples:
- Llama-3.2-1B/Copa_zo_original/20241224_120000/
- Llama-3.2-1B/RTE_lora_synthetic_gpt4o_v1/20241224_130000/

data_type should include:
- Data source: original / synthetic / mixed
- If synthetic: add generation model and version, e.g. synthetic_gpt4o_v1
"""

        if self.dry_run:
            print(f"\n[DRY RUN] Will create README file: {readme_path}")
        else:
            with open(readme_path, 'w', encoding='utf-8') as f:
                f.write(readme_content)
            print(f"✓ Created README file: {readme_path}")

    def create_migration_report(self):
        """Generate migration report"""
        report_path = self.new_structure["pending"] / f"migration_report_{self.timestamp}.json"

        report = {
            "timestamp": datetime.now().isoformat(),
            "dry_run": self.dry_run,
            "summary": {
                "original_data": "copied to Data_v2/original/",
                "synthetic_data": "moved to Pending/data/synthetic_legacy/",
                "results": "organized based on structure"
            },
            "next_steps": [
                "Review Pending/data/synthetic_legacy/ and classify manually",
                "Review Pending/results/legacy_results/ for special experiments",
                "Update config files for new experiments"
            ]
        }

        if self.dry_run:
            print(f"\n[DRY RUN] Migration report preview:")
            print(json.dumps(report, indent=2, ensure_ascii=False))
        else:
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            print(f"\n✓ Migration report: {report_path}")

    def run(self):
        """Execute complete migration"""
        print("\n" + "="*80)
        print(f"Data Migration Tool {'[DRY RUN MODE]' if self.dry_run else '[EXECUTE MODE]'}")
        print("="*80)

        self.create_new_directories()
        self.migrate_original_data()
        self.migrate_synthetic_data_to_pending()
        self.migrate_results_to_pending()
        self.create_migration_report()

        print("\n" + "="*80)
        print("Migration complete!")
        print("="*80)

        if self.dry_run:
            print("\nThis is DRY RUN mode, no files were actually modified.")
            print("Please review the output, then use --execute parameter to perform actual migration.")
        else:
            print("\nActual migration completed. Please check:")
            print(f"  - Data_v2/: New data directory")
            print(f"  - Results_v2/: New results directory")
            print(f"  - Pending_Manual_Classification/: Data pending manual classification")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Migrate existing data to new organization structure")
    parser.add_argument("--execute", action="store_true", help="Execute actual migration (default is dry run)")
    args = parser.parse_args()

    migrator = DataMigrator(dry_run=not args.execute)
    migrator.run()

if __name__ == "__main__":
    main()
