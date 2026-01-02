#!/usr/bin/env python3
"""
Resolve data path - Convert batch path to shared path, or vice versa

Features:
  1. batch path → shared path
  2. shared path → all batch paths referencing this data
  3. Check if path exists

Usage:
    # Resolve batch path to shared path
    python resolve_data_path.py "Data_v2/synthetic/batch_20241229_temperature/Copa/temp07_topp10_gpt4o/Copa"

    # Resolve shared path, find all batches using it
    python resolve_data_path.py "Data_v2/synthetic/_shared/Copa/temp07_topp10_gpt4o/Copa"
"""

import argparse
import os
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent


def resolve_path(input_path):
    """Resolve path"""

    path = Path(input_path)

    # Determine if batch path or shared path
    if '_shared' in path.parts:
        # Shared path → Find all batches referencing it
        resolve_shared_to_batches(path)
    elif 'batch_' in str(path):
        # Batch path → Resolve to shared path
        resolve_batch_to_shared(path)
    else:
        print(f"❌ Unable to recognize path type: {input_path}")
        print("   Path should contain 'batch_' or '_shared'")
        return 1

    return 0


def resolve_batch_to_shared(batch_path):
    """Batch path → Shared path"""

    print("=" * 100)
    print("🔍 Path Resolution: Batch → Shared")
    print("=" * 100)

    # Check if path exists
    abs_path = PROJECT_ROOT / batch_path

    if not abs_path.exists():
        print(f"\n❌ Batch path does not exist: {batch_path}")
        return

    # If it's a symbolic link, resolve it
    if abs_path.is_symlink():
        target = abs_path.resolve()
        relative_target = target.relative_to(PROJECT_ROOT)

        print(f"\n📁 Batch path (symbolic link):")
        print(f"   {batch_path}")

        print(f"\n➡️  Resolves to (Shared path):")
        print(f"   {relative_target}")

        print(f"\n✅ Both paths can be used for training config:")
        print(f"\n   # Method 1: Use Batch path (recommended - clear experiment purpose)")
        print(f'   path: "{batch_path}"')
        print(f"\n   # Method 2: Use Shared path (physical path)")
        print(f'   path: "{relative_target}"')

        # Check if data files exist
        check_data_files(abs_path)

    else:
        print(f"\n⚠️  This is not a symbolic link, may be an actual directory")
        print(f"   Path: {batch_path}")


def resolve_shared_to_batches(shared_path):
    """Shared path → All Batch paths"""

    print("=" * 100)
    print("🔍 Path Resolution: Shared → Batches")
    print("=" * 100)

    # Check if shared path exists
    abs_shared_path = PROJECT_ROOT / shared_path

    if not abs_shared_path.exists():
        print(f"\n❌ Shared path does not exist: {shared_path}")
        return

    print(f"\n📦 Shared path (physical storage):")
    print(f"   {shared_path}")

    # Check data files
    check_data_files(abs_shared_path)

    # Find all batches referencing this shared path
    synthetic_base = PROJECT_ROOT / "Data_v2" / "synthetic"

    batches_using_this = []

    for batch_dir in synthetic_base.iterdir():
        if not batch_dir.is_dir():
            continue
        if batch_dir.name == '_shared':
            continue
        if not batch_dir.name.startswith('batch_'):
            continue

        # Iterate through datasets in batch
        for dataset_dir in batch_dir.iterdir():
            if not dataset_dir.is_dir():
                continue

            # Iterate through experiments
            for exp_dir in dataset_dir.iterdir():
                if not exp_dir.is_dir():
                    continue

                # Check if it's a symbolic link pointing to target shared path
                if exp_dir.is_symlink():
                    target = exp_dir.resolve()
                    if target.parent == abs_shared_path.parent:
                        # Found it!
                        batch_path = exp_dir.relative_to(PROJECT_ROOT)
                        batches_using_this.append({
                            'batch_id': batch_dir.name,
                            'dataset': dataset_dir.name,
                            'experiment': exp_dir.name,
                            'batch_path': str(batch_path)
                        })

    if batches_using_this:
        print(f"\n✅ This data is referenced by the following {len(batches_using_this)} batches:")

        current_batch = None
        for item in batches_using_this:
            if item['batch_id'] != current_batch:
                current_batch = item['batch_id']
                print(f"\n   📂 {current_batch}")

            print(f"      └─ {item['dataset']} / {item['experiment']}")
            print(f"         Path: {item['batch_path']}/{item['dataset']}")

        print(f"\n💡 Training config can use any of the batch paths:")
        for item in batches_using_this[:3]:  # Show only first 3
            print(f'   path: "{item["batch_path"]}/{item["dataset"]}"')
        if len(batches_using_this) > 3:
            print(f"   ... and {len(batches_using_this) - 3} more")

    else:
        print(f"\n⚠️  This data is not referenced by any batch")
        print(f"   This may be an isolated experiment data")


def check_data_files(data_path):
    """Check if data files exist"""

    # Infer dataset name
    dataset_name = data_path.name

    # Possible filenames
    possible_files = [
        f"{dataset_name.lower()}_train.jsonl",
        f"{dataset_name.lower()}_validation.jsonl",
        f"{dataset_name.lower()}_test.jsonl",
        # ArcC special naming
        "ARC-Challenge_train.jsonl",
        "ARC-Challenge_validation.jsonl",
        "ARC-Challenge_test.jsonl",
    ]

    found_files = []
    for filename in possible_files:
        file_path = data_path / filename
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            found_files.append(f"{filename} ({size_mb:.2f} MB)")

    if found_files:
        print(f"\n📄 Data files:")
        for f in found_files:
            print(f"   ✓ {f}")
    else:
        print(f"\n⚠️  No data files found")


def main():
    parser = argparse.ArgumentParser(
        description='Resolve data path (batch ↔ shared)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Resolve batch path
  python resolve_data_path.py "Data_v2/synthetic/batch_20241229_temperature/Copa/temp07_topp10_gpt4o/Copa"

  # Resolve shared path, find all batches using it
  python resolve_data_path.py "Data_v2/synthetic/_shared/Copa/temp07_topp10_gpt4o/Copa"
        """
    )

    parser.add_argument('path', help='Path to resolve (batch path or shared path)')

    args = parser.parse_args()

    return resolve_path(args.path)


if __name__ == '__main__':
    sys.exit(main())
