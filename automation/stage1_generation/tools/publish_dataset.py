#!/usr/bin/env python3
"""
Publish Dataset Tool - Publish validated data from Data_v2 to training script expected path

Features:
  1. Copy data from Data_v2/synthetic/{purpose}/{experiment_id}/{Dataset}/
  2. Publish to Data/synthetic/{Dataset}/ (compatible with training scripts)
  3. Optional: Archive different versions to subdirectory

Usage:
    # Basic usage: publish data to Data/synthetic/{Dataset}/
    python publish_dataset.py \
        --source Data_v2/synthetic/prompt_engineering/copa_mezo_v1/Copa \
        --dataset Copa

    # Specify publish directory
    python publish_dataset.py \
        --source Data_v2/synthetic/prompt_engineering/copa_mezo_v1/Copa \
        --dataset Copa \
        --target Data/rejection_sampling/0_data

    # Archive version simultaneously
    python publish_dataset.py \
        --source Data_v2/synthetic/prompt_engineering/copa_mezo_v1/Copa \
        --dataset Copa \
        --archive mezo_gpt/version_1

    # Use symbolic links (save space)
    python publish_dataset.py \
        --source Data_v2/synthetic/prompt_engineering/copa_mezo_v1/Copa \
        --dataset Copa \
        --symlink

Output:
    Data/synthetic/{Dataset}/
    ├── {dataset}_train.jsonl         # Copy or symbolic link
    ├── {dataset}_validation.jsonl
    └── {dataset}_test.jsonl
"""

import argparse
import json
import os
import shutil
from pathlib import Path
from datetime import datetime


def validate_source(source_path: str, dataset_name: str) -> bool:
    """Validate if source data directory contains necessary files

    Args:
        source_path: Source data directory
        dataset_name: Dataset name (e.g., Copa, CB, BOOLQ, etc.)

    Returns:
        True if valid, False otherwise
    """
    required_files = [
        f"{dataset_name.lower()}_train.jsonl",
        f"{dataset_name.lower()}_validation.jsonl",
    ]

    missing_files = []
    for file_name in required_files:
        file_path = os.path.join(source_path, file_name)
        if not os.path.exists(file_path):
            missing_files.append(file_name)

    if missing_files:
        print(f"❌ Source directory missing required files:")
        for f in missing_files:
            print(f"   - {f}")
        return False

    return True


def publish_files(
    source_dir: str,
    target_dir: str,
    dataset_name: str,
    use_symlink: bool = False,
    archive_path: str = None
):
    """Publish data files to target directory

    Args:
        source_dir: Source data directory
        target_dir: Target publish directory
        dataset_name: Dataset name
        use_symlink: Whether to use symbolic links
        archive_path: Optional archive subdirectory path
    """
    # Create target directory
    os.makedirs(target_dir, exist_ok=True)

    # Files to publish
    files_to_publish = [
        f"{dataset_name.lower()}_train.jsonl",
        f"{dataset_name.lower()}_validation.jsonl",
        f"{dataset_name.lower()}_test.jsonl",
    ]

    published_files = []

    for file_name in files_to_publish:
        source_file = os.path.join(source_dir, file_name)
        target_file = os.path.join(target_dir, file_name)

        # Check if source file exists
        if not os.path.exists(source_file):
            print(f"⚠️  Skipping non-existent file: {file_name}")
            continue

        # If target file already exists, remove it first
        if os.path.exists(target_file) or os.path.islink(target_file):
            os.remove(target_file)

        # Copy or create symbolic link
        if use_symlink:
            # Create symbolic link using absolute path
            abs_source = os.path.abspath(source_file)
            os.symlink(abs_source, target_file)
            print(f"🔗 Symbolic link: {file_name}")
        else:
            shutil.copy2(source_file, target_file)
            print(f"📄 Copied file: {file_name}")

        published_files.append(file_name)

    # If archive path is specified, also publish to archive directory
    if archive_path:
        archive_dir = os.path.join(target_dir, archive_path)
        os.makedirs(archive_dir, exist_ok=True)

        print(f"\n📦 Archiving to: {archive_path}")
        for file_name in published_files:
            source_file = os.path.join(source_dir, file_name)
            archive_file = os.path.join(archive_dir, file_name)

            # Archive directory always uses copy (not symbolic links)
            if os.path.exists(archive_file):
                os.remove(archive_file)

            shutil.copy2(source_file, archive_file)
            print(f"   ✓ {file_name}")

    return published_files


def create_metadata(target_dir: str, source_dir: str, dataset_name: str, args: argparse.Namespace):
    """Create metadata file to record publish information

    Args:
        target_dir: Target directory
        source_dir: Source directory
        dataset_name: Dataset name
        args: Command line arguments
    """
    metadata = {
        "dataset": dataset_name,
        "source": source_dir,
        "target": target_dir,
        "published_at": datetime.now().isoformat(),
        "use_symlink": args.symlink,
        "archive_path": args.archive,
    }

    metadata_file = os.path.join(target_dir, ".publish_metadata.json")
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\n📝 Metadata saved: .publish_metadata.json")


def main():
    parser = argparse.ArgumentParser(
        description='Publish validated dataset to training script expected path',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Publish to default path Data/synthetic/Copa/
  python publish_dataset.py \\
      --source Data_v2/synthetic/prompt_engineering/copa_mezo_v1/Copa \\
      --dataset Copa

  # Publish to rejection_sampling path (compatible with old training scripts)
  python publish_dataset.py \\
      --source Data_v2/synthetic/prompt_engineering/copa_mezo_v1/Copa \\
      --dataset Copa \\
      --target Data/rejection_sampling/0_data

  # Archive version simultaneously
  python publish_dataset.py \\
      --source Data_v2/synthetic/prompt_engineering/copa_mezo_v1/Copa \\
      --dataset Copa \\
      --archive mezo_gpt/version_1

  # Use symbolic links (save disk space)
  python publish_dataset.py \\
      --source Data_v2/synthetic/prompt_engineering/copa_mezo_v1/Copa \\
      --dataset Copa \\
      --symlink
        """
    )

    parser.add_argument('--source', required=True,
                       help='Source data directory (e.g., Data_v2/synthetic/prompt_engineering/copa_mezo_v1/Copa)')
    parser.add_argument('--dataset', required=True,
                       help='Dataset name (e.g., Copa, CB, BOOLQ, RTE)')
    parser.add_argument('--target', default='Data/synthetic',
                       help='Target publish root directory (default: Data/synthetic)')
    parser.add_argument('--archive', default=None,
                       help='Optional archive subdirectory path (e.g., mezo_gpt/version_1)')
    parser.add_argument('--symlink', action='store_true',
                       help='Use symbolic links instead of copying files (saves space)')
    parser.add_argument('--force', action='store_true',
                       help='Force overwrite existing files')

    args = parser.parse_args()

    # Build complete target path
    target_dir = os.path.join(args.target, args.dataset)

    print("\n" + "=" * 80)
    print("📦 Publish Dataset")
    print("=" * 80)
    print(f"Source directory:   {args.source}")
    print(f"Dataset:   {args.dataset}")
    print(f"Target directory: {target_dir}")
    if args.archive:
        print(f"Archive path: {args.archive}")
    print(f"Use symbolic links: {'Yes' if args.symlink else 'No'}")
    print("=" * 80 + "\n")

    # Validate source directory
    if not os.path.exists(args.source):
        print(f"❌ Source directory does not exist: {args.source}")
        return 1

    if not validate_source(args.source, args.dataset):
        return 1

    # Check if target directory already exists
    if os.path.exists(target_dir) and not args.force:
        print(f"⚠️  Target directory already exists: {target_dir}")
        choice = input("Overwrite? [y/N]: ").lower().strip()
        if choice != 'y':
            print("❌ Publish cancelled")
            return 0

    # Publish files
    try:
        published_files = publish_files(
            source_dir=args.source,
            target_dir=target_dir,
            dataset_name=args.dataset,
            use_symlink=args.symlink,
            archive_path=args.archive
        )

        # Create metadata
        create_metadata(target_dir, args.source, args.dataset, args)

        print("\n" + "=" * 80)
        print("✅ Publish completed!")
        print("=" * 80)
        print(f"Published {len(published_files)} files to: {target_dir}")
        print("\nTraining scripts can use the following path:")
        print(f"  TASK={os.path.abspath(target_dir)}")
        print()

        return 0

    except Exception as e:
        print(f"\n❌ Publish failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
