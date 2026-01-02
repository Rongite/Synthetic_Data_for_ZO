#!/usr/bin/env python3
"""
Batch fix hardcoded path issues

Automatically replace hardcoded PROJECT_ROOT in all files with config.py import
"""

import re
from pathlib import Path

# List of files to fix
FILES_TO_FIX = [
    # stage1_generation tools
    "stage1_generation/tools/review_top20.py",
    "stage1_generation/tools/annotate_samples.py",
    "stage1_generation/tools/extract_samples.py",
    "stage1_generation/tools/publish_dataset.py",

    # stage1_generation batch_tools
    "stage1_generation/batch_tools/compare_experiments.py",
    "stage1_generation/batch_tools/list_batch_experiments.py",
    "stage1_generation/batch_tools/list_batches.py",
    "stage1_generation/batch_tools/list_shared_experiments.py",

    # stage1_generation main scripts
    "stage1_generation/list_experiments.py",

    # stage2_training
    "stage2_training/list_results.py",
]

# Hardcoded path pattern
HARDCODED_PATTERN = r'PROJECT_ROOT = "/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO"'

# Replace with import statement
IMPORT_REPLACEMENT = """# Import unified configuration
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import PROJECT_ROOT"""


def fix_file(file_path: Path):
    """Fix hardcoded paths in a single file"""

    if not file_path.exists():
        print(f"⚠️  Skipped (file does not exist): {file_path}")
        return False

    # Read file content
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Check if file contains hardcoded path
    if not re.search(HARDCODED_PATTERN, content):
        print(f"✓ Already fixed or no fix needed: {file_path.name}")
        return False

    # Check if config import already exists
    if 'from config import' in content:
        print(f"✓ Already fixed: {file_path.name}")
        return False

    # Replace hardcoded path
    new_content = re.sub(HARDCODED_PATTERN, IMPORT_REPLACEMENT, content)

    # Check if Path import statement exists, avoid duplicate imports
    if 'from pathlib import Path' in content and 'from pathlib import Path' in IMPORT_REPLACEMENT:
        # Remove duplicate Path import
        new_content = new_content.replace(IMPORT_REPLACEMENT,
            IMPORT_REPLACEMENT.replace('\nfrom pathlib import Path', ''))

    # Backup original file
    backup_path = file_path.with_suffix(file_path.suffix + '.backup')
    with open(backup_path, 'w', encoding='utf-8') as f:
        f.write(content)

    # Write new content
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(new_content)

    print(f"✅ Fix completed: {file_path.name} (backup: {backup_path.name})")
    return True


def main():
    """Main function"""

    automation_dir = Path(__file__).parent

    print("=" * 80)
    print("🔧 Batch fix hardcoded paths")
    print("=" * 80)
    print()

    fixed_count = 0
    total_count = len(FILES_TO_FIX)

    for relative_path in FILES_TO_FIX:
        file_path = automation_dir / relative_path
        if fix_file(file_path):
            fixed_count += 1

    print()
    print("=" * 80)
    print(f"✅ Done! Fixed {fixed_count}/{total_count} files")
    print("=" * 80)

    if fixed_count > 0:
        print()
        print("💡 Tips:")
        print("   1. Original files have been backed up as *.backup")
        print("   2. Please test if modified scripts work correctly")
        print("   3. Use backup files to rollback if needed")


if __name__ == '__main__':
    main()
