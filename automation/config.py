#!/usr/bin/env python3
"""
Global Configuration - Centralized Project Path Management

Usage:
    from config import PROJECT_ROOT, SYNTHETIC_BASE, RESULTS_BASE
"""

import os
from pathlib import Path

# Method 1: Get from environment variable (recommended)
# User can set: export SYNTHETIC_DATA_PROJECT_ROOT="/path/to/project"
_env_root = os.environ.get('SYNTHETIC_DATA_PROJECT_ROOT')

if _env_root:
    PROJECT_ROOT = Path(_env_root)
else:
    # Method 2: Auto-detect (fallback)
    # Find project root directory from current file location
    current_file = Path(__file__).resolve()

    # automation/config.py → Synthetic_Data_for_ZO/
    PROJECT_ROOT = current_file.parent.parent

    # Verify if correct project root directory is found
    # Check for landmark files/directories
    if not (PROJECT_ROOT / "PromptZO").exists():
        # If not found, try hardcoded path (last fallback)
        PROJECT_ROOT = Path("/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO")

# Common subdirectories
AUTOMATION_DIR = PROJECT_ROOT / "automation"
DATA_DIR = PROJECT_ROOT / "Data"
DATA_V2_DIR = PROJECT_ROOT / "Data_v2"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_V2_DIR = PROJECT_ROOT / "Results_v2"

# Stage 1 related
SYNTHETIC_BASE = DATA_V2_DIR / "synthetic"
ORIGINAL_DATA_DIR = DATA_DIR / "original"

# Stage 2 related
TRAINING_SCRIPTS_DIR = PROJECT_ROOT / "PromptZO" / "MeZO" / "large_models"

# Print configuration (for debugging)
def print_config():
    """Print current configuration"""
    print("=" * 80)
    print("🔧 Project Configuration")
    print("=" * 80)
    print(f"PROJECT_ROOT:         {PROJECT_ROOT}")
    print(f"AUTOMATION_DIR:       {AUTOMATION_DIR}")
    print(f"DATA_DIR:             {DATA_DIR}")
    print(f"DATA_V2_DIR:          {DATA_V2_DIR}")
    print(f"RESULTS_DIR:          {RESULTS_DIR}")
    print(f"RESULTS_V2_DIR:       {RESULTS_V2_DIR}")
    print(f"SYNTHETIC_BASE:       {SYNTHETIC_BASE}")
    print(f"TRAINING_SCRIPTS_DIR: {TRAINING_SCRIPTS_DIR}")
    print("=" * 80)

    # Verify paths
    missing = []
    for name, path in [
        ("PROJECT_ROOT", PROJECT_ROOT),
        ("PromptZO", PROJECT_ROOT / "PromptZO"),
        ("automation", AUTOMATION_DIR),
    ]:
        if not path.exists():
            missing.append(name)

    if missing:
        print(f"⚠️  Warning: Following paths do not exist: {', '.join(missing)}")
        print(f"   Please check if PROJECT_ROOT is set correctly")
    else:
        print("✅ All critical paths verified successfully")
    print("=" * 80)


if __name__ == '__main__':
    print_config()
