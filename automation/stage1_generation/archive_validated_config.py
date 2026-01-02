#!/usr/bin/env python3
"""Archive validated configuration as template"""
import argparse, sys, yaml
from pathlib import Path
from datetime import datetime

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True)
    parser.add_argument('--data-dir', required=True)
    args = parser.parse_args()

    config_path, data_dir = Path(args.config), Path(args.data_dir)
    if not config_path.exists() or not data_dir.exists():
        print("❌ Config or data directory does not exist"); return 1

    flag_file = data_dir / "validation_checkpoints/prompt_test_passed.flag"
    if not flag_file.exists():
        print("❌ Validation passed flag not found"); return 1

    with open(config_path) as f: config = yaml.safe_load(f)
    config.setdefault('experiment', {}).update({'status': 'validated', 'validated_at': datetime.now().isoformat()})

    templates_dir = Path("automation/configs/stage1/templates")
    templates_dir.mkdir(parents=True, exist_ok=True)
    template_path = templates_dir / f"{config.get('task_name','task').lower()}_{config.get('training_method','method')}_validated.yaml"
    with open(template_path, 'w') as f: yaml.dump(config, f, allow_unicode=True)

    archive_dir = Path("automation/configs/stage1/archive") / datetime.now().strftime("%Y-%m")
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / f"{config.get('task_name','task').lower()}_{config.get('training_method','method')}_{config.get('version','v1')}_complete_{datetime.now().strftime('%Y%m%d_%H%M%S')}.yaml"
    with open(archive_path, 'w') as f: yaml.dump(config, f, allow_unicode=True)

    print(f"✅ Template: {template_path}\n✅ Archive: {archive_path}")
    return 0

if __name__ == "__main__":
    sys.exit(main())
