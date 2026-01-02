#!/usr/bin/env python3
"""
Create tuning experiment configuration based on validated template

Usage:
  python automation/stage1_generation/create_experiment.py \
      --template automation/configs/stage1/templates/copa_mezo_validated.yaml \
      --param generation.temperature=0.9 \
      --param generation.top_p=0.95 \
      --batch-id batch_20241230_temperature
"""
import argparse, sys, yaml
from pathlib import Path
from datetime import datetime

def main():
    parser = argparse.ArgumentParser(description='Create tuning experiment configuration')
    parser.add_argument('--template', required=True, help='Validated template path')
    parser.add_argument('--param', action='append', help='Parameter modification (format: key=value)')
    parser.add_argument('--batch-id', help='Batch ID')
    args = parser.parse_args()

    template_path = Path(args.template)
    if not template_path.exists():
        print(f"❌ Template does not exist: {template_path}"); return 1

    with open(template_path) as f: config = yaml.safe_load(f)

    if config.get('experiment', {}).get('status') != 'validated':
        print("❌ Template not marked as validated"); return 1

    # Apply parameter modifications
    changes = []
    if args.param:
        for param in args.param:
            key, value = param.split('=', 1)
            keys = key.split('.')
            target = config
            for k in keys[:-1]:
                target = target.setdefault(k, {})
            old_value = target.get(keys[-1], 'N/A')
            try:
                target[keys[-1]] = float(value) if '.' in value else int(value) if value.isdigit() else value
            except:
                target[keys[-1]] = value
            changes.append(f"  {key}: {old_value} → {value}")

    # Update batch_id
    if args.batch_id:
        config.setdefault('experiment', {})['batch_id'] = args.batch_id

    # Update status
    config['experiment']['status'] = 'experiment'
    config['experiment']['created_from_template'] = str(template_path)
    config['experiment']['created_at'] = datetime.now().isoformat()
    if changes:
        config['experiment']['parameter_changes'] = '\n'.join(changes)

    # Generate output filename
    task = config.get('task_name', 'task').lower()
    method = config.get('training_method', 'method')

    # Generate descriptive name
    param_desc = '_'.join([p.split('=')[0].split('.')[-1] + p.split('=')[1].replace('.', '') for p in (args.param or [])])
    output_name = f"{task}_{method}_{param_desc}.yaml" if param_desc else f"{task}_{method}_experiment.yaml"

    output_dir = Path("automation/configs/stage1/experiments")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / output_name

    with open(output_path, 'w') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False)

    print(f"✅ Experiment configuration created: {output_path}")
    if changes:
        print("\nParameter changes:\n" + '\n'.join(changes))
    print("\nNext step: python automation/stage1_generation/generator.py " + str(output_path))

    return 0

if __name__ == "__main__":
    sys.exit(main())
