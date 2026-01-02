#!/usr/bin/env python3
"""
Checkpoint 1 Tool - Review first 20 samples and automate processing

Features:
1. Display all 20 original vs rephrased comparisons at once
2. User inputs indices of unqualified samples (comma-separated)
3. Automatic rejection sampling (unqualified → replace with original data)
4. Automatically generate rephrase few-shot examples from qualified samples
5. Automatically inject few-shot into rephrase_rest.py

Usage:
  cd Data_v2/synthetic/_shared/{Dataset}/{experiment_dir}/scripts/
  python /path/to/automation/stage1_generation/tools/review_top20.py
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Set


def load_jsonl(file_path: str) -> List[Dict]:
    """Load JSONL file"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def save_jsonl(data: List[Dict], file_path: str):
    """Save JSONL file"""
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def display_all_samples(original_data: List[Dict], rephrased_data: List[Dict], field_to_rephrase: str = "premise"):
    """Display all 20 sample comparisons at once"""
    print("\n" + "=" * 100)
    print("First 20 samples comparison - Please carefully review original vs rephrased data")
    print("=" * 100)

    for i in range(min(len(original_data), len(rephrased_data))):
        original = original_data[i]
        rephrased = rephrased_data[i]

        print(f"\n[Sample {i+1}]")
        print(f"  Original {field_to_rephrase}:  {original.get(field_to_rephrase, 'N/A')}")
        print(f"  Rephrased {field_to_rephrase}:  {rephrased.get(field_to_rephrase, 'N/A')}")

        # Display other fields as context
        if "choice1" in original:
            print(f"  Choice1: {original['choice1']}")
        if "choice2" in original:
            print(f"  Choice2: {original['choice2']}")
        if "question" in original:
            print(f"  Question: {original['question']}")

    print("\n" + "=" * 100)


def get_rejected_indices() -> Set[int]:
    """Get user input for indices of unqualified samples"""
    print("\nPlease enter indices of unqualified samples (1-20), separated by commas")
    print("Example: 3,7,15  indicates samples 3, 7, 15 are unqualified")
    print("If all are qualified, press Enter directly")

    while True:
        user_input = input("\nUnqualified sample indices: ").strip()

        if not user_input:
            return set()

        try:
            indices = set()
            for part in user_input.split(','):
                part = part.strip()
                if part:
                    idx = int(part)
                    if 1 <= idx <= 20:
                        indices.add(idx - 1)  # Convert to 0-based index
                    else:
                        print(f"❌ Index {idx} is out of range (1-20), please re-enter")
                        indices = None
                        break

            if indices is not None:
                return indices

        except ValueError:
            print("❌ Input format error, please enter numbers separated by commas")


def perform_rejection_sampling(
    original_data: List[Dict],
    rephrased_data: List[Dict],
    rejected_indices: Set[int]
) -> List[Dict]:
    """Perform rejection sampling: replace unqualified samples with original data"""
    result = []

    for i in range(min(len(original_data), len(rephrased_data))):
        if i in rejected_indices:
            # Unqualified: use original data
            result.append(original_data[i])
        else:
            # Qualified: use rephrased data
            result.append(rephrased_data[i])

    return result


def generate_fewshot_examples(
    original_data: List[Dict],
    rephrased_data: List[Dict],
    rejected_indices: Set[int],
    field_to_rephrase: str,
    max_examples: int = 15
) -> List[Dict]:
    """Generate few-shot examples from qualified samples"""
    examples = []

    for i in range(min(len(original_data), len(rephrased_data))):
        if i not in rejected_indices:
            # Qualified sample
            example = {
                'original': original_data[i].get(field_to_rephrase, ''),
                'rephrased': rephrased_data[i].get(field_to_rephrase, '')
            }
            # Preserve other fields as context
            for key in original_data[i]:
                if key != field_to_rephrase:
                    example[key] = original_data[i][key]

            examples.append(example)

            if len(examples) >= max_examples:
                break

    return examples


def inject_fewshot_to_rephrase_rest(
    script_path: Path,
    fewshot_examples: List[Dict],
    field_to_rephrase: str
):
    """Inject few-shot examples into rephrase_rest.py"""
    if not script_path.exists():
        print(f"⚠️  Warning: {script_path} does not exist, skipping injection")
        return

    # Read script content
    with open(script_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Generate few-shot text
    fewshot_text = "# Few-shot examples from approved top20 samples\nFEWSHOT_EXAMPLES = [\n"
    for ex in fewshot_examples:
        fewshot_text += "    {\n"
        fewshot_text += f"        'original': {json.dumps(ex['original'], ensure_ascii=False)},\n"
        fewshot_text += f"        'rephrased': {json.dumps(ex['rephrased'], ensure_ascii=False)},\n"
        # Add other fields
        for key in ex:
            if key not in ['original', 'rephrased']:
                fewshot_text += f"        {json.dumps(key)}: {json.dumps(ex[key], ensure_ascii=False)},\n"
        fewshot_text += "    },\n"
    fewshot_text += "]\n"

    # Find injection position (before prompt generation function)
    # Simple method: inject at beginning of file
    lines = content.split('\n')

    # Find where import statements end
    insert_line = 0
    for i, line in enumerate(lines):
        if line.startswith('import ') or line.startswith('from '):
            insert_line = i + 1

    # Insert few-shot
    lines.insert(insert_line + 1, '\n' + fewshot_text)

    # Save
    backup_path = script_path.with_suffix('.py.backup')
    script_path.rename(backup_path)

    with open(script_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"✓ Few-shot examples injected into: {script_path}")
    print(f"  Backup saved: {backup_path}")


def main():
    """Main process"""
    # Detect current directory
    current_dir = Path.cwd()

    # Read config to get dataset_name
    config_file = current_dir.parent / "generation_config.yaml"
    if not config_file.exists():
        print(f"❌ Error: Config file does not exist: {config_file}")
        print("   Please ensure you run this script in the correct directory")
        return 1

    # Read config
    import yaml
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    dataset_cfg = config.get('dataset', {})
    gen_cfg = config.get('generation', {})

    dataset_name = dataset_cfg.get('task_name', 'dataset')
    dataset_display_name = dataset_cfg.get('dataset_name', dataset_name.capitalize())

    # Find necessary files (using dynamic dataset_name)
    rephrased_file = current_dir.parent / dataset_display_name / f"{dataset_name}_train_top20.jsonl"

    # Read original data path and field_to_rephrase from config
    input_path = dataset_cfg.get('input_path', '')
    field_to_rephrase = gen_cfg.get('field_to_rephrase', 'premise')

    if not input_path:
        print("❌ Error: Missing dataset.input_path in config file")
        return 1

    # Parse original data path
    project_root = Path("/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO")
    original_file = project_root / input_path

    # Validate file existence
    if not original_file.exists():
        print(f"❌ Error: Original data does not exist: {original_file}")
        return 1

    if not rephrased_file.exists():
        print(f"❌ Error: Rephrased data does not exist: {rephrased_file}")
        print("   Please run rephrase_top20.py first")
        return 1

    # Load data
    print(f"\nLoading data...")
    print(f"  Original: {original_file}")
    print(f"  Rephrased: {rephrased_file}")

    original_data = load_jsonl(str(original_file))[:20]
    rephrased_data = load_jsonl(str(rephrased_file))[:20]

    if len(rephrased_data) < 20:
        print(f"⚠️  Warning: Rephrased data has only {len(rephrased_data)} entries")

    # Display all samples
    display_all_samples(original_data, rephrased_data, field_to_rephrase)

    # Get unqualified sample indices
    rejected_indices = get_rejected_indices()

    approved_count = 20 - len(rejected_indices)
    print(f"\nStatistics:")
    print(f"  Qualified samples: {approved_count}")
    print(f"  Unqualified samples: {len(rejected_indices)}")

    if approved_count < 10:
        print(f"\n⚠️  Warning: Only {approved_count} qualified samples, may be insufficient for high-quality few-shot")
        proceed = input("Continue? (y/N): ").lower().strip()
        if proceed != 'y':
            print("Cancelled")
            return 0

    # Perform rejection sampling
    print(f"\nPerforming rejection sampling...")
    final_data = perform_rejection_sampling(original_data, rephrased_data, rejected_indices)

    # Save results
    output_dir = current_dir.parent / "validation_checkpoints"
    output_dir.mkdir(exist_ok=True)

    # Save rejection sampling data to dataset subdirectory
    dataset_output_dir = current_dir.parent / dataset_display_name
    output_file = dataset_output_dir / f"{dataset_name}_train_top20_validated.jsonl"
    save_jsonl(final_data, str(output_file))
    print(f"✓ Rejection sampling completed: {output_file}")

    # Generate few-shot examples
    print(f"\nGenerating few-shot examples...")
    fewshot_examples = generate_fewshot_examples(
        original_data, rephrased_data, rejected_indices, field_to_rephrase
    )
    print(f"✓ Generated {len(fewshot_examples)} few-shot examples")

    # Save few-shot to checkpoint
    fewshot_file = output_dir / "top20_fewshot.json"
    with open(fewshot_file, 'w', encoding='utf-8') as f:
        json.dump({
            'field_to_rephrase': field_to_rephrase,
            'examples': fewshot_examples
        }, f, ensure_ascii=False, indent=2)
    print(f"✓ Few-shot saved: {fewshot_file}")

    # Inject into rephrase_rest.py
    rephrase_rest_script = current_dir / "rephrase_rest.py"
    if rephrase_rest_script.exists():
        print(f"\nInjecting few-shot into rephrase_rest.py...")
        inject_fewshot_to_rephrase_rest(
            rephrase_rest_script, fewshot_examples, field_to_rephrase
        )
    else:
        print(f"\n⚠️  {rephrase_rest_script} does not exist, skipping injection")

    # Save review record
    review_record = {
        'total': 20,
        'approved': approved_count,
        'rejected': len(rejected_indices),
        'rejected_indices': sorted(list(rejected_indices)),
        'field_to_rephrase': field_to_rephrase
    }

    review_file = output_dir / "top20_review.json"
    with open(review_file, 'w', encoding='utf-8') as f:
        json.dump(review_record, f, ensure_ascii=False, indent=2)
    print(f"✓ Review record saved: {review_file}")

    print("\n" + "=" * 100)
    print("✅ Checkpoint 1 processing completed!")
    print("=" * 100)
    print(f"\nNext step:")
    print(f"  Run rephrase_rest.py to generate remaining data")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("\n\nUser interrupted")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
