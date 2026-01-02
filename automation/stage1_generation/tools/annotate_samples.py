#!/usr/bin/env python3
"""
Checkpoint 2 Tool - Process samples 21-80

Features:
Process 21-40 (Checkpoint 2A):
  1. Display all comparisons at once
  2. User inputs indices of unqualified samples
  3. Automatic rejection sampling (unqualified → replace with original data)
  4. Automatically generate validation few-shot examples from qualified samples (labeled as "same")

Process 41-80 (Checkpoint 2B):
  1. Display all comparisons at once
  2. User inputs indices of unqualified samples
  3. Automatic rejection sampling (unqualified → replace with original data)
  4. Automatic labeling: qualified → "same", unqualified → "not the same"
  5. Automatically generate test_set (for testing AI judge accuracy)

Usage:
  cd Data_v2/synthetic/_shared/{Dataset}/{experiment_dir}/scripts/

  # Process 21-40
  python /path/to/automation/stage1_generation/tools/annotate_samples.py --range 21-40

  # Process 41-80
  python /path/to/automation/stage1_generation/tools/annotate_samples.py --range 41-80
"""

import json
import argparse
import sys
from pathlib import Path
from typing import List, Dict, Set, Tuple


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


def parse_range(range_str: str) -> Tuple[int, int]:
    """Parse range string, e.g., '21-40' -> (20, 40)"""
    parts = range_str.split('-')
    if len(parts) != 2:
        raise ValueError(f"Invalid range format: {range_str}")

    start = int(parts[0]) - 1  # Convert to 0-based index
    end = int(parts[1])

    return start, end


def display_samples(
    original_data: List[Dict],
    rephrased_data: List[Dict],
    start_idx: int,
    end_idx: int,
    field_to_rephrase: str = "premise"
):
    """Display sample comparisons for the specified range at once"""
    print("\n" + "=" * 100)
    print(f"Samples {start_idx+1}-{end_idx} Comparison - Please carefully review original vs rephrased data")
    print("=" * 100)

    for i in range(start_idx, min(end_idx, len(original_data), len(rephrased_data))):
        original = original_data[i]
        rephrased = rephrased_data[i]

        print(f"\n[Sample {i+1}]")
        print(f"  Original {field_to_rephrase}:  {original.get(field_to_rephrase, 'N/A')}")
        print(f"  Rephrased {field_to_rephrase}:  {rephrased.get(field_to_rephrase, 'N/A')}")

        # Dynamically display other fields as context (excluding field_to_rephrase)
        for key in original:
            if key != field_to_rephrase and not key.startswith('_'):
                value = original[key]
                # Limit display length to avoid overly long output
                if isinstance(value, str) and len(value) > 100:
                    value = value[:100] + "..."
                print(f"  {key}: {value}")

    print("\n" + "=" * 100)


def get_rejected_indices(start_idx: int, end_idx: int) -> Set[int]:
    """Get user input for indices of unqualified samples"""
    print(f"\nPlease enter indices of unqualified samples ({start_idx+1}-{end_idx}), separated by commas")
    print(f"Example: {start_idx+3},{start_idx+7},{start_idx+15}  indicates these samples are unqualified")
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
                    if start_idx + 1 <= idx <= end_idx:
                        indices.add(idx - 1)  # Convert to 0-based index
                    else:
                        print(f"❌ Index {idx} is out of range ({start_idx+1}-{end_idx}), please re-enter")
                        indices = None
                        break

            if indices is not None:
                return indices

        except ValueError:
            print("❌ Input format error, please enter numbers separated by commas")


def perform_rejection_sampling(
    original_data: List[Dict],
    rephrased_data: List[Dict],
    rejected_indices: Set[int],
    start_idx: int,
    end_idx: int
) -> List[Dict]:
    """Perform rejection sampling: replace unqualified samples with original data"""
    result = []

    for i in range(start_idx, min(end_idx, len(original_data), len(rephrased_data))):
        if i in rejected_indices:
            # Unqualified: use original data
            result.append(original_data[i])
        else:
            # Qualified: use rephrased data
            result.append(rephrased_data[i])

    return result


def generate_validation_fewshot(
    original_data: List[Dict],
    rephrased_data: List[Dict],
    rejected_indices: Set[int],
    start_idx: int,
    end_idx: int,
    field_to_rephrase: str
) -> List[Dict]:
    """Generate validation few-shot examples from qualified samples 21-40"""
    examples = []

    for i in range(start_idx, min(end_idx, len(original_data), len(rephrased_data))):
        if i not in rejected_indices:
            # Qualified sample - mark as "same"
            example = {
                f'original_{field_to_rephrase}': original_data[i].get(field_to_rephrase, ''),
                f'rephrased_{field_to_rephrase}': rephrased_data[i].get(field_to_rephrase, ''),
                'evaluation': 'same'
            }
            # Preserve other fields
            for key in original_data[i]:
                if key != field_to_rephrase:
                    example[key] = original_data[i][key]

            examples.append(example)

    return examples


def generate_test_set(
    original_data: List[Dict],
    rephrased_data: List[Dict],
    rejected_indices: Set[int],
    start_idx: int,
    end_idx: int,
    field_to_rephrase: str
) -> List[Dict]:
    """Generate test_set: all samples 41-80 with ground_truth labels"""
    test_set = []

    for i in range(start_idx, min(end_idx, len(original_data), len(rephrased_data))):
        # Auto-label: unqualified → not the same, qualified → same
        ground_truth = 'not the same' if i in rejected_indices else 'same'

        test_item = {
            'index': i,
            f'original_{field_to_rephrase}': original_data[i].get(field_to_rephrase, ''),
            f'rephrased_{field_to_rephrase}': rephrased_data[i].get(field_to_rephrase, ''),
            'ground_truth': ground_truth
        }
        # Preserve other fields
        for key in original_data[i]:
            if key != field_to_rephrase:
                test_item[key] = original_data[i][key]

        test_set.append(test_item)

    return test_set


def main():
    """Main process"""
    parser = argparse.ArgumentParser(description='Checkpoint 2 Tool - Process samples 21-80')
    parser.add_argument(
        '--range',
        required=True,
        help='Processing range: 21-40 or 41-80'
    )
    args = parser.parse_args()

    # Parse range
    try:
        start_idx, end_idx = parse_range(args.range)
    except ValueError as e:
        print(f"❌ Error: {e}")
        return 1

    # Validate range
    if not ((start_idx == 20 and end_idx == 40) or (start_idx == 40 and end_idx == 80)):
        print(f"❌ Error: Only supports --range 21-40 or --range 41-80")
        return 1

    # Detect current directory
    current_dir = Path.cwd()

    # Try to read from config file
    config_file = current_dir.parent / "generation_config.yaml"
    if not config_file.exists():
        print(f"❌ Error: Config file does not exist: {config_file}")
        return 1

    # Read config
    import yaml
    with open(config_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    dataset_cfg = config.get('dataset', {})
    gen_cfg = config.get('generation', {})

    input_path = dataset_cfg.get('input_path', '')
    field_to_rephrase = gen_cfg.get('field_to_rephrase', 'premise')
    dataset_name = dataset_cfg.get('task_name', 'copa')

    # Parse original data path
    project_root = Path("/home/ubuntu/LLM-inference/jikai-project/Synthetic_Data_for_ZO")
    original_file = project_root / input_path

    # Find synthetic data file - should be already merged top20 and rest
    train_file = current_dir.parent / f"{dataset_name}_train.jsonl"

    # Validate file existence
    if not original_file.exists():
        print(f"❌ Error: Original data does not exist: {original_file}")
        return 1

    if not train_file.exists():
        print(f"❌ Error: Training data does not exist: {train_file}")
        print("   Please run rephrase_top20.py and rephrase_rest.py first, and merge data")
        return 1

    # Load data
    print(f"\nLoading data...")
    print(f"  Original: {original_file}")
    print(f"  Rephrased: {train_file}")

    original_data = load_jsonl(str(original_file))
    rephrased_data = load_jsonl(str(train_file))

    if len(rephrased_data) < end_idx:
        print(f"⚠️  Warning: Rephrased data has only {len(rephrased_data)} entries, less than {end_idx}")

    # Display samples
    display_samples(original_data, rephrased_data, start_idx, end_idx, field_to_rephrase)

    # Get unqualified sample indices
    rejected_indices = get_rejected_indices(start_idx, end_idx)

    total_count = end_idx - start_idx
    approved_count = total_count - len(rejected_indices)

    print(f"\nStatistics:")
    print(f"  Qualified samples: {approved_count}")
    print(f"  Unqualified samples: {len(rejected_indices)}")

    # Create output directory
    output_dir = current_dir.parent / "validation_checkpoints"
    output_dir.mkdir(exist_ok=True)

    # Perform different operations based on range
    if start_idx == 20 and end_idx == 40:
        # ========== 21-40 (Checkpoint 2A): Rejection Sampling + Validation Few-shot ==========
        print(f"\n[Checkpoint 2A: Process samples 21-40]")

        # 1. Perform rejection sampling
        print(f"\nPerforming rejection sampling...")
        final_data = perform_rejection_sampling(
            original_data, rephrased_data, rejected_indices, start_idx, end_idx
        )

        # Save data after rejection sampling
        output_file = output_dir / f"samples_{start_idx+1}_{end_idx}_validated.jsonl"
        save_jsonl(final_data, str(output_file))
        print(f"✓ Rejection sampling completed: {output_file}")

        # 2. Generate validation few-shot (only from qualified samples, labeled as "same")
        print(f"\nGenerating validation few-shot examples...")
        fewshot_examples = generate_validation_fewshot(
            original_data, rephrased_data, rejected_indices,
            start_idx, end_idx, field_to_rephrase
        )
        print(f"✓ Generated {len(fewshot_examples)} validation few-shot examples")

        # Save few-shot
        fewshot_file = output_dir / "validation_fewshot.json"
        with open(fewshot_file, 'w', encoding='utf-8') as f:
            json.dump({
                'field_to_rephrase': field_to_rephrase,
                'examples': fewshot_examples
            }, f, ensure_ascii=False, indent=2)
        print(f"✓ Validation few-shot saved: {fewshot_file}")

    elif start_idx == 40 and end_idx == 80:
        # ========== 41-80 (Checkpoint 2B): Rejection Sampling + Test Set ==========
        print(f"\n[Checkpoint 2B: Process samples 41-80]")

        # 1. Perform rejection sampling
        print(f"\nPerforming rejection sampling...")
        final_data = perform_rejection_sampling(
            original_data, rephrased_data, rejected_indices, start_idx, end_idx
        )

        # Save data after rejection sampling
        output_file = output_dir / f"samples_{start_idx+1}_{end_idx}_validated.jsonl"
        save_jsonl(final_data, str(output_file))
        print(f"✓ Rejection sampling completed: {output_file}")

        # 2. Generate test_set (auto-label: qualified → same, unqualified → not the same)
        print(f"\nGenerating test_set...")
        test_set = generate_test_set(
            original_data, rephrased_data, rejected_indices,
            start_idx, end_idx, field_to_rephrase
        )
        print(f"✓ Generated {len(test_set)} test samples")

        # Statistics for test_set
        same_count = sum(1 for item in test_set if item['ground_truth'] == 'same')
        not_same_count = len(test_set) - same_count

        print(f"  Ground Truth label statistics:")
        print(f"  - same (qualified): {same_count}")
        print(f"  - not the same (unqualified): {not_same_count}")

        # Save test_set
        test_file = output_dir / "validation_test_set.json"
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump({
                'field_to_rephrase': field_to_rephrase,
                'test_set': test_set
            }, f, ensure_ascii=False, indent=2)
        print(f"✓ Test set saved: {test_file}")
        print(f"  Purpose: Test the accuracy of AI judge validation prompt")

    # Save annotation record
    annotation_record = {
        'range': f'{start_idx+1}-{end_idx}',
        'total': total_count,
        'approved': approved_count,
        'rejected': len(rejected_indices),
        'rejected_indices': sorted(list(rejected_indices)),
        'field_to_rephrase': field_to_rephrase
    }

    record_file = output_dir / f"samples_{start_idx+1}_{end_idx}_annotation.json"
    with open(record_file, 'w', encoding='utf-8') as f:
        json.dump(annotation_record, f, ensure_ascii=False, indent=2)
    print(f"✓ Annotation record saved: {record_file}")

    print("\n" + "=" * 100)
    print(f"✅ Samples {start_idx+1}-{end_idx} processing completed!")
    print("=" * 100)

    if start_idx == 20 and end_idx == 40:
        print(f"\n✅ Checkpoint 2A completion summary:")
        print(f"  1. Rejection sampling: {approved_count}/{total_count} samples retained rephrased version")
        print(f"  2. Validation few-shot: generated {len(fewshot_examples) if 'fewshot_examples' in locals() else 0} examples")
        print(f"\nNext step:")
        print(f"  Run: python annotate_samples.py --range 41-80")
    elif start_idx == 40 and end_idx == 80:
        print(f"\n✅ Checkpoint 2B completion summary:")
        print(f"  1. Rejection sampling: {approved_count}/{total_count} samples retained rephrased version")
        print(f"  2. Test set: generated {len(test_set) if 'test_set' in locals() else 0} labeled samples")
        print(f"  3. Ground Truth: same={same_count}, not the same={not_same_count}")
        print(f"\nNext step:")
        print(f"  Use test_set to test validation prompt accuracy")
        print(f"  Run: python generate_validation_test.py")

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
