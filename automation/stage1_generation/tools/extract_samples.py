#!/usr/bin/env python3
"""
Extract samples in specified range for manual annotation

Usage:
    python extract_samples.py --range 21-80 --input Copa/copa_train.jsonl
    python extract_samples.py --range 21-40 --input Copa/copa_train.jsonl
    python extract_samples.py --range 41-80 --input Copa/copa_train.jsonl
"""

import argparse
import json
import os
from pathlib import Path


def parse_range(range_str):
    """Parse range string, e.g., '21-80' -> (20, 80)

    Args:
        range_str: Range string in 'start-end' format, using 1-based index

    Returns:
        (start_index, end_index): 0-based index, end_index is exclusive
    """
    parts = range_str.split('-')
    if len(parts) != 2:
        raise ValueError(f"Invalid range format: {range_str}, should be 'start-end'")

    start = int(parts[0])
    end = int(parts[1])

    if start < 1 or end < 1:
        raise ValueError("Range index must start from 1")
    if start > end:
        raise ValueError(f"Start index ({start}) cannot be greater than end index ({end})")

    # Convert to 0-based index
    return start - 1, end


def extract_samples(input_file, start_idx, end_idx, output_file):
    """Extract samples in specified range

    Args:
        input_file: Input file path
        start_idx: Start index (0-based, inclusive)
        end_idx: End index (0-based, exclusive)
        output_file: Output file path
    """
    # Read all data
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line.strip()))

    total_count = len(data)
    print(f"📊 Total samples: {total_count}")

    # Validate range
    if start_idx >= total_count:
        raise ValueError(f"Start index ({start_idx+1}) exceeds data range (1-{total_count})")
    if end_idx > total_count:
        print(f"⚠️  End index ({end_idx}) exceeds data range, adjusted to {total_count}")
        end_idx = total_count

    # Extract samples
    extracted = data[start_idx:end_idx]
    extracted_count = len(extracted)

    print(f"📤 Extracting samples: {start_idx+1}-{end_idx} (total {extracted_count})")

    # Create output directory
    output_dir = os.path.dirname(output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        for item in extracted:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')

    print(f"✅ Saved to: {output_file}")

    return extracted_count


def main():
    parser = argparse.ArgumentParser(description='Extract samples in specified range for manual annotation')
    parser.add_argument('--range', required=True,
                       help='Sample range (1-based index, e.g., 21-80)')
    parser.add_argument('--input', required=True,
                       help='Input file path (relative to current directory or absolute path)')
    parser.add_argument('--output', default=None,
                       help='Output file path (default: validation_checkpoints/samples_{range}.jsonl)')

    args = parser.parse_args()

    # Parse range
    try:
        start_idx, end_idx = parse_range(args.range)
    except ValueError as e:
        print(f"❌ Error: {e}")
        return 1

    # Determine output file
    if args.output:
        output_file = args.output
    else:
        # Default output to validation_checkpoints directory
        range_str = args.range.replace('-', '_')
        output_file = f"validation_checkpoints/samples_{range_str}.jsonl"

    # Determine input file
    input_file = args.input
    if not os.path.isabs(input_file):
        # If relative path, try to find from current directory
        if not os.path.exists(input_file):
            # Try to find from parent directory of script location
            script_dir = Path(__file__).parent.parent
            potential_path = script_dir / input_file
            if potential_path.exists():
                input_file = str(potential_path)

    if not os.path.exists(input_file):
        print(f"❌ Input file does not exist: {input_file}")
        return 1

    print(f"\n{'='*60}")
    print(f"🔍 Extract Samples Tool")
    print(f"{'='*60}")
    print(f"Input file: {input_file}")
    print(f"Extract range: {start_idx+1}-{end_idx} (1-based)")
    print(f"Output file: {output_file}")
    print(f"{'='*60}\n")

    try:
        count = extract_samples(input_file, start_idx, end_idx, output_file)
        print(f"\n✅ Successfully extracted {count} samples!")
        return 0
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
