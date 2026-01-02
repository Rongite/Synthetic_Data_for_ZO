# Validated Prompt Template Library

This directory stores prompt configurations that have been fully manually validated and can be reused for parameter tuning experiments.

## What is a "Validated" Template?

Validated templates must meet the following conditions:

1. ✅ **Checkpoint 1 Passed**: The first 20 samples have been manually reviewed, and approved samples have been made into few-shot examples for rephrase_prompt
2. ✅ **Checkpoint 2 Passed**: Samples 21-80 have been manually annotated, and few-shot examples and test_set for validation_prompt have been automatically generated
3. ✅ **Checkpoint 3 Passed**: validation prompt achieves accuracy ≥95% on test_set
4. ✅ **Batch Validation Complete**: All samples have undergone rejection sampling using the validated prompt

## Template File Format

```yaml
# Example: copa_mezo_validated.yaml
task_name: "Copa"
training_method: "mezo"
version: "v1"
status: "validated"  # Marked as validated

# Validation history (automatically recorded)
validation_history:
  validated_at: "2024-12-24 15:30:00"
  top20_review:
    total: 20
    approved: 18
    rejected: 2
    review_file: "Data_v2/synthetic/Copa_mezo_gpt4o_v1/validation_checkpoints/top20_review.json"
  samples_21_80_annotation:
    total: 60
    same: 57
    not_the_same: 3
    annotation_file: "Data_v2/synthetic/Copa_mezo_gpt4o_v1/validation_checkpoints/samples_21_80_annotated.json"
  validation_prompt_test:
    accuracy: 0.975
    passed: true
    test_file: "Data_v2/synthetic/Copa_mezo_gpt4o_v1/validation_checkpoints/prompt_test_results.json"

generation:
  model: "gpt-4o"
  temperature: 0.5
  field_to_rephrase: "premise"

  # Contains few-shot examples generated from manual review
  rephrase_prompt: |
    You are tasked with rephrasing...

    ### Few-shot Examples:
    Example 1:
    Original premise: The man broke his toe.
    Rephrased premise: The man sustained a toe fracture.
    ...
    (From samples approved in checkpoint 1 manual review)

validation:
  model: "gpt-4o"

  # Contains few-shot examples generated from manual annotation
  validation_prompt: |
    Judge if the rephrased premise...

    ### Few-shot Examples:
    Example 1:
    Original: The tenant misplaced his keys.
    Rephrased: The tenant lost his apartment keys.
    Judgment: same
    ...
    (From annotated samples 21-40 in checkpoint 2)

  # Test set (samples 41-80)
  test_set:
    - original_premise: "..."
      rephrased_premise: "..."
      ground_truth: "same"
    ...
```

## Using Templates to Create Parameter Tuning Experiments

```bash
# Create experiment configuration based on template
python automation/stage1_generation/create_experiment.py \
       --template automation/configs/stage1/templates/copa_mezo_validated.yaml \
       --version v2 \
       --param generation.temperature=0.7

# Generate scripts
python automation/stage1_generation/generator.py \
       automation/configs/stage1/experiments/copa_mezo_v2_temperature07.yaml

# Directly generate complete dataset (no manual review needed)
cd Data_v2/synthetic/Copa_mezo_gpt4o_v2/scripts/
python rephrase_all.py

# Validate
python validate.py
```

## Template Naming Convention

Format: `{task}_{method}_validated.yaml`

Examples:
- `copa_mezo_validated.yaml` - Copa task, MeZO method
- `rte_mezo_validated.yaml` - RTE task, MeZO method
- `boolq_lora_validated.yaml` - BOOLQ task, LoRA method

## Existing Templates

(To be added)

- [ ] copa_mezo_validated.yaml
- [ ] rte_mezo_validated.yaml
- [ ] cb_mezo_validated.yaml
- [ ] boolq_mezo_validated.yaml
- [ ] arcc_mezo_validated.yaml

## Important Notes

1. ⚠️ **Do not directly modify template files**
   - Templates are the "gold standard" that have been manually validated
   - If you need to adjust the prompt, create a new draft configuration and re-validate

2. ⚠️ **Templates are only for parameter tuning experiments**
   - After inheriting a template, you can only modify generation parameters (temperature, model, etc.)
   - Cannot modify prompt content
   - If you need to modify the prompt, you must go through the complete validation process

3. ✅ **Parameter tuning experiments can skip manual checkpoints**
   - Because the prompt has already been validated
   - Directly use rephrase_all.py to generate the complete dataset
   - Use the validated validation prompt to verify data
