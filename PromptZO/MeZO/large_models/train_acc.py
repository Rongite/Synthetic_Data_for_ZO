import logging
import os
import json
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Define task-specific logic for input preparation and accuracy calculation
def prepare_inputs_and_labels(example, task):
    """
    Prepare task-specific input prompts and correct labels for evaluation.
    """
    if task == "copa":
        premise = example["premise"]
        question = example["question"]
        choices = [example["choice1"], example["choice2"]]
        inputs = [
            f"Premise: {premise}\nQuestion: {question}\nAnswer: {choice}" for choice in choices
        ]
        correct_label = example["label"]  # Index of the correct choice
        return inputs, correct_label
    elif task == "cb":
        premise = example["premise"]
        hypothesis = example["hypothesis"]
        inputs = [f"Premise: {premise}\nHypothesis: {hypothesis}"]
        correct_label = example["label"]
        return inputs, correct_label
    elif task == "wsc":
        text = example["text"]
        inputs = [text]
        correct_label = example["label"]
        return inputs, correct_label
    else:
        raise ValueError(f"Unsupported task: {task}")

def evaluate_model(model, tokenizer, dataset, task, device="cuda", max_new_tokens=50):
    """
    Evaluate the model on the dataset and compute accuracy.
    """
    logger.info(f"Evaluating model on task: {task}")
    model.eval()
    model.to(device)
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    correct = 0
    total = 0

    for example in tqdm(dataset):
        inputs, correct_label = prepare_inputs_and_labels(example, task)

        # Tokenize inputs
        encoded_inputs = tokenizer(
            inputs,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)

        # Generate predictions
        outputs = model.generate(
            input_ids=encoded_inputs["input_ids"],
            attention_mask=encoded_inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)

        # Task-specific accuracy calculation
        if task == "copa":
            # Compare predictions with choices
            predicted_choice = predictions[0].strip()
            if predicted_choice in inputs:
                predicted_label = inputs.index(predicted_choice)
            else:
                predicted_label = -1  # Invalid prediction
            correct += int(predicted_label == correct_label)
        elif task in ["cb", "wsc"]:
            predicted_label = predictions[0].strip()
            correct += int(predicted_label.lower() == correct_label.lower())

        total += 1

    accuracy = correct / total if total > 0 else 0
    logger.info(f"Accuracy for task {task}: {accuracy * 100:.2f}%")
    return accuracy

def load_local_dataset(dataset_path, split):
    """
    Load a local dataset file for the specified split.
    """
    file_name = None
    for file in os.listdir(dataset_path):
        if file.endswith(f"_{split}.jsonl"):
            file_name = os.path.join(dataset_path, file)
            break

    if not file_name or not os.path.exists(file_name):
        raise FileNotFoundError(f"No matching file for split '{split}' in {dataset_path}")

    with open(file_name, "r", encoding="utf-8") as f:
        return [json.loads(line.strip()) for line in f]


def determine_task_from_dataset(dataset_path):
    """
    Infer task type based on file naming convention in the dataset folder.
    """
    for file in os.listdir(dataset_path):
        if "copa_" in file.lower():
            return "copa"
        elif "cb_" in file.lower():
            return "cb"
        elif "wsc_" in file.lower():
            return "wsc"

    raise ValueError("Unable to determine task type from dataset file names.")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument("--model", required=True, help="Path to the model or Hugging Face model name.")
    parser.add_argument("--dataset", required=True, help="Path to the dataset folder.")
    parser.add_argument("--split", default="validation", choices=["train", "validation", "test"],
                        help="Specify the dataset split to evaluate.")
    args = parser.parse_args()

    # Load model and tokenizer
    logger.info(f"Loading model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(args.model)

    # Load dataset
    dataset_path = args.dataset
    task = determine_task_from_dataset(dataset_path)
    dataset = load_local_dataset(dataset_path, args.split)

    # Evaluate model
    evaluate_model(model, tokenizer, dataset, task)


if __name__ == "__main__":
    main()
