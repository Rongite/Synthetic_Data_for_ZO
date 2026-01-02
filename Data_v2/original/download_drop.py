from datasets import load_dataset
import json
import os

# Load the DROP dataset
dataset = load_dataset("drop")

# Define the output directory
output_dir = "/home/jlong1/Downloads/Data/zo/0_original_data/DROP"
os.makedirs(output_dir, exist_ok=True)

# Function to save data in JSONL format
def save_to_jsonl(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        for sample in data:
            json.dump(sample, f, ensure_ascii=False)
            f.write("\n")

# Save the training set
if "train" in dataset:
    train_file = os.path.join(output_dir, "drop_train.jsonl")
    save_to_jsonl(dataset["train"], train_file)
    print(f"Training set saved to: {train_file}")

# Save the validation set
if "validation" in dataset:
    validation_file = os.path.join(output_dir, "drop_validation.jsonl")
    save_to_jsonl(dataset["validation"], validation_file)
    print(f"Validation set saved to: {validation_file}")
