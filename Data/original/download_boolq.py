from datasets import load_dataset
import json
import os

# Load the dataset
dataset = load_dataset("boolq")

# Define the output directory
output_dir = "/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/BOOLQ"
os.makedirs(output_dir, exist_ok=True)

# Access the training set, validation set, and test set
train_data = dataset['train']
val_data = dataset['validation']

# Save the training set as a JSON file
train_data.to_json("/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/BOOLQ/boolq_train.jsonl")

# Save the validation set as JSON files
val_data.to_json("/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/BOOLQ/boolq_validation.jsonl")
