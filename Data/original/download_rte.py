from datasets import load_dataset
import json
import os

# Load the dataset
dataset = load_dataset("super_glue", "rte")

# Define the output directory
output_dir = "/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/RTE"
os.makedirs(output_dir, exist_ok=True)

# Access the training set, validation set, and test set
train_data = dataset['train']
val_data = dataset['validation']
test_data = dataset['test']

# Print a sample from the training set
print(train_data[0])

# Save the training set as a JSON file
train_data.to_json("/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/RTE/rte_train.jsonl")

# Save the validation set and test set as JSON files
val_data.to_json("/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/RTE/rte_validation.jsonl")
test_data.to_json("/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/RTE/rte_test.jsonl")