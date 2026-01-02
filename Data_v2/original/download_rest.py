from datasets import load_dataset
import json
import os

# Load the dataset
dataset = load_dataset("super_glue", "wic")

# Access the training set, validation set, and test set
train_data = dataset['train']
val_data = dataset['validation']
test_data = dataset['test']

# Print a sample from the training set
print(train_data[0])

# Save the training set as a JSON file
train_data.to_json("/home/jlong1/Downloads/Data/zo/0_original_data/WIC/wic_train.jsonl")

# Save the validation set and test set as JSON files
val_data.to_json("/home/jlong1/Downloads/Data/zo/0_original_data/WIC/wic_validation.jsonl")
test_data.to_json("/home/jlong1/Downloads/Data/zo/0_original_data/WIC/wic_test.jsonl")

#####

# Load the dataset
dataset = load_dataset("super_glue", "wsc")

# Access the training set, validation set, and test set
train_data = dataset['train']
val_data = dataset['validation']
test_data = dataset['test']

# Print a sample from the training set
print(train_data[0])

# Save the training set as a JSON file
train_data.to_json("/home/jlong1/Downloads/Data/zo/0_original_data/WSC/wsc_train.jsonl")

# Save the validation set and test set as JSON files
val_data.to_json("/home/jlong1/Downloads/Data/zo/0_original_data/WSC/wsc_validation.jsonl")
test_data.to_json("/home/jlong1/Downloads/Data/zo/0_original_data/WSC/wsc_test.jsonl")

#####

# Load the dataset
dataset = load_dataset("super_glue", "rte")

# Access the training set, validation set, and test set
train_data = dataset['train']
val_data = dataset['validation']
test_data = dataset['test']

# Print a sample from the training set
print(train_data[0])

# Save the training set as a JSON file
train_data.to_json("/home/jlong1/Downloads/Data/zo/0_original_data/RTE/rte_train.jsonl")

# Save the validation set and test set as JSON files
val_data.to_json("/home/jlong1/Downloads/Data/zo/0_original_data/RTE/rte_validation.jsonl")
test_data.to_json("/home/jlong1/Downloads/Data/zo/0_original_data/RTE/rte_test.jsonl")

#####

# Load the dataset
dataset = load_dataset("super_glue", "boolq")

# Access the training set, validation set, and test set
train_data = dataset['train']
val_data = dataset['validation']
test_data = dataset['test']

# Print a sample from the training set
print(train_data[0])

# Save the training set as a JSON file
train_data.to_json("/home/jlong1/Downloads/Data/zo/0_original_data/BOOLQ/boolq_train.jsonl")

# Save the validation set and test set as JSON files
val_data.to_json("/home/jlong1/Downloads/Data/zo/0_original_data/BOOLQ/boolq_validation.jsonl")
test_data.to_json("/home/jlong1/Downloads/Data/zo/0_original_data/BOOLQ/boolq_test.jsonl")

#####

# Load the dataset
dataset = load_dataset("super_glue", "multirc")

# Access the training set, validation set, and test set
train_data = dataset['train']
val_data = dataset['validation']
test_data = dataset['test']

# Print a sample from the training set
print(train_data[0])

# Save the training set as a JSON file
train_data.to_json("/home/jlong1/Downloads/Data/zo/0_original_data/MultiRC/multirc_train.jsonl")

# Save the validation set and test set as JSON files
val_data.to_json("/home/jlong1/Downloads/Data/zo/0_original_data/MultiRC/multirc_validation.jsonl")
test_data.to_json("/home/jlong1/Downloads/Data/zo/0_original_data/MultiRC/multirc_test.jsonl")

#####

# Load the dataset
dataset = load_dataset("super_glue", "record")

# Access the training set, validation set, and test set
train_data = dataset['train']
val_data = dataset['validation']
test_data = dataset['test']

# Print a sample from the training set
print(train_data[0])

# Save the training set as a JSON file
train_data.to_json("/home/jlong1/Downloads/Data/zo/0_original_data/ReCoRD/record_train.jsonl")

# Save the validation set and test set as JSON files
val_data.to_json("/home/jlong1/Downloads/Data/zo/0_original_data/ReCoRD/record_validation.jsonl")
test_data.to_json("/home/jlong1/Downloads/Data/zo/0_original_data/ReCoRD/record_test.jsonl")
