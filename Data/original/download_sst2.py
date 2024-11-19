from datasets import load_dataset

# Load the SST-2 dataset
dataset = load_dataset("glue", "sst2")

# Access the training set, validation set, and test set
train_data = dataset['train']
val_data = dataset['validation']
test_data = dataset['test']

# Print a sample from the training set
print(train_data[0])

# Save the training set as a JSON file
train_data.to_json("/home/jlong1/Downloads/Data/zo/0_original_data/sst2_train.jsonl")

# Save the validation set and test set as JSON files
val_data.to_json("/home/jlong1/Downloads/Data/zo/0_original_data/sst2_validation.jsonl")
test_data.to_json("/home/jlong1/Downloads/Data/zo/0_original_data/sst2_test.jsonl")
