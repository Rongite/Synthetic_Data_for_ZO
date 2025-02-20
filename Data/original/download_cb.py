from datasets import load_dataset

# Load the cb dataset
dataset = load_dataset("super_glue", "cb")

# Access the training set, validation set, and test set
train_data = dataset['train']
val_data = dataset['validation']
test_data = dataset['test']

# Print a sample from the training set
print(train_data[0])

# Save the training set as a JSON file
train_data.to_json("/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/CB/cb_train.jsonl")

# Save the validation set and test set as JSON files
val_data.to_json("/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/CB/cb_validation.jsonl")
test_data.to_json("/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/CB/cb_test.jsonl")
