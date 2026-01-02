from datasets import load_dataset

# Load the copa dataset
dataset = load_dataset("super_glue", "copa")

# Access the training set, validation set, and test set
train_data = dataset['train']
val_data = dataset['validation']
test_data = dataset['test']

# Print a sample from the training set
print(train_data[0])

# Save the training set as a JSON file
train_data.to_json("/home/jlong1/Downloads/Data/zo/0_original_data/COPA/copa_train.jsonl")

# Save the validation set and test set as JSON files
val_data.to_json("/home/jlong1/Downloads/Data/zo/0_original_data/COPA/copa_validation.jsonl")
test_data.to_json("/home/jlong1/Downloads/Data/zo/0_original_data/COPA/copa_test.jsonl")
