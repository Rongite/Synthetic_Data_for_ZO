from datasets import load_dataset

# Access the training set, validation set, and test set
train_data = load_dataset('allenai/ai2_arc', 'ARC-Easy', trust_remote_code=True, split='train')
val_data = load_dataset('allenai/ai2_arc', 'ARC-Easy', trust_remote_code=True, split='validation')
test_data = load_dataset('allenai/ai2_arc', 'ARC-Easy', trust_remote_code=True, split='test')

# Print a sample from the training set
print(train_data[0])

# Save the training set as a JSON file
train_data.to_json("/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/ArcE/ARC-Easy_train.jsonl")

# Save the validation set and test set as JSON files
val_data.to_json("/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/ArcE/ARC-Easy_validation.jsonl")
test_data.to_json("/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/ArcE/ARC-Easy_test.jsonl")
