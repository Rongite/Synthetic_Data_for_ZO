from datasets import load_dataset
import os

# Access the training set, validation set, and test set
train_data = load_dataset('Rowan/hellaswag', trust_remote_code=True, split='train') # modify
val_data = load_dataset('Rowan/hellaswag', trust_remote_code=True, split='validation') # modify
test_data = load_dataset('Rowan/hellaswag', trust_remote_code=True, split='test') # modify

# Print a sample from the training set
print(train_data[0])

# Save the dataset as a JSON file
save_path = "/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/HellaSwag" # modify
if not os.path.exists(save_path):
    os.makedirs(save_path)

train_data.to_json(os.path.join(save_path, "hellaswag_train.jsonl")) # modify
val_data.to_json(os.path.join(save_path, "hellaswag_validation.jsonl")) # modify
test_data.to_json(os.path.join(save_path, "hellaswag_test.jsonl")) # modify
