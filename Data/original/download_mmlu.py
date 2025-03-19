from datasets import load_dataset
import os

# Access the training set, validation set, and test set
train_data = load_dataset(
            "cais/mmlu", "all", trust_remote_code=True, split='auxiliary_train'
        ) # modify
val_data = load_dataset(
            "cais/mmlu", "all", trust_remote_code=True, split='test'
        ) # modify

# Print a sample from the training set
print(train_data[0])

# Save the dataset as a JSON file
save_path = "/home/jlong1/Downloads/Synthetic_Data_for_ZO/Data/original/MMLU" # modify
if not os.path.exists(save_path):
    os.makedirs(save_path)

train_data.to_json(os.path.join(save_path, "mmlu_train.jsonl")) # modify
val_data.to_json(os.path.join(save_path, "mmlu_validation.jsonl")) # modify
