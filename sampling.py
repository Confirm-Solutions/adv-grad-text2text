import json
import random

def sample_jsonl(input_file, output_train_file, output_test_file, train_samples=8000, test_samples=2000):
    # Read all lines from the input JSONL file
    with open(input_file, 'r', encoding='utf-8') as file:
        all_lines = file.readlines()

    # Ensure the number of requested samples is not greater than the total number of lines
    total_lines = len(all_lines)
    train_samples = min(train_samples, total_lines)
    test_samples = min(test_samples, total_lines - train_samples)

    # Randomly shuffle the lines
    random.shuffle(all_lines)

    # Split the lines into training and testing sets
    train_lines = all_lines[:train_samples]
    test_lines = all_lines[train_samples:train_samples + test_samples]

    # Write the training samples to the output file
    with open(output_train_file, 'w', encoding='utf-8') as train_file:
        train_file.writelines(train_lines)

    # Write the testing samples to the output file
    with open(output_test_file, 'w', encoding='utf-8') as test_file:
        test_file.writelines(test_lines)

if __name__ == "__main__":
    # Replace 'input.jsonl' with the path to your input JSONL file
    input_file_path = 'webtext.train.jsonl'

    # Replace 'output_train.jsonl' and 'output_test.jsonl' with the desired output paths
    output_train_file_path = 'output_train.jsonl'
    output_test_file_path = 'output_test.jsonl'

    # Call the function to sample lines
    sample_jsonl(input_file_path, output_train_file_path, output_test_file_path)