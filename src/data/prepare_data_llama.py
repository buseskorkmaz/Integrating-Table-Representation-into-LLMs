from datasets import Dataset, load_from_disk
import argparse
from transformers import AutoTokenizer
import os  
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../'))

data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data/')

def load_data(file_path):
    data = {"content": []}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            # Adding each line to the data dictionary under 'content'
            data["content"].append(line.strip())
    return data


def process_line(tokenizer, source_line, target_line):
    # Process source line
    processed_source = f"<s>[INST] Explain what the table demonstrates.  {source_line.strip()}"
    processed_target = f" [/INST] {target_line.strip()} </s>"

    # Combine processed source line with target line
    # Tokenize the source and target separately
    source_encoding = tokenizer(processed_source, truncation=True, padding="max_length", max_length=512)
    target_encoding = tokenizer(processed_target, truncation=True, padding="max_length", max_length=512)

    return {"input_ids": source_encoding["input_ids"], "attention_mask": source_encoding["attention_mask"], "labels": target_encoding["input_ids"]}


def process_files(tokenizer, source_files, target_files, output_data):
    for source_file, target_file in zip(source_files, target_files):
        with open(source_file, 'r') as src_file, open(target_file, 'r') as tgt_file:
            for source_line, target_line in zip(src_file, tgt_file):
                encoding = process_line(tokenizer, source_line, target_line)
                output_data['input_ids'].append(encoding['input_ids'])
                output_data['attention_mask'].append(encoding['attention_mask'])
                output_data['labels'].append(encoding['labels'])


def main(task):
    
    tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-chat-hf")
    special_tokens_dict = {'additional_special_tokens': ['<R>','<C>','<CAP>', '[EMPTY]', '[BOLD]']}
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    # model.resize_token_embeddings(len(tokenizer))
    output_data = {'input_ids': [], 'attention_mask': [], 'labels': []}

    if "train" in task:                    
        # List of file paths
        source_files = [
            os.path.join(data_path,"scigen_large_original/large_train.source"),
            # Add more source files as needed
        ]

        target_files = [
            os.path.join(data_path,"scigen_large_original/large_train.target"),
            # Add more target files as needed
        ]


    elif "test" in task:

        if task == 'test_other':

            source_files = [
                os.path.join(data_path,"scigen_test/test_other.source"),
            ]

            target_files = [
                os.path.join(data_path,"scigen_test/test_other.target"),
            ]

        elif task == 'test_cl':
            source_files = [
                os.path.join(data_path,"scigen_test/test-CL.source"),
                # Add more source files as needed
            ]

            target_files = [
                os.path.join(data_path,"scigen_test/test-CL.target"),
                # Add more target files as needed
            ]
        else:
            return FileNotFoundError
    else:
        return FileNotFoundError
  
    # Process and combine source and target files
    process_files(tokenizer, source_files, target_files, output_data)

    # Create a Hugging Face dataset from the processed data
    dataset = Dataset.from_dict(output_data)

    # Save the dataset to disk
    dataset.save_to_disk(os.path.join(data_path, f'llama_compliant_hf_{task}'))

    # Load and check the dataset
    loaded_dataset = load_from_disk(os.path.join(data_path, f'llama_compliant_hf_{task}'))
    print(loaded_dataset[0])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Prepare data for training/test')
    parser.add_argument('task', type=str, help='Task type: train or test')
    args = parser.parse_args()

    main(args.task)