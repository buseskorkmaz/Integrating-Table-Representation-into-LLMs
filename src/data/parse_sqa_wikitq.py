from datasets import load_dataset, load_from_disk
from transformers import AutoTokenizer
import os  
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../'))

data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../data/')

# Load the dataset
msr_sqa = load_dataset("msr_sqa")

splits = ['train', 'validation', 'test']
print(msr_sqa)

# Process each split
for split in splits:
    # Assuming 'table_header' is a column containing header names for all rows
    # and 'table_data' is a column where each row contains a list of table rows
    msr_sqa[split] = msr_sqa[split].filter(lambda x: x["position"] == 0)
    msr_sqa[split] = msr_sqa[split].map(lambda x: {
        "table_column_names": x['table_header'],
        "table_content_values": [row for row in x['table_data']],
        "answers": x['answer_text']  # Assuming answer_text is a list of lists
    }, batched=True, remove_columns=['table_header', 'table_data', 'answer_text'])
    msr_sqa[split] = msr_sqa[split].remove_columns(["id", "annotator", "position",
                                                    "question_and_history", "answer_coordinates"])
# Print the first example of the train split to check the result
print(msr_sqa['train'][0])
print(msr_sqa)

wikitablequestions = load_dataset("wikitablequestions")
splits = ['train', 'validation', 'test']
print(wikitablequestions)
# print(wikitablequestions["train"][0]['table']['header'])
# Print the structure of the first element in the train split
print(wikitablequestions["train"][0])

# Process each split
for split in splits:
    # Use batched map to process the dataset efficiently
    wikitablequestions[split] = wikitablequestions[split].map(lambda batch: {
        "table_column_names": [x['header'] for x in batch['table']],
        "table_content_values": [x['rows'] for x in batch['table']],
    }, batched=True, remove_columns=['id', 'table'])

# Constants for separators
row_separator = '<R>'
cell_separator = '<C>'
caption_separator = '<CAP>'


def flatten_table(example):
    text = row_separator + ' ' + cell_separator
    row_len = len(example["table_column_names"])
    for i, c in enumerate(example['table_column_names']):
        text += ' ' + c
        if i < row_len - 1:
            text += ' ' + cell_separator

    for row in example['table_content_values']:
        text += ' ' + row_separator + ' ' + cell_separator
        for i, c in enumerate(row):
            text += ' ' + c
            if i < row_len - 1:
                text += ' ' + cell_separator
    
    # Answers could be a list, so we join them if there are multiple answers
    answers = ', '.join(example['answers']) if isinstance(example['answers'], list) else example['answers']

    example = {"text": text,
               "answers": answers}
    
    return example

datasets = [msr_sqa, wikitablequestions]
dataset_names = ["msr_sqa", "wikitable"]

for dataset in datasets:
    for split in splits:
        dataset[split] = dataset[split].map(lambda x: flatten_table(x))

# print(wikitablequestions["train"][0])
print(msr_sqa["train"][0])

def process_splits(tokenizer, row):
    
    output_data = {}
    processed_source = row['question'] + row['text']
    processed_target = row['answers']

    # Tokenize the source and target separately
    source_encoding = tokenizer(processed_source, truncation=True, padding="max_length", max_length=512)
    target_encoding = tokenizer(processed_target, truncation=True, padding="max_length", max_length=512)

    output_data['input_ids'] = source_encoding['input_ids']
    output_data['attention_mask'] = source_encoding['attention_mask']
    output_data['labels'] = target_encoding['input_ids']

    return output_data


tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-xl")

for dataset, name in zip(datasets, dataset_names):
    for split in splits:
        dataset[split] = dataset[split].map(lambda x: process_splits(tokenizer, x))
    dataset.save_to_disk(os.path.join(data_path, f't5_compliant_hf_{name}'))
    # Load and check the dataset
    loaded_dataset = load_from_disk(os.path.join(data_path, f't5_compliant_hf_{name}'))
    print(loaded_dataset['train'][0])
