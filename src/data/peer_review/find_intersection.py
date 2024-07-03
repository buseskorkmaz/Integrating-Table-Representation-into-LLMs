import os
import json
from pathlib import Path
from datasets import load_dataset, load_from_disk

data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../data/')

# Define the file path to the JSON file
base_path = os.path.join(data_path, 'processed_full_large_dataset.json')

# Variable to store the loaded data
scigen_papers = {}

# Check if the file exists
if base_path.is_file():
    with open(base_path, 'r', encoding='utf-8') as file:
        # Load the data from the JSON file
        scigen_papers = json.load(file)
else:
    print(f"File not found: {base_path}")

scigen_titles = []
for i in range(len(scigen_papers)):
    scigen_titles.append(scigen_papers[str(i)]['paper'].lower().replace("//", ""))

# Base path to the directories
# Download PeerRead into /data first
base_paths = {
    'acl': Path(os.path.join(data_path, 'PeerRead/data/acl_2017')),
    'conll': Path(os.path.join(data_path, 'PeerRead/data/conll_2016')),
    'iclr': Path(os.path.join(data_path, 'PeerRead/data/iclr_2017'))
}

# Placeholder for loaded data for each conference
loaded_data = {}

# Define the base path where the JSON files are saved
base_path = Path(os.path.join(data_path, 'peerread_processed'))

# List of conferences
conferences = ['acl', 'conll', 'iclr']
# List of subsets
subsets = ['train', 'dev', 'test']

# Replace this with the actual list of SciGen titles
scigen_titles = list(set(scigen_titles))

matched_papers = {'acl': [], 'conll': [], 'iclr': []}

# Iterate over each conference and subset
for conf in conferences:
    loaded_data[conf] = {}
    for subset in subsets:
        file_path = base_path / conf / f'{subset}.json'
        # Check if the file exists
        if file_path.is_file():
            with open(file_path, 'r', encoding='utf-8') as f:
                # Load the data from the JSON file
                papers = json.load(f)
                loaded_data[conf][subset] = papers
                # Filter and append matched papers
                for paper in papers:
                    if paper['paper_info_metadata']['title'] is not None and paper['paper_info_metadata']['title'].lower().replace("//", "") in scigen_titles:
                        matched_papers[conf].append(paper['paper_info_metadata']['title'])
        else:
            print(f"File not found: {file_path}")

# Save the matched papers to JSON files
for conf in matched_papers:
    matched_papers_path = base_path / f"{conf}_matched_papers.json"
    with open(matched_papers_path, 'w', encoding='utf-8') as f:
        json.dump(matched_papers[conf], f, ensure_ascii=False, indent=4)
