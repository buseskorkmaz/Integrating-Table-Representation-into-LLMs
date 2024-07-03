import requests
import concurrent.futures
import json
import os  
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../'))

data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../data/')

# Load JSON data from file
with open(os.path.join(data_path, 'scigen_large_original/train.json'), 'r') as file:
    papers_data = json.load(file)

# Extract paper IDs
paper_ids = [info['paper_id'] for key, info in papers_data.items()]

# Function to download a single source file
def download_source(paper_id):
    filename = os.path.join(data_path, f'arxiv_full/{paper_id}.tar.gz')
    # Check if the file already exists to avoid re-downloading
    if not os.path.exists(filename):
        url = f'https://arxiv.org/e-print/{paper_id}'
        try:
            response = requests.get(url, allow_redirects=True)
            if response.status_code == 200:
                with open(filename, 'wb') as file:
                    file.write(response.content)
                print(f'Downloaded {filename}')
            else:
                print(f'Failed to download source for {paper_id}')
        except Exception as e:
            print(f'Error downloading {paper_id}: {e}')
    else:
        print(f'File {filename} already exists. Skipping download.')

unique_paper_ids = list(set(paper_ids))  # Remove duplicates to ensure efficiency
print("Number of unique paper_ids:", len(unique_paper_ids))

# Use ThreadPoolExecutor to download files in parallel
with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
    # Submit download tasks
    future_to_paper_id = {executor.submit(download_source, paper_id): paper_id for paper_id in unique_paper_ids}

    # Wait for the futures to complete
    for future in concurrent.futures.as_completed(future_to_paper_id):
        paper_id = future_to_paper_id[future]
        try:
            future.result()  # This is where you can handle results or exceptions if needed
        except Exception as e:
            print(f'Error downloading file for {paper_id}: {e}')
