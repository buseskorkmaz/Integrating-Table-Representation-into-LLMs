import tarfile
import json
import re
import os  
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../'))

data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../../data/')
# Directory where the .tar.gz files are located
directory = os.path.join(data_path, f'arxiv_full/')

# Directory where extracted files will be placed
extracted_dir = os.path.join(data_path, f'extracted_arxiv_full/')

# Load JSON data from file
with open(os.path.join(data_path, 'scigen_large_original/train.json'), 'r') as file:
    dataset = json.load(file)

# Function to extract .tar.gz files into separate directories
def extract_files():
    for filename in os.listdir(directory):
        if filename.endswith('.tar.gz'):
            # Strip the .tar.gz extension to get the paper ID
            paper_id = filename[:-7]
            file_path = os.path.join(directory, filename)
            paper_dir = os.path.join(extracted_dir, paper_id)
            try:
                os.makedirs(paper_dir, exist_ok=True)  # Create a directory for the paper ID
                with tarfile.open(file_path, 'r:gz') as tar:
                    tar.extractall(path=paper_dir)
                print(f'Extracted {filename} into {paper_dir}')
            except tarfile.ReadError as e:
                print(f'Not a gzip file or archive is corrupted for {filename}: {e}')
            except Exception as e:
                print(f'An error occurred while extracting {filename}: {e}')


# Function to parse LaTeX files and combine sections
def parse_latex(paper_id):
    paper_dir = os.path.join(extracted_dir, paper_id)
    full_body = ''
    # Regular expression patterns
    env_pattern = re.compile(r'\\begin\{(table|figure)\*?\}.*?\\end\{\1\*?\}', re.DOTALL)
    comment_pattern = re.compile(r'^\s*%.*$', re.MULTILINE)
    bibliography_pattern = re.compile(r'\\bibliography\{.*?\}')
    usepackage_pattern = re.compile(r'\\usepackage\{.*?\}')
    
    for root, dirs, files in os.walk(paper_dir):
        for file in files:
            if file.endswith('.tex'):
                file_path = os.path.join(root, file)
                with open(file_path, 'r') as f:
                    content = f.read()
                    # Remove tables and figures
                    content = re.sub(env_pattern, '', content)
                    # Remove comment lines
                    content = re.sub(comment_pattern, '', content)
                    # Remove bibliography lines
                    content = re.sub(bibliography_pattern, '', content)
                    # Remove package inclusion lines
                    content = re.sub(usepackage_pattern, '', content)
                    
                    full_body += content  # You may want to add additional processing to combine the sections
                    print(f"Processed file {file}")
    print("Full body", full_body)
    # if full_body == "" or full_body == " ":
    #     return False
    return full_body

# Function to update JSON data
def update_json(full_body, paper_id):
    for index in dataset:
        entry = dataset[index]
        if entry['paper_id'] == paper_id:
            entry['full_body_text'] = full_body
            return

# Extract all .tar.gz files
extract_files()

# Process each paper and update the dataset
counter = 0
for index in dataset:
    entry = dataset[index]
    paper_id = entry['paper_id']
    print(paper_id)
    try:
        full_body = parse_latex(paper_id)
        # if not full_body:
        #     counter += 1
        #     if paper_id != "1912.03832v1": break 
    except FileNotFoundError:
        print(f'Not extracted, corrupted file {paper_id}')
        full_body = ""
    except Exception as e:
        print(f'An error occurred while extracting {paper_id}: {e}')
        full_body = ""
    update_json(full_body, paper_id)

# Save the updated dataset to a new JSON file
with open(os.path.join(data_path, f'updated_large_dataset.json'), 'w') as file:
    json.dump(dataset, file, indent=4)

print("Updated dataset with full body texts.")
