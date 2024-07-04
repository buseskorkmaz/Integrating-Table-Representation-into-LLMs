import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../'))
from evaluate import load
from bert_score import score
from datasets import load_metric
import sacrebleu
import re

# Initialize the METEOR and ROUGE metrics
meteor = load('meteor')
rouge = load('rouge')

# Base directory containing the files
base_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../generations/')

# Model sizes to evaluate
# to obtain table 1, use only pre-trained LLMs with False condition in the loop
model_sizes = ['flan-t5-small', 'flan-t5-base', 'flan-t5-large', 'flan-t5-xl', 'flan-t5-xl-wikitable-mrs_sqa', 'flan-t5-xl-wikitable', 'Llama-2-7b-chat-hf','TableLlama', 'wikitable-scigen-table', 'wikitable-mrs_sqa-scigen-table']

# Tests to evaluate
tests = ['test_cl', 'test_other']

# Log file to store the scores
log_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../logs/generate_table2.log')

# Create or clear the log file before writing the scores
with open(log_file_path, 'w', encoding='utf-8') as log_file:
    log_file.write("METEOR, ROUGE, BLEU, and BERTScore Evaluation Results:\n")

def preprocess_text(text):
    # Remove specific tokens
    text = re.sub(r'\[INST\].*?\[/INST\]', '', text)
    text = text.replace("<pad>", "").replace("<s>", "").replace("</s>", "").replace("<unk>", "").strip()
    
    # Split text into sentences to remove any repeated ones
    sentences = text.split('.')
    unique_sentences = []
    for sentence in sentences:
        sentence = sentence.strip()
        if sentence not in unique_sentences and sentence != "":
            unique_sentences.append(sentence)
    
    # Join the unique sentences back into a single string
    cleaned_text = '. '.join(unique_sentences).strip()
    if cleaned_text and not cleaned_text.endswith('.'):
        cleaned_text += '.'
    return cleaned_text

def evaluate_metrics(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        predictions = []
        references = []
        for line in file:
            if line.startswith('prediction:'):
                cleaned_prediction = preprocess_text(line.strip()[len('prediction:'):].strip())
                predictions.append(cleaned_prediction)  # Correct for METEOR, ROUGE, BERTScore
            elif line.startswith('reference:'):
                cleaned_reference = preprocess_text(line.strip()[len('reference:'):].strip())
                references.append(cleaned_reference)  # Correct for METEOR, ROUGE, BERTScore

    # Compute METEOR and ROUGE scores
    meteor_results = meteor.compute(predictions=predictions, references=references)
    rouge_results = rouge.compute(predictions=predictions, references=references)

    # Compute BERTScore
    P, R, F1 = score(predictions, references, lang="en", verbose=True)
    bert_score_result = F1.mean().item()  # Taking mean F1 score for simplicity

    # # Correct format for BLEU: references need to be a list of lists
    
    # bleu_predictions = [[pred] for pred in predictions]
    # bleu = load_metric('bleu')
    # bleu_results = bleu.compute(predictions=bleu_predictions, references=bleu_references)

    # Calculate BLEU score for all predictions
    bleu_references = [[ref] for ref in references]  # Adjust for BLEUs
    bleu_results = sacrebleu.corpus_bleu(predictions, bleu_references).score


    # Write the scores to the log file
    with open(log_file_path, 'a', encoding='utf-8') as log_file:
        log_file.write(f'Model: {file_path.replace(base_dir + "/", "")}\n')
        log_file.write(f'METEOR score: {meteor_results["meteor"]}\n')
        log_file.write(f'ROUGE scores: {rouge_results}\n')
        log_file.write(f'BLEU score: {bleu_results}\n')
        log_file.write(f'BERTScore: {bert_score_result}\n')
        log_file.write("\n")  # Add an empty line for better readability


# Iterate through the models, tests, and boolean conditions
for model_size in model_sizes:
    for test in tests:
        # True means finetuned, False is only pre-trained LLM
        # example: flan-t5-xl-True means flan-t5-xl finetuned on scigen, flan-t5-xl-False is just pre-trained FlanT5-xl
        # also, that's why some models doesn't have False version, try/except block skips those models.
        for condition in ['True','False']:
            # Construct file path
            # file_name = f'log-{test}-flan-t5-{model_size}-{condition}.txt'
            file_name = f'log-{test}-{model_size}-{condition}.txt'
            file_path = os.path.join(base_dir, file_name)
            
            try:
                # Check if the file exists
                if os.path.isfile(file_path):
                    evaluate_metrics(file_path)
                else:
                    with open(log_file_path, 'a', encoding='utf-8') as log_file:
                        log_file.write(f'File does not exist: {file_path}\n\n')
            except Exception as e:
                print("skipping ", file_path)
                print(e)

print(f'Evaluation completed. Results are logged in {log_file_path}')
