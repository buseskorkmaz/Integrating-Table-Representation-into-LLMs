import argparse
from datasets import load_metric, load_from_disk
import sacrebleu
from transformers import pipeline
from bert_score import score
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os
import re

root_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../')
def main(task, model, peft, finetuned, log_file):
    base_model = f"osunlp/TableLlama"
    if finetuned:
        if peft:
            # Load the model and tokenizer
            tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
            load_model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16)
            new_model = os.path.join(root_path, f'models/{model}')
            model = PeftModel.from_pretrained(load_model, new_model).merge_and_unload()
        else:
            new_model = os.path.join(root_path, f'models/{model}-1e-6')
            tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(base_model)  # Load base model
            model_state_dict = torch.load(os.path.join(new_model, "model.pkl"), map_location='cpu')
            model.load_state_dict(model_state_dict)
    else:
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(base_model)  # Load base model

    model.to("cuda" if torch.cuda.is_available() else "cpu")

    # Load the test dataset
    test_dataset_path = os.path.join(root_path, f'data/llama_compliant_hf_{task}')
    test_dataset = load_from_disk(test_dataset_path)

    # Prepare the pipeline
    pipe = pipeline(task="text2text-generation", model=model, tokenizer=tokenizer, device=0 if torch.cuda.is_available() else -1)

    # Initialize metrics
    bleu_metric = load_metric("sacrebleu")
    bert_scores = []
    bleu_scores= []

    # Function for generating predictions with truncation
    def generate_prediction(input_ids, attention_mask):
        # Truncate input_ids and attention_mask to model's max length
        input_ids = input_ids[:512]
        attention_mask = attention_mask[:512]

        # Decode input_ids to text
        input_text = tokenizer.decode(input_ids, skip_special_tokens=True)

        # Generate outputs using text input
        outputs = pipe(input_text, max_new_tokens=512)
        return outputs[0]['generated_text'] if outputs else None

    predictions = []
    references  = []

    # Evaluate the dataset
    for data in test_dataset:
        try:
            # Generate prediction
            prediction = generate_prediction(data['input_ids'], data['attention_mask'])

            # Decode the reference
            reference = tokenizer.decode(data['labels'])
            reference = re.sub(r'\[INST\].*?\[/INST\]', '', reference)
            print("reference:", reference)
            log_file.write(f"reference: {reference}\n")
            print("prediction:", prediction)
            log_file.write(f"prediction: {prediction}\n")

        # Append the prediction and reference to their respective lists
            prediction = prediction.replace("<pad>", "")
            prediction = prediction.replace("<unk>", "")
            reference = reference.replace("<pad>", "")
            reference = reference.replace("<unk>", "")
            reference = reference.replace("</s>", "")

            predictions.append(prediction)
            references.append([reference])  # Note that references need to be a list of lists

            # Compute BERTScore
            P, R, F1 = score([prediction], [reference], lang="en")
            bert_scores.append(F1.mean().item())

        except Exception as e:
            print(f"Error processing data: {e}")

    # Make sure predictions are generated
    if not predictions:
        print("No predictions were generated.")
        log_file.write("No predictions were generated.")
    else:
        # Calculate BLEU score for all predictions
        bleu_score = sacrebleu.corpus_bleu(predictions, references).score

        # Calculate average BERTScore
        average_bert_score = sum(bert_scores) / len(bert_scores) if bert_scores else 0

        print(f"BLEU score: {bleu_score}")
        log_file.write(f"BLEU score: {bleu_score}\n")
        print(f"Average BERTScore: {average_bert_score}")
        log_file.write(f"Average BERTScore: {average_bert_score}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Llama model training/test')
    parser.add_argument('--task', type=str, help='Task type: test_cl or test_other')
    parser.add_argument('--model', type=str, help='Model name')
    parser.add_argument('--finetuned', type=bool, help='True for to load finetuned model False to use pre-trained model', default=False)
    parser.add_argument('--use_peft', type=bool, help='Load with peft: True or False', default=False)
    args = parser.parse_args()

    # Open log file
    with open(os.path.join(root_path, f'generations/log-{args.task}-{args.model}-{str(args.finetuned)}.txt'), "a") as log_file:
        # Log the arguments
        print("Arguments:", args)
        log_file.write(f"Arguments: {args}\n")

        # Check if the task argument starts with 'test'
        if args.task.startswith('test'):
            main(args.task, args.model, args.use_peft, args.finetuned,  log_file)
        else:
            error_message = "Task must start with 'test'"
            print(error_message)
            log_file.write(error_message + "\n")
