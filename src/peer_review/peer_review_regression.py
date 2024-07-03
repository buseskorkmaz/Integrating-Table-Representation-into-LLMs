import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../'))

import json
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, accuracy_score
from typing import List, Dict, Any
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import torch.nn.functional as F
import xgboost as xgb
from xgboost import XGBClassifier
import numpy as np
np.random.seed(42)
from sklearn.metrics import accuracy_score, classification_report

root_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../')

class FlanT5Model:
    def __init__(self, model_type:bool):
        self.model_type = model_type
        base_model = "google/flan-t5-xl"
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        if model_type == "finetuned":
            self.model = AutoModelForSeq2SeqLM.from_pretrained("buseskorkmaz/flant5-xl-table-1e-6")
            self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        elif model_type == "wiki_sci":
            # remove mrs_sqa
            new_model = os.path.join(root_path, "models/wikitable-scigen-table-1e-6")
            self.model = AutoModelForSeq2SeqLM.from_pretrained(base_model)  # Load base model
            self.model_state_dict = torch.load(os.path.join(new_model, "model.pkl"), map_location='cpu')
            self.model.load_state_dict(self.model_state_dict)
            self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        elif model_type == "wiki":
            new_model = os.path.join(root_path, 'models/flan-t5-xl-wikitable-1e-6')
            self.model = AutoModelForSeq2SeqLM.from_pretrained(base_model)  # Load base model
            self.model_state_dict = torch.load(os.path.join(new_model, "model.pkl"), map_location='cpu')
            self.model.load_state_dict(self.model_state_dict)
            self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        elif model_type == "all":
            new_model = os.path.join(root_path, 'models/wikitable-mrs_sqa-scigen-table-1e-6')
            self.model = AutoModelForSeq2SeqLM.from_pretrained(base_model)  # Load base model
            self.model_state_dict = torch.load(os.path.join(new_model, "model.pkl"), map_location='cpu')
            self.model.load_state_dict(self.model_state_dict)
            self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        else:
            # base_model and model_id is same, this is pre-trained LLM
            model_id = "google/flan-t5-xl"
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id, device_map="auto")
        print("DEVICE", self.model.device)

    def __call__(self, title, abstract=None, introduction=None, tables=None):
        text =  f"{title.upper()}"
        max_length= 256  
        if abstract:
            text += f"\n{abstract}"
            max_length= 2048
        if introduction:
            text += f"\nIntroduction\n{introduction}"
            max_length= 4096
        if tables:
            text += f"\n{tables}"
            max_length= 4096

        inputs =self.tokenizer(text, return_tensors="pt", max_length=max_length, padding=True, truncation=True)
        inputs = inputs.to('cuda')
        embeddings = self.model.generate(**inputs, max_new_tokens=2048)
        # print(embeddings)
        return embeddings[0]
     
    
    def process_embeddings_tensor(self, embeddings):

        if type(embeddings[0]) == List:
            for idx, embed in enumerate(embeddings):
                embeddings[idx] = torch.tensor(embed)

        max_length = max(t.shape[0] for t in embeddings)
        padded_batch_embed = [F.pad(t, (0, max_length - t.shape[0]), value=self.tokenizer.pad_token_id) for t in embeddings]
        [print(t.shape) for t in padded_batch_embed]       
        stacked_tensor = torch.stack(padded_batch_embed, dim=0)
        print(stacked_tensor.shape)
        print("Tensor", stacked_tensor)
        return stacked_tensor

def extract_introduction(sections):
        for section in sections:
            if section["heading"] and "introduction" in section['heading'].lower():
                return section['text']

def process_table_content(data, table_caption=False):

    if table_caption:
        return data['table_caption']
    
    row_seperator = '<R>'
    cell_separator = '<C>'
    caption_separator = '<CAP>'

    text = row_seperator + ' ' + cell_separator
    row_len = len(data['table_column_names'])
    for i,c in enumerate(data['table_column_names']):
        text += ' ' + c
        if i < row_len-1: 
            text += ' ' + cell_separator

    for row in data['table_content_values']:
        text += ' ' + row_seperator + ' ' + cell_separator
        for i, c in enumerate(row):
            text += ' ' + c
            if i < row_len -1:
                text += ' ' + cell_separator

    # text += ' ' + caption_separator + ' ' +data['table_caption'] + '\n'
    return text


def extract_tables(paper_title, full_text_table, table_caption):
    tables = []
    for paper in full_text_table:
        if paper['paper'].lower() == paper_title.lower():
            tables.append(process_table_content(paper, table_caption))
    
    print("Tables: \n")
    [print(table, "\n") for table in tables]
    return tables

class Evaluator:

    def __init__(self, model_type):
        self.model_type = model_type
        self.model = FlanT5Model(model_type)
        # self.dataset_path = dataset_path
    
    def get_regressor(self, regressor_name):
        if regressor_name == "SVR":
            return SVR()
        elif regressor_name == "RandomForest":
            return RandomForestRegressor(random_state=42)
        elif regressor_name == "GradientBoosting":
            return GradientBoostingRegressor(random_state=42)
        elif regressor_name == "LinearRegression":
            return LinearRegression()
        elif regressor_name == "XGBoost":
            return xgb.XGBRegressor(objective="reg:squarederror", random_state=42)
        elif regressor_name == "XGBoostC":
            return XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
        else:
            raise ValueError(f"Unsupported regressor: {regressor_name}")

    def read_dataset(self):
      
        # Define the file path to the JSON file
        base_path = Path(os.path.join(root_path, 'data/processed_full_large_dataset.json'))

        # Variable to store the loaded data
        scigen_papers = {}

        # Check if the file exists
        if base_path.is_file():
            with open(base_path, 'r', encoding='utf-8') as file:
                # Load the data from the JSON file
                scigen_papers = json.load(file)
        else:
            print(f"File not found: {base_path}")

        print("First loaded sample:", scigen_papers['0']['paper'])

        conferences = ['acl', 'conll', 'iclr']
        matched_papers_titles = []
        matched_peer_review_dataset = []
        full_body_dataset = []

        for conference in conferences:
            # Define the file path to the JSON file
            matched_base_path = Path(os.path.join(root_path, f'data/peerread_processed/{conference}_matched_papers.json'))

            # Check if the file exists
            if matched_base_path.is_file():
                with open(matched_base_path, 'r', encoding='utf-8') as file:
                    # Load the data from the JSON file
                    matched_papers = json.load(file)
            else:
                print(f"File not found: {matched_base_path}")

            matched_papers_titles += [title.lower().replace("\\", "") for title in matched_papers]
            subsets = ["train", "test", "dev"]
            for subset in subsets:
                conference_path = os.path.join(root_path, f"data/peerread_processed/{conference}/{subset}.json")

                # Check if the file exists
                try:
                    with open(conference_path, 'r', encoding='utf-8') as file:
                        # Load the data from the JSON file
                        subset_papers = json.load(file)
                except:
                    print(f"File not found: {conference_path}")

                for i in range(len(subset_papers)):
                    if subset_papers[i]['paper_info_metadata']['title'] and subset_papers[i]['paper_info_metadata']['title'].lower().replace("\\", "") in matched_papers_titles:
                        matched_peer_review_dataset.append(subset_papers[i])
                
                print("Matched review papers dataset:", len(matched_peer_review_dataset))
                        
        # full body dataset
        for i in range(len(scigen_papers)):
            if scigen_papers[str(i)]['paper'].lower().replace("\\", "") in matched_papers_titles:
                full_body_dataset.append(scigen_papers[str(i)])
        
        print("Matched full body dataset:", len(full_body_dataset))
        return matched_peer_review_dataset, full_body_dataset 

    
    def generate_embeddings(self, abstract=False, intro=False, captions= False, tables=False):
        peer_reviews, full_text_table = self.read_dataset()
        batch_embed = []
        embeddings_filename = os.path.join(root_path, f"data/peerreview_embeddings/paper_embeddings_{self.model_type}_{abstract}_{intro}_{captions}_{tables}.json")

        for paper in peer_reviews:
            title = paper['paper_info_metadata']['title']

            if tables:
                abstract = paper['paper_info_metadata']['abstractText']
                sections = paper['paper_info_metadata']['sections']
                introduction = extract_introduction(sections)
                tables = extract_tables(title, full_text_table, table_caption=False)
                embeddings = self.model(title, abstract, introduction, tables)
            elif captions:
                abstract = paper['paper_info_metadata']['abstractText']
                sections = paper['paper_info_metadata']['sections']
                introduction = extract_introduction(sections)
                tables = extract_tables(title, full_text_table, table_caption=True)
                embeddings = self.model(title, abstract, introduction, tables)
            elif intro:
                abstract = paper['paper_info_metadata']['abstractText']
                sections = paper['paper_info_metadata']['sections']
                introduction = extract_introduction(sections)
                embeddings = self.model(title, abstract, introduction, tables=None)
            elif abstract:
                abstract = paper['paper_info_metadata']['abstractText']
                embeddings = self.model(title, abstract, introduction=None, tables=None)
            else:
                embeddings = self.model(title, abstract=None, introduction=None, tables=None)
            # Convert embeddings to a list (or a nested list if it's multidimensional) to ensure JSON serializability
            batch_embed.append(embeddings)
        
        # Ensure embeddings are in a serializable format
        serializable_embeddings = [embed.tolist() if hasattr(embed, 'tolist') else embed for embed in batch_embed]
        with open(embeddings_filename, 'w') as f:
            json.dump(serializable_embeddings, f)
        return batch_embed

    def get_reviews(self):
        peer_reviews, full_text_table = self.read_dataset()
        recommendations, impacts, originality = [], [], []
        
        for paper in peer_reviews:
            print(paper.keys())
            print(paper["paper_info_metadata"]['title'])
            recommendations.append(paper['review_RECOMMENDATION'])
            # impacts.append(paper["review_IMPACT"])
            # originality.append(paper['review_ORIGINALITY'])
            # Convert embeddings to a list (or a nested list if it's multidimensional) to ensure JSON serializability
        
        return recommendations

    def peer_review_classification(self, processed_embeddings):
    
        recommendations = self.get_reviews()
        reviews = {"recommendation": recommendations, 
                #    "impact": impacts,
                #    "originality": originality
                   }
        
        print(reviews['recommendation'][0])
        # print(reviews['impact'][0])
        # print(reviews['originality'][0])


        results = {}
        for feature in ['recommendation']:
            # Encode class labels to start from 0
            label_encoder = LabelEncoder()
            y_encoded = label_encoder.fit_transform(reviews['recommendation'])

            # Split the data
            X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(
                processed_embeddings, y_encoded, test_size=0.2, random_state=42
            )
            
            results[feature] = {}
           
            print(f"Evaluating with XGBoost classifier")
            model = self.get_regressor("XGBoostC")
            
            # Fit the model on the training data using encoded labels
            model.fit(X_train, y_train_encoded)
            
            # Predict on the training and test set
            prediction_train = model.predict(X_train)
            prediction_test = model.predict(X_test)
            
            # Decode the predictions back to original labels if necessary
            prediction_train_decoded = label_encoder.inverse_transform(prediction_train)
            prediction_test_decoded = label_encoder.inverse_transform(prediction_test)
            
            # Evaluate the model using the original labels
            accuracy_train = accuracy_score(label_encoder.inverse_transform(y_train_encoded), prediction_train_decoded)
            accuracy_test = accuracy_score(label_encoder.inverse_transform(y_test_encoded), prediction_test_decoded)
            report_test = classification_report(label_encoder.inverse_transform(y_test_encoded), prediction_test_decoded)
            
            # Store results
            self.results["XGBoostC"] = {
                'accuracy_train': accuracy_train,
                'accuracy_test': accuracy_test,
                'classification_report': report_test,
                'predictions_train': prediction_train_decoded,
                'predictions_test': prediction_test_decoded,
                'true_train': label_encoder.inverse_transform(y_train_encoded),
                'true_test': label_encoder.inverse_transform(y_test_encoded)
            }

        results = self.results
        return results

    def predict_score(self, 
                      embeddings=None, 
                    #   regressor_names=["SVR", "RandomForest", "GradientBoosting", "LinearRegression", "XGBoost", "XGBoostC"],
                    regressor_names=["XGBoost", "XGBoostC"],
        ):
        embeddings_filename = os.path.join(root_path, f"data/peerreview_embeddings/paper_embeddings_{self.model_type}.json")
        if embeddings is None:
            with open(embeddings_filename, 'r', encoding='utf-8') as f:
                embeddings = json.load(f)

        processed_embeddings = self.model.process_embeddings_tensor(embeddings)
        if isinstance(processed_embeddings, torch.Tensor):
            processed_embeddings = processed_embeddings.cpu().numpy()  # Convert to numpy array

        recommendations = self.get_reviews()
        reviews = {"recommendation": recommendations, 
                #    "impact": impacts,
                #    "originality": originality
                   }
        
        print(reviews['recommendation'][0])
        # print(reviews['impact'][0])
        # print(reviews['originality'][0])

        results = {}
        for feature in ['recommendation']:
            print(f"Processing feature: {feature}")
            y = reviews[feature]
            X_train, X_test, y_train, y_test = train_test_split(processed_embeddings, y, test_size=0.2, random_state=42)
            
            self.results = {}
            for regressor_name in regressor_names:
                if regressor_name == "XGBoostC":
                    results = self.peer_review_classification(processed_embeddings)
                else:
                    print(f"Evaluating with {regressor_name}")
                    regressor = self.get_regressor(regressor_name)
                    
                    # Fit the model on the training data
                    regressor.fit(X_train, y_train)
                    
                    # Predict on the training and test set
                    prediction_train = regressor.predict(X_train)
                    prediction_test = regressor.predict(X_test)
                    
                    # Round predictions to nearest integer
                    # prediction_train_rounded = np.rint(prediction_train)
                    # prediction_test_rounded = np.rint(prediction_test)

                    # Evaluate using rounded predictions
                    train_error = mean_squared_error(y_train, prediction_train)
                    test_error = mean_squared_error(y_test, prediction_test)
                    
                    self.results[regressor_name] = {
                        'train_error': train_error,
                        'test_error': test_error,
                        'predictions_train': prediction_train,
                        'predictions_test': prediction_test,
                        'true_train': y_train,
                        'true_test': y_test
                    }
                

        results = self.results
        return results

    def print_results(self, file):
        for regressor_name, metrics in self.results.items():
            print(f"Model: {regressor_name}, Feature: recommendation")
            if regressor_name == "XGBoostC":   
                print(f"  Training Accuracy: {metrics['accuracy_train']:.2f}")
                print(f"  Test Accuracy: {metrics['accuracy_test']:.2f}")
                print("  Classification Report on Test Set:")
                print(metrics['classification_report'])
                print("\n")

                file.write(f"  Training Accuracy: {metrics['accuracy_train']:.2f}\n")
                file.write(f"  Test Accuracy: {metrics['accuracy_test']:.2f}\n")
                file.write("  Classification Report on Test Set:\n")
                file.write(f"{metrics['classification_report']}\n\n")

            else:
                print(f"Regressor: {regressor_name}")
                print(f"  Training Error/Accuracy: {metrics['train_error']:.4f}")
                print(f"  Test Error/Accuracy: {metrics['test_error']:.4f}")
                
                # Print sample training predictions and true scores
                print(f"  Sample Training Predictions: {metrics['predictions_train'][:5]}")
                print(f"  Sample True Training Scores: {metrics['true_train'][:5]}")

                file.write(f"Regressor: {regressor_name}\n")
                file.write(f"  Training Error/Accuracy: {metrics['train_error']:.4f}\n")
                file.write(f"  Test Error/Accuracy: {metrics['test_error']:.4f}\n")
                file.write(f"  Sample Training Predictions: {metrics['predictions_train'][:5]}\n")
                file.write(f"  Sample True Training Scores: {metrics['true_train'][:5]}\n")
                file.write(f"  Sample Test Predictions: {metrics['predictions_test'][:5]}\n")
                file.write(f"  Sample True Test Scores: {metrics['true_test'][:5]}\n\n")
                
            # Print sample test predictions and true scores
            print(f"  Sample Test Predictions: {metrics['predictions_test'][:5]}")
            print(f"  Sample True Test Scores: {metrics['true_test'][:5]}\n")

def evaluate_model(model_type):
    evaluator = Evaluator(model_type)
    experiments = [
        {"abstract": False, "intro": False, "captions": False, "tables": False},
        {"abstract": True, "intro": False, "captions": False, "tables": False},
        {"abstract": True, "intro": True, "captions": False, "tables": False},
        {"abstract": True, "intro": True, "captions": True, "tables": False},
        {"abstract": True, "intro": True, "captions": False, "tables": True},
    ]
    # table 3
    with open("peer_review_fixed_seed.log", "a") as file:
        file.write(f"Experiments started for {model_type}...")
        for experiment in experiments:
            print("Experiment:", experiment)
            file.write(f"Experiment: {experiment}")
            embeddings = evaluator.generate_embeddings(experiment["abstract"], experiment["intro"], experiment["captions"], experiment["tables"])
            results = evaluator.predict_score(embeddings)
            evaluator.print_results(file)
    
    return results

# Evaluate both the base and the fine-tuned models
base_results = evaluate_model("base")
finetuned_results = evaluate_model("finetuned")
all_results = evaluate_model("all")
wiki_sci_results = evaluate_model("wiki_sci")
wiki_results = evaluate_model("wiki")
