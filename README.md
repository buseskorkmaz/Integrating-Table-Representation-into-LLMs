# Integrating Table Representations into Large Language Models for Improved Scholarly Document Comprehension
---
## Environment
---

This code developed with Python 3.10. The packages needed can be found in `requirements.txt`. 

## Dataset
---

### Table Question-Answering

To adept the models into table representations we apply intermediate pre-training (or continual training) with general purpose table question-answering datasets of WikiTable and Microsoft's Sequential Question Answering. Both datasets are available in HuggingFace. We process them using `/src/data/parse_sqa_wikitq.py` to obtain FlanT5 compatible training dataset. The processed versions can be found in `data/t5_compliant_hf_msr_sqa` and `data/t5_compliant_hf_wikitable`.

### Scientific Tables

We used SciGen's large split to fine-tune the models on scientific table understanding. The training data taken from SciGen is given in `/data/scigen_large_original`. This data is further processed to have FlanT5 and Llama-2 compliant training datasets using `src/data/prepare_data_t5.py` and `src/data/prepare_data_llama.py`, respectively. The final version of the training datasets are shared as `data/t5_compliant_hf_train_large` and `data/llama_compliant_hf_train_large_spec_tokens`. Also, we used the test split of SciGen dataset to test our models. With followig same procedure with the training dataset, processed versions can be found in `/data/scigen_test`.

### Peer Review (PeerRead)

The data used in the peer review prediction task has been obtained through the intersection of papers in PeerRead and SciGen. PeerRead has the reviews for the papers while SciGen provides the structured table interpretations with captions. We enrich the original SciGen dataset with full body text through crawling arXiv submissions of the papers in SciGen via `src/data/peer_review/download_arxiv_full.py` and `src/data/peer_review/parse_latex_files.py`. This dataset is available with HuggingFace id `buseskorkmaz/scigen_enriched_with_full_body`.  

The intersection of SciGen and PeerRead is identified through `src/data/peer_review/find_intersection` and includes papers from ACL 2017, CoNLL 2016, and ICLR 2017. These papers can be found in `data/peerread_processed`. Finally, peer review prediction experiments use the embeddings of formed data for that particular experiment (more detail in `src/peer_review/peer_review_prediction.py`). We cache the embeddings of the designed experiments `src/data/peer_review_embeddings`.

## Models

We train the models using the scripts under `src/train/`. The trained models are available at HuggingFace and `scripts/download_models.sh` share tips regarding how to obtain these models.

---
## Training

All models are trained on A100 80GB GPU(s). FlanT5 models are usually fit into 1 GPU while Llama-2-7B may require 2 GPUs. We log the experiments to `wandb`. Training scripts utilize `accelerate` default config in your environment. Also, the hyperparameters needed for training is placed in train_cfg in respective training scripts.

## Evaluation

We evaluate our models first generating the explanations for tables in test datasets `data/t5_compliant_hf_test_cl` and `data/t5_compliant_hf_test_other` through `src/eval/evaluate_t5.py` or `src/eval/evaluate_llama.py`. These scripts generate the explanations under the `generations/` directory. Then, we compute the reported scores of ROUGE, BERT Score and METEOR using the `src/eval/compute_meteor_rouge.py`.

---
## Peer Review Score Prediction

After obtaining the datasets, we run the peer review score prediction experiments with the script `src/peer_review/peer_review_prediction.py`.


