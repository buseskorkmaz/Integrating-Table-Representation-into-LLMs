# clone models using the huggingface repos below
# buseskorkmaz/Llama-2-7b-chat-hf-table - LLama-2-chat-hf fine-tuned on SciGen

cd ../models

# Fine-tuned on Scigen (base-model FlanT5-small)
git clone https://huggingface.co/buseskorkmaz/flan-t5-small-table

# Fine-tuned on Scigen (base-model FlanT5-base)
git clone https://huggingface.co/buseskorkmaz/flan-t5-base-table

# Fine-tuned on Scigen (base-model FlanT5-large)
git clone https://huggingface.co/buseskorkmaz/flan-t5-large-table

# Fine-tuned on Scigen (base-model FlanT5-xl)
git clone https://huggingface.co/buseskorkmaz/flant5-xl-table

# Intermediate pre-training with WikiTable (base model FlanT5-xl)
git clone https://huggingface.co/buseskorkmaz/flan-t5-xl-wikitable

# Intermediate pre-training with WikiTable - fine-tuned on SciGen (base model FlanT5-xl)
git clone https://huggingface.co/buseskorkmaz/wikitable-scigen-table/

# Intermediate pre-training with WikiTable+SQA (base model FlanT5-xl)
git clone https://huggingface.co/buseskorkmaz/wikitable-mrs_sqa-scigen-table

# Intermediate pre-training with WikiTable+SQA - fine-tuned on SciGen (base model FlanT5-xl)
git clone https://huggingface.co/buseskorkmaz/wikitable-mrs_sqa-scigen-table
