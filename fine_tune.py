import time
import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    GPT2LMHeadModel,
    GPT2Tokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset
import argparse

# Import required libraries
import os
import argparse
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling, set_seed
from datasets import load_dataset
from torch.utils.data import Dataset
from fine_tune import *

# Define the dataset class

# Set random seed for reproducibility
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

os.environ["WANDB_DISABLED"] = "true"
def measure_single_token_generation_time(target_model, tokenizer, device, prompt="Test prompt"):
    """
    Measures the time taken by the target model to generate a single token.

    Args:
        target_model (AutoModelForCausalLM): The target language model.
        tokenizer (AutoTokenizer): The tokenizer associated with the model.
        device (str): The device to run the model on ('cuda' or 'cpu').
        prompt (str): The prompt to start generation from. Defaults to "Test prompt".

    Returns:
        float: The average time taken for single token generation (in seconds).
    """
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids)

    # Warm-up to ensure fair timing
    with torch.no_grad():
        _ = target_model(input_ids, attention_mask=attention_mask)

    # Measure time for single token generation
    num_runs = 100  # Number of repetitions for averaging
    total_time = 0

    for _ in range(num_runs):
        start_time = time.time()
        with torch.no_grad():
            outputs = target_model(input_ids, attention_mask=attention_mask, return_dict=True)
            logits = outputs.logits[:, -1, :]  # Logits for the last token
            _ = torch.argmax(logits, dim=-1)  # Simulate single token prediction
        end_time = time.time()
        total_time += (end_time - start_time)

    avg_time_per_token = total_time / num_runs
    print(f"Average time per token (single token generation): {avg_time_per_token:.6f} seconds")
    return avg_time_per_token

# Example usage
# Instantiate SpeculativeDecoder and measure single token time
decoder = SpeculativeDecoder(target_model_name="EleutherAI/gpt-neo-1.3B", draft_model_name="distilgpt2")
single_token_time = measure_single_token_generation_time(
    decoder.Mp, decoder.tokenizer, decoder.device
)


class CNNDailyMailGPT2Dataset(Dataset):
    def __init__(self, data, tokenizer, max_length, prompt_prefix="summarize: "):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.prompt_prefix = prompt_prefix
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        article = self.data[idx]['article']
        highlights = self.data[idx]['highlights']
        input_text = self.prompt_prefix + article
        target_text = highlights

        input_tokens = self.tokenizer.encode_plus(
            input_text, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt'
        )
        target_tokens = self.tokenizer.encode_plus(
            target_text, max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt'
        )

        input_ids = input_tokens['input_ids'].squeeze()
        attention_mask = input_tokens['attention_mask'].squeeze()
        labels = target_tokens['input_ids'].squeeze()

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    

model_name = 'distilgpt2'  # Modify to 'gpt2', 'gpt2-large', etc. as needed

batch_size = 4
max_length = 1024
epochs = 1
seed = 42
# Set up argument parsing for Kaggle notebook
output_dir = f'/kaggle/working/{model_name}_epochs{epochs}_batch{batch_size}_max_len{max_length}'  # Kaggle's working directory for output
os.makedirs(output_dir, exist_ok=True)


# Set random seed
set_seed(seed)

# Load dataset from Hugging Face
dataset = load_dataset('cnn_dailymail', '3.0.0')



train_data = dataset['train']
val_data = dataset['validation']

# Limit train_data to the first 10,000 samples
# train_data = train_data.select(range(50000))
val_data = val_data.select(range(3000))
# Initialize tokenizer
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Create datasets
train_dataset = CNNDailyMailGPT2Dataset(train_data, tokenizer, max_length)

val_dataset = CNNDailyMailGPT2Dataset(val_data, tokenizer, max_length)


# Initialize model
model = GPT2LMHeadModel.from_pretrained(model_name)
model.resize_token_embeddings(len(tokenizer))

# Define data collator
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# Define training arguments
training_args = TrainingArguments(
    output_dir=output_dir,
    num_train_epochs=epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
)

# Create Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Train model
trainer.train()

# Save the model and tokenizer
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)