import json
import torch
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
import os

# --- Configuration ---
BASE_MODEL = "./phi-2"
# The script reads from the pre-split training data file.
CSV_DATA_PATH = "train_data.csv"
OUTPUT_MODEL_PATH = "./phi2-csv-finetuned"
# Set this to match the context you want the model to learn
CONTEXT_WINDOW = 20


def prepare_dataset_from_csv():
    """
    Loads the pre-split training data and formats it for fine-tuning.
    """
    if not os.path.exists(CSV_DATA_PATH):
        print(f"Training data '{CSV_DATA_PATH}' not found.")
        print("Please run the 'prepare_data.py' script first to create it.")
        exit()

    print(f"Loading training data from {CSV_DATA_PATH}")
    df = pd.read_csv(CSV_DATA_PATH)

    header = ",".join(df.columns)
    data_rows = [",".join(map(str, row)) for row in df.values]

    # Create prompt-completion pairs from the training data
    formatted_data = []
    for i in range(len(data_rows) - CONTEXT_WINDOW):
        context = data_rows[i: i + CONTEXT_WINDOW]
        completion = data_rows[i + CONTEXT_WINDOW]

        prompt_text = f"###HEADER:\n{header}\n###CONTEXT:\n" + "\n".join(context)
        full_text = f"{prompt_text}\n###NEXT_ROW:\n{completion}"

        formatted_data.append({"text": full_text})

    print(f"Created {len(formatted_data)} training examples.")
    return Dataset.from_list(formatted_data)


# --- Main Fine-Tuning Logic ---
# 1. Prepare Dataset
dataset = prepare_dataset_from_csv()

# 2. Load Tokenizer and Model from local path
print(f"Loading base model and tokenizer from local path: {BASE_MODEL}")
if not os.path.exists(BASE_MODEL):
    print(f"Error: Model directory not found at '{BASE_MODEL}'.")
    print("Please download the model manually into the 'phi-2' folder.")
    exit()

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True,
    dtype=torch.bfloat16
)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# 3. Tokenize the dataset
def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=512)


tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# 4. Set Up Training Arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_MODEL_PATH,
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-5,
    warmup_steps=10,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=5,
    save_strategy="epoch",
)

# 5. Initialize Trainer
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

# 6. Start Fine-Tuning
print("Starting fine-tuning on the training data...")
trainer.train()
print("Fine-tuning complete.")

# 7. Save the Final Model and Tokenizer
print(f"Saving the fine-tuned model to {OUTPUT_MODEL_PATH}")
trainer.save_model(OUTPUT_MODEL_PATH)
tokenizer.save_pretrained(OUTPUT_MODEL_PATH)
print("Model saved successfully!")

