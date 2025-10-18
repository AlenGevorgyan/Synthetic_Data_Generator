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
# MODIFIED: The script now points to a local folder for the model.
# Make sure to download the model into a folder named "phi-2" in the same directory.
BASE_MODEL = "./phi-2"
# The path to your sample CSV data.
CSV_DATA_PATH = "sample_data.csv"
# Where the final, fine-tuned model will be saved.
OUTPUT_MODEL_PATH = "./phi2-csv-finetuned"
# How many preceding rows the model should consider as context.
CONTEXT_WINDOW = 20


def prepare_dataset_from_csv():
    """
    Loads data from a CSV and formats it for fine-tuning. The model learns
    to predict the next row based on a 'CONTEXT_WINDOW' of previous rows.
    """
    # 1. Create a dummy CSV if it doesn't exist, so the script can run.
    if not os.path.exists(CSV_DATA_PATH):
        print(f"Sample data '{CSV_DATA_PATH}' not found. Creating a dummy file.")
        dummy_df = pd.DataFrame({
            'ProductID': ['P001', 'P002', 'P003', 'P004', 'P005', 'P006'],
            'Category': ['Electronics', 'Books', 'Electronics', 'Home Goods', 'Books', 'Home Goods'],
            'Price': [299.99, 19.99, 49.99, 129.50, 24.95, 89.99],
            'InStock': [True, True, False, True, True, False]
        })
        dummy_df.to_csv(CSV_DATA_PATH, index=False)

    print(f"Loading data from {CSV_DATA_PATH}")
    df = pd.read_csv(CSV_DATA_PATH)
    header = ",".join(df.columns)

    data_rows = [",".join(map(str, row)) for row in df.values]

    # 2. Create prompt-completion pairs
    formatted_data = []
    for i in range(len(data_rows) - CONTEXT_WINDOW):
        context = data_rows[i: i + CONTEXT_WINDOW]
        completion = data_rows[i + CONTEXT_WINDOW]

        # We provide the header and context rows as the prompt
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
    print("Please follow the instructions to download the model manually.")
    exit()

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, trust_remote_code=True, dtype=torch.bfloat16)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token


# 3. Tokenize the dataset
def tokenize_function(examples):
    # Note: Increased max_length to handle multiple rows of context
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
print("Starting fine-tuning...")
trainer.train()
print("Fine-tuning complete.")

# 7. Save the Final Model and Tokenizer
print(f"Saving the fine-tuned model to {OUTPUT_MODEL_PATH}")
trainer.save_model(OUTPUT_MODEL_PATH)
tokenizer.save_pretrained(OUTPUT_MODEL_PATH)
print("Model saved successfully!")

