import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

# --- Configuration ---
FINETUNED_MODEL_PATH = "./phi2-csv-finetuned"
CSV_DATA_PATH = "test_data.csv"
# --- MODIFIED: Set to your training context window ---
CONTEXT_WINDOW = 20
# The file where the newly generated data will be saved
OUTPUT_CSV_PATH = "generated_output.csv"


def run_demo():
    """
    Loads the fine-tuned model and generates multiple new CSV rows,
    saving them to a new file.
    """
    # 1. Check if model and data exist
    if not os.path.exists(FINETUNED_MODEL_PATH) or not os.path.exists(CSV_DATA_PATH):
        print(f"Error: Model or test data not found. Please run preparation and training scripts first.")
        return

    # --- NEW: Get user input for number of rows ---
    while True:
        try:
            num_rows_to_generate = int(input("How many new rows would you like to generate? "))
            if num_rows_to_generate > 0:
                break
            else:
                print("Please enter a positive number.")
        except ValueError:
            print("Invalid input. Please enter a whole number.")
    # --- END NEW ---

    print(f"Loading fine-tuned model from: {FINETUNED_MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(FINETUNED_MODEL_PATH, dtype=torch.bfloat16)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model loaded on device: {device}")

    # 2. Prepare the initial prompt from the TEST CSV
    print(f"\nPreparing initial prompt from '{CSV_DATA_PATH}'...")
    df = pd.read_csv(CSV_DATA_PATH)
    header = ",".join(df.columns)

    available_rows = len(df)
    rows_to_use = min(available_rows, CONTEXT_WINDOW)

    if rows_to_use == 0:
        print("Error: The CSV file has no data rows to use as context.")
        return

    print(f"Using the last {rows_to_use} rows as the starting context.")

    # This context will be updated in a sliding window
    context_rows = [",".join(map(str, row)) for row in df.tail(rows_to_use).values]

    # 3. Generate new rows in a loop
    generated_rows = []
    print(f"\nGenerating {num_rows_to_generate} new rows...")
    for i in range(num_rows_to_generate):
        # Format the prompt with the current context
        prompt_text = f"###HEADER:\n{header}\n###CONTEXT:\n" + "\n".join(context_rows) + "\n###NEXT_ROW:\n"

        inputs = tokenizer(prompt_text, return_tensors="pt").to(device)

        outputs = model.generate(
            **inputs,
            max_new_tokens=50,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.encode("\n")[0]
        )

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        new_row = generated_text.split("###NEXT_ROW:\n")[-1].strip()

        if not new_row:  # Handle cases where generation might fail
            print(f"Warning: Generation failed at step {i + 1}. Stopping.")
            break

        print(f"  ({i + 1}/{num_rows_to_generate}) Generated: {new_row}")

        # Add the new row to our results
        generated_rows.append(new_row)

        # Update the context for the next iteration (sliding window)
        # Remove the oldest row and add the newest one
        if len(context_rows) >= CONTEXT_WINDOW:
            context_rows.pop(0)
        context_rows.append(new_row)

    # 4. Save the generated rows to a new CSV file
    if generated_rows:
        print(f"\nSaving {len(generated_rows)} new rows to '{OUTPUT_CSV_PATH}'...")
        # Split the string rows back into lists of values
        output_data = [row.split(',') for row in generated_rows]
        output_df = pd.DataFrame(output_data, columns=df.columns)
        output_df.to_csv(OUTPUT_CSV_PATH, index=False)
        print("Save complete.")
    else:
        print("\nNo rows were generated.")

    print("\nDemo complete.")


if __name__ == "__main__":
    run_demo()

