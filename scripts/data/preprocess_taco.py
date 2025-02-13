import json
import os
import re
import numpy as np
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

# Load dataset
ds = load_dataset("kings-crown/Isabelle_Proofs")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")

# Convert dataset to DataFrame
for split in ds.keys():
    df = pd.DataFrame(ds[split])

    # Assign unique task IDs
    df["task_id"] = np.arange(len(df))

    # Define mappings to match dataset format
    df.rename(
        columns={
            "natural_language_statement": "problem",
            "isabelle_translation": "translation",
            "informal_proof": "informal_proof",
            "formal_proof": "formal_proof",
            "isabelle_body": "isabelle_body",
        },
        inplace=True,
    )

    # Remove leading instructions like "Here's how you can structure the proof..."
    def clean_proof_text(text):
        if isinstance(text, str):
            text = re.sub(r".*?(Here's how you can structure the proof in Isabelle:|Here is a structured Isabelle proof for the lemma:|Here's how you might structure the proof:)", "", text, flags=re.DOTALL).strip()
            return text
        return text
        
    df["isabelle_body"] = df["isabelle_body"].apply(clean_proof_text)

    # Extract only content between ':' and 'qed' in Isabelle body
    def extract_between_colon_qed(text):
        if isinstance(text, str):
            match = re.search(r":(.*?)qed", text, re.DOTALL)
            return match.group(1).strip() + " qed" if match else text
        return text

    df["isabelle_body"] = df["isabelle_body"].apply(extract_between_colon_qed)

    # Filter out excessively long statements for training
    df["problem_len"] = df["problem"].apply(lambda x: len(tokenizer.tokenize(x)))
    df = df[(df["problem_len"] < 2048) & (df["problem_len"] > 64)]

    # Deduplicate based on problem statement
    df = df.drop_duplicates(subset=["problem"])

    # Save to JSONL format for training
    os.makedirs("scripts/data/isabelle", exist_ok=True)
    df.to_json(
        f"scripts/data/isabelle/{split}.jsonl",
        lines=True,
        orient="records",
        force_ascii=False,
    )

    print(f"Processed {split} split with {len(df)} examples.")

# Display modified dataset
import ace_tools as tools
tools.display_dataframe_to_user(name="Processed Isabelle Proofs", dataframe=df)
