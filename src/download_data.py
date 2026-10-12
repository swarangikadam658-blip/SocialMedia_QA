from datasets import load_dataset
import pandas as pd
import os

print("Loading Reddit QA dataset...")

dataset = load_dataset(
    "EnDevSols/reddit-qa-dataset",
    split="train",
    streaming=True
)

rows = []

# Take records at regular intervals instead of taking
# only the first 10,000 records.
interval = 350

print("Collecting records from different parts of the dataset...")

for i, item in enumerate(dataset):

    if i % interval == 0:
        rows.append({
            "question_title": item.get("question_title", ""),
            "body": item.get("body", ""),
            "metadata": item.get("metadata", {})
        })

        print(f"Collected {len(rows)} records...", end="\r")

    if len(rows) >= 10000:
        break

print()
print(f"Collected {len(rows)} records.")

df = pd.DataFrame(rows)

os.makedirs("data/raw", exist_ok=True)

df.to_json(
    "data/raw/reddit_qa_raw.json",
    orient="records",
    indent=2
)

print("Dataset saved successfully!")
print("File: data/raw/reddit_qa_raw.json")
print(df.head())