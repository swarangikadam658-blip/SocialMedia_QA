import pandas as pd
import re
import os
import ast

# Load raw Reddit data
df = pd.read_json("data/raw/reddit_qa_raw.json")

print("Original records:", len(df))

# Remove rows with missing question or body
df = df.dropna(subset=["question_title", "body"])

# Convert to string
df["question_title"] = df["question_title"].astype(str)
df["body"] = df["body"].astype(str)


# Clean text function
def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# Create cleaned versions
df["question_clean"] = df["question_title"].apply(clean_text)
df["answer_clean"] = df["body"].apply(clean_text)


# Remove very short questions/answers
df = df[
    (df["question_clean"].str.len() >= 20) &
    (df["answer_clean"].str.len() >= 20)
]

# Remove duplicate questions
df = df.drop_duplicates(subset=["question_clean"])

# Extract subreddit and score from metadata
def get_metadata_value(metadata, key):
    try:
        if isinstance(metadata, dict):
            return metadata.get(key, "")
        return ""
    except:
        return ""


df["subreddit"] = df["metadata"].apply(
    lambda x: get_metadata_value(x, "subreddit")
)

df["score"] = df["metadata"].apply(
    lambda x: get_metadata_value(x, "score")
)


# Keep only the columns needed for our project
df = df[
    [
        "question_title",
        "question_clean",
        "body",
        "answer_clean",
        "subreddit",
        "score"
    ]
]

# Create processed folder
os.makedirs("data/processed", exist_ok=True)

# Save cleaned dataset
df.to_csv(
    "data/processed/reddit_qa_cleaned.csv",
    index=False
)

print("Cleaning completed!")
print("Final records:", len(df))
print("Saved to: data/processed/reddit_qa_cleaned.csv")

print("\nSample:")
print(df.head(3))