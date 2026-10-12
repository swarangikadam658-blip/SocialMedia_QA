import pandas as pd
import re
import nltk

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Load dataset
df = pd.read_csv("data/processed/reddit_qa_cleaned.csv")

print("Records loaded:", len(df))


# NLP tools
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()


# NLP preprocessing function
def preprocess_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Keep only letters and spaces
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    # Tokenization
    tokens = word_tokenize(text)

    # Remove stopwords and lemmatize
    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words
    ]

    # Convert tokens back to text
    return " ".join(tokens)


# Apply NLP preprocessing
df["question_nlp"] = df["question_clean"].apply(preprocess_text)


# Remove empty results
df = df[df["question_nlp"].str.strip() != ""]


# Save NLP-ready dataset
df.to_csv(
    "data/processed/reddit_qa_nlp.csv",
    index=False
)


print("NLP preprocessing completed!")
print("Final records:", len(df))

print("\nExample:")
print("Original:")
print(df["question_title"].iloc[0])

print("\nAfter NLP preprocessing:")
print(df["question_nlp"].iloc[0])