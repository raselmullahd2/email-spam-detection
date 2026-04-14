import re
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

DATA_PATH = "Nazario_5.csv"


def clean_text(text):
    if pd.isna(text):
        return ""

    text = str(text)
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)
    text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " ", text)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_labels(series: pd.Series) -> pd.Series:
    mapping = {
        "spam": 1,
        "phishing": 1,
        "malicious": 1,
        "junk": 1,
        "ham": 0,
        "legitimate": 0,
        "safe": 0,
        "normal": 0,
        "not spam": 0,
    }

    # Case 1: already numeric 0/1
    if pd.api.types.is_numeric_dtype(series):
        vals = set(series.dropna().unique().tolist())
        if vals.issubset({0, 1}):
            return series.astype(int)

    lowered = series.astype(str).str.strip().str.lower()
    mapped = lowered.map(mapping)

    if mapped.isna().any():
        print("Unmapped labels found:")
        print(series[mapped.isna()].dropna().unique())
        raise ValueError("Please inspect label values and update normalize_labels().")

    return mapped.astype(int)


df = pd.read_csv(DATA_PATH)

print("Columns found:", df.columns.tolist())
print("Raw label values:")
print(df["label"].value_counts(dropna=False))

df["subject"] = df["subject"].fillna("").astype(str)
df["body"] = df["body"].fillna("").astype(str)

df["combined_text"] = (df["subject"] + " " + df["body"]).apply(clean_text)
df["target"] = normalize_labels(df["label"])

df = df[df["combined_text"].str.len() > 0].copy()
df = df[["combined_text", "target"]].reset_index(drop=True)

print("\nFinal target distribution:")
print(df["target"].value_counts(dropna=False))

X_train, X_test, y_train, y_test = train_test_split(
    df["combined_text"],
    df["target"],
    test_size=0.2,
    random_state=42,
    stratify=df["target"],
)

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    stop_words="english"
)

X_train_tfidf = vectorizer.fit_transform(X_train)

model = MultinomialNB(alpha=1.0)
model.fit(X_train_tfidf, y_train)

print("\nModel classes:", model.classes_)

joblib.dump(model, "best_nb_model.pkl")
joblib.dump(vectorizer, "tfidf_vectorizer.pkl")

print("\nSaved:")
print("- best_nb_model.pkl")
print("- tfidf_vectorizer.pkl")