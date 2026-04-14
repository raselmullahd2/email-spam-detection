import re
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Dense, Dropout, LSTM, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping

warnings.filterwarnings("ignore")

# -----------------------------
# Reproducibility
# -----------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

# -----------------------------
# Paths
# -----------------------------
DATA_PATH = Path("Nazario_5.csv")   # keep CSV in same folder as script
OUTPUT_DIR = Path("outputs_nazario")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Config
# -----------------------------
TEST_SIZE = 0.2
MAX_FEATURES_TFIDF = 10000
MAX_WORDS = 10000
MAX_SEQUENCE_LEN = 250
EMBEDDING_DIM = 100
BATCH_SIZE = 32
EPOCHS = 10


def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""

    text = str(text)

    # remove HTML
    text = re.sub(r"<.*?>", " ", text)

    # remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", " ", text)

    # remove emails
    text = re.sub(r"\b[\w\.-]+@[\w\.-]+\.\w+\b", " ", text)

    # lowercase
    text = text.lower()

    # keep letters, numbers, spaces
    text = re.sub(r"[^a-z0-9\s]", " ", text)

    # collapse spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_labels(series: pd.Series) -> pd.Series:
    # try string labels first
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

    if pd.api.types.is_numeric_dtype(series):
        vals = set(series.dropna().unique().tolist())
        if vals.issubset({0, 1}):
            return series.astype(int)

    lowered = series.astype(str).str.strip().str.lower()

    # direct mapping
    mapped = lowered.map(mapping)

    if mapped.isna().any():
        print("\nUnmapped labels found:")
        print(series[mapped.isna()].dropna().unique())
        raise ValueError("Please inspect label values and update normalize_labels().")

    return mapped.astype(int)


def evaluate_predictions(y_true, y_pred, y_prob=None):
    results = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_score": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_prob is not None:
        try:
            results["roc_auc"] = roc_auc_score(y_true, y_prob)
        except Exception:
            results["roc_auc"] = np.nan
    else:
        results["roc_auc"] = np.nan

    return results


def save_confusion_matrix(cm, labels, title, filename):
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation="nearest")
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(labels))
    plt.xticks(ticks, labels)
    plt.yticks(ticks, labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def load_nazario_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Could not find {DATA_PATH.resolve()}")

    df = pd.read_csv(DATA_PATH)

    required_cols = ["subject", "body", "label"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print("Columns found:", df.columns.tolist())
    print("Rows before cleaning:", len(df))
    print("\nRaw label values:")
    print(df["label"].value_counts(dropna=False))

    df = df.copy()
    df["subject"] = df["subject"].fillna("").astype(str)
    df["body"] = df["body"].fillna("").astype(str)

    # final variable used in models: subject + body
    df["combined_text"] = (df["subject"] + " " + df["body"]).apply(clean_text)
    df["target"] = normalize_labels(df["label"])

    df = df[df["combined_text"].str.len() > 0].copy()
    df = df[["combined_text", "target"]].reset_index(drop=True)

    print("\nRows after cleaning:", len(df))
    print("\nFinal binary target distribution:")
    print(df["target"].value_counts(dropna=False))

    return df


def run_classical_models(X_train_text, X_test_text, y_train, y_test):
    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES_TFIDF,
        ngram_range=(1, 2),
        stop_words="english",
    )

    X_train_tfidf = vectorizer.fit_transform(X_train_text)
    X_test_tfidf = vectorizer.transform(X_test_text)

    results = []

    # Logistic Regression
    lr = LogisticRegression(
        max_iter=1000,
        solver="liblinear",
        random_state=SEED,
    )
    lr.fit(X_train_tfidf, y_train)
    lr_pred = lr.predict(X_test_tfidf)
    lr_prob = lr.predict_proba(X_test_tfidf)[:, 1]

    lr_metrics = evaluate_predictions(y_test, lr_pred, lr_prob)
    lr_metrics["model"] = "Logistic Regression"
    results.append(lr_metrics)

    save_confusion_matrix(
        confusion_matrix(y_test, lr_pred),
        ["Legitimate", "Spam/Phishing"],
        "Logistic Regression Confusion Matrix",
        "cm_logistic_regression.png",
    )

    # Naive Bayes
    nb = MultinomialNB(alpha=1.0)
    nb.fit(X_train_tfidf, y_train)
    nb_pred = nb.predict(X_test_tfidf)
    nb_prob = nb.predict_proba(X_test_tfidf)[:, 1]

    nb_metrics = evaluate_predictions(y_test, nb_pred, nb_prob)
    nb_metrics["model"] = "Naive Bayes"
    results.append(nb_metrics)

    save_confusion_matrix(
        confusion_matrix(y_test, nb_pred),
        ["Legitimate", "Spam/Phishing"],
        "Naive Bayes Confusion Matrix",
        "cm_naive_bayes.png",
    )

    return pd.DataFrame(results)


def build_dnn_model(vocab_size, max_len):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=max_len),
        GlobalAveragePooling1D(),
        Dense(128, activation="relu"),
        Dropout(0.3),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_lstm_model(vocab_size, max_len):
    model = Sequential([
        Embedding(input_dim=vocab_size, output_dim=EMBEDDING_DIM, input_length=max_len),
        LSTM(128, dropout=0.2, recurrent_dropout=0.2),
        Dense(64, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def run_deep_learning_models(X_train_text, X_test_text, y_train, y_test):
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train_text)

    X_train_seq = tokenizer.texts_to_sequences(X_train_text)
    X_test_seq = tokenizer.texts_to_sequences(X_test_text)

    X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_SEQUENCE_LEN, padding="post", truncating="post")
    X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_SEQUENCE_LEN, padding="post", truncating="post")

    vocab_size = min(MAX_WORDS, len(tokenizer.word_index) + 1)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)
    ]

    results = []

    # DNN
    dnn = build_dnn_model(vocab_size, MAX_SEQUENCE_LEN)
    dnn.fit(
        X_train_pad,
        y_train,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        callbacks=callbacks,
    )
    dnn_prob = dnn.predict(X_test_pad).ravel()
    dnn_pred = (dnn_prob >= 0.5).astype(int)

    dnn_metrics = evaluate_predictions(y_test, dnn_pred, dnn_prob)
    dnn_metrics["model"] = "Dense Neural Network"
    results.append(dnn_metrics)

    save_confusion_matrix(
        confusion_matrix(y_test, dnn_pred),
        ["Legitimate", "Spam/Phishing"],
        "Dense Neural Network Confusion Matrix",
        "cm_dnn.png",
    )

    # LSTM
    lstm = build_lstm_model(vocab_size, MAX_SEQUENCE_LEN)
    lstm.fit(
        X_train_pad,
        y_train,
        validation_split=0.1,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        verbose=1,
        callbacks=callbacks,
    )
    lstm_prob = lstm.predict(X_test_pad).ravel()
    lstm_pred = (lstm_prob >= 0.5).astype(int)

    lstm_metrics = evaluate_predictions(y_test, lstm_pred, lstm_prob)
    lstm_metrics["model"] = "LSTM"
    results.append(lstm_metrics)

    save_confusion_matrix(
        confusion_matrix(y_test, lstm_pred),
        ["Legitimate", "Spam/Phishing"],
        "LSTM Confusion Matrix",
        "cm_lstm.png",
    )

    return pd.DataFrame(results)


def main():
    print("Loading Nazario dataset...")
    df = load_nazario_dataset()

    X = df["combined_text"]
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=SEED,
        stratify=y,
    )

    print("\nTrain size:", len(X_train))
    print("Test size :", len(X_test))

    print("\nRunning classical models...")
    classical_results = run_classical_models(X_train, X_test, y_train, y_test)

    print("\nRunning deep learning models...")
    dl_results = run_deep_learning_models(X_train, X_test, y_train, y_test)

    final_results = pd.concat([classical_results, dl_results], ignore_index=True)
    final_results = final_results[["model", "accuracy", "precision", "recall", "f1_score", "roc_auc"]]
    final_results = final_results.sort_values(by="f1_score", ascending=False).reset_index(drop=True)

    print("\nFinal Results:")
    print(final_results.round(4))

    final_results.to_csv(OUTPUT_DIR / "nazario_model_results.csv", index=False)

    print(f"\nSaved outputs to: {OUTPUT_DIR.resolve()}")


if __name__ == "__main__":
    main()