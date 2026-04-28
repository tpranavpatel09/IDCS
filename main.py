from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

try:
    import mlflow
except ImportError:  # MLflow is optional for quick local runs.
    mlflow = None

try:
    import spacy
except ImportError:
    spacy = None


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = (
    BASE_DIR / "investment_documents_10000.csv"
    if (BASE_DIR / "investment_documents_10000.csv").exists()
    else BASE_DIR / "investment_documents_1000.csv"
)
MODEL_DIR = BASE_DIR / "models"
TAGGER_PATH = MODEL_DIR / "idcs_tagger.joblib"
RETRIEVER_PATH = MODEL_DIR / "idcs_retriever.joblib"
RISK_MODEL_PATH = MODEL_DIR / "chronic_disease_risk_model.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

_NLP = None


def _get_nlp():
    global _NLP
    if _NLP is None and spacy is not None:
        try:
            _NLP = spacy.load("en_core_web_sm", disable=["parser", "ner"])
        except OSError:
            _NLP = False
    return _NLP


def preprocess_text(text: str) -> str:
    text = str(text).lower()
    nlp = _get_nlp()
    if nlp:
        doc = nlp(text)
        return " ".join(
            token.lemma_
            for token in doc
            if token.is_alpha and not token.is_stop
        )
    tokens = re.findall(r"[a-z]+", text)
    return " ".join(token for token in tokens if len(token) > 2)


def preprocess_batch(texts):
    return [preprocess_text(text) for text in texts]


def load_documents(path: Path = DATA_PATH) -> pd.DataFrame:
    data = pd.read_csv(path)
    required_columns = {"doc_id", "text", "label"}
    missing = required_columns.difference(data.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return data.dropna(subset=["text", "label"]).reset_index(drop=True)


def build_tagging_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("clean", FunctionTransformer(preprocess_batch, validate=False)),
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=50000)),
            (
                "classifier",
                LogisticRegression(max_iter=1000, class_weight="balanced"),
            ),
        ]
    )


def train_document_tagger(data: pd.DataFrame) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(
        data["text"],
        data["label"],
        test_size=0.3,
        random_state=42,
        stratify=data["label"],
    )

    tagger = build_tagging_pipeline()
    tagger.fit(X_train, y_train)
    predictions = tagger.predict(X_test)

    report = classification_report(y_test, predictions, output_dict=True)
    weighted_precision = precision_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(tagger, TAGGER_PATH)

    metrics = {
        "tagging_accuracy": report["accuracy"],
        "weighted_precision": weighted_precision,
        "manual_tagging_effort_reduction_target": 0.60,
        "production_performance_threshold": 0.90,
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2))

    if mlflow is not None:
        with mlflow.start_run(run_name="idcs-document-tagger"):
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(tagger, "document_tagger")

    return metrics


def build_retriever(data: pd.DataFrame) -> dict:
    cleaned_text = preprocess_batch(data["text"])
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=75000)
    matrix = vectorizer.fit_transform(cleaned_text)
    retriever = {
        "vectorizer": vectorizer,
        "matrix": matrix,
        "documents": data[["doc_id", "text", "label"]].to_dict(orient="records"),
    }
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(retriever, RETRIEVER_PATH)
    return retriever


def query_documents(question: str, top_k: int = 5) -> dict:
    retriever = joblib.load(RETRIEVER_PATH)
    query_vector = retriever["vectorizer"].transform([preprocess_text(question)])
    scores = cosine_similarity(query_vector, retriever["matrix"]).ravel()
    ranked_indexes = scores.argsort()[::-1][:top_k]

    sources = []
    for index in ranked_indexes:
        document = retriever["documents"][int(index)]
        sources.append(
            {
                "doc_id": document["doc_id"],
                "label": document["label"],
                "score": round(float(scores[index]), 4),
                "snippet": document["text"][:300],
            }
        )

    answer = (
        "The most relevant investment research documents are listed in sources. "
        "Connect this retrieval context to a HuggingFace/LangChain LLM chain for "
        "full natural-language answer generation with citations."
    )
    return {"question": question, "answer": answer, "sources": sources}


def train_chronic_disease_risk_model() -> dict:
    training_data = pd.DataFrame(
        [
            [63, 8.2, 145, 32, 7200, 1],
            [41, 5.5, 118, 24, 10500, 0],
            [58, 7.4, 138, 29, 6100, 1],
            [36, 5.1, 112, 22, 12000, 0],
            [70, 8.8, 152, 34, 4300, 1],
            [49, 6.0, 124, 26, 8800, 0],
        ],
        columns=["age", "a1c", "systolic_bp", "bmi", "daily_steps", "high_risk"],
    )

    features = ["age", "a1c", "systolic_bp", "bmi", "daily_steps"]
    X = training_data[features]
    y = training_data["high_risk"]
    model = Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("classifier", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]
    )
    model.fit(X, y)
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump({"model": model, "features": features}, RISK_MODEL_PATH)
    return {"risk_model": "trained", "features": features}


def monitor_model_performance(metrics: dict) -> dict:
    threshold = metrics.get("production_performance_threshold", 0.90)
    accuracy = metrics.get("tagging_accuracy", 0)
    return {
        "accuracy": accuracy,
        "threshold": threshold,
        "auto_retraining_triggered": accuracy < threshold,
    }


def train_all() -> dict:
    data = load_documents()
    tagging_metrics = train_document_tagger(data)
    build_retriever(data)
    risk_metrics = train_chronic_disease_risk_model()
    monitoring = monitor_model_performance(tagging_metrics)
    return {
        "tagging": tagging_metrics,
        "retriever": str(RETRIEVER_PATH),
        "risk_model": risk_metrics,
        "monitoring": monitoring,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="IDCS RAG and ML tagging pipeline")
    parser.add_argument("--train", action="store_true", help="Train and save all models")
    parser.add_argument("--query", type=str, help="Ask a natural-language document question")
    parser.add_argument("--top-k", type=int, default=5, help="Number of documents to retrieve")
    args = parser.parse_args()

    if args.train:
        print(json.dumps(train_all(), indent=2))
    elif args.query:
        print(json.dumps(query_documents(args.query, args.top_k), indent=2))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
