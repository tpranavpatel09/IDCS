# IDCS - Investment Document Classification System

IDCS is a prototype of the production AI system described in the project summary:

- Retrieval-Augmented Generation style querying for investment research documents.
- ML-based document tagging with Python, spaCy-style preprocessing, TF-IDF, and scikit-learn.
- Model persistence for repeatable API access.
- MLflow-ready monitoring metrics and auto-retraining trigger logic.
- REST APIs for document querying, document tagging, and risk prediction.
- A small chronic disease risk prediction model to represent the healthcare ML use case.

## Architecture

```text
Investment research documents
        |
        v
Text preprocessing with spaCy fallback
        |
        +--> TF-IDF + Logistic Regression document tagger
        |
        +--> TF-IDF retrieval index for natural-language search
        |
        v
FastAPI endpoints for /query, /tag-document, /predict-risk
        |
        v
MLflow metrics + saved model artifacts
```

The current repository uses TF-IDF retrieval so it can run locally on a laptop. The `requirements.txt` includes LangChain, HuggingFace Transformers, sentence-transformers, and FAISS so the retrieval layer can be upgraded to a full embedding-based RAG pipeline.

## Main files

```text
main.py                         Training, retrieval, monitoring, CLI access
app.py                          FastAPI service
investment_documents_1000.csv   Sample investment document training data
investment_documents_10000.csv  Optional generated 10,000 document dataset
randomdata.py                   Synthetic 10,000 document data generator
requirements.txt                Python dependencies
models/                         Saved model artifacts, created after training
```

## Run locally

Install dependencies:

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

Train the tagging model, retriever, monitoring metrics, and risk model:

```bash
python main.py --train
```

Generate a larger 10,000 document dataset first:

```bash
python randomdata.py
python main.py --train
```

Ask a natural-language question:

```bash
python main.py --query "Which documents discuss equity valuation and market risk?"
```

Start the REST API:

```bash
uvicorn app:app --reload
```

Open the interactive API docs:

```text
http://localhost:8000/docs
```

## API endpoints

### Query investment documents

```http
POST /query
```

```json
{
  "question": "What are the key risks in fixed income research?",
  "top_k": 5
}
```

### Tag an investment document

```http
POST /tag-document
```

```json
{
  "text": "This report reviews earnings, valuation, price target, and sector growth."
}
```

### Predict chronic disease risk

```http
POST /predict-risk
```

```json
{
  "age": 63,
  "a1c": 8.2,
  "systolic_bp": 145,
  "bmi": 32,
  "daily_steps": 7200
}
```

## Production mapping

This project maps to the resume description as follows:

- RAG pipeline: `query_documents()` retrieves the most relevant research documents for natural-language questions.
- HuggingFace and LangChain: dependencies are included for replacing TF-IDF retrieval with transformer embeddings and LLM generation.
- Document tagging: `train_document_tagger()` trains a TF-IDF + Logistic Regression classifier.
- Model monitoring: metrics are saved to `models/metrics.json` and logged to MLflow when available.
- Auto-retraining: `monitor_model_performance()` flags retraining when accuracy falls below the 90% threshold.
- REST integration: `app.py` exposes production-style endpoints for platform integration.
- Healthcare ML: `train_chronic_disease_risk_model()` demonstrates the chronic disease risk prediction model surface.
