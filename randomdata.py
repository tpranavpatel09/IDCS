import random
import pandas as pd

NUM_DOCS = 1000
labels = {
    "Equity Research": [
        "equity", "stocks", "earnings", "valuation", "analyst",
        "growth", "shares", "price target", "recommendation", "sector"
    ],
    "Fixed Income": [
        "bonds", "yield", "duration", "interest rates", "credit spread",
        "treasury", "coupon", "maturity", "default risk", "fixed income"
    ],
    "Risk & Compliance": [
        "compliance", "regulation", "risk management", "KYC",
        "AML", "governance", "audit", "controls", "policy", "exposure"
    ],
    "Market Outlook": [
        "macroeconomic", "GDP", "inflation", "market trends",
        "economic cycle", "global markets", "forecast", "outlook", "policy"
    ],
    "Financial Statements": [
        "balance sheet", "income statement", "cash flow",
        "revenue", "expenses", "profit", "assets", "liabilities", "net income"
    ]
}
filler_phrases = [
    "This report provides an analysis of",
    "The document focuses on",
    "We examine recent developments in",
    "This analysis highlights",
    "The following report discusses",
    "A detailed review of"
]
noise_words = [
    "performance", "overview", "quarterly", "annual", "strategic",
    "operational", "key drivers", "historical data", "future expectations"
]
data = []
doc_id = 1
docs_per_label = NUM_DOCS // len(labels)
for label, keywords in labels.items():
    for _ in range(docs_per_label):
        sentence = random.choice(filler_phrases)

        keyword_sample = random.sample(keywords, k=4)
        noise_sample = random.sample(noise_words, k=2)

        text = (
            f"{sentence} "
            f"{', '.join(keyword_sample)}, "
            f"and {noise_sample[0]}. "
            f"The report also covers {noise_sample[1]} "
            f"related to {random.choice(keywords)}."
        )

        data.append({
            "doc_id": doc_id,
            "text": text,
            "label": label
        })
        doc_id += 1

df = pd.DataFrame(data)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
output_path = "investment_documents_1000.csv"
df.to_csv(output_path, index=False)
print(f"Dataset saved to {output_path}")
print(df.head())