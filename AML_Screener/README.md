# Bitcoin AML Screener

This is a Flask-based Bitcoin Wallet Risk Classification tool created to detect potentially risky wallets using machine learning, clustering, and heuristic scoring.

---

## Features

- Real-time risk analysis of Bitcoin addresses
- Manual scoring system based on transaction patterns
- Machine learning classification (Random Forest, SVM, etc.)
- Cluster analysis to improve model understanding
- Trigger system for risk indicators (e.g., large transfers, high tx count)
- Combined scoring (ML + manual + cluster bumping)
- Clean chatbot-style web UI with risk history

---

## Project Structure

AML_SCREENER/
├── templates/index.html # Flask template
├── api_extracted_suspicious_features.csv # Features from suspicious wallets
├── api_extracted_wallet_features.csv # Features from labeled wallets
├── app.py # Main Flask app
├── btc_wallets_data.csv # Raw wallet info for suspicious wallets
├── combined_wallet_features.csv # Final merged features used for training
├── features_address.ipynb # Jupyter notebook for feature extraction
├── kmeans_cluster_model.pkl # Saved clustering model
├── labels_transactionsagg.csv # Raw wallet for labeled wallets
├── wallet_risk_model.pkl # Best-performing ML model
├── wallets_labeled.csv # Labeled wallet dataset
├── .gitignore # Git ignore file
└── README.md # Project documentation

---

## How It Works

1. Data Preparation: Combined wallet data labeled as risky or non-risky
2. Feature Engineering: Used BlockCypher API to extract real wallet activity features
3. Manual Risk Scoring: Weighted formula considers tx count, confirmations, etc.
4. Model Training: Chose the best classifier using macro-F1 score
5. Clustering: Applied KMeans to find behavior patterns and bump the score of risky clusters
6. Combined Score: Final score is a weighted mix of ML probability + manual score we calculated
7. Flask App: Enter address, get instant risk score, result, and triggers

---

## .gitignore

This project ignores:

- Virtual environments (`.venv/`)
- Bytecode and caches (`__pycache__`, `.pyc`)
- IDE configs (`.idea/`)
- API keys/configs (`.env`)

---

## How To Run the Project

1. Make sure a .env file exists in the root directory with your BlockCypher API key, like this:
   BLOCKCYPHER_API_KEY=your_key_here
2. Run the Flask app:
   python app.py
