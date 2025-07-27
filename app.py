from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
from blockcypher import get_address_details

# Calculate manual score by attaching weightages to each feature
def calculate_score(features):
    total_confirmations = features[8]
    avg_tx_value = features[9]
    num_txrefs = features[7]
    final_n_tx = features[6]
    n_tx = features[3]

    score = (
        0.45 * min(total_confirmations / 1000, 1) +
        0.13 * min(avg_tx_value / 1e7, 1) +
        0.11 * min(num_txrefs / 100, 1) +
        0.08 * min(final_n_tx / 100, 1) +
        0.07 * min(n_tx / 100, 1)
    )
    return round(score * 100, 2)


app = Flask(__name__)
model = joblib.load('wallet_risk_model.pkl')
kmeans = joblib.load('kmeans_cluster_model.pkl')

chat_history = []

# Define the features the model was trained on
FEATURE_NAMES = [
    'balance', 
    'total_received', 
    'total_sent', 
    'n_tx', 
    'final_balance', 
    'unconfirmed_balance', 
    'final_n_tx', 
    'num_txrefs', 
    'total_confirmations', 
    'avg_tx_value'
]

def get_wallet_features(address):
    try:
        data = get_address_details(address)

        balance = data.get('balance', 0)
        total_received = data.get('total_received', 0)
        total_sent = data.get('total_sent', 0)
        n_tx = data.get('n_tx', 0)
        final_balance = data.get('final_balance', 0)
        unconfirmed_balance = data.get('unconfirmed_balance', 0)
        final_n_tx = data.get('final_n_tx', 0)

        txrefs = data.get('txrefs', [])
        num_txrefs = len(txrefs)
        total_confirmations = sum(tx.get('confirmations', 0) for tx in txrefs)
        avg_tx_value = sum(tx.get('value', 0) for tx in txrefs) / num_txrefs if num_txrefs > 0 else 0

        recent_txrefs = txrefs[:5]
        recent_total_value = sum(tx.get('value', 0) for tx in recent_txrefs)
        recent_avg_value = recent_total_value / len(recent_txrefs) if recent_txrefs else 0

        return [
            balance, total_received, total_sent, n_tx, final_balance, unconfirmed_balance, 
            final_n_tx, num_txrefs, total_confirmations, avg_tx_value, recent_avg_value
        ]

    except Exception as e:
        print("Error fetching wallet features:", e)
        return None


@app.route('/', methods=['POST', 'GET'])
def wallet_screener():
    global chat_history
    result = None
    address = None
    risk_score = None
    risk_level = None
    error = None
    manual_score = None

    try:
        if request.method == 'POST':
            address = request.form['wallet_address']
            features = get_wallet_features(address)

            if features:
                input_data = pd.DataFrame([features[:10]], columns=FEATURE_NAMES)
                cluster = int(kmeans.predict(input_data)[0])    # Predict cluster
                probability = model.predict_proba(input_data)[0][1]
                manual_score = calculate_score(features) / 100
                
                # Bump score depending on cluster
                if cluster == 0:  # ~50% risky
                    probability += 0.05
                    probability = min(probability, 1.0)

                # Combine model + manual
                combined_score = 0.8 * probability + 0.2 * manual_score
                risk_score = f"{combined_score:.2%}"

                # Define custom risk levels
                if combined_score >= 0.85:
                    risk_level = "High Risk"
                elif combined_score >= 0.6:
                    risk_level = "Moderate Risk"
                else:
                    risk_level = "Low Risk"

                result = "Risky wallet" if combined_score >= 0.6 else "Not a risky wallet"

                # Trigger notes
                trigger_flag = []
                if features[3] > 1000:
                    trigger_flag.append("High transaction count")
                if features[2] > 1e7:
                    trigger_flag.append("Large amount sent")
                if features[9] < 10:
                    trigger_flag.append("Few confirmations")
                if features[10] > features[9] * 3:
                    trigger_flag.append("Spike in transaction value")

                triggers = "; ".join(trigger_flag)

                chat_history.append({
                    "address": address,
                    "risk_score": risk_score,
                    "risk_level": risk_level,
                    "result": result,
                    "triggers": triggers,
                    "risk_class": risk_level.lower().replace(" ", "-")
                })
            else:
                error = "Could not retrieve features for this address."
    
    except Exception as e:
        error = f"Internal error: {str(e)}"

    return render_template(
        'index.html',
        chat_history=chat_history,
        error=error
    )
    
if __name__ == '__main__':
    app.run(debug=True)
