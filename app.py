from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import os
import pandas as pd

app = Flask(__name__)

# Global variables for model and scalers
model = None
f_scaler = None
t_scaler = None
model_ready = False

def load_ai_assets():
    global model, f_scaler, t_scaler, model_ready
    try:
        model_path = 'gold_model.h5'
        f_scaler_path = 'f_scaler.pkl'
        t_scaler_path = 't_scaler.pkl'

        if os.path.exists(model_path) and os.path.exists(f_scaler_path) and os.path.exists(t_scaler_path):
            print("--- Loading AI Assets ---")
            model = load_model(model_path)
            with open(f_scaler_path, 'rb') as f:
                f_scaler = pickle.load(f)
            with open(t_scaler_path, 'rb') as f:
                t_scaler = pickle.load(f)
            model_ready = True
            print("--- AI Assets Loaded Successfully ---")
        else:
            print("--- ERROR: AI Asset Files Missing ---")
    except Exception as e:
        print(f"--- ERROR loading assets: {str(e)} ---")

# Execute load during startup
load_ai_assets()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if not model_ready:
        return jsonify({'status': 'error', 'message': 'AI Model is warming up... please wait 30s.'}), 503

    try:
        data = request.json
        features = np.array(data['features'])
        
        # Scaling
        features_scaled = f_scaler.transform(features.reshape(-1, features.shape[-1]))
        features_reshaped = features_scaled.reshape(1, 60, features.shape[-1])
        
        # Prediction
        prediction_scaled = model.predict(features_reshaped, verbose=0)
        prediction_inr = t_scaler.inverse_transform(prediction_scaled)
        
        return jsonify({
            'status': 'success',
            'predicted_gold_price_inr': float(prediction_inr[0][0])
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route("/fetch_live", methods=["GET"])
def fetch_live():
    if not model_ready:
        return jsonify({'status': 'error', 'message': 'AI Model is warming up... please wait 30s.'}), 503

    try:
        import yfinance as yf
        
        gold = yf.download('GC=F', period='1y', progress=False)
        usd_inr = yf.download('INR=X', period='1y', progress=False)
        
        gold_price = gold['Close'].iloc[:, 0] if isinstance(gold['Close'], pd.DataFrame) else gold['Close']
        usd_rate = usd_inr['Close'].iloc[:, 0] if isinstance(usd_inr['Close'], pd.DataFrame) else usd_inr['Close']
        
        df = pd.DataFrame({
            'Gold_Price_USD': gold_price,
            'USD_INR': usd_rate
        }).dropna()
        
        df['Gold_Price_INR'] = df['Gold_Price_USD'] * df['USD_INR']
        df['MA10'] = df['Gold_Price_INR'].rolling(10).mean()
        df['MA50'] = df['Gold_Price_INR'].rolling(50).mean()
        df['EMA10'] = df['Gold_Price_INR'].ewm(span=10).mean()
        df['Volatility'] = df['Gold_Price_INR'].rolling(10).std()
        
        delta = df['Gold_Price_INR'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        df = df.dropna()
        feature_cols = ['Gold_Price_USD', 'USD_INR', 'MA10', 'MA50', 'EMA10', 'Volatility', 'RSI']
        final_df = df[feature_cols].tail(60)
        
        return jsonify({
            'status': 'success',
            'features': final_df.values.tolist(),
            'dates': [d.strftime('%Y-%m-%d') for d in final_df.index]
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)