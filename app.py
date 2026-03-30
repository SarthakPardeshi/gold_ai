from flask import Flask, request, jsonify, render_template
import numpy as np
from tensorflow.keras.models import load_model
import pickle

app = Flask(__name__)

# 1. Load the AI components (Ensure you save them first in your training script)
# model.save('gold_model.h5')
# pickle.dump(f_scaler, open('f_scaler.pkl', 'wb'))
# pickle.dump(t_scaler, open('t_scaler.pkl', 'wb'))

try:
    model = load_model('gold_model.h5', compile=False)
    f_scaler = pickle.load(open('f_scaler.pkl', 'rb'))
    t_scaler = pickle.load(open('t_scaler.pkl', 'rb'))
except Exception as e:
    print(f"Warning: Model/Scalers not found. Ensure they exist. Error: {e}")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["GET", "POST"])
def predict():
    try:
        # Expecting a JSON with the last 60 days of feature data
        data = request.get_json()
        input_features = np.array(data['features']) # Shape: (60, 7)
        
        # Scale features
        scaled_input = f_scaler.transform(input_features)
        
        # Reshape for LSTM: (1, 60, 7)
        scaled_input = scaled_input.reshape(1, 60, 7)
        
        # Make Prediction
        prediction_scaled = model.predict(scaled_input)
        
        # Inverse Scale to get INR Price
        final_price_inr = t_scaler.inverse_transform(prediction_scaled)
        
        return jsonify({
            'status': 'success',
            'predicted_gold_price_inr': float(final_price_inr[0][0])
        })
    
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

@app.route("/fetch_live", methods=["GET"])
def fetch_live():
    try:
        import yfinance as yf
        import pandas as pd
        
        # Download last 1 year to safely cover 110+ days for moving averages
        gold = yf.download('GC=F', period='1y', progress=False)
        usd_inr = yf.download('INR=X', period='1y', progress=False)
        
        gold_price = gold['Close'].iloc[:, 0] if isinstance(gold['Close'], pd.DataFrame) else gold['Close']
        usd_rate = usd_inr['Close'].iloc[:, 0] if isinstance(usd_inr['Close'], pd.DataFrame) else usd_inr['Close']
        
        df = pd.DataFrame({
            'Gold_Price_USD': gold_price,
            'USD_INR': usd_rate
        }).dropna()
        
        # Engineering accurate AI Features
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
        final_df = df[feature_cols].tail(60) # The strict 60 day sequence
        
        return jsonify({
            'status': 'success',
            'features': final_df.values.tolist(),
            'dates': pd.Series(final_df.index).dt.strftime('%Y-%m-%d').tolist()
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5000)