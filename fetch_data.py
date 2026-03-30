import yfinance as yf
import pandas as pd
import numpy as np

print("Fetching GC=F (Gold USD)...")
gold = yf.download('GC=F', start='2010-01-01')
print("Fetching INR=X (USD/INR)...")
usd_inr = yf.download('INR=X', start='2010-01-01')

# Handle MultiIndex
gold_price = gold['Close'].iloc[:, 0] if isinstance(gold['Close'], pd.DataFrame) else gold['Close']
usd_rate = usd_inr['Close'].iloc[:, 0] if isinstance(usd_inr['Close'], pd.DataFrame) else usd_inr['Close']

df = pd.DataFrame({
    'Gold_Price_USD': gold_price,
    'USD_INR': usd_rate
}).dropna()

print("Engineering features...")
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

# Extract only the 7 features
feature_cols = ['Gold_Price_USD', 'USD_INR', 'MA10', 'MA50', 'EMA10', 'Volatility', 'RSI']
final_df = df[feature_cols]

print(f"Saving to latest_engineered_gold.csv with shape {final_df.shape}...")
final_df.to_csv('latest_engineered_gold.csv', index=False, header=False)
print("Done! You can now upload 'latest_engineered_gold.csv' to the web UI.")
