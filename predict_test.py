import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import pickle
import logging
import os

# Suppress tf warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  
logging.getLogger('tensorflow').setLevel(logging.FATAL)

try:
    df = pd.read_csv('latest_engineered_gold.csv', header=None)
    last_60 = df.tail(60).values
    
    print("Loading model...")
    model = load_model('gold_model.h5', compile=False)
    f_scaler = pickle.load(open('f_scaler.pkl', 'rb'))
    t_scaler = pickle.load(open('t_scaler.pkl', 'rb'))

    print("Scaling...")
    scaled_input = f_scaler.transform(last_60)
    scaled_input = scaled_input.reshape(1, 60, 7)

    print("Predicting...")
    prediction_scaled = model.predict(scaled_input, verbose=0)
    
    print("Inverse transforming...")
    final_price_inr = t_scaler.inverse_transform(prediction_scaled)

    print(f"SUCCESS_PREDICTION: {float(final_price_inr[0][0]):.2f}")
except Exception as e:
    import traceback
    print(f"ERROR: {e}")
    traceback.print_exc()
