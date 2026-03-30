import traceback
from tensorflow.keras.models import load_model

with open("err.txt", "w") as f:
    try:
        model = load_model('gold_model.h5', compile=False)
        f.write("Model loaded successfully")
    except Exception as e:
        f.write(traceback.format_exc())
