import pickle
f = pickle.load(open('f_scaler.pkl', 'rb'))
t = pickle.load(open('t_scaler.pkl', 'rb'))
print("Feature scaler max:", f.data_max_)
print("Target scaler max:", t.data_max_)
