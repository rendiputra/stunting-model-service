from flask import Flask, jsonify, request, make_response

import joblib
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

app = Flask(__name__)
app.config["DEBUG"] = True

@app.route('/', methods=['POST'])
def hello():
  data = request.get_json()
  umur = data['umur']
  jk = data['jk']
  tinggi = data['tinggi']

  knn = joblib.load('knn.model')
  test = pd.DataFrame({'Umur (bulan)': [umur], 'Jenis Kelamin': [jk], 'Tinggi Badan (cm)': [tinggi]})
  pred = knn.predict(test)
  a = np.array(pred)
  b = a.tolist()

  return make_response(jsonify({'data': b}), 200)

app.run()