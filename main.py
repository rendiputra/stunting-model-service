# python3 -m flask run
# gunicorn main:app -b 127.0.0.1:5000
from flask import Flask, jsonify, request, make_response

import joblib
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
# app.config["DEBUG"] = False

@app.route('/', methods=['GET'])
def hello():
  data = 'Hallo World'

  # return make_response(jsonify({'data': data}), 200)
  return jsonify({'data': data})

@app.route('/tt-u', methods=['GET'])
def stunting():
  data = request.get_json()
  umur = data['umur']
  jk = data['jk']
  tinggi = data['tinggi']

  knn = joblib.load('knn_stunting.model')
  test = pd.DataFrame({'Umur (bulan)': [umur], 'Jenis Kelamin': [jk], 'Tinggi Badan (cm)': [tinggi]})
  pred = knn.predict(test)
  a = np.array(pred)
  b = a.tolist()

  # return make_response(jsonify({'data': b}), 200)
  return jsonify({'data': b})

@app.route('/bb-u', methods=['GET'])
def berat():
  data = request.get_json()
  umur = data['umur']
  jk = data['jk']
  berat = data['berat']

  knn = joblib.load('knn_berat_status_gizi.model')
  test = pd.DataFrame({'Umur (bulan)': [umur], 'Berat Badan (kg)': [berat], 'Jenis Kelamin': [jk]})
  pred = knn.predict(test)
  a = np.array(pred)
  b = a.tolist()

  # return make_response(jsonify({'data': b}), 200)
  return jsonify({'data': b})

  
# app.run(host='0.0.0.0', port=8080)
# app.run(debug=False, use_debugger=False, use_reloader=False, port=os.getenv("PORT", default=5000))
if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))