
import numpy as np
import pandas as pd
import os
from sklearn import linear_model as OSlm
from sklearn.metrics import mean_squared_error
from flask import Flask, render_template, request
from pandas.io.json import json_normalize

flaskapp = Flask(__name__)

# Run or refresh model
def model(x):
    if x == 1:
        data = pd.read_csv('D:\\2011CS010144\\finalised_dataset.csv')
    data=data.fillna(data.mean())
    dataX = data[['temperature','pressure','area']]
    dataY = data['Yield']

    global OSLM
    OSLM = OSlm.LinearRegression()
    global bbmodel
    OSLM.fit(dataX, dataY)
    bbpred = OSLM.predict(dataX)
    rmse = mean_squared_error(dataY, bbpred)
    return OSLM, rmse


# Predict from the model build
@flaskapp.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        input_values = request.form

    inputX = pd.DataFrame(json_normalize(input_values))
    input = inputX[['temperature', 'pressure', 'area']]
    predval = OSLM.predict(input)
    input['predval'] = predval
    input.columns = ['temperature', 'pressure', 'area', 'Yield Predicted']
    return render_template('predict.html', tables=[input.to_html()], titles=input.columns.values)


# Home page that renders for every web call
@flaskapp.route("/")
def home():
    return render_template("home.html")


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 900))
    global OSLM, Error
    OSLM, Error = model(1)
    flaskapp.run(host='0.0.0.0', port=port, debug=True)

