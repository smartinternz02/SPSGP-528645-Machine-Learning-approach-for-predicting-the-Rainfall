import numpy as np
import pickle
import joblib
import matplotlib
import matplotlib.pyplot as plt
import time
import pandas
import os
from flask import Flask, request, jsonify, render_template


app = Flask(__name__)
model = pickle.load(open("C:/Users/ASUS/Desktop/Final/IBM endpoint deploy/Rainfall prediction/rainfall.pkl", 'rb'))
scale = pickle.load(open("C:/Users/ASUS/Desktop/Final/IBM endpoint deploy/Rainfall prediction/scale.pkl",'rb'))


@app.route('/') # route to display the home page
def home():
    return render_template('index.html') #rendering the home page

@app.route('/predict',methods=["POST","GET"]) # route to show the predictions in a web UI
def predict():
    #  reading the inputs given by the user
    input_feature=[float(x) for x in request.form.values()]
    features_values = [np.array(input_feature)]
    names = ['Location', 'WindDir3pm', 'MinTemp', 'MaxTemp', 'Rainfall', 'WindGustSpeed',
             'WindSpeed9am', 'WindSpeed3pm', 'Humidity9am', 'Humidity3pm',
             'Pressure9am', 'Pressure3pm', 'Temp9am', 'Temp3pm', 'RainToday',
             'WindGustDir', 'WindDir9am', 'year', 'month', 'day']

    data = pandas.DataFrame(features_values, columns=names)
    data_scaled = scale.transform(data)  # Scale the data using the previously loaded scale object
    data = pandas.DataFrame(data_scaled, columns=names)

    # Predictions using the loaded model file
    prediction = model.predict(data)
    pred_prob = model.predict_proba(data)

    print(prediction)
    if prediction == "yes":
        return render_template("chance.html")
    else:
        return render_template("nochance.html")
     # showing the prediction results in a UI
if __name__=="__main__":
    
    # app.run(host='0.0.0.0', port=8000,debug=True)    # running the app
    port=int(os.environ.get('PORT',5000))
    app.run(port=port,debug=True,use_reloader=False)