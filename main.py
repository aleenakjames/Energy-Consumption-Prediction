from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import joblib
import numpy as np
import json

app = Flask(__name__)

def calculate_kseb_bill(units):
    if units <= 40:
        return units * 2.5
    elif units <= 80:
        return 100 + (units - 40) * 3.5
    elif units <= 120:
        return 100 + 140 + (units - 80) * 4.6
    elif units <= 200:
        return 100 + 140 + 184 + (units - 120) * 6.6
    elif units <= 300:
        return 100 + 140 + 184 + 480 + (units - 200) * 7.5
    else:
        return 100 + 140 + 184 + 480 + 750 + (units - 300) * 7.9

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    microwave = float(request.form['Microwave'])
    humidity = float(request.form['Humidity'])
    windspeed = float(request.form['Windspeed'])
    home_office = float(request.form['HomeOffice'])
    precipitation_intensity = float(request.form['PrecipitationIntensity'])
    fridge = float(request.form['Fridge'])
    solar = float(request.form['Solar'])
    living_room = float(request.form['Living_room'])
    temperature = float(request.form['Temperature'])

    # Load the trained model
    loaded_model = joblib.load('energy_model.pkl')

    # Load the scaler object
    scaler = joblib.load('scaler.pkl')
    features_to_scale =[windspeed, home_office, microwave]

    data = [fridge,living_room,windspeed,microwave,precipitation_intensity,home_office,temperature,humidity,solar]
    for i,feature in enumerate(data):
        if feature in features_to_scale:
           data[i] = scaler.transform([[data[i]]])[0][0]
    # Make prediction using the loaded model
    predicted_energy = loaded_model.predict([data])
    pred = predicted_energy* 60 

    print(f"Predicted energy consumption: {predicted_energy[0]} kWh")

    # Calculate KSEB bill
    bill_amount = calculate_kseb_bill(pred[0])
    print(f"KSEB Bill Amount: Rs. {bill_amount}")

    response = {'result': bill_amount}
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)