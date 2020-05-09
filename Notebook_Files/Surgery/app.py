from flask import Flask, render_template, request, url_for
from sklearn.externals import joblib
# import pandas as pd
import numpy as np
import os

app = Flask(__name__)


@app.route("/")
def main():
    return render_template("home.html")

@app.route("/predict", methods = ['GET', 'POST'])
def predict():
    if request.method == "POST":
        try:
            Pregnancies = int(request.form['Pregnancies'])
            Glucose = int(request.form['Glucose'])
            BloodPressure = int(request.form['BloodPressure'])
            SkinThickness = int(request.form['SkinThickness'])
            Insulin = int(request.form['Glucose'])
            BMI = float(request.form['Glucose'])
            DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
            Age = int(request.form['Age'])
            pred_args = [Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]
            pred_args_arr = np.array(pred_args)
            pred_args_arr = pred_args_arr.reshape(1, -1)
            svm = open('SVM.pkl', 'rb')
            ml_model = joblib.load(svm)
            model_prediction = ml_model.predict(pred_args_arr)
            if model_prediction == 1:
                model_prediction = "You have Diabetes, please consult a Doctor."
            elif model_prediction == 0:
                model_prediction = "You don't have Diabetes."
        except ValueError:
            return "Please check whether the values are entered correctly"

    return render_template("predict.html", prediction = model_prediction)
    # return "Heyy!"


if __name__ == "__main__":
    app.run()
    
