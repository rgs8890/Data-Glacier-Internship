import pandas as pd
import numpy as np
from flask import Flask, request, render_template
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

standard_scaler = joblib.load("models\label_encoder_grade.pkl")
label_encoder = joblib.load("models\scaler_grade.pkl")
app = Flask(__name__)

model_path = "models\svm_model.pkl"
loaded_svm = pickle.load(open(model_path, "rb"))

@app.route('/')
def home():
    return render_template("schoolgrade1.html")

@app.route('/predict', methods=["POST"])
def predict():
    if request.method == "POST":
        
        boolean_variables = {
            "schoolsup": "schoolsup",
            "famsup": "famsup",
            "paid": "paid",
            "activities": "activities",
            "nursery": "nursery",
            "higher": "higher",
            "internet": "internet",
            "romantic": "romantic"
        }
        
        binary_values = {}
        for variable, form_name in boolean_variables.items():
            binary_values[variable] = 1 if request.form[form_name] == "Yes" else 0
            
        form_data = {
            'school': request.form['school'],
            'sex': request.form['sex'],
            'age': int(request.form['age']),
            'address': request.form['address'],
            'famsize': request.form['famsize'],
            'Pstatus': request.form['Pstatus'],
            'Medu': int(request.form['Medu']),
            'Fedu': int(request.form['Fedu']),
            'Mjob': request.form['Mjob'],
            'Fjob': request.form['Fjob'],
            'reason': request.form['reason'],
            'guardian': request.form['guardian'],
            'traveltime': int(request.form['traveltime']),
            'studytime': int(request.form['studytime']),
            'failures': int(request.form['failures']),
            **binary_values,
            'famrel': int(request.form['famrel']),
            'freetime': int(request.form['freetime']),
            'goout': int(request.form['goout']),
            'Dalc': int(request.form['Dalc']),
            'Walc': int(request.form['Walc']),
            'health': int(request.form['health']),
            'absences': int(request.form['absences']),
            'G1': int(request.form['G1']),
            'G2': int(request.form['G2'])
        }
        
        variables_to_encode = ['guardian', 'Mjob', 'Fjob', 'reason', 'Pstatus', "famsize", "address", "sex", "school"]
        for variable in variables_to_encode:
            form_data[variable] = label_encoder.transform([form_data[variable]])[0]
        
        data_list = [list(form_data.values())]
        scaled_data = standard_scaler.transform(data_list)
        for i, (key, _) in enumerate(form_data.items()):
            form_data[key] = scaled_data[0][i]
        
        student_prediction = loaded_svm.predict(form_data)
        return render_template("result.html", prediction=student_prediction)

if __name__ == '__main__':
    app.run(debug=True)