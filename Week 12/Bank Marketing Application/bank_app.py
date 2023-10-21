import numpy as np
from flask import Flask, request, render_template
import pickle
import joblib
from sklearn.preprocessing import RobustScaler, LabelEncoder
import sklearn
import sys
import flask
# print(f"numpy version: {np.__version__}")
# print(f"pickle version: {pickle.__version__}")
# print(f"joblib version: {joblib.__version__}")
# print(f"sklearn version: {sklearn.__version__}")
# print(f"sys version: {sys.version}")
# print(f"Flask version: {flask.__version__}")
app=Flask(__name__)
model_path = "models\\final-model-bank-stacked.pkl"
bank_model = pickle.load(open(model_path, "rb"))
label_encoder = joblib.load("models\\label_encoder.pkl")
robust_scaler = joblib.load("models\\robust_scaler.pkl")

@app.route('/')
def home():
    return render_template("templates\\bankmarketing.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        if request.method == "POST":
            
            housing = 1 if request.form['housing'] == "1" else 0
            loan = 1 if request.form['loan'] == "1" else 0
            
            form_data = {
                'marital': request.form['marital'],
                'education': request.form['education'],
                'housing': housing,
                'loan': loan,
                'balance': float(request.form['balance']),
                'contact': request.form['contact'],
                'campaign': int(request.form['campaign']),
                'job': request.form['job'],
                'poutcome': request.form['poutcome'],
                'previous': int(request.form['previous']),  
                'duration': int(request.form['duration']), 
                'pdays': int(request.form['pdays']), 
            }
            print(form_data)
            features_to_be_encoded = ["marital", "education", "contact", "poutcome", "job"]
            features_to_be_standardized = ["pdays", "previous", "campaign", "duration", "balance"]
            
            # Encoding categorical features
            for feature in features_to_be_encoded:
                form_data[feature] = label_encoder.transform([form_data[feature]])[0]
            
            # Standardizing numeric features
            for feature in features_to_be_standardized:
                form_data[feature] = robust_scaler.transform([[form_data[feature]]])[0]
                
            int_features = list(form_data.values())
            bank_prediction = bank_model.predict([int_features]) 
            
            return render_template("templates\\bankresult.html", prediction=bank_prediction[0])
        
    except Exception as e:
        print("Error Occurred!")
        print("Form Data: ")
        print(request.form)
        print("\nError Message: ")
        print(str(e))
        return "An error occurred. Check the server logs for details.", 500


if __name__ == "__main__":
    app.run(debug=True)
    