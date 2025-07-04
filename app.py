from flask import Flask, render_template, request
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import joblib

app = Flask(__name__)

# Load the scaler and model
scaler = joblib.load('scaler.pkl')
classifier = joblib.load('model.pkl')

# Define feature names used in training
feature_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']  # 9 features

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data and convert to float
    input_values = [float(request.form.get(feat)) for feat in feature_names]

    # Create DataFrame with feature names
    input_df = pd.DataFrame([input_values], columns=feature_names)

    # Standardize the input
    std_data = scaler.transform(input_df)

    # Predict
    prediction = classifier.predict(std_data)[0]

    result = "Diabetic" if prediction == 1 else "Non-Diabetic"
    return render_template('index.html', prediction_text=f"The person is {result}")

if __name__ == "__main__":
    app.run(debug=True)


import joblib

# After fitting
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(classifier, 'model.pkl')
