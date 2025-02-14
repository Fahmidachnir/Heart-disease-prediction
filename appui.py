import streamlit as st
import joblib
import numpy as np

# Load model and scaler
model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Heart Disease Prediction App")

# Input fields
age = st.number_input("Age", 20, 100, 50)
sex = st.selectbox("Sex (0 = Female, 1 = Male)", [0, 1])
cp = st.selectbox("Chest Pain Type (1-4)", [1, 2, 3, 4])
trestbps = st.number_input("Resting Blood Pressure", 80, 200, 120)
chol = st.number_input("Cholesterol", 100, 400, 200)
fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl (1 = Yes, 0 = No)", [0, 1])
restecg = st.selectbox("Rest ECG (0, 1, 2)", [0, 1, 2])
thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150)
exang = st.selectbox("Exercise Induced Angina (1 = Yes, 0 = No)", [0, 1])
oldpeak = st.number_input("ST Depression", 0.0, 6.0, 1.0)
slope = st.selectbox("Slope of ST Segment (0, 1, 2)", [0, 1, 2])
ca = st.number_input("Number of Major Vessels", 0, 4, 0)
thal = st.selectbox("Thalassemia (3 = Normal, 6 = Fixed, 7 = Reversible)", [3, 6, 7])

# Make prediction
if st.button("Predict"):
    features = np.array([age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]).reshape(1, -1)
    scaled_features = scaler.transform(features)
    prediction = model.predict(scaled_features)[0]
    st.write(f"**Prediction: {'Heart Disease' if prediction > 0 else 'No Heart Disease'}**")


