import functions
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
import streamlit as st
BASE_DIR = Path("Heart attack and diseases model").resolve().parent
scaler_path = BASE_DIR / "NN_scaler.joblib"
scaler = joblib.load(scaler_path)
NN_model_path = BASE_DIR / "NN_model.joblib"
NN_model = joblib.load(NN_model_path)
rf_model_path = BASE_DIR / "random_forest_model.joblib"
rf_model = joblib.load(rf_model_path)
rf_scaler_path = BASE_DIR / "random_forest_scaler.joblib"
rf_scaler = joblib.load(rf_scaler_path)

st.title("Heart Attack and Disease Risk Predictor")
st.write("Enter your health information below to check your risk of heart disease.")
age = st.number_input("Age", min_value=1, max_value=120, value=30)
gender = st.selectbox("Gender", options=[("Male", 1), ("Female", 0)], format_func=lambda x: x[0])[1]
glucose_mg_dl = st.number_input("Glucose level (mg/dL)", min_value=0.0, value=90.0)
cholesterol_mg_dl = st.number_input("Cholesterol level (mg/dL)", min_value=0.0, value=180.0)
systolic_bp = st.number_input("systolic blood pressure (mmHg)", min_value=0.0, value=80.0)
diastolic_bp = st.number_input("Diastolic blood pressure (mmHg)", min_value=0.0, value=80.0)
bmi = st.number_input("BMI", min_value=0.0, value=22.0)
heart_rate = st.number_input("Heart rate (bpm)", min_value=0.0, value=70.0)
smoking_status = st.selectbox("Smoking status", options=[("Smoker", 1), ("Non-Smoker", 0)], format_func=lambda x: x[0])[1]
alcohol_consumption = st.selectbox("Alcohol consumption", options=[("Yes", 1), ("No", 0)], format_func=lambda x: x[0])[1]
physical_activity = st.selectbox("Physical activity level", options=[("High", 2), ("Medium", 1), ("Low", 0)], format_func=lambda x: x[0])[1]
family_history = st.selectbox("Family history of heart disease", options=[("Yes", 1), ("No", 0)], format_func=lambda x: x[0])[1]
if st.button("Predict"):
    Map = functions.MAP(systolic_bp, diastolic_bp)
    Rpp = functions.RPP(systolic_bp, heart_rate)
    Pp = functions.PP(systolic_bp, diastolic_bp)
    unhealthy_lifestyle_score = functions.UnhealthyLifeScore(smoking_status, alcohol_consumption, physical_activity)
    atherogenic_index_coefficient = functions.AtherogenicIndexCoefficient(cholesterol_mg_dl, systolic_bp)
    smoking_hypertension_interaction = functions.SmokingHypertensionInteraction(smoking_status, systolic_bp)
    cardiac_adiposity_proxy = functions.CardiacAdiposityProxy(bmi, heart_rate)
    cardiovascular_stress_index = functions.CardiovascularStressIndex(Map, heart_rate)
    columns = [
    'age', 'gender', 'glucose_mg_dl', 'cholesterol_mg_dl', 'systolic_bp',
    'diastolic_bp', 'heart_rate', 'alcohol_consumption', 'smoking_status',
    'bmi', 'physical_activity', 'family_history', 'MAP',
    'RPP Rate Pressure Product', 'PP Pulse Pressure', 'unhealthy_lifestyle_score',
    'Atherogenic Index Coefficient', 'Smoking-Hypertension Interaction',
    'Cardiac Adiposity Proxy', 'Cardiovascular Stress Index'
    ]
    values = [age, gender, glucose_mg_dl, cholesterol_mg_dl, systolic_bp,
          diastolic_bp, heart_rate, alcohol_consumption, smoking_status,
          bmi, physical_activity, family_history, Map, Rpp, Pp,
          unhealthy_lifestyle_score, atherogenic_index_coefficient,
          smoking_hypertension_interaction, cardiac_adiposity_proxy,
          cardiovascular_stress_index]
    input_df = pd.DataFrame([values], columns=columns)
    unimportant_cols = ['gender', 'alcohol_consumption', 'heart_rate']
    scale_cols = ['age', 'glucose_mg_dl', 'cholesterol_mg_dl', 'systolic_bp',
              'diastolic_bp', 'bmi', 'MAP',
              'RPP Rate Pressure Product', 'PP Pulse Pressure',
              'Atherogenic Index Coefficient', 'Smoking-Hypertension Interaction',
              'Cardiac Adiposity Proxy', 'Cardiovascular Stress Index']
    nn_scaled_input = scaler.transform(input_df[scale_cols])         
    nn_nonscaled_input = input_df.drop(columns=scale_cols + unimportant_cols).values   
    nn_input_final = np.concatenate((nn_scaled_input, nn_nonscaled_input), axis=1) 
   
    mapping = {"low":0 ,"medium":1 ,"high":2}
    input_df["physical_activity"] = input_df["physical_activity"].map(mapping)
    rf_input = pd.get_dummies(input_df, columns=["physical_activity"])
    rf_scaled_input = scaler.transform(rf_input[scale_cols])
    rf_nonscaled_input = rf_input.drop(columns=scale_cols + unimportant_cols).values
    rf_input_final = np.concatenate((rf_scaled_input, rf_nonscaled_input), axis=1)


    output_rf = rf_model.predict(rf_input_final)
    output_NN = NN_model.predict(nn_input_final)
    st.subheader("Results")
    if output_NN[0][0] > 0.5:
       st.error("Neural Network: High risk of heart disease.")
    else:
       st.success("Neural Network: Low risk of heart disease.")
    if output_rf[0] > 0.5:
       st.error("Random Forest: High risk of heart disease.")
    else:
       st.success("Random Forest: Low risk of heart disease.")
