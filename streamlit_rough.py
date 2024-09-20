import numpy as np
import pickle
import pandas as pd
import streamlit as st

with open('./gaussian_trained_model.pkl' , "rb") as pickle_in:
    gnb_best_model = pickle.load(pickle_in)

def welcome():
    return "Predicting 30-day readmission of patients"

def predict_readmission(features):
    prediction = gnb_best_model.predict([features])
    return prediction

def main():
    st.title("30-Day Patient Readmission Prediction")
    
    # HTML layout for header
    html_temp = """
    <div style="background-color:#4CAF50; padding:10px">
        <h2 style="color:white; text-align:center;">Patient Readmission Prediction ML App</h2>
    </div>
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    
    st.header("Input Features")
    
    # Input fields for model features
    full_utilization = st.number_input("Full Utilization", min_value=0.0, step=0.1)
    number_inpatient = st.number_input("Number of Inpatient", min_value=0)
    discharge_disposition_id = st.number_input("Discharge Disposition ID", min_value=1)
    diabetesMed = st.number_input("Diabetes Medication (1 for Yes, 0 for No)", min_value=0, max_value=1)
    admission_source_id = st.number_input("Admission Source ID", min_value=1)
    diag_1 = st.number_input("Diagnosis 1", min_value=0)
    num_lab_procedures = st.number_input("Number of Lab Procedures", min_value=0)
    diag_2 = st.number_input("Diagnosis 2", min_value=0)
    number_emergency = st.number_input("Number of Emergency Visits", min_value=0)
    number_diagnoses = st.number_input("Number of Diagnoses", min_value=0)
    num_medications = st.number_input("Number of Medications", min_value=0)
    citoglipton = st.number_input("Citoglipton (1 for Yes, 0 for No)", min_value=0, max_value=1)
    acetohexamide = st.number_input("Acetohexamide (1 for Yes, 0 for No)", min_value=0, max_value=1)
    change = st.number_input("Change in Medication (1 for Yes, 0 for No)", min_value=0, max_value=1)
    metformin_pioglitazone = st.number_input("Metformin-Pioglitazone (1 for Yes, 0 for No)", min_value=0, max_value=1)
    metformin_rosiglitazone = st.number_input("Metformin-Rosiglitazone (1 for Yes, 0 for No)", min_value=0, max_value=1)
    glimepiride_pioglitazone = st.number_input("Glimepiride-Pioglitazone (1 for Yes, 0 for No)", min_value=0, max_value=1)
    
    result = ""
    
    if st.button("Predict"):
        features = [
            full_utilization,
            number_inpatient,
            discharge_disposition_id,
            diabetesMed,
            admission_source_id,
            diag_1,
            num_lab_procedures,
            diag_2,
            number_emergency,
            number_diagnoses,
            num_medications,
            citoglipton,
            acetohexamide,
            change,
            metformin_pioglitazone,
            metformin_rosiglitazone,
            glimepiride_pioglitazone
        ]
        result = predict_readmission(features)
        st.success('The predicted readmission status is: {}'.format(result[0]))

    if st.button("About"):
        st.text("This app predicts 30-day patient readmission based on various features.")
        st.text("Built with Streamlit and powered by a Gaussian Naive Bayes model.")

if __name__ == '__main__':
    main()
