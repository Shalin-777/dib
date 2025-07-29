from pycaret.classification import *
import pandas as pd
import streamlit as st

model = load_model('tuned_rf_diabetes')

def predict(model, input_df):
    predictions_df = predict_model(estimator=model, data=input_df)
    predictions = predictions_df.iloc[0]['prediction_label']
    return predictions

def run():
    pregancies = st.text_input("Pregnancies")
    glucose = st.text_input("Glucose")
    bloodPressure = st.text_input("Blood Pressure")
    skinThickness = st.text_input("Skin Thickness")
    insulin = st.text_input("Insulin")
    bmi = st.text_input("BMI")
    debpedfunc = st.text_input("Diabetes Pedigree Function")
    age = st.text_input("Age")
    output = ""

    input_dict = {
        'Pregnancies': pregancies,
        'Glucose': glucose,
        'BloodPressure': bloodPressure,
        'SkinThickness': skinThickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': debpedfunc,
        'Age': age      
    }   
    input_df = pd.DataFrame([input_dict])   
    if st.button("Predict"):
        output = predict(model=model, input_df=input_df)
        output = str(output)
        st.success('The output is {}'.format(output))

if __name__ == '__main__':
    run()