
import streamlit as st
import pandas as pd
import joblib

model = joblib.load("credit_model.pkl")

st.title("German Credit Risk Prediction")

age = st.slider("Age", 18, 75)
credit_amount = st.number_input("Credit Amount")
duration = st.slider("Duration (in months)", 6, 60)
job = st.selectbox("Job Type", [0, 1, 2, 3])
housing = st.selectbox("Housing", ['own', 'free', 'rent'])
sex = st.selectbox("Sex", ['male', 'female'])

input_dict = {
    'Age': age,
    'Credit amount': credit_amount,
    'Duration': duration,
    'Job': job,
    'Sex_male': 1 if sex == 'male' else 0,
    'Housing_own': 1 if housing == 'own' else 0,
    'Housing_rent': 1 if housing == 'rent' else 0,
}

input_data = pd.DataFrame([input_dict])
for col in model.feature_names_in_:
    if col not in input_data.columns:
        input_data[col] = 0

input_data = input_data[model.feature_names_in_]

prediction = model.predict(input_data)[0]
st.write("Prediction:", "✅ Good Credit Risk" if prediction == 1 else "❌ Bad Credit Risk")
