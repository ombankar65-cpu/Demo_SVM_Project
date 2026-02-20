import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("🎓 Job Placement Predictor")
st.write("Enter the student details below to predict placement status.")

# Create two columns for a cleaner layout
col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", options=["Male", "Female"])
    ssc_p = st.number_input("Secondary School % (ssc_p)", min_value=0.0, max_value=100.0, value=65.0)
    hsc_p = st.number_input("Higher Secondary % (hsc_p)", min_value=0.0, max_value=100.0, value=65.0)

with col2:
    hsc_s = st.selectbox("HSC Specialization", options=["Commerce", "Science", "Arts"])
    degree_p = st.number_input("Degree % (degree_p)", min_value=0.0, max_value=100.0, value=65.0)
    mba_p = st.number_input("MBA % (mba_p)", min_value=0.0, max_value=100.0, value=65.0)

# Preprocessing: Map categorical strings to numbers (adjust based on your training encoding)
# Assuming: Male=1, Female=0 | Commerce=0, Science=1, Arts=2 (Modify if your encoding was different)
gender_val = 1 if gender == "Male" else 0
hsc_map = {"Commerce": 0, "Science": 1, "Arts": 2}
hsc_s_val = hsc_map[hsc_s]

# Organize input for prediction
input_data = pd.DataFrame([[gender_val, ssc_p, hsc_p, hsc_s_val, degree_p, mba_p]], 
                          columns=['gender', 'ssc_p', 'hsc_p', 'hsc_s', 'degree_p', 'mba_p'])

if st.button("Predict Placement"):
    prediction = model.predict(input_data)
    result = prediction[0]
    
    if result == "Placed":
        st.success(f"Result: {result} 🎉")
    else:
        st.error(f"Result: {result} ⚠️")
