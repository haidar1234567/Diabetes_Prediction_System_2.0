import pickle
import streamlit as st
import numpy as np

# path of file of model 
file_path = r"C:\Users\A\Documents\python\webApp\Diabetes_Prediction.pkl"

# load the model  
Classifier = pickle.load(open(file_path, "rb"))

#  Streamlit App UI
st.title("🩺 Diabetes Prediction App")
st.write("Enter your details below to check for diabetes risk.")

#  User Inputs (Ensure float64 type)
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, step=1)  # step=int
glucose = st.number_input("Glucose Level", min_value=0, max_value=200, step=1)  # step=int
bp = st.number_input("Blood Pressure", min_value=0, max_value=150, step=1)  # step=int
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, step=1)  # step=int
insulin = st.number_input("Insulin Level", min_value=0, max_value=900, step=1)  # step=int
bmi = st.number_input("BMI", min_value=0.0, max_value=50.0, step=0.1)  # step=float
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=50.0, step=0.01)  # step=float
age = st.number_input("Age", min_value=0, max_value=120, step=1)  # step=int

#  Convert Inputs to Float (Ensuring Uniform Data Type)
user_input = np.array([[float(pregnancies), float(glucose), float(bp), float(skin_thickness),float(insulin), float(bmi), float(dpf), float(age)]], dtype=np.float64)

# Prediction Button
if st.button("Predict"):
    # Convert Input to Proper Float64 Array
    user_input = np.array([[pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]], dtype=np.float64)

    # Debugging: Print Data Types (Optional)
    st.write("Data Types:", user_input.dtype)
    
    # take pridiction from model
    prediction = Classifier.predict(user_input)
    
    # Show Result
    if prediction[0] == 1:
        st.error("🚨 Diabetes positive! Please consult a doctor.")
    else:
        st.success("✅ Diabetes negative! Stay Healthy. 😃")




