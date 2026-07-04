import streamlit as st
import pandas as pd
import joblib

# Load saved model, scaler, and expected columns
model = joblib.load("KNN_heart.pkl")
scaler = joblib.load("scaler.pkl")
expected_columns = joblib.load("columns.pkl")

st.title("Heart💓Disease Prediction App")
st.markdown("Provide the following details to predict the likelihood of heart disease.")



# Collect user input
age = st.slider("Age", 18, 100, 40)
sex = st.selectbox("Sex", ["M", "F"])
chest_pain = st.selectbox("Chest Pain Type", ["ATA", "NAP", "TA", "ASY"])
resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120)
cholesterol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
fasting_bs = st.selectbox("Fasting Blood Sugar > 120 mg/dL", [0, 1])
resting_ecg = st.selectbox("Resting ECG", ["Normal", "ST", "LVH"])
max_hr = st.slider("Max Heart Rate", 60, 220, 150)
exercise_angina = st.selectbox("Exercise-Induced Angina", ["Y", "N"])
oldpeak = st.slider("Oldpeak (ST Depression)", 0.0, 6.0, 1.0)
st_slope = st.selectbox("ST Slope", ["Up", "Flat", "Down"])

# When Predict is clicked
if st.button("Predict"):
    # Create a DataFrame from user input
    raw_input = {
        "Age": age,
        "RestingBP": resting_bp,
        "Cholesterol": cholesterol,
        "FastingBS": fasting_bs,
        "MaxHR": max_hr,
        "Oldpeak": oldpeak,
        "Sex_" + sex: 1,
        "ChestPainType_" + chest_pain: 1,
        "RestingECG_" + resting_ecg: 1,
        "ExerciseAngina_" + exercise_angina: 1,
        "ST_Slope_" + st_slope: 1,
    }

    input_df = pd.DataFrame([raw_input])

    # Fill in missing columns with 0s
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns to match the expected order
    input_df = input_df[expected_columns]

    # Scale the input
    scaled_input = scaler.transform(input_df)

    # Make prediction
    prediction = model.predict(scaled_input)[0]

    # Show result
    if prediction == 1:
        st.error("⚠️ High Risk of Heart Disease")
    else:
        st.success("✅ Low Risk of Heart Disease")

st.markdown("---")

st.subheader("📘 Feature Guide (What each input means)")

with st.expander("Click to understand medical terms"):
    st.markdown("""
### 🫀 Chest Pain Types
- **ATA (Atypical Angina):** Chest pain not fully typical of heart disease  
- **NAP (Non-Anginal Pain):** Pain usually not related to heart  
- **TA (Typical Angina):** Classic heart-related chest pain  
- **ASY (Asymptomatic):** No chest pain even if disease exists  

---

### ⚡ ST Slope
- **Up:** Normal recovery pattern (good sign)  
- **Flat:** Possible abnormal blood flow  
- **Down:** Strong sign of reduced heart blood flow  

---

### 📉 Oldpeak
- Measures ST depression during exercise ECG  
- Higher value → higher stress on heart  

---

### ❤️ Exercise-Induced Angina
- **Y:** Chest pain during exercise  
- **N:** No chest pain during exercise  

---

### 🧪 Resting ECG
- **Normal:** No abnormality  
- **ST:** ST-T wave abnormality (possible ischemia)  
- **LVH:** Possible enlarged heart  

---

### 🍬 Fasting Blood Sugar
- **0:** Normal (≤120 mg/dL)  
- **1:** High (>120 mg/dL)  

---

### 📊 Max Heart Rate
- Higher values usually indicate better heart fitness during exercise
""")