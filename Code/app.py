import streamlit as st
import pandas as pd
import numpy as np
import joblib

# 1. Page Configuration
st.set_page_config(page_title="NephroGuard: CKD Predictor", page_icon="🩺", layout="wide")
st.title("🩺 NephroGuard: Early-Stage CKD Prediction System")
st.markdown("""
    This application uses a Machine Learning model to predict the probability of Chronic Kidney Disease (CKD) 
    based on routine clinical test results. 
""")

# 2. Load the Saved Artifacts
# We use st.cache_resource so these large files are only loaded once, making the app fast
@st.cache_resource
def load_artifacts():
    model = joblib.load('saved_models/ckd_random_forest_model.pkl')
    scaler = joblib.load('saved_models/ckd_scaler.pkl')
    label_encoders = joblib.load('saved_models/ckd_label_encoders.pkl')
    expected_features = joblib.load('saved_models/ckd_expected_features.pkl')
    return model, scaler, label_encoders, expected_features

model, scaler, label_encoders, expected_features = load_artifacts()

# 3. Build the User Input Interface
st.header("Enter Patient Clinical Data")

# We will use columns to make the UI look professional and organized
col1, col2, col3 = st.columns(3)

# Define the categorical columns (based on our earlier EDA)
categorical_cols = ['red_blood_cells', 'pus_cell', 'pus_cell_clumps', 'bacteria', 
                    'hypertension', 'diabetes_mellitus', 'coronary_artery_disease', 
                    'appetite', 'pedal_edema', 'anemia']

# Create a dictionary to store user inputs
user_data = {}

# Programmatically generate input fields based on expected features
for i, feature in enumerate(expected_features):
    # Distribute inputs across the 3 columns
    target_col = col1 if i % 3 == 0 else col2 if i % 3 == 1 else col3
    
    with target_col:
        if feature in categorical_cols:
            # Reconstruct the original text classes from the label encoder
            original_classes = label_encoders[feature].classes_
            user_data[feature] = st.selectbox(f"{feature.replace('_', ' ').title()}", original_classes)
        else:
            # Default numeric inputs to 0.0
            user_data[feature] = st.number_input(f"{feature.replace('_', ' ').title()}", value=0.0, step=0.1)

# 4. Prediction Logic
st.markdown("---")
if st.button("Predict Kidney Disease Risk", type="primary"):
    
    # A. Convert user dictionary to a Pandas DataFrame
    input_df = pd.DataFrame([user_data])
    
    # B. Preprocessing: Label Encode the categorical variables
    for col in categorical_cols:
        if col in input_df.columns:
            le = label_encoders[col]
            input_df[col] = le.transform(input_df[col])
            
    # C. Preprocessing: Scale the numerical variables
    # We must ensure the columns are in the exact same order as training
    input_df = input_df[expected_features] 
    input_df_scaled = scaler.transform(input_df)
    
    # D. Make the Prediction
    prediction = model.predict(input_df_scaled)[0]
    prediction_proba = model.predict_proba(input_df_scaled)[0]
    
    # 5. Display the Results
    st.header("Diagnostic Result")
    
    if prediction == 1:
        st.error(f"⚠️ **High Risk of Chronic Kidney Disease (CKD)** detected.")
        st.write(f"**Model Confidence:** {prediction_proba[1] * 100:.2f}%")
        st.write("Recommendation: Patient requires immediate nephrology consultation.")
    else:
        st.success(f"✅ **Low Risk. No CKD detected.**")
        st.write(f"**Model Confidence:** {prediction_proba[0] * 100:.2f}%")
        st.write("Recommendation: Continue standard preventative care.") 