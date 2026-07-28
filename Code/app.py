import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import json
from PIL import Image
import google.generativeai as genai

# 1. Page Configuration
st.set_page_config(page_title="NephroGuard: CKD Predictor", page_icon="🩺", layout="wide")
st.title("🩺 NephroGuard: Early-Stage CKD Prediction System")
st.markdown("""
    This application uses a Machine Learning model to predict the probability of Chronic Kidney Disease (CKD).
    **New Feature:** Upload a clinical lab report image to automatically extract and fill patient data!
""")

# 2. Sidebar for API Key Setup
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("Enter Gemini API Key (for Image Extraction)", type="password")
if api_key:
    genai.configure(api_key=api_key)

# 3. Load the Saved Artifacts
@st.cache_resource
def load_artifacts():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model = joblib.load(os.path.join(current_dir, 'saved_models', 'ckd_random_forest_model.pkl'))
    scaler = joblib.load(os.path.join(current_dir, 'saved_models', 'ckd_scaler.pkl'))
    label_encoders = joblib.load(os.path.join(current_dir, 'saved_models', 'ckd_label_encoders.pkl'))
    expected_features = joblib.load(os.path.join(current_dir, 'saved_models', 'ckd_expected_features.pkl'))
    return model, scaler, label_encoders, expected_features

model, scaler, label_encoders, expected_features = load_artifacts()

# 4. Initialize Streamlit Session State for Extracted Data
if 'extracted_data' not in st.session_state:
    st.session_state.extracted_data = {}

# 5. Image Upload & Extraction Section
st.header("📄 Step 1: Auto-Fill from Lab Report (Optional)")
uploaded_file = st.file_uploader("Upload Lab Report Image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Lab Report", width=400)
    
    if st.button("Extract Data with AI"):
        if not api_key:
            st.error("Please enter your Gemini API Key in the sidebar first.")
        else:
            with st.spinner("Analyzing medical report (finding the best available AI model)..."):
                try:
                    # Define exactly what we want Gemini to extract
                    prompt = f"""
                    You are a medical data extraction assistant. 
                    Analyze this clinical lab report and extract the following clinical metrics: {expected_features}.
                    
                    Rules:
                    1. If a value is missing or cannot be found, DO NOT include it in the output.
                    2. For categorical features (like hypertension, diabetes_mellitus), extract them as 'yes', 'no', 'normal', 'abnormal', 'present', 'notpresent', 'good', 'poor' based on the report.
                    3. Return ONLY a raw JSON object (no markdown, no backticks, no explanation). 
                    Example format: {{"blood_pressure": 120.0, "sugar": 0.0, "hypertension": "no"}}
                    """
                    
                    extracted_dict = None
                    last_error = None
                    successful_model = ""

                    # --- THE FAILOVER LOOP ---
                    # Dynamically loops through all models you have access to.
                    # If one fails (e.g., 404 deprecated), it silently tries the next.
                    for m in genai.list_models():
                        if 'generateContent' in m.supported_generation_methods:
                            try:
                                vision_model = genai.GenerativeModel(m.name)
                                response = vision_model.generate_content([prompt, image])
                                
                                # Clean the response
                                raw_text = response.text.strip().removeprefix('```json').removesuffix('```').strip()
                                extracted_dict = json.loads(raw_text)
                                successful_model = m.name
                                break # It worked! Exit the loop immediately.
                                
                            except Exception as e:
                                last_error = e
                                continue # Model failed, silently try the next one in the list

                    if extracted_dict is None:
                        # If the loop finishes and absolutely nothing worked
                        raise Exception(f"All available AI models failed. Last error: {last_error}")
                    
                    # Save to session state
                    st.session_state.extracted_data = extracted_dict
                    st.success(f"✅ Data extracted successfully using `{successful_model}`! Review the populated form below.")
                    
                except Exception as e:
                    st.error(f"Extraction failed: {e}")

# 6. Build the User Input Interface
st.header("📋 Step 2: Verify & Enter Patient Data")
col1, col2, col3 = st.columns(3)

categorical_cols = ['red_blood_cells', 'pus_cell', 'pus_cell_clumps', 'bacteria', 
                    'hypertension', 'diabetes_mellitus', 'coronary_artery_disease', 
                    'appetite', 'pedal_edema', 'anemia']

user_data = {}

for i, feature in enumerate(expected_features):
    target_col = col1 if i % 3 == 0 else col2 if i % 3 == 1 else col3
    
    extracted_val = st.session_state.extracted_data.get(feature, None)
    
    with target_col:
        if feature in categorical_cols:
            original_classes = list(label_encoders[feature].classes_)
            
            default_index = 0
            if extracted_val is not None:
                extracted_val = str(extracted_val).lower().strip()
                if extracted_val in original_classes:
                    default_index = original_classes.index(extracted_val)
                    
            user_data[feature] = st.selectbox(f"{feature.replace('_', ' ').title()}", original_classes, index=default_index)
        else:
            default_num = 0.0
            if extracted_val is not None:
                try:
                    default_num = float(extracted_val)
                except ValueError:
                    pass 
                    
            user_data[feature] = st.number_input(f"{feature.replace('_', ' ').title()}", value=default_num, step=0.1)

# 7. Prediction Logic
st.markdown("---")
if st.button("Predict Kidney Disease Risk", type="primary"):
    
    input_df = pd.DataFrame([user_data])
    
    for col in categorical_cols:
        if col in input_df.columns:
            le = label_encoders[col]
            input_df[col] = le.transform(input_df[col])
            
    input_df = input_df[expected_features] 
    input_df_scaled = scaler.transform(input_df)
    
    prediction = model.predict(input_df_scaled)[0]
    prediction_proba = model.predict_proba(input_df_scaled)[0]
    
    st.header("Diagnostic Result")
    
    if prediction == 1:
        st.error(f"⚠️ **High Risk of Chronic Kidney Disease (CKD)** detected.")
        st.write(f"**Model Confidence:** {prediction_proba[1] * 100:.2f}%")
        st.write("Recommendation: Patient requires immediate nephrology consultation.")
    else:
        st.success(f"✅ **Low Risk. No CKD detected.**")
        st.write(f"**Model Confidence:** {prediction_proba[0] * 100:.2f}%")
        st.write("Recommendation: Continue standard preventative care.")