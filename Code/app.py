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
    Upload a clinical lab report image to automatically extract and fill patient data, followed by AI-driven clinical recommendations.
""")

# 2. Secure API Key Configuration (Streamlit Secrets)
# It tries to read from .streamlit/secrets.toml first. 
try:
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
except FileNotFoundError:
    st.error("Missing `.streamlit/secrets.toml` file. Please create it and add your GEMINI_API_KEY.")
    api_key = None
except KeyError:
    st.error("`GEMINI_API_KEY` not found in secrets.toml. Please add it.")
    api_key = None

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
            st.error("API Key is missing. Cannot run extraction.")
        else:
            with st.spinner("Analyzing medical report..."):
                try:
                    prompt = f"""
                    You are a medical data extraction assistant. 
                    Analyze this clinical lab report and extract the following clinical metrics: {expected_features}.
                    Rules:
                    1. If missing, DO NOT include it.
                    2. For categorical features, use: 'yes', 'no', 'normal', 'abnormal', 'present', 'notpresent', 'good', 'poor'.
                    3. Return ONLY a raw JSON object. 
                    """
                    
                    extracted_dict = None
                    last_error = None
                    successful_model = ""

                    # Dynamically loops through models for the fastest vision support
                    for m in genai.list_models():
                        if 'generateContent' in m.supported_generation_methods:
                            try:
                                vision_model = genai.GenerativeModel(m.name)
                                response = vision_model.generate_content([prompt, image])
                                raw_text = response.text.strip().removeprefix('```json').removesuffix('```').strip()
                                extracted_dict = json.loads(raw_text)
                                successful_model = m.name
                                break 
                            except Exception as e:
                                last_error = e
                                continue 

                    if extracted_dict is None:
                        raise Exception(f"All models failed. Last error: {last_error}")
                    
                    st.session_state.extracted_data = extracted_dict
                    st.success(f"✅ Data extracted successfully using `{successful_model}`!")
                    
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

# 7. Prediction & Doctor's Note Logic
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
        st.write(f"**Random Forest Confidence:** {prediction_proba[1] * 100:.2f}%")
        risk_level = "HIGH RISK"
    else:
        st.success(f"✅ **Low Risk. No CKD detected.**")
        st.write(f"**Random Forest Confidence:** {prediction_proba[0] * 100:.2f}%")
        risk_level = "LOW RISK"

    # --- AI DOCTOR'S NOTE GENERATION (STRICT JSON API) ---
    st.subheader("👨‍⚕️ AI Nephrologist Clinical Summary")
    if not api_key:
        st.warning("API Key missing. Cannot generate clinical summary.")
    else:
        with st.spinner("Generating personalized clinical recommendations..."):
            try:
                # We force the API to output a strict JSON dictionary. 
                # This mathematically prevents it from generating bullet points or extra text.
                doc_prompt = f"""
                Analyze the data below and return a strictly formatted JSON object. 
                Do NOT use markdown. Do NOT use backticks. Output ONLY the raw JSON.
                
                Risk Level: {risk_level}
                Patient Data: {user_data}
                
                Output Format Required:
                {{
                    "clinical_note": "Write exactly 3 sentences here. State the risk level. Mention specific abnormal numbers (e.g., Creatinine or Blood Pressure). Recommend a renal ultrasound."
                }}
                """
                
                doc_text = None
                successful_doc_model = ""

                for m in genai.list_models():
                    if 'generateContent' in m.supported_generation_methods:
                        try:
                            doc_model = genai.GenerativeModel(m.name)
                            
                            # Lower temperature to 0.1 to remove creativity and enforce strict compliance
                            doc_response = doc_model.generate_content(
                                doc_prompt,
                                generation_config=genai.types.GenerationConfig(temperature=0.1)
                            )
                            
                            # Clean the output in case the API wraps it in ```json ... ```
                            raw_json = doc_response.text.strip().removeprefix('```json').removesuffix('```').strip()
                            
                            # Convert the string to a Python dictionary
                            parsed_response = json.loads(raw_json)
                            
                            # Extract just the clean paragraph
                            doc_text = parsed_response["clinical_note"]
                            successful_doc_model = m.name
                            break # Success! Exit the loop.
                            
                        except Exception as e:
                            # If the model tries to output bullet points, json.loads() will fail,
                            # triggering this exception and forcing it to try the next model.
                            continue 

                if doc_text is None:
                    raise Exception("All API models failed to return proper JSON structure.")
                
                st.info(doc_text)
                st.caption(f"Note generated securely via JSON API using `{successful_doc_model}`")
                
            except Exception as e:
                st.error(f"Could not generate clinical note at this time. Error Details: {e}")