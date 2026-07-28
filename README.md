<div align="center">
  <h1>🏥 NephroGuard: Predictive Modeling of Early-Stage Chronic Kidney Disease Using Explainable AI</h1>
  <p><strong>A robust, interpretable, and multimodal Machine Learning system for early-stage CKD detection, featuring Gemini Vision AI for automated clinical report extraction and SHAP for Explainable AI (XAI).</strong></p>
</div>

---

## 📸 Screenshots

| Home |
|:---:|
| <img src="/SS/Screenshot 2026-07-28 at 12.28.51 PM.png" width="800"> |


| Verify & Enter Patient Data | 
|:---:|
| <img src="/SS/Screenshot 2026-07-28 at 12.31.48 PM.png" width="800"> | 


| Diagnostic Result(Disease Detected) | Diagnostic Result(Disease does not Detected) | 
|:---:|:---:|
| <img src="/SS/Screenshot 2026-07-28 at 12.32.46 PM.png" width="400"> | <img src="/SS/Screenshot 2026-07-28 at 12.29.38 PM.png" width="400"> |
---

## 📑 Table of Contents
- [📸 Screenshots](#-screenshots)
- [|  |  |](#----)
- [📑 Table of Contents](#-table-of-contents)
- [🏥 1. Clinical Problem Statement \& Project Overview](#-1-clinical-problem-statement--project-overview)
  - [The Research Gap](#the-research-gap)
- [✨ 2. Key Innovations \& Features](#-2-key-innovations--features)
- [🏗️ 3. System Architecture \& ML Pipeline](#️-3-system-architecture--ml-pipeline)
- [📊 4. Dataset \& Clinical Features](#-4-dataset--clinical-features)
  - [Expected Clinical Features (24 Independent Variables)](#expected-clinical-features-24-independent-variables)
- [🧹 5. Data Cleaning, Imputation, \& Preprocessing](#-5-data-cleaning-imputation--preprocessing)
- [⚙️ 6. Model Training, Evaluation, \& Hyperparameter Tuning](#️-6-model-training-evaluation--hyperparameter-tuning)
  - [The Metric Shift](#the-metric-shift)
  - [Hyperparameter Tuning (`GridSearchCV`)](#hyperparameter-tuning-gridsearchcv)
- [🧠 7. Explainable AI (XAI) Integration](#-7-explainable-ai-xai-integration)
- [💻 8. Deployment \& Multimodal AI Interface](#-8-deployment--multimodal-ai-interface)
  - [Complete Pipeline Serialization](#complete-pipeline-serialization)
  - [Multimodal Vision Architecture (Google Gemini Flash 1.5)](#multimodal-vision-architecture-google-gemini-flash-15)
- [🚀 9. Installation \& Setup Guide](#-9-installation--setup-guide)
  - [Prerequisites](#prerequisites)
  - [Step-by-Step Installation](#step-by-step-installation)
- [📂 10. Repository Structure](#-10-repository-structure)
- [🎓 11. Academic Context \& Developer Profile](#-11-academic-context--developer-profile)
- [📜 12. License \& Acknowledgments](#-12-license--acknowledgments)

---

## 🏥 1. Clinical Problem Statement & Project Overview

**Chronic Kidney Disease (CKD)** is a "silent killer." Patients often remain completely asymptomatic during the early stages (Stages 1-3). By the time physical symptoms manifest in Stages 4 or 5, the kidneys have irreversibly lost their ability to filter waste from the blood, necessitating expensive dialysis or organ transplantation. However, if detected early, disease progression can be halted or significantly slowed through diet, lifestyle modifications, and basic medication.

Most existing diagnostic methodologies require specialized, expensive laboratory setups, making widespread early screening unfeasible in resource-constrained regions. 

**NephroGuard** addresses this critical gap by translating the clinical medical problem of CKD into a well-defined Machine Learning classification task. By analyzing routine, inexpensive, and easily accessible blood and urine test data, this project provides a reliable early-screening tool. Furthermore, by addressing the severe challenge of missing clinical data through advanced mathematical imputation, and employing **Explainable AI (SHAP)** techniques, this work bridges the trust gap between black-box AI models and medical professionals.

### The Research Gap
1. **Handling Missing Data:** Medical data is inherently messy. Instead of deleting rows with missing data (which destroys clinical variance), this system utilizes advanced multidimensional K-Nearest Neighbors (KNN) Imputation.
2. **The "Black Box" Problem:** Doctors will not (and legally should not) trust a model they cannot understand. This project integrates SHAP to explicitly reveal which medical biomarkers caused a specific diagnosis.
3. **Manual Data Entry Bottlenecks:** A major barrier to healthcare IT adoption is manual data entry. NephroGuard integrates Google's Gemini Vision AI for Multimodal Optical Character Recognition (OCR), allowing clinicians to simply upload a photo of a lab report for automated, intelligent data population.

---

## ✨ 2. Key Innovations & Features

*   **Supervised Binary Classification Engine:** A highly tuned Random Forest classifier optimized specifically for **Recall (Sensitivity)** to minimize False Negatives (missing a sick patient), a critical requirement for medical AI.
*   **Explainable AI (XAI) via SHAP:** Utilizes Cooperative Game Theory to calculate the marginal contribution of every clinical feature, providing both Global Summary plots and Local Patient-Specific Force Plots.
*   **Advanced Clinical Preprocessing:** Implements KNN Imputation for numerical features and Mode Imputation for categorical variables, ensuring no data leakage occurs across training boundaries.
*   **Multimodal Vision AI (Gemini Flash 1.5):** Employs LLM-driven OCR to intelligently read, extract, and structure messy clinical lab report images directly into the diagnostic pipeline.
*   **Human-in-the-Loop Design:** Automatically populated data is presented to the clinician for verification before inference, ensuring medical safety protocols are met.

---

## 🏗️ 3. System Architecture & ML Pipeline

The project follows a rigorous, production-grade Machine Learning lifecycle, serialized for continuous deployment.

1.  **Data Ingestion:** Sourcing raw, noisy data (featuring typos, hidden characters, and missing values).
2.  **Cleaning & Preprocessing:** Stripping whitespace, coercing data types, and mapping medical abbreviations to standard terminology.
3.  **Imputation:** Filling missing values dynamically (Mode for categorical, KNN for continuous).
4.  **Stratified Splitting & Scaling:** Segregating the dataset into an 80/20 train/test split (Stratified to maintain class distributions) and applying `StandardScaler` *only* to the training set to prevent Data Leakage.
5.  **Modeling & Optimization:** Benchmarking Logistic Regression, SVM, Random Forest, and Gradient Boosting. Running `GridSearchCV` on the top performer to tune hyperparameters (`max_depth`, `class_weight='balanced'`).
6.  **Pipeline Serialization:** Exporting the full state (Model, Scaler, Imputer, Encoders, Feature Order) via `joblib`.
7.  **Inference Server:** A Streamlit-powered REST-style frontend that dynamically scales and encodes future unseen patient data identically to the training environment.

---

## 📊 4. Dataset & Clinical Features

The dataset requires careful clinical consideration. Features must not be treated merely as numbers; they represent deep physiological metrics. The target variable is heavily imbalanced, necessitating F1-Score and Recall as our primary evaluation metrics over raw Accuracy.

### Expected Clinical Features (24 Independent Variables)
**Target Variable:** `target` (1 = CKD Positive, 0 = CKD Negative)

| Feature | Clinical Name | Type | Description / Significance |
| :--- | :--- | :--- | :--- |
| `age` | Age | Numerical | Age of the patient in years. |
| `blood_pressure` | Blood Pressure (bp) | Numerical | Diastolic blood pressure; high BP is both a cause and complication of CKD. |
| `specific_gravity` | Specific Gravity (sg) | Categorical/Num | Measures kidney concentration capacity. |
| `albumin` | Albumin (al) | Categorical/Num | Protein level in urine; a key marker of kidney damage. |
| `sugar` | Sugar (su) | Categorical/Num | Glucose in urine; heavily tied to diabetic nephropathy. |
| `red_blood_cells` | Red Blood Cells (rbc) | Categorical | Normal or Abnormal; indicates bleeding in the urinary tract. |
| `pus_cell` | Pus Cells (pc) | Categorical | Normal or Abnormal; indicates infection. |
| `pus_cell_clumps` | Pus Cell Clumps (pcc) | Categorical | Present or Not Present. |
| `bacteria` | Bacteria (ba) | Categorical | Present or Not Present; indicates severe UTI. |
| `blood_glucose_random`| Random Blood Glucose (bgr)| Numerical | Random sugar levels. |
| `blood_urea` | Blood Urea (bu) | Numerical | Waste product filtered by kidneys. High levels indicate failure. |
| `serum_creatinine` | Serum Creatinine (sc) | Numerical | **Primary indicator** of kidney function. Rises when kidneys fail. |
| `sodium` | Sodium (sod) | Numerical | Electrolyte balance managed by kidneys. |
| `potassium` | Potassium (pot) | Numerical | Electrolyte balance managed by kidneys. |
| `hemoglobin` | Hemoglobin (hemo) | Numerical | Kidneys produce EPO which stimulates RBC production. Low = Anemia/CKD. |
| `packed_cell_volume` | Packed Cell Volume (pcv)| Numerical | Volume percentage of red blood cells in blood. |
| `white_blood_cell_count`| WBC Count (wc) | Numerical | Immune response metric. |
| `red_blood_cell_count`| RBC Count (rc) | Numerical | Closely tied to Hemoglobin and EPO production. |
| `hypertension` | Hypertension (htn) | Categorical | Yes/No; High blood pressure diagnosis. |
| `diabetes_mellitus` | Diabetes (dm) | Categorical | Yes/No; Leading cause of CKD globally. |
| `coronary_artery_disease`| Coronary Artery Disease| Categorical | Yes/No; Cardiovascular health indicator. |
| `appetite` | Appetite (appet) | Categorical | Good/Poor; Uremia (toxin buildup) causes loss of appetite. |
| `pedal_edema` | Pedal Edema (pe) | Categorical | Yes/No; Swelling in legs due to fluid retention. |
| `anemia` | Anemia (ane) | Categorical | Yes/No; Direct side effect of failed EPO production. |

---

## 🧹 5. Data Cleaning, Imputation, & Preprocessing

Clinical datasets are notoriously dirty. The raw data contained hidden string characters (e.g., `	43`, `	yes`), missing values represented by `?`, and incorrect data typing.

1. **Typo Eradication:** Applied `.str.strip()` and explicit mapping to eradicate tab spaces and carriage returns from categorical text columns.
2. **Numeric Coercion:** Forced columns like `packed_cell_volume` and `white_blood_cell_count` into `float64` structures using `pd.to_numeric(errors='coerce')`.
3. **Categorical Imputation:** Handled via Mode Imputation (filling missing text categories with the most frequent occurrence).
4. **Research-Grade Numerical Imputation (KNN):** Simple mean imputation introduces bias. We utilized `sklearn.impute.KNNImputer(n_neighbors=5)`. This calculates multidimensional Euclidean distance between patients. If a patient is missing a hemoglobin reading, the algorithm finds 5 patients with nearly identical age, blood pressure, and creatinine profiles, and averages their hemoglobin. This preserves true clinical variance.
5. **Label Encoding:** Dynamically instantiated `LabelEncoder()` for all text columns, saving them into a dictionary artifact for production decoding.

---

## ⚙️ 6. Model Training, Evaluation, & Hyperparameter Tuning

According to the *No Free Lunch Theorem*, no single algorithm works best universally. We benchmarked four distinct architectures against our scaled data:
1.  **Logistic Regression** (Linear baseline)
2.  **Support Vector Machines (SVM)** (High-dimensional boundary mapping)
3.  **Gradient Boosting** (Sequential ensemble method)
4.  **Random Forest** (Bagged ensemble of decision trees)

### The Metric Shift
Accuracy is a dangerous metric in medical AI. We optimized strictly for **Recall (Sensitivity)**. A False Negative (sending a sick patient home) is fatal. A False Positive (recommending a healthy patient get a secondary test) is an acceptable safety net.

### Hyperparameter Tuning (`GridSearchCV`)
The Random Forest model outperformed the baseline and was passed through an exhaustive Grid Search to squeeze out the absolute maximum performance, pushing the Bayesian Error Rate as low as possible.
*   `n_estimators`: [50, 100, 200]
*   `max_depth`: [None, 10, 20, 30]
*   `class_weight`: 'balanced' (Forces the model to heavily penalize errors on the minority class to boost Recall).
*   `scoring`: 'recall'

*The best-tuned model achieved an unparalleled Recall score (near 0.98+), ensuring almost zero sick patients slip through the diagnostic net.*

---

## 🧠 7. Explainable AI (XAI) Integration

Doctors cannot rely on a Black Box. We implemented **SHAP (SHapley Additive exPlanations)** to provide mathematical transparency to every prediction.

*   **Global Explainability (Summary Plot):** Proves the model learned medically sound rules across the entire test set. It visually confirms that parameters like `Serum Creatinine` and `Hemoglobin` possess the highest Gini Importance and directly drive the prediction toward a positive CKD diagnosis when their levels are elevated or depleted, respectively.
*   **Local Explainability (Force Plot):** Provides a unique, patient-specific visualization for the clinician interface. It shows exactly which lab results pushed the specific patient toward a CKD diagnosis (Red forces) and which lab results attempted to pull them toward a healthy classification (Blue forces). 

---

## 💻 8. Deployment & Multimodal AI Interface

The trained model is useless if isolated in a Jupyter Notebook. We developed a robust **Streamlit Web Application** (`app.py`) to serve the model.

### Complete Pipeline Serialization
To prevent Data Drift and ensure mathematical consistency, the entire pipeline is serialized in the `saved_models/` directory using `joblib`:
*   `ckd_random_forest_model.pkl` (Inference Engine)
*   `ckd_scaler.pkl` (Standardization mapping)
*   `ckd_knn_imputer.pkl` (Dynamic NaN handling)
*   `ckd_label_encoders.pkl` (String-to-Int translation)
*   `ckd_expected_features.pkl` (Strict array sequencing)

### Multimodal Vision Architecture (Google Gemini Flash 1.5)
We integrated Google's Generative AI SDK to overcome manual data entry bottlenecks. 
1. The user uploads a photo (JPG/PNG) of a physical lab report.
2. The image is passed to `gemini-1.5-flash` with a strict JSON-enforced prompt demanding extraction of the specific 24 expected clinical features.
3. The response is parsed, and Streamlit's `st.session_state` is updated to pre-fill the UI forms.
4. The clinician acts as a Human-in-the-Loop, verifying the AI's extraction before executing the Random Forest prediction.

---

## 🚀 9. Installation & Setup Guide

### Prerequisites
*   Python 3.8+
*   Google Gemini API Key (Available free from Google AI Studio)

### Step-by-Step Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/NephroGuard-CKD.git
   cd NephroGuard-CKD
   ```

2. **Create a virtual environment (Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scriptsctivate
   ```

3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *(Required packages include: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `shap`, `streamlit`, `joblib`, `google-generativeai`, `pillow`)*

4. **Run the Application:**
   ```bash
   streamlit run Code/app.py
   ```

5. **Using the Application:**
   * Open the Local URL provided by Streamlit in your web browser.
   * Input your Gemini API key in the sidebar.
   * Upload a sample lab report image OR manually adjust the sliders/dropdowns.
   * Click **"Predict Kidney Disease Risk"** to view the diagnosis and model confidence.

---

## 📂 10. Repository Structure

```text
📦 Predictive-Modeling-of-Early-Stage-Chronic-Kidney-Disease
 ┣ 📂 Code/
 ┃ ┣ 📂 .ipynb_checkpoints/
 ┃ ┣ 📂 saved_models/
 ┃ ┃ ┣ 📜 ckd_expected_features.pkl
 ┃ ┃ ┣ 📜 ckd_knn_imputer.pkl
 ┃ ┃ ┣ 📜 ckd_label_encoders.pkl
 ┃ ┃ ┣ 📜 ckd_random_forest_model.pkl
 ┃ ┃ ┗ 📜 ckd_scaler.pkl
 ┃ ┣ 📜 app.py                                # Streamlit Deployment Script
 ┃ ┣ 📜 datasetCollection_&_EDA.ipynb         # Data Ingestion & Visualization
 ┃ ┣ 📜 datasetCollection_&_EDA.py
 ┃ ┗ 📜 Data Cleaning and Preprocessing.py    # Pipeline logic & Training
 ┣ 📂 Documents/
 ┃ ┗ 📜 Predictive-Modeling...Explainable-AI.docx # Project Report
 ┣ 📜 requirements.txt                        # Dependency definitions
 ┗ 📜 README.md                               # This file
```

---

## 🎓 11. Academic Context & Developer Profile

**NephroGuard** was developed as a comprehensive thesis-grade research project, designed to bridge the gap between abstract academic machine learning and highly functional, deployable medical technology. 

**About the Developer:**
Developed by a final-year Computer Science and Engineering (CSE) undergraduate student at **Jahangirnagar University**. With a strong academic background (CGPA 3.62/4.00) and a professional specialization in Software Quality Assurance (SQA) and cross-platform mobile development (Flutter), this project represents a passionate intersection of those disciplines. 

The developer maintains a sustained interest in the **Internet of Things (IoT) and Healthcare AI**, specifically focusing on solutions tailored for resource-constrained regions like Bangladesh. By combining rigorous ML evaluation metrics, SHAP explainability for clinician trust, and bleeding-edge Vision LLMs to eliminate manual data entry barriers, NephroGuard serves as a blueprint for accessible, next-generation telehealth infrastructure.

---

## 📜 12. License & Acknowledgments

*   **Dataset:** Sourced originally from the UCI Machine Learning Repository (Apollo Hospitals, India).
*   **Libraries:** Built entirely on open-source frameworks including `scikit-learn`, `Streamlit`, and `SHAP`.
*   **LLM API:** Utilizing Google Generative AI (Gemini 1.5 Flash) for multimodal vision extraction.

---
<div align="center">
  <p><i>Building accessible AI for early intervention. Because every diagnosis matters.</i></p>
</div>
