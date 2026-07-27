from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
# from datasetCollection_&_EDA import df  # Import the DataFrame from the EDA notebook

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ssl 

# Bypass SSL verification for macOS
ssl._create_default_https_context = ssl._create_unverified_context

# Set visualization style for professional looking plots
sns.set_theme(style="whitegrid")

# 2. Dataset Collection (Loading the data from a stable mirror)
url = "https://raw.githubusercontent.com/patilgirish815/Kidney_Cancer_Prediction_Using_Machine_Learning/main/dataset/kidney_disease.csv"


# Some datasets use '\t?' as well as '?' for missing values
df = pd.read_csv(url, na_values=['?', '\t?'])

# Drop the 'id' column if it exists, as it has no clinical/predictive value
if 'id' in df.columns:
    df.drop('id', axis=1, inplace=True)

# 3. Rename columns for better readability
columns_mapping = {
    'bp': 'blood_pressure', 'sg': 'specific_gravity', 'al': 'albumin', 'su': 'sugar',
    'rbc': 'red_blood_cells', 'pc': 'pus_cell', 'pcc': 'pus_cell_clumps', 'ba': 'bacteria',
    'bgr': 'blood_glucose_random', 'bu': 'blood_urea', 'sc': 'serum_creatinine', 
    'sod': 'sodium', 'pot': 'potassium', 'hemo': 'hemoglobin', 'pcv': 'packed_cell_volume',
    'wc': 'white_blood_cell_count', 'rc': 'red_blood_cell_count', 'htn': 'hypertension',
    'dm': 'diabetes_mellitus', 'cad': 'coronary_artery_disease', 'appet': 'appetite',
    'pe': 'pedal_edema', 'ane': 'anemia', 'classification': 'target' # Updated mapping
}
df.rename(columns=columns_mapping, inplace=True)
print(df.head())

# Clean target variable (removing any invisible tabs/spaces)
df['target'] = df['target'].str.strip()
df['target'] = df['target'].map({'ckd': 1, 'notckd': 0}) 

print("Starting Data Cleaning...")

# 1. Clean invisible typos (tabs and spaces) in categorical columns
# The dataset has hidden strings like '\tyes', ' yes', or '\tno'
categorical_cols = df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    # Convert to string, strip whitespace and tabs, then replace 'nan' back to actual np.nan
    df[col] = df[col].astype(str).str.strip()
    df[col] = df[col].replace('nan', np.nan)

# 2. Fix Incorrect Data Types
# These columns are numerical but got parsed as objects due to typos in the raw file
cols_to_numeric = ['packed_cell_volume', 'white_blood_cell_count', 'red_blood_cell_count']
for col in cols_to_numeric:
    # errors='coerce' forces any unreadable text (like '?') to become NaN
    df[col] = pd.to_numeric(df[col], errors='coerce')

# 3. Handle Missing Values (Imputation)
# Update our lists of categorical and numerical columns after the dtype fix
categorical_cols = df.select_dtypes(include=['object']).columns
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns

# Do NOT impute the target variable! Drop it from the numerical list temporarily.
if 'target' in numerical_cols:
    numerical_cols = numerical_cols.drop('target')

# A. Categorical Imputation: Fill missing with Mode (most frequent value)
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

# B. Numerical Imputation: Use KNN Imputer (Research-level)
print("Applying KNN Imputation for numerical features...")
knn_imputer = KNNImputer(n_neighbors=5)
df[numerical_cols] = knn_imputer.fit_transform(df[numerical_cols])

# 4. Encode Categorical Variables (Label Encoding)
# Machine Learning models need numbers, not text like 'normal' or 'abnormal'
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le # We save the encoder for deployment later!

# 5. Final Sanity Check
print("\n--- Data Types After Cleaning ---")
print(df.info())
print("\n--- Total Missing Values After Cleaning ---")
print(df.isnull().sum().sum()) # Should print 0