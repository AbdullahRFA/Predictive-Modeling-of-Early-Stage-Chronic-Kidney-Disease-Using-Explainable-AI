# 1. Import necessary libraries
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

print(df.head())

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

# 4. Basic Inspection
print("Dataset Shape:", df.shape)
print("\n--- Data Types & Missing Values ---")
print(df.info())

print("Null value counts:\n", df.isnull().sum())
print("Cross-tabulation of target variable:\n", df['target'].value_counts())

# 5. Visualizing the Target Distribution
plt.figure(figsize=(6, 4))
sns.countplot(data=df, x='target', palette=['#2ecc71', '#e74c3c'])
plt.title("Distribution of CKD vs Not CKD")
plt.xticks(ticks=[0, 1], labels=['Not CKD (0)', 'CKD (1)'])
plt.show()

# 6. Visualizing Missing Data Patterns
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis', yticklabels=False)
plt.title("Missing Data Heatmap (Yellow = Missing)")
plt.show()

# 7. Correlation matrix for numerical features
plt.figure(figsize=(12, 8))
numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
sns.heatmap(df[numerical_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix of Numerical Features")
plt.show()