import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import streamlit as st

st.set_page_config(page_title='Hypertension Predictor', layout='centered')

# Load data
df = pd.read_csv('hypertension_dataset.csv')

# Clean column names (strip spaces, replace spaces with underscores, remove weird chars)
df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace(r'[\n\r\t]', '', regex=True)

# Define columns
numeric_cols = ['Age', 'Salt_Intake', 'Stress_Score', 'Sleep_Duration', 'BMI']
cat_cols = ['BP_History', 'Medication', 'Family_History', 'Exercise_Level', 'Smoking_Status']

# Convert numeric columns, coerce errors to NaN, fill NaNs with median
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    median_val = df[col].median()
    df[col].fillna(median_val, inplace=True)

# Convert categorical columns to string (to avoid issues with mixed types)
for col in cat_cols:
    df[col] = df[col].astype(str).str.strip()

# Remove rows with missing target or map target properly
df = df[df['Has_Hypertension'].isin(['Yes', 'No'])].copy()
y = df['Has_Hypertension'].map({'Yes': 1, 'No': 0})

# Prepare feature matrix X with the cleaned columns
X = df[numeric_cols + cat_cols]

# Check for any remaining NaNs (should be none, but just in case)
if X.isnull().any().any():
    st.error("Warning: Found NaNs in features after cleaning. Filling with medians or mode now.")

    for col in numeric_cols:
        if X[col].isnull().any():
            X[col].fillna(X[col].median(), inplace=True)
    for col in cat_cols:
        if X[col].isnull().any():
            # Fill missing categorical with most frequent value
            mode_val = X[col].mode()[0]
            X[col].fillna(mode_val, inplace=True)

# Define preprocessor and pipeline
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), numeric_cols),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cat_cols)
])

model_pipeline = Pipeline([
    ('pre', preprocessor),
    ('clf', RandomForestClassifier(n_estimators=200, random_state=42))
])

@st.cache_resource
def train_and_save():
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model_pipeline.fit(X_train, y_train)
    joblib.dump(model_pipeline, 'hypertension_model.pkl')
    return model_pipeline

model = train_and_save()

st.title('Hypertension Predictor')

st.write("Columns in dataset:", df.columns.tolist())
st.write("BMI sample data:", df['BMI'].head())
st.write("BMI dtype:", df['BMI'].dtype)
st.write("Salt_Intake sample data:", df['Salt_Intake'].head())
st.write("Salt_Intake dtype:", df['Salt_Intake'].dtype)

with st.sidebar:
    age = st.number_input('Age', int(df['Age'].min()), int(df['Age'].max()), int(df['Age'].median()))
    salt = st.number_input('Salt Intake (grams/day)', float(df['Salt_Intake'].min()), float(df['Salt_Intake'].max()), float(df['Salt_Intake'].median()), step=0.1, format="%.1f")
    stress = st.number_input('Stress Score (1-10)', int(df['Stress_Score'].min()), int(df['Stress_Score'].max()), int(df['Stress_Score'].median()))
    bp_hist = st.selectbox('BP History', sorted(df['BP_History'].unique().tolist()))
    sleep = st.number_input('Sleep Duration (hours)', float(df['Sleep_Duration'].min()), float(df['Sleep_Duration'].max()), float(df['Sleep_Duration'].median()), step=0.1, format="%.1f")
    bmi = st.number_input('BMI', float(df['BMI'].min()), float(df['BMI'].max()), float(df['BMI'].median()), step=0.1, format="%.1f")
    med = st.selectbox('Medication', sorted(df['Medication'].unique().tolist()))
    fam = st.selectbox('Family History', sorted(df['Family_History'].unique().tolist()))
    exercise = st.selectbox('Exercise Level', sorted(df['Exercise_Level'].unique().tolist()))
    smoke = st.selectbox('Smoking Status', sorted(df['Smoking_Status'].unique().tolist()))

input_df = pd.DataFrame([{
    'Age': age,
    'Salt_Intake': salt,
    'Stress_Score': stress,
    'BP_History': bp_hist,
    'Sleep_Duration': sleep,
    'BMI': bmi,
    'Medication': med,
    'Family_History': fam,
    'Exercise_Level': exercise,
    'Smoking_Status': smoke
}])

if st.button('Predict'):
    prob = model.predict_proba(input_df)[0][1]
    pred = model.predict(input_df)[0]
    st.subheader('Prediction')
    st.write('Has Hypertension: Yes' if pred == 1 else 'Has Hypertension: No')
    st.progress(int(prob * 100))
    st.write(f'Probability of hypertension: {prob * 100:.2f}%')

if st.button('Retrain model'):
    with st.spinner('Retraining...'):
        model = train_and_save()
    st.success('Model retrained and saved to hypertension_model.pkl')

st.markdown('---')
st.write(f'Dataset rows: {len(df)}')
