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
st.title('Hypertension Predictor')

# Load and clean data
try:
    df = pd.read_csv('hypertension_dataset.csv')
    df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace(r'\W', '', regex=True)
    st.success("Dataset loaded successfully.")
except Exception as e:
    st.error(f"Failed to load dataset: {e}")
    st.stop()

# Debug: Show cleaned column names
st.write("Dataset Columns (cleaned):", df.columns.tolist())

# Check for expected features
expected_numeric = ['Age', 'Salt_Intake', 'Stress_Score', 'Sleep_Duration', 'BMI']
expected_cat = ['BP_History', 'Medication', 'Family_History', 'Exercise_Level', 'Smoking_Status']
expected_target = 'Has_Hypertension'

all_expected = expected_numeric + expected_cat + [expected_target]
missing_feats = [col for col in all_expected if col not in df.columns]

if missing_feats:
    st.error(f"Missing expected columns: {missing_feats}")
    st.stop()

# Convert numeric columns safely to float, handling any non-numeric values
for col in expected_numeric:
    if col not in df.columns:
        st.error(f"Missing expected column: {col}")
        st.stop()
    
    # Try to convert column to numeric, invalid parsing will turn into NaN
    df[col] = pd.to_numeric(df[col], errors='coerce')
    if df[col].isnull().any():
        st.warning(f"Column {col} contains non-numeric values and has been converted to NaN. Filling NaN with median.")
        df[col].fillna(df[col].median(), inplace=True)  # Fill NaNs with median

# Ensure all categorical columns are of type 'str'
for col in expected_cat:
    if col not in df.columns:
        st.error(f"Missing expected column: {col}")
        st.stop()
    df[col] = df[col].astype(str).str.strip()  # Ensure they're treated as strings

# Clean target column
if 'Has_Hypertension' not in df.columns:
    st.error("Missing target column: Has_Hypertension")
    st.stop()
else:
    # Drop rows where the target column is NaN or invalid
    df = df[df['Has_Hypertension'].isin(['Yes', 'No'])].copy()
    y = df['Has_Hypertension'].map({'Yes': 1, 'No': 0})

# Prepare feature matrix X
X = df[expected_numeric + expected_cat]

# Handle any remaining NaNs in features (fill them)
X[expected_numeric] = X[expected_numeric].fillna(X[expected_numeric].median(numeric_only=True))
for col in expected_cat:
    X[col] = X[col].fillna(X[col].mode()[0])

# Check for NaNs before proceeding with model training
if X.isnull().any().any():
    st.error("NaN values detected in X features. Please clean the data before training.")
    st.stop()
if y.isnull().any():
    st.error("NaN values detected in y target. Please clean the data before training.")
    st.stop()

# Preprocessing pipeline
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), expected_numeric),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), expected_cat)
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
    
    # Debugging information for training
    st.write("X_train shape:", X_train.shape)
    st.write("y_train distribution:", y_train.value_counts())
    
    # Ensure no NaNs in X_train or y_train
    if X_train.isnull().any().any() or y_train.isnull().any():
        st.error("NaN values detected in X_train or y_train!")
        st.stop()

    # Fit model
    model_pipeline.fit(X_train, y_train)
    joblib.dump(model_pipeline, 'hypertension_model.pkl')
    return model_pipeline

# Train and save the model
model = train_and_save()

# Sidebar for inputs
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
    try:
        prob = model.predict_proba(input_df)[0][1]
        pred = model.predict(input_df)[0]
        st.subheader('Prediction')
        st.write('Has Hypertension: Yes' if pred == 1 else 'Has Hypertension: No')
        st.progress(int(prob * 100))
        st.write('Probability of hypertension: {:.2f}%'.format(prob * 100))
    except Exception as e:
        st.error(f"Prediction failed: {e}")

if st.button('Retrain model'):
    with st.spinner('Retraining...'):
        model = train_and_save()
    st.success('Model retrained and saved to hypertension_model.pkl')

st.markdown('---')
st.write('Dataset rows:', len(df))
