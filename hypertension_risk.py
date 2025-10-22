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

# Load data and clean column names
df = pd.read_csv('hypertension_dataset.csv')
df.columns = df.columns.str.strip().str.replace(' ', '_').str.replace(r'[\n\r\t]', '', regex=True)

# Convert numeric columns to numeric dtype, coerce errors to NaN, then fill with median
numeric_cols = ['Age', 'Salt_Intake', 'Stress_Score', 'Sleep_Duration', 'BMI']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    median_val = df[col].median()
    df[col].fillna(median_val, inplace=True)

# Check for missing columns in categorical columns too
cat_cols = ['BP_History', 'Medication', 'Family_History', 'Exercise_Level', 'Smoking_Status']
for col in cat_cols:
    if col not in df.columns:
        st.error(f"Missing expected categorical column: {col}")

# Create X and y datasets
X = df[numeric_cols + cat_cols]
y = df['Has_Hypertension'].map({'Yes': 1, 'No': 0})

# Build preprocessor and model pipeline
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
    # Use global df here if needed
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    model_pipeline.fit(X_train, y_train)
    joblib.dump(model_pipeline, 'hypertension_model.pkl')
    return model_pipeline

model = train_and_save()

st.title('Hypertension Predictor')

# Show some info about dataset
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
    st.write('Probability of hypertension: {:.2f}%'.format(prob * 100))

if st.button('Retrain model'):
    with st.spinner('Retraining...'):
        model = train_and_save()
    st.success('Model retrained and saved to hypertension_model.pkl')

st.markdown('---')
st.write('Dataset rows:', len(df))
