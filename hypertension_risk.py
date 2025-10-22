# app.py — Robust Hypertension Predictor (copy-paste)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import traceback

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

st.set_page_config(page_title='Hypertension Predictor', layout='centered')
st.title('Hypertension Predictor')

# ---------------------------
# Helpers
# ---------------------------
def clean_column_name(col: str) -> str:
    """Normalize column name to Title_Case without special chars (keeps underscores)."""
    s = str(col).strip()
    s = s.replace(' ', '_')
    # remove non-word except underscore
    s = ''.join(ch for ch in s if ch.isalnum() or ch == '_')
    # convert to Title case words separated by underscore, then join with underscore
    parts = [p.capitalize() for p in s.split('_') if p != ""]
    return '_'.join(parts)

def robust_to_numeric_series(s: pd.Series) -> pd.Series:
    """Convert a series to numeric safely (strip units, commas, <, >, etc.)"""
    s2 = s.astype(str).str.strip()
    # empty-like values -> NaN
    s2 = s2.replace({'nan': None, 'none': None, 'na': None, '': None, 'null': None})
    # remove commas, percent signs, letters, angle signs
    s2 = s2.str.replace(r'[\,\%\sA-Za-z]+', '', regex=True)
    s2 = s2.str.lstrip('<>').replace('', pd.NA)
    return pd.to_numeric(s2, errors='coerce')

def ensure_min_max_for_input(col_series, default_min=None, default_max=None):
    """Return safe min and max for st.number_input if min==max or min>max."""
    if col_series.dropna().empty:
        return (default_min if default_min is not None else 0, default_max if default_max is not None else 1)
    mn = float(col_series.min())
    mx = float(col_series.max())
    if mn == mx:
        # widen range a bit
        mn = mn - abs(mn * 0.1) - 1
        mx = mx + abs(mx * 0.1) + 1
    if mn > mx:
        mn, mx = mx, mn
    return (mn, mx)

# ---------------------------
# Load dataset
# ---------------------------
try:
    df = pd.read_csv('hypertension_dataset.csv', low_memory=False)
except FileNotFoundError:
    st.error("hypertension_dataset.csv not found in app folder. Upload dataset or place it in the same folder as app.py.")
    st.stop()
except Exception as e:
    st.error(f"Failed to read CSV: {e}")
    st.stop()

# Normalize column names and show to user
orig_columns = df.columns.tolist()
cleaned_map = {orig: clean_column_name(orig) for orig in orig_columns}
df.rename(columns=cleaned_map, inplace=True)
st.write("Detected columns (cleaned):", df.columns.tolist())

# ---------------------------
# Expected feature names (Title_Case with underscores)
# ---------------------------
expected_numeric = ['Age', 'Salt_Intake', 'Stress_Score', 'Sleep_Duration', 'Bmi']   # note 'Bmi' instead of 'BMI' due to cleaning
expected_cat = ['Bp_History', 'Medication', 'Family_History', 'Exercise_Level', 'Smoking_Status']
expected_target = 'Has_Hypertension'

# Check existence of expected columns (try tolerant matching)
missing = [c for c in (expected_numeric + expected_cat + [expected_target]) if c not in df.columns]
if missing:
    # attempt tolerant matching by lower-case substring matching
    lower_map = {c.lower(): c for c in df.columns}
    resolved = []
    for c in list(missing):
        found = None
        for col in df.columns:
            if c.lower().replace('_','') == col.lower().replace('_',''):
                found = col
                break
        if found:
            df.rename(columns={found: c}, inplace=True)
            resolved.append((found, c))
            missing.remove(c)
    if resolved:
        st.info(f"Auto-resolved columns: {resolved}")
    if missing:
        st.error(f"Missing expected columns after attempted resolution: {missing}")
        st.stop()

# ---------------------------
# Convert numeric columns safely
# ---------------------------
for col in expected_numeric:
    st.write(f"Processing numeric column: {col}")
    df[col] = robust_to_numeric_series(df[col])
    if df[col].isnull().any():
        st.warning(f"Column {col} had non-numeric entries. Filling NaNs with median.")
        try:
            median = float(df[col].median())
        except Exception:
            median = 0.0
        df[col].fillna(median, inplace=True)

# Make categorical columns strings and fill NaNs
for col in expected_cat:
    df[col] = df[col].astype(str).str.strip()
    if df[col].isnull().any() or (df[col] == 'nan').any():
        # replace 'nan' strings
        df[col] = df[col].replace({'nan': None})
    if df[col].isnull().any():
        mode = df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown"
        df[col].fillna(mode, inplace=True)

# Clean target column: allow values like 'yes', 'no', 1, 0, True/False
if expected_target not in df.columns:
    st.error(f"Target column {expected_target} not found.")
    st.stop()

# Normalize target values to 0/1
def normalize_target(v):
    if pd.isna(v):
        return np.nan
    s = str(v).strip().lower()
    if s in ('1', 'yes', 'y', 'true', 't'):
        return 1
    if s in ('0', 'no', 'n', 'false', 'f'):
        return 0
    # fallback: numbers
    try:
        nv = float(v)
        return 1 if nv >= 1 else 0
    except Exception:
        return np.nan

df[expected_target] = df[expected_target].apply(normalize_target)
df = df[df[expected_target].notnull()].copy()
y = df[expected_target].astype(int)

# ---------------------------
# Prepare X and final NaN checks
# ---------------------------
X = df[expected_numeric + expected_cat].copy()

# Filling remaining numeric NaNs (if any)
for col in expected_numeric:
    if X[col].isnull().any():
        X[col].fillna(X[col].median(), inplace=True)

# Filling remaining categorical NaNs
for col in expected_cat:
    if X[col].isnull().any():
        X[col].fillna(X[col].mode().iloc[0] if not X[col].mode().empty else "Unknown", inplace=True)

if X.isnull().any().any() or y.isnull().any():
    st.error("NaNs remain in features/target after cleaning. Aborting.")
    st.stop()

st.write("Final dataset shape:", X.shape, "Target distribution:", y.value_counts().to_dict())

# ---------------------------
# Preprocessing + model pipeline
# ---------------------------
preprocessor = ColumnTransformer([
    ('num', StandardScaler(), expected_numeric),
    ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), expected_cat)
])

model_pipeline = Pipeline([
    ('pre', preprocessor),
    ('clf', RandomForestClassifier(n_estimators=200, random_state=42))
])

# Cache training so retrain button can re-run when requested
@st.cache_resource
def train_pipeline(X_, y_):
    X_train, X_test, y_train, y_test = train_test_split(X_, y_, test_size=0.20, random_state=42, stratify=y_)
    model_pipeline.fit(X_train, y_train)
    # save
    joblib.dump(model_pipeline, 'hypertension_model.pkl')
    return model_pipeline

# Train or load
try:
    # prefer loading saved model if exists
    model = joblib.load('hypertension_model.pkl')
    st.success("Loaded existing model (hypertension_model.pkl).")
except Exception:
    st.info("Training new model (may take a few seconds)...")
    model = train_pipeline(X, y)
    st.success("Model trained and saved as hypertension_model.pkl")

# ---------------------------
# Sidebar inputs (safe ranges)
# ---------------------------
st.sidebar.header("Enter your values")

# compute safe min/max for numeric inputs
age_min, age_max = ensure_min_max_for_input(df['Age'], 18, 90)
salt_min, salt_max = ensure_min_max_for_input(df['Salt_Intake'], 0.0, 20.0)
stress_min, stress_max = ensure_min_max_for_input(df['Stress_Score'], 1, 10)
sleep_min, sleep_max = ensure_min_max_for_input(df['Sleep_Duration'], 3.0, 12.0)
bmi_min, bmi_max = ensure_min_max_for_input(df['Bmi'], 10.0, 45.0)

age = st.sidebar.number_input('Age', min_value=float(age_min), max_value=float(age_max), value=float(df['Age'].median()))
salt = st.sidebar.number_input('Salt Intake (grams/day)', min_value=float(salt_min), max_value=float(salt_max), value=float(df['Salt_Intake'].median()), step=0.1, format="%.1f")
stress = st.sidebar.number_input('Stress Score (1-10)', min_value=float(stress_min), max_value=float(stress_max), value=float(df['Stress_Score'].median()), step=1.0, format="%.0f")
bp_hist = st.sidebar.selectbox('BP History', sorted(X['Bp_History'].unique().tolist()))
sleep = st.sidebar.number_input('Sleep Duration (hours)', min_value=float(sleep_min), max_value=float(sleep_max), value=float(df['Sleep_Duration'].median()), step=0.1, format="%.1f")
bmi = st.sidebar.number_input('BMI', min_value=float(bmi_min), max_value=float(bmi_max), value=float(df['Bmi'].median()), step=0.1, format="%.1f")
med = st.sidebar.selectbox('Medication', sorted(X['Medication'].unique().tolist()))
fam = st.sidebar.selectbox('Family History', sorted(X['Family_History'].unique().tolist()))
exercise = st.sidebar.selectbox('Exercise Level', sorted(X['Exercise_Level'].unique().tolist()))
smoke = st.sidebar.selectbox('Smoking Status', sorted(X['Smoking_Status'].unique().tolist()))

input_df = pd.DataFrame([{
    'Age': age,
    'Salt_Intake': salt,
    'Stress_Score': stress,
    'Bp_History': bp_hist,
    'Sleep_Duration': sleep,
    'Bmi': bmi,
    'Medication': med,
    'Family_History': fam,
    'Exercise_Level': exercise,
    'Smoking_Status': smoke
}])

# ---------------------------
# Predict / Retrain UI
# ---------------------------
if st.button('Predict'):
    try:
        # Defensive: ensure same column order and dtypes as training
        pred_proba = model.predict_proba(input_df)[0][1]
        pred = model.predict(input_df)[0]
        st.subheader('Prediction')
        st.write('Has Hypertension: **Yes**' if pred == 1 else 'Has Hypertension: **No**')
        st.progress(int(pred_proba * 100))
        st.write('Probability of hypertension: {:.2f}%'.format(pred_proba * 100))
    except Exception as e:
        st.error("Prediction failed. See traceback:")
        st.code(traceback.format_exc())

if st.button('Retrain model'):
    with st.spinner('Retraining...'):
        try:
            model = train_pipeline(X, y)
            st.success('Model retrained and saved to hypertension_model.pkl')
        except Exception:
            st.error("Retrain failed:")
            st.code(traceback.format_exc())

st.markdown('---')
st.write('Dataset rows:', len(df))
