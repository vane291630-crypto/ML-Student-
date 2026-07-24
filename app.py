import os
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ============================================================
# CUSTOM STYLING
# ============================================================
st.markdown(
    """
    <style>
    .main { background-color: #f8f9fa; }
    h1, h2, h3 { color: #2c3e50; }
    .stMetric {
        background-color: #ffffff;
        border: 1px solid #e9ecef;
        border-radius: 10px;
        padding: 15px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    div[data-testid="stMetricValue"] { color: #3498db; font-weight: bold; }
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover { background-color: #2980b9; transform: translateY(-2px); }
    </style>
    """,
    unsafe_allow_html=True,
)

FEATURES = ["Hours Studied", "Previous Scores", "Sleep Hours", "Sample Question Papers Practiced"]
TARGET = "Performance Index"

# ============================================================
# DATA LOADING & CLEANING
# ============================================================
@st.cache_data
def load_and_clean_data():
    file_path = "Student_Performance.csv"
    
    if not os.path.exists(file_path):
        return None
        
    df = pd.read_csv(file_path)
    df.drop_duplicates(inplace=True)
    df.drop(columns=["Extracurricular Activities"], inplace=True, errors="ignore")
    
    required_cols = [c for c in FEATURES + [TARGET] if c in df.columns]
    df.dropna(subset=required_cols, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

# ============================================================
# MODEL TRAINING (Upgraded to Random Forest)
# ============================================================
@st.cache_resource
def train_model(df):
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # Random Forest generally provides higher accuracy for tabular data
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    return model, r2, mae

def get_feature_bounds(df):
    bounds = {}
    for col in FEATURES:
        bounds[col] = (int(np.floor(df[col].min())), int(np.ceil(df[col].max())))
    return bounds

# ============================================================
# MAIN APP EXECUTION
# ============================================================
df = load_and_clean_data()

if df is None:
    st.error("⚠️ Dataset not found! Please ensure 'Student_Performance.csv' is in the same folder as this script.")
    st.stop()

model, model_r2, model_mae = train_model(df)
bounds = get_feature_bounds(df)

# Header
st.title("🎓 Student Performance Predictor")
st.markdown("Use this tool to predict student test scores based on their study habits and historical data.")

# Create main tabs
tab_predict, tab_insights = st.tabs(["🎯 Make a Prediction", "📊 Data Insights"])

# ------------------------------------------------------------
# TAB 1: PREDICTION
# ------------------------------------------------------------
with tab_predict:
    st.markdown("### Adjust Student Profiles")
    
    col1, col2 = st.columns(2)
    
    with col1:
        hours_studied = st.slider(
            "Hours Studied per Day", 
            min_value=bounds["Hours Studied"][0], 
            max_value=bounds["Hours Studied"][1],
            value=5, step=1
        )
        sleep_hours = st.slider(
            "Sleep Hours per Night", 
            min_value=bounds["Sleep Hours"][0], 
            max_value=bounds["Sleep Hours"][1],
            value=7, step=1
        )

    with col2:
        previous_scores = st.number_input(
            "Previous Test Scores", 
            min_value=bounds["Previous Scores"][0], 
            max_value=bounds["Previous Scores"][1],
            value=75, step=1
        )
        question_papers = st.number_input(
            "Sample Papers Practiced", 
            min_value=bounds["Sample Question Papers Practiced"][0], 
            max_value=bounds["Sample Question Papers Practiced"][1],
            value=2, step=1
        )

    st.markdown("---")

    if st.button("Predict Performance Index", type="primary", use_container_width=True):
        input_data = pd.DataFrame([{
            "Hours Studied": hours_studied,
            "Previous Scores": previous_scores,
            "Sleep Hours": sleep_hours,
            "Sample Question Papers Practiced": question_papers,
        }])

        prediction = model.predict(input_data)[0]
        prediction = max(0.0, min(100.0, prediction))

        st.success("Prediction generated successfully!")
        
        # Display Prediction Result nicely
        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            st.metric(label="Predicted Score", value=f"{prediction:.1f} / 100")
            
        with res_col2:
            st.markdown("#### Feature Importance")
            # Show feature importance from the Random Forest model
            importance_df = pd.DataFrame({
                "Feature": FEATURES,
                "Importance": model.feature_importances_
            }).sort_values(by="Importance", ascending=True)

            fig, ax = plt.subplots(figsize=(6, 2.5))
            ax.barh(importance_df["Feature"], importance_df["Importance"], color="#3498db")
            ax.set_xlabel("Impact on Prediction")
            ax.spines[['top', 'right']].set_visible(False)
            st.pyplot(fig)

# ------------------------------------------------------------
# TAB 2: DATA INSIGHTS
# ------------------------------------------------------------
with tab_insights:
    st.markdown("### Model & Dataset Metrics")
    
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Records", df.shape[0])
    m2.metric("Avg Score", f"{df[TARGET].mean():.1f}")
    m3.metric("Model Accuracy (R²)", f"{model_r2:.3f}")
    m4.metric("Avg Error (MAE)", f"{model_mae:.2f} pts")

    st.markdown("---")
    
    viz_col1, viz_col2 = st.columns(2)
    
    with viz_col1:
        st.subheader("Feature Correlation")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.heatmap(df.corr(), annot=True, cmap="Blues", fmt=".2f", ax=ax, cbar=False)
        st.pyplot(fig)
        
    with viz_col2:
        st.subheader("Previous Scores vs Performance")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(x=df["Previous Scores"], y=df[TARGET], color="#3498db", alpha=0.5, ax=ax)
        ax.set_xlabel("Previous Scores")
        ax.set_ylabel("Final Performance Index")
        st.pyplot(fig)
        
