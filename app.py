import os
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
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
# CUSTOM STYLING (Dark/Light Mode Compatible)
# ============================================================
st.markdown(
    """
    <style>
    .stButton>button {
        background-color: #3498db;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
        transition: all 0.3s;
    }
    .stButton>button:hover { 
        background-color: #2980b9; 
        color: white;
        transform: translateY(-2px); 
    }
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
# MODEL TRAINING (Reverted to Linear Regression)
# ============================================================
@st.cache_resource
def train_model(df):
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # Linear Regression will correctly extrapolate high/low scores for this dataset
    model = LinearRegression()
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
            value=8, step=1  # Defaulted to your test values
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
            value=89, step=1
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
        
        res_col1, res_col2 = st.columns([1, 2])
        with res_col1:
            # The score should now correctly reflect a much higher value!
            st.metric(label="Predicted Score", value=f"{prediction:.1f} / 100")
            
        with res_col2:
            st.markdown("#### Feature Impact (Coefficients)")
            
            # Map Linear Regression coefficients instead of RF importances
            importance_df = pd.DataFrame({
                "Feature": FEATURES,
                "Impact": model.coef_
            }).sort_values(by="Impact", key=abs, ascending=True)

            fig, ax = plt.subplots(figsize=(6, 2.5), facecolor='white')
            
            # Color positive impact blue, negative impact red
            colors = ['#e74c3c' if x < 0 else '#3498db' for x in importance_df["Impact"]]
            
            ax.barh(importance_df["Feature"], importance_df["Impact"], color=colors)
            ax.set_xlabel("Points added/lost per unit", color="black")
            ax.tick_params(colors='black') 
            ax.spines[['top', 'right']].set_visible(False)
            ax.axvline(0, color='black', linewidth=1) # Add a center line for reference
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
    
    sns.set_theme(style="whitegrid", rc={"figure.facecolor": "white", "axes.facecolor": "white", "text.color": "black", "axes.labelcolor": "black", "xtick.color": "black", "ytick.color": "black"})
    numeric_df = df.select_dtypes(include='number')

    viz_tab1, viz_tab2, viz_tab3 = st.tabs(["🔗 Relationships & Correlation", "📊 Distributions", "📦 Box Plots"])

    with viz_tab1:
        st.markdown("#### Feature Correlation Heatmap")
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, vmin=-1, vmax=1, ax=ax1)
        st.pyplot(fig1)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        vcol1, vcol2 = st.columns(2)
        with vcol1:
            st.markdown("#### Previous Scores vs Performance")
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            sns.scatterplot(x=df['Previous Scores'], y=df[TARGET], color="#2ecc71", alpha=0.6, ax=ax2)
            st.pyplot(fig2)
            
        with vcol2:
            st.markdown("#### Hours Studied Regression")
            fig3, ax3 = plt.subplots(figsize=(6, 4))
            sns.regplot(x='Hours Studied', y=TARGET, data=df, 
                        scatter_kws={'alpha':0.5, 'color': '#34495e'}, 
                        line_kws={'color': '#e74c3c', 'linewidth': 2}, ax=ax3)
            st.pyplot(fig3)

    with viz_tab2:
        st.markdown("#### Distribution of Performance Index")
        fig4, ax4 = plt.subplots(figsize=(10, 5))
        sns.histplot(df[TARGET], bins=20, kde=True, color="#f1c40f", edgecolor='black', ax=ax4)
        ax4.set_ylabel("Frequency")
        st.pyplot(fig4)

    with viz_tab3:
        bcol1, bcol2 = st.columns(2)
        
        with bcol1:
            st.markdown("#### Box Plot: Hours Studied")
            fig5, ax5 = plt.subplots(figsize=(6, 4))
            sns.boxplot(x=df['Hours Studied'], color="#e74c3c", ax=ax5)
            st.pyplot(fig5)
            
        with bcol2:
            st.markdown("#### Box Plot: Performance Index")
            fig6, ax6 = plt.subplots(figsize=(6, 4))
            sns.boxplot(x=df[TARGET], color="#9b59b6", ax=ax6)
            st.pyplot(fig6)
