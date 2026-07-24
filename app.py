import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# --- Page Configuration ---
st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")

# --- Data Caching & Loading ---
@st.cache_data
def load_and_clean_data():
    # Load the dataset
    df = pd.read_csv('Student_Performance.csv')
    
    # Cleaning steps derived from your notebook
    df.drop_duplicates(inplace=True) 
    
    # Add errors='ignore' so it doesn't crash if the column is already gone
    df.drop(columns=['Extracurricular Activities'], inplace=True, errors='ignore') 
    
    df.reset_index(drop=True, inplace=True)
    
    return df

@st.cache_resource
def train_model(df):
    # Features and Target
    X = df[['Hours Studied', 'Previous Scores', 'Sleep Hours', 'Sample Question Papers Practiced']]
    y = df['Performance Index']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
    
    # Initialize and train Linear Regression (Best performing and fastest)
    lr = LinearRegression(fit_intercept=True, positive=True)
    lr.fit(X_train, y_train)
    
    # Calculate R2 for display
    y_pred = lr.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    
    return lr, r2

# Load data and model
df = load_and_clean_data()
model, model_r2 = train_model(df)

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
menu = st.sidebar.radio("Go to:", ["📊 Data Overview", "📈 Visualizations", "🎯 Make a Prediction"])

st.sidebar.markdown("---")
st.sidebar.info("This app uses a Machine Learning model to predict a student's Performance Index based on their study habits and historical scores.")

# --- Main Content ---
st.title("🎓 Student Performance Prediction App")

if menu == "📊 Data Overview":
    st.header("Dataset Overview")
    st.write("This section provides a look at the cleaned dataset used to train our prediction models. We removed duplicate rows and dropped the `Extracurricular Activities` column to focus purely on numeric features.")
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Data Preview")
        st.dataframe(df.head(10))
    with col2:
        st.subheader("Summary Statistics")
        st.dataframe(df.describe())
        
    st.info(f"**Dataset Shape:** {df.shape[0]} rows and {df.shape[1]} columns.")

elif menu == "📈 Visualizations":
    st.header("Exploratory Data Analysis")
    st.write("Visualizing the relationships between different features and the final Performance Index.")
    
    tab1, tab2, tab3 = st.tabs(["Correlation Heatmap", "Previous Scores vs Performance", "Hours Studied vs Performance"])
    
    with tab1:
        st.subheader("Feature Correlation")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
        st.pyplot(fig)
        
    with tab2:
        st.subheader("Impact of Previous Scores")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(x=df['Previous Scores'], y=df['Performance Index'], color='#3498db', alpha=0.6)
        plt.xlabel("Previous Scores")
        plt.ylabel("Performance Index")
        st.pyplot(fig)
        
    with tab3:
        st.subheader("Impact of Hours Studied")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.regplot(x='Hours Studied', y='Performance Index', data=df, scatter_kws={'alpha':0.5}, line_kws={'color':'red'})
        plt.xlabel("Hours Studied")
        plt.ylabel("Performance Index")
        st.pyplot(fig)

elif menu == "🎯 Make a Prediction":
    st.header("Predict Student Performance")
    st.write(f"Adjust the parameters below to predict the student's final index. Our underlying Linear Regression model operates with an **R² Score of {model_r2:.4f}**.")
    
    st.markdown("### Input Student Data")
    
    # Input fields structured in columns
    col1, col2 = st.columns(2)
    
    with col1:
        hours_studied = st.slider("Hours Studied per Day", min_value=1, max_value=24, value=5, step=1)
        sleep_hours = st.slider("Sleep Hours per Night", min_value=1, max_value=24, value=7, step=1)
        
    with col2:
        previous_scores = st.number_input("Previous Test Scores (0-100)", min_value=0, max_value=100, value=75, step=1)
        question_papers = st.number_input("Sample Papers Practiced", min_value=0, max_value=50, value=2, step=1)
    
    st.markdown("---")
    
    # Prediction execution
    if st.button("Predict Performance Index", type="primary", use_container_width=True):
        input_data = pd.DataFrame({
            'Hours Studied': [hours_studied],
            'Previous Scores': [previous_scores],
            'Sleep Hours': [sleep_hours],
            'Sample Question Papers Practiced': [question_papers]
        })
        
        prediction = model.predict(input_data)[0]
        
        # Ensure prediction doesn't exceed logical bounds
        prediction = max(0.0, min(100.0, prediction))
        
        st.success("Prediction complete!")
        st.metric(label="Predicted Performance Index", value=f"{prediction:.2f} / 100")
