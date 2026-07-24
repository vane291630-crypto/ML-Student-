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
    initial_sidebar_state="expanded",
)

# ============================================================
# CUSTOM STYLING
# ============================================================
st.markdown(
    """
    <style>
    .main { background-color: #f7f9fc; }
    .stMetric {
        background-color: #ffffff;
        border: 1px solid #e6e9f0;
        border-radius: 12px;
        padding: 12px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    }
    div[data-testid="stMetricValue"] { color: #2c3e83; }
    .block-container { padding-top: 2rem; }
    h1, h2, h3 { color: #1f2a5c; }
    .stButton>button {
        background-color: #2c3e83;
        color: white;
        border-radius: 8px;
        border: none;
        padding: 0.6rem 1.2rem;
        font-weight: 600;
    }
    .stButton>button:hover { background-color: #1f2a5c; color: white; }
    .stTabs [data-baseweb="tab-list"] { gap: 6px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #eef1f9;
        border-radius: 8px 8px 0 0;
        padding: 8px 16px;
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
    df = pd.read_csv("Student_Performance.csv")

    # Drop duplicate rows
    df.drop_duplicates(inplace=True)

    # Drop the categorical column so every remaining feature is numeric
    df.drop(columns=["Extracurricular Activities"], inplace=True, errors="ignore")

    # Drop rows with missing values in the columns we actually use
    required_cols = [c for c in FEATURES + [TARGET] if c in df.columns]
    df.dropna(subset=required_cols, inplace=True)

    df.reset_index(drop=True, inplace=True)
    return df


# ============================================================
# MODEL TRAINING
# ============================================================
@st.cache_resource
def train_model(df):
    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )

    # NOTE: The original code forced positive=True, which mathematically forbids
    # any feature (including Sleep Hours) from ever having a negative effect on
    # the prediction. That is why increasing sleep hours could never lower the
    # predicted score, even when it should. A plain LinearRegression lets each
    # coefficient be whatever the data actually supports.
    lr = LinearRegression()
    lr.fit(X_train, y_train)

    y_pred = lr.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    return lr, r2, mae


def get_feature_bounds(df):
    """Real min/max values from the training data, used to keep sliders
    inside the range the model was actually trained on. Letting a slider go
    to 24 'hours studied' or '24 sleep hours' pushes the model far outside
    the data it learned from, which is why predictions looked nonsensical."""
    bounds = {}
    for col in FEATURES:
        col_min = int(np.floor(df[col].min()))
        col_max = int(np.ceil(df[col].max()))
        bounds[col] = (col_min, col_max)
    return bounds


# ============================================================
# LOAD DATA + TRAIN MODEL (with error handling)
# ============================================================
try:
    df = load_and_clean_data()
except FileNotFoundError:
    st.error(
        "Couldn't find `Student_Performance.csv`. Please make sure the file "
        "is in the same folder as this app before running it."
    )
    st.stop()

if df.empty or not all(c in df.columns for c in FEATURES + [TARGET]):
    st.error(
        "The dataset is missing one or more required columns: "
        f"{FEATURES + [TARGET]}. Please check the CSV file."
    )
    st.stop()

model, model_r2, model_mae = train_model(df)
bounds = get_feature_bounds(df)

# ============================================================
# SIDEBAR
# ============================================================
st.sidebar.title("🎓 Navigation")
menu = st.sidebar.radio(
    "Go to:", ["📊 Data Overview", "📈 Visualizations", "🎯 Make a Prediction"]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "This app uses a Linear Regression model to predict a student's "
    "**Performance Index** from study habits and historical scores."
)
st.sidebar.metric("Model R² Score", f"{model_r2:.3f}")
st.sidebar.metric("Mean Abs. Error", f"{model_mae:.2f} pts")

# ============================================================
# MAIN CONTENT
# ============================================================
st.title("🎓 Student Performance Prediction App")

# ------------------------------------------------------------
# DATA OVERVIEW
# ------------------------------------------------------------
if menu == "📊 Data Overview":
    st.header("Dataset Overview")
    st.write(
        "This section shows the cleaned dataset used to train the model. "
        "Duplicate rows were removed and the categorical "
        "`Extracurricular Activities` column was dropped so the model only "
        "uses numeric features."
    )

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
    with col2:
        st.subheader("Summary Statistics")
        st.dataframe(df.describe(), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Avg. Performance Index", f"{df[TARGET].mean():.1f}")

# ------------------------------------------------------------
# VISUALIZATIONS
# ------------------------------------------------------------
elif menu == "📈 Visualizations":
    st.header("Exploratory Data Analysis")
    st.write(
        "Visualizing the relationships between each feature and the final "
        "Performance Index."
    )

    tab1, tab2, tab3 = st.tabs(
        ["Correlation Heatmap", "Previous Scores vs Performance", "Hours Studied vs Performance"]
    )

    with tab1:
        st.subheader("Feature Correlation")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            df.corr(numeric_only=True),
            annot=True,
            cmap="coolwarm",
            fmt=".2f",
            ax=ax,
        )
        st.pyplot(fig)

    with tab2:
        st.subheader("Impact of Previous Scores")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(
            x=df["Previous Scores"], y=df[TARGET], color="#2c3e83", alpha=0.6, ax=ax
        )
        ax.set_xlabel("Previous Scores")
        ax.set_ylabel("Performance Index")
        st.pyplot(fig)

    with tab3:
        st.subheader("Impact of Hours Studied")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.regplot(
            x="Hours Studied",
            y=TARGET,
            data=df,
            scatter_kws={"alpha": 0.5, "color": "#2c3e83"},
            line_kws={"color": "red"},
            ax=ax,
        )
        ax.set_xlabel("Hours Studied")
        ax.set_ylabel("Performance Index")
        st.pyplot(fig)

# ------------------------------------------------------------
# PREDICTION
# ------------------------------------------------------------
elif menu == "🎯 Make a Prediction":
    st.header("Predict Student Performance")
    st.write(
        f"Adjust the parameters below to predict a student's final index. "
        f"The underlying Linear Regression model has an **R² Score of "
        f"{model_r2:.3f}** and an average error of about **{model_mae:.1f} points**."
    )

    st.markdown("### Input Student Data")
    st.caption(
        "Sliders are limited to the range seen in the training data. "
        "Values far outside this range are unreliable because the model "
        "never learned from data like that."
    )

    hs_min, hs_max = bounds["Hours Studied"]
    ps_min, ps_max = bounds["Previous Scores"]
    sh_min, sh_max = bounds["Sleep Hours"]
    qp_min, qp_max = bounds["Sample Question Papers Practiced"]

    col1, col2 = st.columns(2)

    with col1:
        hours_studied = st.slider(
            "Hours Studied per Day", min_value=hs_min, max_value=hs_max,
            value=min(max(5, hs_min), hs_max), step=1,
        )
        sleep_hours = st.slider(
            "Sleep Hours per Night", min_value=sh_min, max_value=sh_max,
            value=min(max(7, sh_min), sh_max), step=1,
        )

    with col2:
        previous_scores = st.number_input(
            "Previous Test Scores", min_value=ps_min, max_value=ps_max,
            value=min(max(75, ps_min), ps_max), step=1,
        )
        question_papers = st.number_input(
            "Sample Papers Practiced", min_value=qp_min, max_value=qp_max,
            value=min(max(2, qp_min), qp_max), step=1,
        )

    st.markdown("---")

    if st.button("Predict Performance Index", type="primary", use_container_width=True):
        input_data = pd.DataFrame(
            {
                "Hours Studied": [hours_studied],
                "Previous Scores": [previous_scores],
                "Sleep Hours": [sleep_hours],
                "Sample Question Papers Practiced": [question_papers],
            }
        )

        prediction = model.predict(input_data)[0]
        prediction = max(0.0, min(100.0, prediction))

        st.success("Prediction complete!")
        st.metric(label="Predicted Performance Index", value=f"{prediction:.2f} / 100")

        # Show how each feature is influencing the prediction, so the
        # direction of the effect (positive or negative) is transparent.
        st.markdown("#### How each factor is contributing")
        coef_df = pd.DataFrame(
            {
                "Feature": FEATURES,
                "Coefficient": model.coef_,
            }
        ).sort_values("Coefficient", key=abs, ascending=False)

        fig, ax = plt.subplots(figsize=(7, 3.5))
        colors = ["#2c3e83" if c >= 0 else "#c0392b" for c in coef_df["Coefficient"]]
        ax.barh(coef_df["Feature"], coef_df["Coefficient"], color=colors)
        ax.set_xlabel("Effect on Performance Index (per unit increase)")
        ax.axvline(0, color="black", linewidth=0.8)
        st.pyplot(fig)

        st.caption(
            "Blue bars increase the predicted score as that factor goes up; "
            "red bars decrease it. This reflects the true relationship learned "
            "from the data, so an unrealistic value (like extreme sleep hours) "
            "won't be artificially forced to help the score."
        )
