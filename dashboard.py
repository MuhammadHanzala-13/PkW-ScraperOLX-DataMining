import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import os

# --- Page Config ---
st.set_page_config(page_title="PakWheels EDA Dashboard", layout="wide")

# --- Custom CSS ---
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    div[data-testid="metric-container"] {
        background-color: #1e3a5f;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.15);
    }
    div[data-testid="metric-container"] label {
        color: #93c5fd !important;
        font-size: 14px;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 26px;
        font-weight: bold;
    }
    h1, h2, h3 {color: #1f2937;}
    </style>
""", unsafe_allow_html=True)

# --- Data Loading ---
@st.cache_data
def load_data():
    processed_path = "data/pakwheels_cars_processed.csv"
    raw_path = "data/pakwheels_cars_raw.csv"
    df = pd.read_csv(processed_path) if os.path.exists(processed_path) else None
    raw_df = pd.read_csv(raw_path) if os.path.exists(raw_path) else None
    return df, raw_df

df, raw_df = load_data()

if df is None:
    st.error("Processed data not found. Please run pakwheels_data_engineering.py first.")
    st.stop()

# --- Header ---
st.title("PakWheels EDA Dashboard")
st.markdown("Automated Exploratory Data Analysis for the Data Mining Project — ANN Ready Dataset")
st.markdown("---")

# --- 1. Dataset Overview ---
st.header("1. Dataset Overview")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Records", f"{len(df):,}")
with col2:
    if 'price' in df.columns:
        st.metric("Avg. Price (PKR)", f"{int(df['price'].mean()):,}")
    else:
        st.metric("Avg. Price (PKR)", "Not Available")
with col3:
    # mileage_km may be named differently - detect it
    mileage_col = next((c for c in ['mileage_km', 'mileage', 'Mileage'] if c in df.columns), None)
    if mileage_col:
        st.metric("Avg. Mileage (km)", f"{int(df[mileage_col].mean()):,}")
    else:
        st.metric("Avg. Mileage (km)", "In Progress...")
with col4:
    if 'car_age' in df.columns:
        st.metric("Avg. Car Age", f"{round(df['car_age'].mean(), 1)} Years")
    elif 'year' in df.columns:
        avg_age = 2024 - df['year'].mean()
        st.metric("Avg. Car Age", f"{round(avg_age, 1)} Years")
    else:
        st.metric("Avg. Car Age", "In progress...")

st.markdown("---")

# --- 2. Price Distribution & Year vs Price ---
st.header("2. Price Analysis")
col_p1, col_p2 = st.columns(2)

with col_p1:
    st.subheader("Price Distribution")
    if 'price' in df.columns:
        fig = px.histogram(df, x='price', nbins=60,
                           title="Price Range Frequency (After Outlier Removal)",
                           color_discrete_sequence=['#3b82f6'])
        fig.update_xaxes(title="Price (PKR)")
        fig.update_yaxes(title="Number of Cars")
        st.plotly_chart(fig, use_container_width=True)

with col_p2:
    st.subheader("Price vs. Manufacturing Year")
    if 'year' in df.columns and 'price' in df.columns:
        color_col = 'fuel_type' if 'fuel_type' in df.columns else None
        fig2 = px.scatter(df, x="year", y="price", color=color_col,
                          opacity=0.6, title="Year vs. Price Scatter")
        st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")

# --- 3. Market Breakdown ---
st.header("3. Market Breakdown")
col_m1, col_m2 = st.columns(2)

with col_m1:
    st.subheader("Brand Market Share")
    brand_col = 'brand_clean' if 'brand_clean' in df.columns else ('brand' if 'brand' in df.columns else None)
    if brand_col:
        brand_counts = df[brand_col].value_counts().reset_index()
        brand_counts.columns = ['Brand', 'Count']
        fig3 = px.bar(brand_counts, x='Brand', y='Count',
                      title="Vehicles Per Brand",
                      color='Count', color_continuous_scale="Blues")
        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Brand column not found.")

with col_m2:
    st.subheader("Transmission Distribution")
    if 'transmission' in df.columns:
        fig4 = px.pie(df, names='transmission', title="Automatic vs Manual",
                      hole=0.4, color_discrete_sequence=px.colors.sequential.Teal)
        st.plotly_chart(fig4, use_container_width=True)
    else:
        st.info("Transmission column not found.")

st.markdown("---")

# --- 4. Missing Value Report ---
st.header("4. Missing Value (NA) Resolution Report")
st.markdown("A required step in EDA: identifying, imputing, and validating missing data before ANN training.")

if raw_df is not None:
    col_na1, col_na2 = st.columns(2)

    with col_na1:
        st.subheader("Before Processing (Raw Data)")
        raw_na = raw_df.isna().sum()
        for col in raw_df.columns:
            raw_na[col] += (raw_df[col].astype(str) == "Unknown").sum()
        raw_na = raw_na[raw_na > 0].reset_index()
        raw_na.columns = ['Feature', 'Missing Count']
        if not raw_na.empty:
            fig5 = px.bar(raw_na, x='Feature', y='Missing Count',
                          title="Missing Values Per Feature (Raw)",
                          color_discrete_sequence=['#ef4444'])
            st.plotly_chart(fig5, use_container_width=True)
        else:
            st.success("No missing values detected in the raw dataset.")

    with col_na2:
        st.subheader("After Statistical Imputation (Processed Data)")
        st.markdown("""
        **Imputation strategy applied:**
        - **Numerical (Mileage, Engine CC):** Filled using **Median** — robust against outliers.
        - **Categorical (Body Type, Fuel, Color):** Filled using **Mode** — most frequent value.
        - **Target Variable (Price):** Rows with missing price are **dropped strictly**.
        """)
        proc_na = df.isna().sum().reset_index()
        proc_na.columns = ['Feature', 'Missing Count']
        proc_na = proc_na[proc_na['Missing Count'] > 0]
        if not proc_na.empty:
            fig6 = px.bar(proc_na, x='Feature', y='Missing Count',
                          title="Remaining NAs After Processing",
                          color_discrete_sequence=['#10b981'])
            st.plotly_chart(fig6, use_container_width=True)
        else:
            st.success("0 Missing Values. The dataset is mathematically complete and ready for ANN.")
else:
    st.warning("Raw data file not found. Cannot generate missing values comparison.")

st.markdown("---")

# --- 5. Feature Correlation Heatmap ---
st.header("5. Feature Correlation Matrix")
st.markdown("This heatmap reveals mathematical relationships between numerical features and the target (Price). High correlation = high predictive power in the ANN.")

numeric_df = df.select_dtypes(include=['float64', 'int64', 'int32'])
if not numeric_df.empty:
    fig_h, ax = plt.subplots(figsize=(12, 7))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm",
                fmt=".2f", ax=ax, linewidths=0.5)
    st.pyplot(fig_h)
else:
    st.warning("No numerical columns available for correlation.")

st.markdown("---")

# --- 6. Processed Data Inspector ---
st.header("6. Processed Dataset Inspector")
st.markdown(f"Showing the first 100 rows of the **{len(df):,}** fully cleaned and engineered records ready for the ANN model.")
st.dataframe(df.head(100), use_container_width=True)
