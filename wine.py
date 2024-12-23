import pickle
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objs as go
from sklearn.preprocessing import MinMaxScaler
import time

# Load pipeline
with open('model_kmeans_wine_quality.pkl', 'rb') as f:
    pipeline = pickle.load(f)

scaler = pipeline['scaler']
kmeans = pipeline['kmeans']
log_features = pipeline['log_transform_cols']

# Page Configuration
st.set_page_config(
    page_title="Wine Predictions",
    page_icon="🍇",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Advanced Custom CSS
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    :root {
        --deep-purple: #5D3FD3;
        --soft-purple: #8A4FFF;
        --accent-pink: #FF6B9E;
        --dark-background: #0F0F1A;
        --mid-background: #1A1A2E;
        --text-color: #E6E6E6;
    }

    body {
        background-color: var(--dark-background);
        color: var(--text-color);
        font-family: 'Poppins', sans-serif;
    }

    .stApp {
        background-color: var(--dark-background);
        max-width: 1400px;
        margin: 0 auto;
    }

    /* Elegant Header */
    .header-container {
        display: flex;
        align-items: center;
        justify-content: center;
        margin-bottom: 2rem;
        background: linear-gradient(135deg, var(--deep-purple), var(--soft-purple));
        border-radius: 20px;
        padding: 1rem;
        box-shadow: 0 10px 30px rgba(93, 63, 211, 0.3);
    }

    .header-title {
        font-size: 3.5rem;
        font-weight: 700;
        color: white;
        text-shadow: 0 4px 6px rgba(0,0,0,0.2);
        margin-left: 1rem;
    }

    /* Sleek Inputs */
    .stNumberInput > div > div > input {
        background-color: var(--mid-background);
        color: var(--text-color);
        border: 2px solid var(--soft-purple);
        border-radius: 15px;
        padding: 12px;
        font-size: 1rem;
        transition: all 0.3s ease;
    }

    .stNumberInput > div > div > input:focus {
        border-color: var(--accent-pink);
        box-shadow: 0 0 15px rgba(138, 79, 255, 0.3);
    }

    /* Modern Button */
    .stButton > button {
        background: linear-gradient(135deg, var(--soft-purple), var(--accent-pink));
        color: white !important;
        font-weight: 600;
        border: none;
        border-radius: 20px;
        padding: 15px 30px;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        transition: all 0.4s ease;
    }

    .stButton > button:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 20px rgba(138, 79, 255, 0.4);
    }

    /* Result Container */
    .result-container {
        background: linear-gradient(145deg, var(--mid-background), #2A2A40);
        border: 2px solid var(--soft-purple);
        border-radius: 20px;
        padding: 30px;
        text-align: center;
        box-shadow: 0 15px 40px rgba(0, 0, 0, 0.3);
        margin-top: 20px;
    }

    .result-cluster {
        font-size: 4rem;
        color: var(--accent-pink);
        font-weight: 800;
        text-shadow: 0 6px 10px rgba(255, 107, 158, 0.3);
    }
    </style>
""", unsafe_allow_html=True)

# Logo and Header
st.markdown("""
    <div class="header-container">
        <svg width="100" height="100" viewBox="0 0 100 100" xmlns="http://www.w3.org/2000/svg">
            <circle cx="50" cy="50" r="45" fill="#8A4FFF"/>
            <path d="M25,50 Q50,20 75,50 T125,50" stroke="white" stroke-width="6" fill="none"/>
            <path d="M30,60 Q50,40 70,60 T110,60" stroke="white" stroke-width="4" fill="none"/>
            <circle cx="50" cy="70" r="10" fill="white"/>
        </svg>
        <div class="header-title">Wine Predictions</div>
    </div>
""", unsafe_allow_html=True)

# Sidebar with Advanced Styling
st.sidebar.title("🍷 Wine Insight")
st.sidebar.markdown("""
    ### Guide
    Masukkan sifat kimia anggur Anda secara tepat.
""")

# Input Columns
col1, col2 = st.columns(2)

with col1:
    fixed_acidity = st.number_input('Fixed Acidity')
    volatile_acidity = st.number_input('Volatile Acidity')
    citric_acid = st.number_input('Citric Acid')
    residual_sugar = st.number_input('Residual Sugar')
    chlorides = st.number_input('Chlorides')

with col2:
    free_sulfur_dioxide = st.number_input('Free Sulfur Dioxide')
    total_sulfur_dioxide = st.number_input('Total Sulfur Dioxide')
    density = st.number_input('Density')
    pH = st.number_input('pH')
    sulphates = st.number_input('Sulphates')

alcohol = st.number_input('Alcohol')


# Predict Cluster Button
if st.button('🔮 Analyze Wine'):
    # Validasi input - cek apakah semua field sudah terisi
    input_fields = [
        fixed_acidity, volatile_acidity, citric_acid, 
        residual_sugar, chlorides, free_sulfur_dioxide, 
        total_sulfur_dioxide, density, pH, 
        sulphates, alcohol
    ]
    
    # Cek apakah ada field yang masih 0 atau belum diisi
    if any(field == 0 or field is None for field in input_fields):
        st.error("⚠️ Harap lengkapi semua field sebelum melakukan analisis!")
    else:
        with st.spinner('🍇 Decoding Wine Characteristics...'):
            time.sleep(2)

            # Prepare the new data
            data_baru = pd.DataFrame([{
                'fixed acidity': fixed_acidity,
                'volatile acidity': volatile_acidity,
                'citric acid': citric_acid,
                'residual sugar': residual_sugar,
                'chlorides': chlorides,
                'free sulfur dioxide': free_sulfur_dioxide,
                'total sulfur dioxide': total_sulfur_dioxide,
                'density': density,
                'pH': pH,
                'sulphates': sulphates,
                'alcohol': alcohol
            }])

            # Transform log features
            for col in log_features:
                if col in data_baru.columns:
                    data_baru[col] = np.log1p(data_baru[col])

            # Normalize the new data
            data_baru_scaled = scaler.transform(data_baru)

            # Predict cluster
            cluster_pred = kmeans.predict(data_baru_scaled)[0]

            # Quality Description
            cluster_quality = {
                0: "kualitas sedang 🍇\nAnggur ini memiliki karakteristik rata-rata. Anggur ini mungkin lebih baik jika disempurnakan lebih lanjut.",
                1: "Kualitas Baik 🏆\nAnggur yang sangat baik dengan sifat kimia yang unggul dan karakteristik yang seimbang!"
            }

            # Result Display
            st.markdown(
                f'''
                <div class="result-container">
                    <h3 style="color: var(--soft-purple);">
                        🍷 Wine Quality Prediction
                    </h3>
                    <div class="result-cluster">
                        Cluster {cluster_pred}
                    </div>
                    <div style="margin-top: 15px; color: var(--text-color);">
                        {cluster_quality.get(cluster_pred, "Unknown Wine Quality")}
                    </div>
                </div>
                ''', 
                unsafe_allow_html=True
            )