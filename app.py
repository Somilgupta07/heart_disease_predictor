import streamlit as st
import pandas as pd
import numpy as np
import joblib, torch, os
import xgboost as xgb
import torch.nn as nn

# --- Neural Network Architecture ---
class HeartNet(nn.Module):
    def __init__(self, input_dim=13):
        super(HeartNet, self).__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 32), nn.ReLU(), nn.Dropout(0.2),
                                 nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())
    def forward(self, x): return self.net(x)

st.set_page_config(page_title="IBM Certified Heart AI", layout="wide")
st.title("💓 Heart Disease Diagnostic System")

# Load Global Scaler
scaler = joblib.load("models/scaler.pkl")

# --- Sidebar Inputs ---
st.sidebar.header("Patient Vitals")
def get_input():
    # Capture all 13 features from your CSV
    age = st.sidebar.slider("Age", 20, 80, 50)
    sex = st.sidebar.selectbox("Sex", [1, 0], format_func=lambda x: "Male" if x==1 else "Female")
    cp = st.sidebar.slider("Chest Pain Type", 0, 3, 1)
    trestbps = st.sidebar.slider("Resting Blood Pressure", 90, 200, 120)
    chol = st.sidebar.slider("Cholesterol", 120, 500, 200)
    fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120", [0, 1])
    restecg = st.sidebar.slider("Resting ECG", 0, 2, 0)
    thalach = st.sidebar.slider("Max Heart Rate", 70, 210, 150)
    exang = st.sidebar.selectbox("Exercise Induced Angina", [0, 1])
    oldpeak = st.sidebar.slider("ST Depression", 0.0, 6.0, 1.0)
    slope = st.sidebar.slider("Slope", 0, 2, 1)
    ca = st.sidebar.slider("Major Vessels", 0, 3, 0)
    thal = st.sidebar.slider("Thal", 0, 3, 2)
    
    return np.array([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])

user_data = get_input()

# --- Model Selection Logic ---
st.subheader("🤖 Diagnostic Core")
model_files = [f for f in os.listdir("models") if f.endswith(('.pkl', '.pth', '.json')) and "scaler" not in f]
selected_model = st.selectbox("Select Model Architecture", model_files)

if st.button("Run Prediction"):
    # 1. Scaling
    scaled_input = scaler.transform(user_data)
    
    # 2. Prediction based on file type
    if selected_model.endswith(".pth"): # PyTorch
        net = HeartNet()
        net.load_state_dict(torch.load(f"models/{selected_model}"))
        net.eval()
        with torch.no_grad():
            res = 1 if net(torch.FloatTensor(scaled_input)) > 0.5 else 0
    elif selected_model.endswith(".json"): # XGBoost
        bst = xgb.XGBClassifier()
        bst.load_model(f"models/{selected_model}")
        res = bst.predict(scaled_input)[0]
    else: # Traditional Scikit-Learn (.pkl)
        clf = joblib.load(f"models/{selected_model}")
        res = clf.predict(scaled_input)[0]

    # --- Result Display ---
    if res == 1:
        st.error("🚨 HIGH RISK: Cardiac abnormalities detected.")
    else:
        st.success("✅ LOW RISK: No significant heart disease markers found.")