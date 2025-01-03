import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pickle

# Load saved models
with open('logistic_regression_model.pkl', 'rb') as file:
    model_lr = pickle.load(file)
with open('decision_tree_model.pkl', 'rb') as file:
    model_dt = pickle.load(file)
with open('random_forest_model.pkl', 'rb') as file:
    model_rf = pickle.load(file)

# Load and preprocess data (for scaler and PCA)
df = pd.read_excel("Bankruptcy (2).xlsx") 
df['class'] = df['class'].astype(str).str.strip().str.lower().replace({'bankruptcy': 1, 'non-bankruptcy': 0})
X = df.drop('class', axis=1) 

# Scale the data (use the same scaler used during training)
scaler = StandardScaler()
scaler.fit(X) # Fit the scaler on the original data

# Apply PCA (use the same PCA object)
pca = PCA(n_components=3) # Use the same number of components
pca.fit(scaler.transform(X)) 

# Create a Streamlit app
def main():
    st.set_page_config(page_title="Bankruptcy Prediction App", page_icon="📈", layout="wide")

    st.markdown("""
    <style>
        body {
            font-family: Arial, sans-serif;
            background-color: #f0f2f6;
        }
        .stApp {
            padding: 1rem;
        }
        .main-title {
            color: #2c3e50;
            text-align: center;
            margin-bottom: 1rem;
        }
        .prediction-box {
            background-color: #ecf0f1;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            margin-top: 0;
            text-align: center;
        }
        .slider-container {
            background-color: #ffffff;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            margin-top: 0;
        }
        .highlight {
            font-weight: bold;
            color: #16a085;
        }
        .result {
            font-size: 1.2rem;
            margin-bottom: 0.5rem;
        }
        .stMarkdown {
            margin: 0 !important;
        }
    </style>
    
    <h1 class="main-title">Bankruptcy Prediction App</h1>
    """, unsafe_allow_html=True)

    # Create two columns: one for sliders and one for results
    col1, col2 = st.columns([2, 2])

    with col1:
        # st.markdown("<div class='slider-container'>", unsafe_allow_html=True)
        st.header("Input Features")
        
        col1_1, col1_2 = st.columns(2)
        with col1_1:
            industrial_risk = st.slider("Industrial Risk", 0.0, 1.0, 0.5, step=0.5)
            financial_flexibility = st.slider("Financial Flexibility", 0.0, 1.0, 0.5, step=0.5)
            competitiveness = st.slider("Competitiveness", 0.0, 1.0, 0.5, step=0.5)
        with col1_2:
            management_risk = st.slider("Management Risk", 0.0, 1.0, 0.5, step=0.5)
            credibility = st.slider("Credibility", 0.0, 1.0, 0.5, step=0.5)
            operating_risk = st.slider("Operating Risk", 0.0, 1.0, 0.5, step=0.5)

        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        # st.markdown("<div class='prediction-box'>", unsafe_allow_html=True)
        st.header("Prediction Results")

        # Create a new sample data point with user input
        new_data = pd.DataFrame({
            'industrial_risk': [industrial_risk],
            'management_risk': [management_risk],
            'financial_flexibility': [financial_flexibility],
            'credibility': [credibility],
            'competitiveness': [competitiveness],
            'operating_risk': [operating_risk]
        })

        # Scale the new data
        new_data_scaled = scaler.transform(new_data) 

        # Apply PCA transformation
        new_data_pca = pca.transform(new_data_scaled)

        # Make predictions
        lr_pred = model_lr.predict(new_data_pca)[0]
        dt_pred = model_dt.predict(new_data_pca)[0]
        rf_pred = model_rf.predict(new_data_pca)[0]

        # Display predictions
        st.markdown(f"<div class='result'><span class='highlight'>Logistic Regression:</span> {'Bankruptcy' if lr_pred == 0 else 'Non-Bankruptcy'}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='result'><span class='highlight'>Decision Tree:</span> {'Bankruptcy' if dt_pred == 0 else 'Non-Bankruptcy'}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='result'><span class='highlight'>Random Forest:</span> {'Bankruptcy' if rf_pred == 0 else 'Non-Bankruptcy'}</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
