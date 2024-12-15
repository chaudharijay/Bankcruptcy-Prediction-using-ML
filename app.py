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
scaler.fit(X)  # Fit the scaler on the original data

# Apply PCA (use the same PCA object)
pca = PCA(n_components=3)  # Use the same number of components
pca.fit(scaler.transform(X)) 

# Create a Streamlit app
def main():
    st.title("Bankruptcy Prediction App")

    # Get user input for features (replace with actual feature names)
    industrial_risk = st.slider("Industrial Risk", 0.0, 1.0, 0.5, step=0.5)
    management_risk = st.slider("Management Risk", 0.0, 1.0, 0.5, step=0.5)
    financial_flexibility = st.slider("Financial Flexibility", 0.0, 1.0, 0.5, step=0.5)
    credibility = st.slider("Credibility", 0.0, 1.0, 0.5, step=0.5)
    competitiveness = st.slider("Competitiveness", 0.0, 1.0, 0.5, step=0.5)
    operating_risk = st.slider("Operating Risk", 0.0, 1.0, 0.5, step=0.5)

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
    st.header("Predictions:")
    st.write("**Logistic Regression:**", "Bankruptcy" if lr_pred == 1 else "Non-Bankruptcy")
    st.write("**Decision Tree:**", "Bankruptcy" if dt_pred == 1 else "Non-Bankruptcy")
    st.write("**Random Forest:**", "Bankruptcy" if rf_pred == 1 else "Non-Bankruptcy")

if __name__ == "__main__":
    main()