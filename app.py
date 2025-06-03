import streamlit as st
import pickle
import numpy as np


# Load the model
with open('house_price_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("House Price Predictor")

# Inputs
sqft = st.number_input("Square Footage", min_value=100, step=10)
bedrooms = st.number_input("Number of Bedrooms", min_value=1, step=1)
bathrooms = st.number_input("Number of Bathrooms", min_value=1, step=1)

if st.button("Predict Price"):
    features = np.array([[sqft, bedrooms, bathrooms]])
    prediction = model.predict(features)[0]
    st.success(f"Predicted Price: ${prediction:,.2f}")