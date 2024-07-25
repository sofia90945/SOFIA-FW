import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load model and encoders
def load_model():
    with open('random_forest_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

def load_encoders():
    with open('label_encoders.pkl', 'rb') as file:
        encoders = pickle.load(file)
    return encoders

model = load_model()
label_encoder_category, label_encoder_item, label_encoder_ingredients, _ = load_encoders()

# Title
st.title("Aplikasi Prediksi Profitabilitas Restoran")

# Input form
st.header("Input Data")
menu_category = st.selectbox("Menu Category", ["Beverages", "Appetizers", "Desserts", "Main Course"])
menu_item = st.text_input("Menu Item")
ingredients = st.text_input("Ingredients")
price = st.number_input("Price", min_value=0.0, value=10.0)

# Preprocessing input with validation
try:
    encoded_menu_category = label_encoder_category.transform([menu_category])[0]
except ValueError:
    st.error("Menu Category tidak dikenali.")
    encoded_menu_category = None

try:
    encoded_menu_item = label_encoder_item.transform([menu_item])[0]
except ValueError:
    st.error("Menu Item tidak dikenali.")
    encoded_menu_item = None

try:
    encoded_ingredients = label_encoder_ingredients.transform([ingredients])[0]
except ValueError:
    st.error("Ingredients tidak dikenali.")
    encoded_ingredients = None

if encoded_menu_category is not None and encoded_menu_item is not None and encoded_ingredients is not None:
    input_data = np.array([[encoded_menu_category, encoded_menu_item, encoded_ingredients, price]])

    # Prediction
    if st.button("Prediksi"):
        prediction = model.predict(input_data)
        st.write(f"Hasil Prediksi: {'High' if prediction[0] == 1 else 'Low'}")
else:
    st.warning("Mohon isi semua input dengan benar.")
