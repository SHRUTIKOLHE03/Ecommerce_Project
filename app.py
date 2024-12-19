import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Streamlit App Title
st.title("E-Commerce Delivery Prediction")
st.write("Predict whether an order will be delivered or not based on shipment details.")

# Input Fields
order_id = st.text_input("Order ID")
mode_of_shipment = st.selectbox("Mode of Shipment", ["Flight", "Ship", "Road"])
warehouse_block = st.selectbox("Warehouse Block", ["A", "B", "C", "D", "F"])

# Dummy Dataset for Training
@st.cache_data
def load_training_data():
    # Example dataset
    data = pd.DataFrame({
        "Mode_of_Shipment": ["Flight", "Ship", "Road", "Ship", "Flight", "Road", "Ship", "Flight"],
        "Warehouse_block": ["A", "B", "C", "D", "A", "C", "D", "F"],
        "Delivery_Status": [1, 0, 1, 0, 1, 1, 0, 1],
    })
    return data

# Load the data
data = load_training_data()

# Preprocessing
@st.cache_data
def preprocess_and_train(data):
    # One-hot encoding
    data_encoded = pd.get_dummies(data, columns=["Mode_of_Shipment", "Warehouse_block"])
    
    # Splitting data
    X = data_encoded.drop("Delivery_Status", axis=1)
    y = data_encoded["Delivery_Status"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train the Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    return rf_model, X.columns

# Train the model
rf_model, feature_columns = preprocess_and_train(data)

# Predict on User Input
if st.button("Predict Delivery Status"):
    # Prepare input data
    input_data = pd.DataFrame({
        "Mode_of_Shipment": [mode_of_shipment],
        "Warehouse_block": [warehouse_block],
    })

    # One-hot encode user input
    input_data = pd.get_dummies(input_data, columns=["Mode_of_Shipment", "Warehouse_block"])
    
    # Add missing columns
    for col in feature_columns:
        if col not in input_data:
            input_data[col] = 0
    
    input_data = input_data[feature_columns]  # Reorder columns to match training data

    # Make Prediction
    prediction = rf_model.predict(input_data)[0]
    status = "Delivered" if prediction == 1 else "Not Delivered"
    
    # Display Prediction
    st.subheader(f"Delivery Status: {status}")
