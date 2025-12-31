import streamlit as st
import numpy as np
import pickle

st.title("Iris Flower Clustering App")

# Load models
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("kmeans.pkl", "rb") as f:
    kmeans = pickle.load(f)

with open("dbscan.pkl", "rb") as f:
    dbscan = pickle.load(f)

st.sidebar.header("Input Iris Features")

# User inputs
sepal_length = st.sidebar.number_input("Sepal Length", 0.0, 10.0, 5.1)
sepal_width  = st.sidebar.number_input("Sepal Width",  0.0, 10.0, 3.5)
petal_length = st.sidebar.number_input("Petal Length", 0.0, 10.0, 1.4)
petal_width  = st.sidebar.number_input("Petal Width",  0.0, 10.0, 0.2)

user_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
user_scaled = scaler.transform(user_data)

# Predictions
km_pred = kmeans.predict(user_scaled)[0]
db_pred = dbscan.fit_predict(user_scaled)[0]

st.write("### KMeans Cluster:", km_pred)
st.write("### DBSCAN Cluster:", db_pred)

st.markdown("""
---
**Notes**
- DBSCAN may return -1 (noise).
""")
