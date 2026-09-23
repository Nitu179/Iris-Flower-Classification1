import streamlit as st
import numpy as np

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# Page settings
st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="centered"
)

# Title
st.title("🌸 Iris Flower Classification")

st.write(
    "Enter the measurements of an Iris flower "
    "to predict its species using Machine Learning."
)

# Load Iris dataset
iris = load_iris()

X = iris.data
y = iris.target

# Scale the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train the model
model = LogisticRegression(max_iter=200)
model.fit(X_scaled, y)

# Input fields
st.subheader("🌿 Flower Measurements")

sepal_length = st.number_input(
    "Sepal Length (cm)",
    min_value=0.0,
    max_value=10.0,
    value=5.1
)

sepal_width = st.number_input(
    "Sepal Width (cm)",
    min_value=0.0,
    max_value=10.0,
    value=3.5
)

petal_length = st.number_input(
    "Petal Length (cm)",
    min_value=0.0,
    max_value=10.0,
    value=1.4
)

petal_width = st.number_input(
    "Petal Width (cm)",
    min_value=0.0,
    max_value=10.0,
    value=0.2
)

# Prediction
if st.button("🔍 Predict Flower"):

    sample = np.array([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]])

    sample_scaled = scaler.transform(sample)

    prediction = model.predict(sample_scaled)[0]

    predicted_species = iris.target_names[prediction]

    st.success(
        f"🌸 Predicted Flower: **Iris-{predicted_species}**"
    )
