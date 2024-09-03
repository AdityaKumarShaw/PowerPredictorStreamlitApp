import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the trained model
# Load the trained model
model = joblib.load('DEPP.joblib')

# Mean and standard deviation values used during training
temp_mean, temp_std = 25.248626, 8.384665      # Example values
rhum_mean, rhum_std = 26.467537, 19.956216     # Example values
wspd_mean, wspd_std = 55.428663, 32.878623       # Example values
pres_mean, pres_std = 239.630952, 423.936913    # Example values

# Title of the app
st.title('Delhi Electricity Consumption Predictor')

# Collect user inputs for prediction
st.header('Input the following features for prediction')

temp = st.number_input('Temperature (°C)', value=25.0)
rhum = st.number_input('Relative Humidity (%)', value=60.0)
wspd = st.number_input('Wind Speed (km/h)', value=10.0)
pres = st.number_input('Pressure (hPa)', value=1010.0)

# Standard scaling formula: (value - mean) / std
if st.button('Predict Electricity Consumption'):
    # Apply standard scaling
    scaled_temp = (temp - temp_mean) / temp_std
    scaled_rhum = (rhum - rhum_mean) / rhum_std
    scaled_wspd = (wspd - wspd_mean) / wspd_std
    scaled_pres = (pres - pres_mean) / pres_std
    
    # Prepare input data for prediction
    scaled_input = np.array([[scaled_temp, scaled_rhum, scaled_wspd, scaled_pres]])
    
    # Make prediction
    prediction = model.predict(scaled_input)
    
    # Display prediction
    st.subheader('Predicted Electricity Consumption')
    st.write(f'{prediction[0]:.2f} units')
