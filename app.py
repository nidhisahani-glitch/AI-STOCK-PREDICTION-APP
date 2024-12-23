import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model
import streamlit as st
import matplotlib.pyplot as plt
import os

# Check for the model file
if not os.path.exists('Stock Predictions Model.keras'):
    st.error("Model file not found. Please check the path and upload the correct file.")
    st.stop()

# Load the pre-trained model
model = load_model('Stock Predictions Model.keras')

# Streamlit Header
st.header('Stock Market Predictor')

# User Input for Stock Symbol and Date Range
stock = st.text_input('Enter Stock Symbol', 'GOOG')
start = st.date_input('Start Date', pd.to_datetime('2012-01-01'))
end = st.date_input('End Date', pd.to_datetime('2022-12-31'))

# Fetching Stock Data with Error Handling
try:
    data = yf.download(stock, start, end)
    if data.empty:
        st.error("No data found for the entered symbol. Please check the symbol.")
        st.stop()
except Exception as e:
    st.error(f"Error fetching data: {e}")
    st.stop()

# Display Stock Data
st.subheader('Stock Data')
st.write(data)

# Data Splitting
data_train = pd.DataFrame(data.Close[0: int(len(data)*0.80)])
data_test = pd.DataFrame(data.Close[int(len(data)*0.80): len(data)])

# Data Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))

# Include last 100 days of training data for prediction continuity
past_100_days = data_train.tail(100)
data_test = pd.concat([past_100_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

# Moving Average Visualizations
st.subheader('Price vs MA50')
ma_50_days = data.Close.rolling(50).mean()
fig1 = plt.figure(figsize=(8, 6))
plt.plot(ma_50_days, 'r', label="MA50")
plt.plot(data.Close, 'g', label="Closing Price")
plt.legend()
st.pyplot(fig1)

st.subheader('Price vs MA50 vs MA100')
ma_100_days = data.Close.rolling(100).mean()
fig2 = plt.figure(figsize=(8, 6))
plt.plot(ma_50_days, 'r', label="MA50")
plt.plot(ma_100_days, 'b', label="MA100")
plt.plot(data.Close, 'g', label="Closing Price")
plt.legend()
st.pyplot(fig2)

st.subheader('Price vs MA100 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8, 6))
plt.plot(ma_100_days, 'r', label="MA100")
plt.plot(ma_200_days, 'b', label="MA200")
plt.plot(data.Close, 'g', label="Closing Price")
plt.legend()
st.pyplot(fig3)

# Preparing Data for Predictions
x = []
y = []

for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i-100:i])
    y.append(data_test_scale[i, 0])

x, y = np.array(x), np.array(y)

# Model Predictions
predict = model.predict(x)

# Reverse Scaling to Original Values
scale = 1 / scaler.scale_
predict = predict * scale
y = y * scale

# Display Prediction Results Table
st.subheader("Prediction Results")
results = pd.DataFrame({'Original Price': y.flatten(), 'Predicted Price': predict.flatten()})
st.write(results.head(10))

# Visualization: Original vs Predicted Price
st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(8, 6))
plt.plot(y, 'r', label='Original Price')
plt.plot(predict, 'g', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig4)
