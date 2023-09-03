import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
import numpy as np
from datetime import datetime, timedelta

# Load the data
df = pd.read_csv('BTC-USD (1).csv')  # replace with the path to your csv file
dates = pd.to_datetime(df['Date'])
df = df['Close'].values
df = df.reshape(-1, 1)

# Scale the data
scaler = MinMaxScaler(feature_range=(0,1))
df = scaler.fit_transform(df)

# Create the training data set
x_train, y_train = [], []

for i in range(60, len(df)):
    x_train.append(df[i-60:i, 0])
    y_train.append(df[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# Build the LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50, return_sequences=False))
model.add(Dense(25))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=1)

# Define the number of days to predict
num_days = 5

# Get the predicted values for the next num_days
predicted_prices = []
for i in range(num_days):
    last_60_days = df[-60:]
    last_60_days = np.reshape(last_60_days, (1, last_60_days.shape[0], 1))
    predicted_price = model.predict(last_60_days)
    predicted_prices.append(predicted_price[0][0])
    df = np.append(df, predicted_price)

predicted_prices = scaler.inverse_transform(np.array(predicted_prices).reshape(-1, 1))

# Get the dates for the predicted prices
last_date = dates.iloc[-1]
predicted_dates = [last_date + timedelta(days=i+1) for i in range(num_days)]

# Print the predicted prices and their corresponding dates
for i in range(num_days):
    print('Predicted price for', predicted_dates[i].strftime('%Y-%m-%d'), ':', predicted_prices[i][0])

# Calculate the error percentage for the last day
real_price = df[-num_days-1]
real_price = scaler.inverse_transform(real_price.reshape(-1, 1))
error_percentage = abs((predicted_prices[-1][0] - real_price[0][0]) / real_price[0][0]) * 100

print('Error percentage for the last day:', error_percentage)