import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.callbacks import EarlyStopping
import numpy as np
from datetime import timedelta

# Load the data
df = pd.read_csv('BTC-USD (1).csv')  # replace with the path to your csv file
dates = pd.to_datetime(df['Date'])
df = df['High'].values
df = df.reshape(-1, 1)

# Scale the data
scaler = MinMaxScaler(feature_range=(0,1))
df = scaler.fit_transform(df)

# Create the training data set
x_train, y_train = [], []

for i in range(35, len(df)):  # increase the number of days used for training
    x_train.append(df[i-35:i, 0])  # use 120 days of data for training
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

# Define the early stopping criteria
early_stop = EarlyStopping(monitor='val_loss', patience=2)

# Train the model
model.fit(x_train, y_train, batch_size=1, epochs=5, validation_split=0.2, callbacks=[early_stop])  # increase the number of epochs and add validation split and early stopping

# Define the number of days to predict
num_days = 12

# Get the predicted values for the next num_days
predicted_prices = []
for i in range(num_days):
    last_120_days = df[-35:]  # use the last 120 days of data for prediction
    last_120_days = np.reshape(last_120_days, (1, last_120_days.shape[0], 1))
    predicted_price = model.predict(last_120_days)
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
# Calculate the error percentage and accuracy
error_percentage = abs((predicted_prices[-1][0] - real_price[0][0]) / real_price[0][0]) * 100
accuracy = 100 - error_percentage

print('Error percentage for the last day:', error_percentage)
print('Accuracy for the last day:', accuracy)
