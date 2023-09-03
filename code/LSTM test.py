import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# Loading the dataset
df = pd.read_csv('BTC-USD (1).csv')

# Preprocessing input data
X = df.drop(columns=['High'])
X = MinMaxScaler().fit_transform(X)

# Preprocessing output data
y = df['High']
y = MinMaxScaler().fit_transform(y.values.reshape(-1, 1))

# Reshaping input data to be 3D [samples, timesteps, features] for LSTM
X = X.reshape((X.shape[0], 1, X.shape[1]))

# Splitting data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Defining the LSTM model
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# Training the model
model.fit(X_train, y_train, epochs=100, verbose=0)

# Predicting 'High' for the next 10 days
num_days = 10
future = np.empty((num_days, X.shape[1], X.shape[2]))
predictions = np.empty((num_days, 1))
for i in range(num_days):
    prediction = model.predict(future[i:i+1])
    predictions[i] = prediction
    future[i] = future[i-1]
    future[i, :, -1] = prediction

# Scaling back the predictions to original scale
predictions = MinMaxScaler().fit(y).inverse_transform(predictions)

print(predictions)