import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import  mean_absolute_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
r = 1092
w = 1070
c = r - w 
# Reading data from CSV file
df = pd.read_csv('BTC-USD (2).csv')
ssd = df[-c:]
df = df[:w]

# Converting date column to datetime format
df['Date'] = pd.to_datetime(df['Date'])

# Extracting year, month, day, and dayofweek as new features
df['year'] = df['Date'].dt.year
df['month'] = df['Date'].dt.month
df['day'] = df['Date'].dt.day
df['dayofweek'] = df['Date'].dt.dayofweek

# Preparing input data
X = df[['year', 'month', 'day', 'dayofweek']]
y_High = df['High']
y_Low = df['Low']

# Splitting data into train andtest sets
X_train, X_test, y_High_train, y_High_test, y_Low_train, y_Low_test = train_test_split(X, y_High, y_Low, test_size=0.2, random_state=11)

# Training model for High using XGBoost
model_High = xgb.XGBRegressor()
model_High.fit(X_train, y_High_train)

# Training model for Low using XGBoost
model_Low = xgb.XGBRegressor()
model_Low.fit(X_train, y_Low_train)

# Predicting High and Low for future days
num_days_future = c
future = pd.DataFrame({'Date': pd.date_range(start=df['Date'].max(), periods=num_days_future+1)[1:]})
future['year'] = future['Date'].dt.year
future['month'] = future['Date'].dt.month
future['day'] = future['Date'].dt.day
future['dayofweek'] = future['Date'].dt.dayofweek
future_x = future[['year', 'month', 'day', 'dayofweek']]

High_forecast = model_High.predict(future_x)
Low_forecast = model_Low.predict(future_x)

# Evaluating the model with test data
High_test_forecast = model_High.predict(X_test)
Low_test_forecast = model_Low.predict(X_test)
High_error_percentage = mean_absolute_error(y_High_test, High_test_forecast) / y_High_test.mean() * 100
Low_error_percentage = mean_absolute_error(y_Low_test, Low_test_forecast) / y_Low_test.mean() * 100

# Preparing new DataFrame to display forecasts along with date and error percentage
forecast_df = pd.DataFrame({'Date': future['Date'], 'High Forecast': High_forecast, 'Low Forecast': Low_forecast})
print(forecast_df)
# Printing the error percentage
print(f"High Error Percentage: {High_error_percentage}%")
print(f"Low Error Percentage: {Low_error_percentage}%")

# Plotting the actual and predicted values
plt.figure(figsize=(10, 6))
plt.plot(df['Date'], df['High'], label='Actual High')
plt.plot(future['Date'], High_forecast , label='Predicted High')
plt.plot(future['Date'], ssd['High'] , label='Actual High_forecast ')
plt.legend()
plt.xlabel('Date')
plt.ylabel('High')
plt.title('Actual vs. Predicted High')
plt.show()

# Plotting the actual and predicted values
plt.figure(figsize=(10, 6))
plt.plot(future['Date'], High_forecast, label='Predicted High')
plt.plot(future['Date'], ssd['High'] , label='Actual High_forecast ')
plt.legend()
plt.xlabel('Date')
plt.ylabel('High')
plt.title('Actual vs. Predicted High')
plt.show()
