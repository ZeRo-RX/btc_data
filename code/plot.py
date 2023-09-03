import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('BTC-USD (1).csv')

X = df[['Date']]
y = df['Adj Close']

plt.plot(df['Date'],df['Adj Close'])
plt.show()