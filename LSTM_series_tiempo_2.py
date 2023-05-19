import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import seaborn as sns
import yfinance as yf
from datetime import datetime
import ipywidgets as widgets
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import mean_absolute_percentage_error,mean_squared_error, r2_score, mean_absolute_error
import ta
from ta import momentum,volatility,trend
from matplotlib import rcParams
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
rcParams['figure.figsize'] = 20,20
rcParams['figure.figsize'] = plt.rcParamsDefault['figure.figsize']

def metrics_custom(y_true, y_predict):
    metric = {'MAPE': mean_absolute_percentage_error(y_true, y_predict),
              'MAE': mean_absolute_error(y_true, y_predict),
              'MSE': mean_squared_error(y_true, y_predict),
              'r2': r2_score(y_true, y_predict)}
    return metric

df_meli_arg = yf.download('MELI.BA',start='2002-01-01', end = '2023-05-12')

df_meli = df_meli_arg.copy()
mask = df_meli.index <= '2020-10-21'
colsmult = ['Open','High','Low','Close','Adj Close']
colsdiv = ['Volume']
df_meli.loc[mask,colsmult] /=25
df_meli.loc[mask,colsdiv] *=25

df_meli['diff_close'] = df_meli['Close'].diff()
df_meli['SMA7'] = ta.trend.sma_indicator(df_meli['Close'], window=7)
df_meli['SMA14'] = ta.trend.sma_indicator(df_meli['Close'], window=14)
df_meli['SMA30'] = ta.trend.sma_indicator(df_meli['Close'], window=30)
df_meli['EMA7'] = ta.trend.ema_indicator(df_meli['Close'], window=7)
df_meli['EMA14'] = ta.trend.ema_indicator(df_meli['Close'], window=14)
df_meli['EMA30'] = ta.trend.ema_indicator(df_meli['Close'], window=30)
df_meli['BB_upper'], df_meli['BB_middle'], df_meli['BB_lower'] = ta.volatility.bollinger_mavg(df_meli['Close']), ta.volatility.bollinger_mavg(df_meli['Close']), ta.volatility.bollinger_mavg(df_meli['Close'])
df_meli['RSI14'] = ta.momentum.rsi(df_meli['Close'], window=14)
df_meli['RSI30'] = ta.momentum.rsi(df_meli['Close'], window=30)
df_meli['stoch_k'], df_meli['stoch_d'] = ta.momentum.stoch(df_meli['High'], df_meli['Low'], df_meli['Close'], window=5, smooth_window=3), ta.momentum.stoch(df_meli['High'], df_meli['Low'], df_meli['Close'], window=5, smooth_window=3)
df_meli['MACD'] = ta.trend.macd_diff(df_meli['Close'])
df_meli['ATR14'] = ta.volatility.average_true_range(df_meli['High'], df_meli['Low'], df_meli['Close'], window=14)
df_meli['ATR30'] = ta.volatility.average_true_range(df_meli['High'], df_meli['Low'], df_meli['Close'], window=30)
df_meli['ADX14'] = ta.trend.adx(df_meli['High'], df_meli['Low'], df_meli['Close'], window=14)
df_meli['ADX30'] = ta.trend.adx(df_meli['High'], df_meli['Low'], df_meli['Close'], window=30)
df_meli['CCI14'] = ta.trend.cci(df_meli['High'], df_meli['Low'], df_meli['Close'], window=14)
df_meli['CCI30'] = ta.trend.cci(df_meli['High'], df_meli['Low'], df_meli['Close'], window=30)
df_meli['ROC14'] = ta.momentum.roc(df_meli['Close'], window=14)
df_meli['ROC30'] = ta.momentum.roc(df_meli['Close'], window=30)
df_meli = df_meli.fillna(0)

corr_matrix = df_meli.corr()

sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')

plt.ylim(0,corr_matrix.shape[0])
plt.xlim(0,corr_matrix.shape[1])

plt.show()

# Separar el conjunto de entrenamiento

train = df_meli.loc[:'2022-12-31']
test = df_meli.loc['2023-01-01':]

# Transformaciones

scaler = MinMaxScaler(feature_range=(0,1))
scaled_train_data = scaler.fit_transform(train.values)

# Spliteo de conjuntos de prueba y entrenamiento

prediction_days = 60

x_train = []
y_train = []

for x in range(prediction_days, len(scaled_train_data)):
    x_train.append(scaled_train_data[x - prediction_days:x, :])
    y_train.append(scaled_train_data[x, train.columns.get_loc('Close')])  # Índice de la columna 'Close' en selected_columns

x_train, y_train = np.array(x_train), np.array(y_train)

# -------------- CREACIÓN DEL MODELO ----------------
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=y_train.shape[0]))
model.compile(optimizer='adam', loss='mean_squared_error')
# -------------- ENTRENAMIENTO -----------------
checkpointer = ModelCheckpoint(filepath='weights_best.hdf5', verbose=2, save_best_only=True)

model.fit(x_train, y_train, epochs=25, batch_size=32, callbacks=[checkpointer])

# -------------- PREDICCIONES -----------------

# Escalar los datos de prueba
scaled_test_data = scaler.transform(test.values)

# Preparar los datos de prueba en secuencias
x_test = []
y_test = []

for x in range(prediction_days, len(scaled_test_data)):
    x_test.append(scaled_test_data[x - prediction_days:x])
    y_test.append(scaled_test_data[x, test.columns.get_loc('Close')])

x_test, y_test = np.array(x_test), np.array(y_test)

# Hacer predicciones
predicted_prices = model.predict(x_test)

# Invertir la escala de las predicciones
predicted_prices = scaler.inverse_transform(predicted_prices)

len(predicted_prices[0])

df_meli