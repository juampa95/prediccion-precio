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
rcParams['figure.figsize'] = 15,6


# functions

def plotting(data_stock,date_start):
    import matplotlib.gridspec as gridspec
    from matplotlib.ticker import MaxNLocator

    # Creacion de un grafico custom
    fig = plt.figure(constrained_layout=True, figsize=(16, 8))

    # determinacion de las ubicaciones de los graficos , pero aqui los graficos ocuparan mas espaciones
    grid = gridspec.GridSpec(ncols=6, nrows=3, figure=fig)

    # get dataframe by stock name

    # set position plot
    ax1 = fig.add_subplot(grid[0, :2])
    ax1.set_title('Closing Price')
    sns.lineplot(data=data_stock.loc[date_start:], x=data_stock.loc[date_start:].index, y="Adj Close", ax=ax1)

    ax2 = fig.add_subplot(grid[0, 2:4])
    ax2.set_title('Volume stock sell')
    sns.lineplot(data=data_stock.loc[date_start:], x=data_stock.loc[date_start:].index, y="Volume", ax=ax2)

    ax4 = fig.add_subplot(grid[0, 4:])
    ax4.set_title('Daily Return stock ')
    sns.lineplot(data=data_stock.loc[date_start:], x=data_stock.loc[date_start:].index,
                 y=data_stock['Adj Close'].loc[date_start:].pct_change(), ax=ax4)

    ax3 = fig.add_subplot(grid[1:, :])
    ax3.set_title('Comparision between moving average')
    ma_columns = moving_avarege_calc(data_stock)

    sns.lineplot(data=data_stock[ma_columns].loc[date_start:], ax=ax3)


def moving_avarege_calc(data_stock):
    ma_day = [10, 20, 50]
    ma_columns = []
    for ma in ma_day:
        column_name = f"Moving_average_{ma}"
        ma_columns.append(column_name)
        data_stock[column_name] = data_stock['Adj Close'].rolling(ma).mean()
    ma_columns.insert(0, 'Adj Close')

    return ma_columns


def metrics_custom(y_true, y_predict):
    metric = {'MAPE': mean_absolute_percentage_error(y_true, y_predict),
              'MAE': mean_absolute_error(y_true, y_predict),
              'MSE': mean_squared_error(y_true, y_predict),
              'r2': r2_score(y_true, y_predict)}
    return metric

# IMPORTAMOS DATASET

df_meli_arg = yf.download('MELI.BA',start='2002-01-01', end = '2023-05-12')

# En el archivo pruebas.py se observó que las acciones de Mercado Libre (MEELI.BA) sufrieron una division
# conocido como stock split entre el 21/10/2020 y 22/10/2020, lo que dividió las acciones a razón de 25:1
# por lo que se hace una transformación inicial al dataset dividiendo los valores anteriores al 21/10/2020
# por 25 y multiplicando el volumen por 25. Esto no se si es correcto, pero fue una manera de normalizar lo
# sucedido en la fecha. Debera probarse realizar todo el proceso partiendo desde 22/10/2020 para no tener que
# modificar ningún valor.


df_meli = df_meli_arg.copy()
mask = df_meli.index <= '2020-10-21'
colsmult = ['Open','High','Low','Close','Adj Close']
colsdiv = ['Volume']
df_meli.loc[mask,colsmult] /=25
df_meli.loc[mask,colsdiv] *=25
df_meli

plotting(df_meli,'2020-05-01')
plt.show()

# ------------- INICIO DE MODELADO ----------------

#split data in train and test , select testing
train = df_meli.loc[:'2022-12-31']
test = df_meli.loc['2023-01-01':]

from matplotlib import rcParams

# figure size in inches
rcParams['figure.figsize'] = 15,6
sns.lineplot(data=train, x=train.index, y="High").set_title('Stock Price MELI.BA from 2002 to now')
sns.lineplot(data=test, x=test.index, y="High")

plt.show()

# -------------- TRANSFORMACIONES ----------------

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(train['High'].values.reshape(-1,1))

# -------------- PREPARADO DE CONJUNTOS TRAIN/TEST ------------------
# Transformo los datos de manera tal que 60 días de datos sirvan como base para calcular el valor
# del día siguiente. Por esto creamos una lista para los valores de x_train en los que cada uno de los valores
# de esta lista serán 60 valores corridos, que me servirán para determinar el valor del día siguiente.

prediction_days = 60

x_train = []
y_train = []

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x - prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# -------------- CREACIÓN DEL MODELO ----------------

def LSTM_model():
    model = Sequential()

    model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))

    model.add(LSTM(units=50))
    model.add(Dropout(0.2))

    model.add(Dense(units=1))

    return model

model = LSTM_model()
model.summary()
model.compile(optimizer='adam',
              loss='mean_squared_error')

# -------------- ENTRENAMIENTO -----------------

checkpointer = ModelCheckpoint(filepath = 'weights_best.hdf5',
                               verbose = 2,
                               save_best_only = True)

model.fit(x_train,
          y_train,
          epochs=25,
          batch_size = 32,
          callbacks = [checkpointer])

# -------------- PREDICCIÓN -------------------

# Obtenemos los datos de test, quitando del Dataframe original los usados para train y también se quitan
# una cantidad igual a los días de predicción con los que hará las primeras predicciones
inputs = df_meli[len(df_meli) - len(test) - prediction_days:]['High']

# Usamos el MinMax scaler que se utilizo para los datos de train
inputs_trasnform = scaler.transform(inputs.values.reshape(-1,1))

# Generamos la lista de listas de 60 días de pruebas para nuestro conjunto test
# (recordar que estos serán la base de la predicción del día siguiente)

x_test = []
for x in range(prediction_days, len(inputs_trasnform)):
    x_test.append(inputs_trasnform[x-prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] ,1))

# Se hacen las predicciones

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Se integran en un dataframe de test combinando el valor real con el valor de la predicción.

test_df = pd.DataFrame(test['High'])
test_df['type'] = 'Real'
predict_df = pd.DataFrame(pd.Series(predicted_prices.reshape(-1,), index=test_df.index, name='High'))
predict_df['type'] = 'Predict'

# Graficamos

all = pd.concat([test_df,predict_df])
sns.lineplot(x='Date', y='High', hue='type', data=all.reset_index()).set_title('Comparison between real data and predict data');
plt.show()

# Métricas de evaulación

f = metrics_custom(test['High'], predicted_prices)
print(pd.DataFrame(f.values(), index=f.keys(), columns=['Metrics']))

# Métricas de evaluación
# MAPE  1.625134e-01
# MAE   1.305200e+03
# MSE   1.935482e+06
# r2   -3.777418e-02
# --------------------- Agregado de variables a conjunto inicial ---------------------

df_meli_2 = df_meli.copy()

# Completamos con dias faltantes y los llenamos con el valor del dia anterior
df_meli_2
df_meli_2 = df_meli_2.asfreq('D')
df_meli_2.fillna(method='ffill', inplace=True)

# Creamos las columnas de diferencias con periodos anteriores 1,2,3,4 y 5 días

df_meli_2['diff_close'] = df_meli_2['Close'].diff()
df_meli_2['diff_close_p1'] = df_meli_2["diff_close"].shift(1)
df_meli_2['diff_close_p2'] = df_meli_2["diff_close"].shift(2)
df_meli_2['diff_close_p3'] = df_meli_2["diff_close"].shift(3)
df_meli_2['diff_close_p4'] = df_meli_2["diff_close"].shift(4)
df_meli_2 = df_meli_2.fillna(0)

# Creamos medias móviles o rolling windows para utilizarlas en el análisis

df_meli_2['rolling_windows_mean'] = df_meli_2['diff_close_p1'].rolling(window=2).mean()
df_meli_2['rolling_windows_mean7'] = df_meli_2['diff_close_p1'].rolling(window=7).mean()
df_meli_2['rolling_windows_mean14'] = df_meli_2['diff_close_p1'].rolling(window=14).mean()
df_meli_2['rolling_windows_mean21'] = df_meli_2['diff_close_p1'].rolling(window=21).mean()

# -------------- INICIO MODELADO  ------------

# Separamos el conjunto

train = df_meli_2.loc[:'2022-12-31']
test = df_meli_2.loc['2023-01-01':]

# Transformaciones

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(train['Close'].values.reshape(-1,1))

# Spliteo de conjuntos test y train

prediction_days = 60

x_train = []
y_train = []

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x - prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# -------------- CREACIÓN DEL MODELO ----------------
# Usamos el mismo que el anterior

# -------------- ENTRENAMIENTO -----------------

checkpointer = ModelCheckpoint(filepath = 'weights_best.hdf5',
                               verbose = 2,
                               save_best_only = True)

model.fit(x_train,
          y_train,
          epochs=25,
          batch_size = 32,
          callbacks = [checkpointer])

# -------------- PREDICCIÓN -------------------

inputs = df_meli_2[len(df_meli_2) - len(test) - prediction_days:]['Close']

inputs_trasnform = scaler.transform(inputs.values.reshape(-1,1))

x_test = []
for x in range(prediction_days, len(inputs_trasnform)):
    x_test.append(inputs_trasnform[x-prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] ,1))

# Se hacen las predicciones

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Se integran en un dataframe de test combinando el valor real con el valor de la predicción.

test_df = pd.DataFrame(test['Close'])
test_df['type'] = 'Real'
predict_df = pd.DataFrame(pd.Series(predicted_prices.reshape(-1,), index=test_df.index, name='Close'))
predict_df['type'] = 'Predict'

# Graficamos

all = pd.concat([test_df,predict_df])
sns.lineplot(x='Date', y='Close', hue='type', data=all.reset_index()).set_title('Comparison between real data and predict data');
plt.show()

# Métricas de evaulación

f = metrics_custom(test['Close'], predicted_prices)
print(pd.DataFrame(f.values(), index=f.keys(), columns=['Metrics']))

# RESULTADO METRICAS
#            Metrics
# MAPE       0.037564
# MAE      299.328733
# MSE   147353.026425
# r2         0.917217

# No guarde las métricas del resultado anterior, pero gráficamente, los valores se aproximan en mayor medida.

# ----------------- PRUEBA BIBLIOTECA TA (Technical analysis) -----------------

# https://technical-analysis-library-in-python.readthedocs.io/en/latest/ta.html

# Según chatGPT el top 10 de indicadores que debería usar son los siguientes

# Media Móvil Simple (SMA): ta.SMA()
# Media Móvil Exponencial (EMA): ta.EMA()
# Bandas de Bollinger (BB): ta.BBANDS()
# Índice de Fuerza Relativa (RSI): ta.RSI()
# Oscilador Estocástico (STOCH): ta.STOCH()
# MACD (Moving Average Convergence Divergence): ta.MACD()
# ATR (Average True Range): ta.ATR()
# ADX (Average Directional Index): ta.ADX()
# CCI (Commodity Channel Index): ta.CCI()
# ROC (Rate of Change): ta.ROC()

df_meli_3 = df_meli.copy()

df_meli_3['SMA'] = ta.trend.sma_indicator(df_meli_3['Close'], window=14)
df_meli_3['EMA'] = ta.trend.ema_indicator(df_meli_3['Close'], window=14)
df_meli_3['BB_upper'], df_meli_3['BB_middle'], df_meli_3['BB_lower'] = ta.volatility.bollinger_mavg(df_meli_3['Close']), ta.volatility.bollinger_mavg(df_meli_3['Close']), ta.volatility.bollinger_mavg(df_meli_3['Close'])
df_meli_3['RSI'] = ta.momentum.rsi(df_meli_3['Close'], window=14)
df_meli_3['stoch_k'], df_meli_3['stoch_d'] = ta.momentum.stoch(df_meli_3['High'], df_meli_3['Low'], df_meli_3['Close'], window=5, smooth_window=3), ta.momentum.stoch(df_meli_3['High'], df_meli_3['Low'], df_meli_3['Close'], window=5, smooth_window=3)
df_meli_3['MACD'] = ta.trend.macd_diff(df_meli_3['Close'])
df_meli_3['ATR'] = ta.volatility.average_true_range(df_meli_3['High'], df_meli_3['Low'], df_meli_3['Close'], window=14)
df_meli_3['ADX'] = ta.trend.adx(df_meli_3['High'], df_meli_3['Low'], df_meli_3['Close'], window=14)
df_meli_3['CCI'] = ta.trend.cci(df_meli_3['High'], df_meli_3['Low'], df_meli_3['Close'], window=14)
df_meli_3['ROC'] = ta.momentum.roc(df_meli_3['Close'], window=10)
df_meli_3 = df_meli_3.fillna(0)

# -------------- INICIO MODELADO  ------------

# Separamos el conjunto

train = df_meli_3.loc[:'2022-12-31']
test = df_meli_3.loc['2023-01-01':]

# Transformaciones

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(train['Close'].values.reshape(-1,1))

# Spliteo de conjuntos test y train

prediction_days = 60

x_train = []
y_train = []

for x in range(prediction_days, len(scaled_data)):
    x_train.append(scaled_data[x - prediction_days:x, 0])
    y_train.append(scaled_data[x, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

# -------------- CREACIÓN DEL MODELO ----------------
# Usamos el mismo que el anterior

# -------------- ENTRENAMIENTO -----------------

checkpointer = ModelCheckpoint(filepath = 'weights_best.hdf5',
                               verbose = 2,
                               save_best_only = True)

model.fit(x_train,
          y_train,
          epochs=25,
          batch_size = 32,
          callbacks = [checkpointer])

# -------------- PREDICCIÓN -------------------

inputs = df_meli_3[len(df_meli_3) - len(test) - prediction_days:]['Close']

inputs_trasnform = scaler.transform(inputs.values.reshape(-1,1))

x_test = []
for x in range(prediction_days, len(inputs_trasnform)):
    x_test.append(inputs_trasnform[x-prediction_days:x, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1] ,1))

# Se hacen las predicciones

predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Se integran en un dataframe de test combinando el valor real con el valor de la predicción.

test_df = pd.DataFrame(test['Close'])
test_df['type'] = 'Real'
predict_df = pd.DataFrame(pd.Series(predicted_prices.reshape(-1,), index=test_df.index, name='Close'))
predict_df['type'] = 'Predict'

# Graficamos
from matplotlib import rcParams
rcParams['figure.figsize'] = 15,6
all = pd.concat([test_df,predict_df])
sns.lineplot(x='Date', y='Close', hue='type', data=all.reset_index()).set_title('Comparison between real data and predict data 3');
plt.show()

# Métricas de evaulación

f = metrics_custom(test['Close'], predicted_prices)
print(pd.DataFrame(f.values(), index=f.keys(), columns=['Metrics']))

# Los resultados fueron muy malos.
# MAPE  1.287206e-01
# MAE   1.044339e+03
# MSE   1.298006e+06
# r2    3.040300e-01
# Esto se puede apreciar visualemnte en el gráfico N°3. Si bien parece que el modelo siguió correctamente la
# tendencia, lo hizo muy por debajo de los valores reales.
