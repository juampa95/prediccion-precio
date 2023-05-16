import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime
import ipywidgets as widgets
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import mean_absolute_percentage_error,mean_squared_error, r2_score, mean_absolute_error


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
