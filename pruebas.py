import pandas as pd
import yfinance as yf
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from bokeh.io import output_file, show
from bokeh.plotting import figure, ColumnDataSource
from bokeh.io import output_notebook


# CONFIGURACIÓN DE PANDAS
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


dolar_blue_hist = pd.read_excel("data/usd_blue_hist.xlsx")

dolar_blue_hist

df_meli_arg = yf.download('MELI.BA',start='2002-01-01', end = '2023-05-12')

df_meli_arg.tail()

fig, ax = plt.subplots(3, 1, figsize=(20, 9), sharex=True)
#
sns.lineplot(x="Date", y="Close", data=df_meli_arg, ax=ax[0])
sns.lineplot(x="Date", y="Adj Close", data=df_meli_arg, ax=ax[1])
sns.lineplot(x="Date", y="Volume", data=df_meli_arg, ax=ax[2])

plt.show()

# En estos gráficos podemos observar que en el 21/10/2020 las acciones caen abruptamente.
# Esto se debio a un stock split que hizo Mercado Libre en proporción 25:1.
# Por lo tanto, voy a modificar los valores antiguos en una relación 25:1 para que coincidan con los actuales

print(df_meli_arg.loc['2020-10-21'])
print(df_meli_arg.loc['2020-10-22'])

# Deberíamos dividir los valores de Open, High, Low, Close, Adj Close por 25 y multiplicar Volume por 25

df_meli = df_meli_arg.copy()
mask = df_meli.index <= '2020-10-21'
colsmult = ['Open','High','Low','Close','Adj Close']
colsdiv = ['Volume']
df_meli.loc[mask,colsmult] /=25
df_meli.loc[mask,colsdiv] *=25

fig, ax = plt.subplots(3, 1, figsize=(20, 9), sharex=True)
#
sns.lineplot(x="Date", y="Close", data=df_meli, ax=ax[0])
sns.lineplot(x="Date", y="Adj Close", data=df_meli, ax=ax[1])
sns.lineplot(x="Date", y="Volume", data=df_meli, ax=ax[2])
plt.show()

# Como podemos observar, ahora no existe ese quiebre abrupto en los datos entre el 21 y 22 de octubre del 2020

df_meli.head(20)

# Lo siguiente que podemos observar es que faltan dias en los datos, ya que sabados y domingo no hay cotizacion
# Vamos a completar el dataframe para no tener problemas

df_meli.shape
df_meli = df_meli.asfreq('D')

# Podemos ver que ahora teneemos mas valores, y ya no existen esos saltos de viernes a lunes. Pero se completo con
# valores Null

df_meli.shape
df_meli.head(20)

# Llenamos los null con los valores del dia anterior utilizando el metodo ffill

df_meli.fillna(method='ffill', inplace=True)
df_meli.head(20)

# Ahora vamos a analizar si existe alguna temporalidad en los datos con statsmodels

resultados = seasonal_decompose(df_meli[['Close']],period=365)
resultados.plot()
plt.show()


# La prueba de Dickey-Fuller Aumentada (ADF por sus siglas en inglés) es una prueba estadística que se utiliza para
# determinar si una serie temporal es estacionaria o no estacionaria. Una serie temporal se considera estacionaria si
# sus estadísticas básicas (media, varianza) permanecen constantes en el tiempo.

def check_test(data):

  result = adfuller(data)
  print('ADF Statistic: %f' % result[0])
  print('p-value: %f' % result[1])
  if result[1]>= 0.05:
    print(f"La variable {data.name} es no estacionario")
  else:
    print(f"La variable  {data.name} es estacionario")


check_test(df_meli['Close'])

# Como el valor de p-value es mayor a 0.05, se busca una transformación o modelo que obtenga datos estacionarios
# Vamos a chequear con el logaritmo del valor.

check_test(np.log(df_meli['Close']))

# El p-value sigue siendo mayor a 0.05 por lo que sigue sin ser estacionario.
# Probamos con la diferenciacion ente el valor de un dia y el valor anterior.

check_test(df_meli['Close'].diff()[1:])

# En este caso la variable se comporta de manera estacionaria, por lo que podremos trabajar con ella.

plt.plot(df_meli.iloc[1:].index, df_meli['Close'].diff()[1:])
plt.show()

# ------------ TRANSFORMACIÓN DE DATOS ---------------
# Se crean columnas con la diferencia a diferentes momentos, es decir haremos una columna que muestre la diferencia
# del valor de cierre de hoy con el de ayer, pero también haremos otra columna con la diferencia del precio de cierre
# al precio que tenía hace 2 días, hace 3 días, hace 4 días y hace 5 días.

df_meli['diff_close'] = df_meli['Close'].diff()
df_meli['diff_close_p1'] = df_meli["diff_close"].shift(1)
df_meli['diff_close_p2'] = df_meli["diff_close"].shift(2)
df_meli['diff_close_p3'] = df_meli["diff_close"].shift(3)
df_meli['diff_close_p4'] = df_meli["diff_close"].shift(4)
df_meli = df_meli.fillna(0)

# Creamos medias móviles o rolling windows para utilizarlas en el análisis

df_meli['rolling_windows_mean'] = df_meli['diff_close_p1'].rolling(window=2).mean()
df_meli['rolling_windows_mean7'] = df_meli['diff_close_p1'].rolling(window=7).mean()
df_meli['rolling_windows_max'] = df_meli['diff_close_p1'].rolling(window=2).max()
df_meli['rolling_windows_min'] = df_meli['diff_close_p1'].rolling(window=2).min()
df_meli.dropna(inplace=True)

df_meli.tail(20)

# ------------ ENTRENAMIENTO MODELO BÁSICO --------------

test_size = 12

train_df = df_meli[:-test_size]
test_df = df_meli[-test_size:]

target = "diff_close"

features = ['diff_close_p1', 'diff_close_p2', 'diff_close_p3','diff_close_p4', 'rolling_windows_mean','rolling_windows_mean7',
       'rolling_windows_max', 'rolling_windows_min']
x_train, y_train = train_df[features], train_df[target]
x_test, y_test = test_df[features], test_df[target]

# obtenemos una línea de base para comparar los resultados, para ello establecemos que el valor predicho para
# el próximo periodo será igual al anterior.

baseline_predict = y_test.shift(1).fillna(0)
baseline_rmse = mean_squared_error(baseline_predict,
                                   y_test, squared=False)
print(f"Baseline rmse: {baseline_rmse}")

# Entrenamos el random forest regresor

regr = RandomForestRegressor(n_estimators=1000,
                              max_depth=50,
                              random_state=0)

regr.fit(x_train, y_train)

predict_test = pd.Series(regr.predict(x_test),
                         y_test.index, name="diff_prediction")

# Vemos los resultados para los valores de test

rfr_rmse = mean_squared_error(predict_test,
                              y_test, squared=False)

print(f"RandomForestRegressor rmse: {rfr_rmse}")

# El error rmse se reduce a la mitad que la linea base.

test_df = pd.concat([test_df , predict_test],axis=1)
test_df['prediction'] = test_df['Close'].shift(1) + test_df['diff_prediction']

# Hacemos una visuializacion con Bokeh que se guardara en un archivo "time_series_plot.html"

output_notebook()
output_file('time_series_plot.html')
# Creamos el objeto ColumnDataSource

data = ColumnDataSource(test_df)

# Hacemos el gráfico de series de tiempo
plot = figure(x_axis_type='datetime',
              title='MELI.BA Close price comparation',
             x_axis_label = "Date",
              y_axis_label = "Prices in $", width=1000, height=300)
plot.line(x='Date', y = 'Close', source = data, color = 'green', legend_label="real Value")
plot.line(x='Date', y = 'prediction', source = data, color = 'blue', legend_label="prediction")

plot.legend.location = "top_right"

# Le agregamos un titulo a las leyendas
plot.legend.title = "Comparasion"
show(plot)