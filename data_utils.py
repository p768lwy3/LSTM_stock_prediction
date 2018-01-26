import numpy as np, pandas as pd
from pandas_datareader import data as web
from sklearn.preprocessing import MinMaxScaler

def data_idx(stock_no, start, end):
  raw_data = pd.read_csv('./input/' + stock_no + '.csv', index_col=0).dropna()
  return raw_data.index

# save model of scaler_adjclose, with same name to convert back
def read_data(stock_no, start, end):
  try:
    print('  Try to find the file on local.')
    raw_data = pd.read_csv('./input/' + stock_no + '.csv')
  except:
    print('  The file doesn\'t exist on the local.')
    print('  Try to get the data from online.')
    for i in range(3):
      try:
        raw_data = web.DataReader(stock_no + '.HK', 'yahoo', start, end)
        break
      except:
        pass
    print('  Save the data to input file.')
    raw_data.to_csv('./input/' + stock_no + '.csv')
  print('  Finish to load the data.')
  
  raw_data = raw_data.dropna()

  adjclose = raw_data['Adj Close'].values.reshape(-1, 1)
  scaler_adjclose = MinMaxScaler().fit(adjclose)
  adjclose = scaler_adjclose.transform(adjclose)

  volume = raw_data['Volume'].values.reshape(-1, 1)
  scaler_volume = MinMaxScaler().fit(volume)
  volume = scaler_volume.transform(volume)

  data = np.append(adjclose, volume)
  data = data.reshape(len(data)//2, 2, order='F')
  
  return data, scaler_adjclose, scaler_volume

def window_transform_series(series, x_window_size, y_window_size):
  X = [series[i:i+x_window_size] for i in range(len(series)-x_window_size-y_window_size+1)]
  y = [series[:, 0][i+x_window_size:i+x_window_size+y_window_size] for i in range(len(X))]
  X = np.asarray(X)
  y = np.asarray(y)
  X.shape = (np.shape(X)[0:3])
  y.shape = (np.shape(y)[0:2])
  return X, y

def tts(X, y, split_ratio):
  spliter = int(len(y) * split_ratio)
  X_train, X_test, y_train, y_test = X[:spliter, :], X[spliter:, :], y[:spliter], y[spliter:]
  X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 2)
  X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 2)
  y_train = y_train.reshape(y_train.shape[0], y_train.shape[1])
  y_test = y_test.reshape(y_test.shape[0], y_test.shape[1])
  return X_train, X_test, y_train, y_test

def testing_values(series, x_window_size, y_window_size):
  # get the last y_window_size values to do prediction,
  X = []
  for i in range(y_window_size):
    if i == 0:
      X.append(series[-x_window_size:])
    else:
      X.append(series[-x_window_size-i:-i])
  X = np.array(X)
  X.shape = (np.shape(X)[0:3])
  return X
