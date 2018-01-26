import data_utils, model_utils, tools
import numpy as np, pandas as pd
from datetime import timedelta

def outputpath(stock_no, pred_time):
  return './output/Prediction_{0}_{1}.csv'.format(stock_no, str(pred_time.date()).replace('-', ''))

def run(stock_no, start, end, x_window_size, y_window_size, 
            split_ratio, layer_num, hidden_dim):

  model = model_utils.read_model(stock_no, x_window_size, y_window_size, layer_num, hidden_dim)

  data, scaler_adjclose, scaler_volume = data_utils.read_data(stock_no, start, end)
  X = data_utils.testing_values(data, x_window_size, y_window_size)
  
  y_pred = model.predict(X.reshape(X.shape[0], X.shape[1], 2))
  y_mean = tools.prediction_mean(y_pred).reshape(-1, 1)
  y_pred = scaler_adjclose.inverse_transform(y_pred)
  y_mean = scaler_adjclose.inverse_transform(y_mean)

  data_idx = data_utils.data_idx(stock_no, start, end)[-y_window_size:]
  y_pred = np.concatenate((y_pred, y_mean), axis=1)
  y_pred = pd.DataFrame(y_pred, index=data_idx, columns=['t+1', 't+2', 't+3', 'mean'])

  filepath = outputpath(stock_no, end)
  y_pred.to_csv(filepath)
