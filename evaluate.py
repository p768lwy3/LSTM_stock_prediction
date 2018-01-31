import data_utils, model_utils, tools
import matplotlib.pyplot as plt, numpy as np
from datetime import datetime

"""
def evaluate(stock_no, start, end, x_window_size, y_window_size, 
            split_ratio, layer_num, hidden_dim):
  
  model = model_utils.read_model(stock_no, x_window_size, y_window_size, layer_num, hidden_dim)

  data, scaler_adjclose, scaler_volume = data_utils.read_data(stock_no, start, end)
  X, y = data_utils.window_transform_series(data, x_window_size=x_window_size, y_window_size=y_window_size)
  X_train, X_test, y_train, y_test = data_utils.tts(X, y, split_ratio)

  print('  Start to evaluate the model...')
  train_error = model.evaluate(X_train, y_train, verbose=0)
  test_error = model.evaluate(X_test, y_test, verbose=0)
  print('  Train Loss: ', train_error[0], ', Test Loss: ', test_error[0])
  print('  Train Error: ', train_error[1], ', Test Error: ', test_error[1])
  
  # plot
  print('  Start to plot the result...')
  train_pred = model.predict(X_train)
  test_pred = model.predict(X_test)

  train_pred = tools.prediction_mean(train_pred)
  test_pred = tools.prediction_mean(test_pred)

  split_pt = x_window_size + len(train_pred)
  true_value = scaler_adjclose.inverse_transform(data[:, 0].reshape(-1, 1))
  train_pred = scaler_adjclose.inverse_transform(train_pred.reshape(-1, 1))
  test_pred = scaler_adjclose.inverse_transform(test_pred.reshape(-1, 1))

  plt.plot(true_value, color='k')
  plt.plot(np.arange(x_window_size, split_pt, 1), train_pred, color='b')
  plt.plot(np.arange(split_pt, split_pt+len(test_pred), 1), test_pred, color='r')
  plt.xlabel('day')
  plt.ylabel('(normalizer) price of stock price')
  plt.title('Using LSTM to predict stock price')
  plt.legend(['original series', 'training', 'testing'])
  plt.savefig('./pic/Evaluation_{0}.png'.format(str(datetime.now()).replace('-', '')))
  print('  The plot is saved to pic folder.')
"""

def evaluate(stock_no, start, end, x_window_size, y_window_size, 
            split_ratio, layer_num, hidden_dim):
  
  model = model_utils.read_model(stock_no, x_window_size, y_window_size, layer_num, hidden_dim)

  data, scaler_adjclose, scaler_volume = data_utils.read_data(stock_no, start, end)
  X, y = data_utils.window_transform_series(data, x_window_size=x_window_size, y_window_size=y_window_size)
  X_train, X_test, y_train, y_test = data_utils.tts(X, y, split_ratio)

  print('  Start to evaluate the model...')

  train_error = model.evaluate(X_train, y_train, verbose=0)
  test_error = model.evaluate(X_test, y_test, verbose=0)

  print('  Train Loss: ', train_error, ', Test Loss: ', test_error)
  #print('  Train Error: ', train_error[1], ', Test Error: ', test_error[1])
  
  # plot
  print('  Start to plot the result...')

  train_pred = model.predict(X_train)
  train_pred_mu = train_pred[:, 0:3]
  train_pred_sd = train_pred[:, 4]
  test_pred = model.predict(X_test)
  test_pred_mu = test_pred[:, 0:3]
  test_pred_sd = test_pred[:, 4]

  train_pred_mu = tools.prediction_mean(train_pred_mu)
  test_pred_mu = tools.prediction_mean(test_pred_mu)

  train_p_upper = train_pred_mu + train_pred_sd
  train_p_lower = train_pred_mu - train_pred_sd
  test_p_upper = test_pred_mu + test_pred_sd
  test_p_lower = test_pred_mu - test_pred_sd

  split_pt = x_window_size + len(train_pred)

  true_value = scaler_adjclose.inverse_transform(data[:, 0].reshape(-1, 1))
  train_pred_mu = scaler_adjclose.inverse_transform(train_pred_mu.reshape(-1, 1))
  train_p_upper = scaler_adjclose.inverse_transform(train_p_upper.reshape(-1, 1))
  train_p_lower = scaler_adjclose.inverse_transform(train_p_lower.reshape(-1, 1))
  test_pred_mu = scaler_adjclose.inverse_transform(test_pred_mu.reshape(-1, 1))
  test_p_upper = scaler_adjclose.inverse_transform(test_p_upper.reshape(-1, 1))
  test_p_lower = scaler_adjclose.inverse_transform(test_p_lower.reshape(-1, 1))

  plt.plot(true_value, color='k')
  plt.plot(np.arange(x_window_size, split_pt, 1), train_pred_mu, color='b')
  plt.fill_between(np.arange(x_window_size, split_pt, 1), 
                  train_p_upper.reshape(train_p_upper.shape[0]),
                  train_p_lower.reshape(train_p_lower.shape[0]), 
                  facecolor='b', alpha=0.5, linestyle='--')
  plt.plot(np.arange(split_pt, split_pt+len(test_pred), 1), test_pred_mu, color='r')
  plt.fill_between(np.arange(split_pt, split_pt+len(test_pred), 1), 
                  test_p_upper.reshape(test_p_upper.shape[0]), 
                  test_p_lower.reshape(test_p_lower.shape[0]), 
                  facecolor='r', alpha=0.5, linestyle='--')
  plt.xlabel('day')
  plt.ylabel('(normalizer) price of stock price')
  plt.title('Using LSTM to predict stock price')
  plt.legend(['original series', 'training', 'testing'])
  plt.savefig('./pic/Evaluation_{0}.png'.format(str(datetime.now()).replace('-', '')))
  print('  The plot is saved to pic folder.')
