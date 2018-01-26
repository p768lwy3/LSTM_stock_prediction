import data_utils, model_utils, tools
import matplotlib.pyplot as plt, numpy as np
from datetime import datetime

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

