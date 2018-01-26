import data_utils, model_utils

def train(stock_no, start, end, x_window_size, y_window_size, 
          split_ratio, batch_size, layer_num, hidden_dim, nb_epoch):
  data, scaler_adjclose, scaler_volume = data_utils.read_data(stock_no, start, end)
  X, y = data_utils.window_transform_series(data, x_window_size=x_window_size, y_window_size=y_window_size)
  X_train, X_test, y_train, y_test = data_utils.tts(X, y, split_ratio)
  filepath = model_utils.model_path(stock_no)
  model_utils.train(X_train, y_train, X_test, y_test, x_window_size, y_window_size, 
                         split_ratio, batch_size, layer_num, hidden_dim, nb_epoch, filepath)
