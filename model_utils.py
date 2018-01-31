import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import BatchNormalization, Dense, Dropout, LeakyReLU, LSTM
from keras.optimizers import Adam

import matplotlib.pyplot as plt, os
from datetime import datetime

"""
def build_model(x_window_size, y_window_size, layer_num, hidden_dim, dr=0.01, lur=0.01, lr=0.01):

  model = Sequential()
  for i in range(layer_num):
    model.add(LSTM(units=hidden_dim, activation='relu', 
              return_sequences=True, input_shape=(x_window_size, 2)))
  model.add(Dropout(dr))
  model.add(LSTM(units=64, activation='relu'))
  model.add(Dropout(dr))
  model.add(Dense(units=y_window_size))
  model.add(BatchNormalization())
  model.add(LeakyReLU(lur))

  optimizer = Adam(lr=lr, epsilon=1e-04, decay=0.0)
  model.compile(loss='mean_squared_error', optimizer=optimizer, metrics=['accuracy'])

  #print(model.summary())
  return model
"""

def build_model(x_window_size, y_window_size, layer_num, hidden_dim, dr=0.01, lur=0.1, lr=0.01):
  # This is the BLSTM MDN model
  # c: the number of outputs we want to predict
  # m: the number of distribution we want to use in the mixture
  c = y_window_size
  m = 1

  def log_sum_exp(x, axis=None):
    """Log-sum-exp trick implementation"""
    x_max = K.max(x, axis=axis, keepdims=True)
    return K.log(K.sum(K.exp(x - x_max),
                      axis=axis, keepdims=True)) + x_max

  def mean_log_Gaussian_like(y_true, params):
    """Mean Log Gaussian Likelihood distribution"""
    """Note: The 'c' variable is obtained as global variable"""
    components = K.reshape(params, [-1, c+2, m])
    mu = components[:, :c, :]
    sigma = components[:, c, :]
    alpha = components[:, c+1, :]
    alpha = K.softmax(K.clip(alpha, 1e-8, 1.))

    exponent = K.log(alpha) - .5*float(c)*K.log(2*np.pi) \
              - float(c) * K.log(sigma) \
              - K.sum(K.expand_dims((y_true,2)-mu)**2, axis=1)/(2*(sigma)**2)

    log_gauss = log_sum_exp(exponent, axis=1)
    res = -K.mean(log_gauss)
    return res

  def mean_log_LaPlace_like(y_true, params):
    """Mean Log Laplace Likelihood distribution"""
    """Note: The 'c' variable is obtained as global variable"""
    components = K.reshape(params, [-1, c+2, m])
    mu = components[:, :c, :]
    sigma = components[:, c, :]
    alpha = components[:, c+1, :]
    alpha = K.softmax(K.clip(alpha, 1e-2, 1.))

    exponent = K.log(alpha) - float(c)*K.log(2*sigma) \
              - K.sum(K.abs(K.expand_dims(y_true,2)-mu), axis=1)/(sigma)

    log_gauss = log_sum_exp(exponent, axis=1)
    res = -K.mean(log_gauss)
    return res

  INPUTS = Input(shape=(x_window_size, 2))
  BLSTM1 = Bidirectional(LSTM(128, activation='tanh', return_sequences=True))(INPUTS)
  BLSTM2 = Bidirectional(LSTM(128, activation='tanh', return_sequences=True))(BLSTM1)
  BLSTM3 = Bidirectional(LSTM(128, activation='tanh', return_sequences=False))(BLSTM2)
  FC1 = Dense(128)(BLSTM3)
  FC2 = Dense(64)(FC1)
  LRU = LeakyReLU(lur)(FC2)
  FC_mus = Dense(c*m)(LRU)
  FC_sigmas = Dense(m, activation=K.exp, kernel_regularizer=l2(1e-3))(LRU)
  FC_alphas = Dense(m, activation='softmax')(LRU)
  OUTPUTS = concatenate([FC_mus, FC_sigmas, FC_alphas], axis=1)
  MODEL = Model(INPUTS, OUTPUTS)
  optimizer = Adam(lr=lr, epsilon=1e-04, decay=0.0)
  MODEL.compile(optimizer=optimizer, loss=mean_log_LaPlace_like)
  print(MODEL.summary())
  return MODEL

def model_path(stock_no):
  # model path format: 'model_stockno_date_weights_base.best.hdf5'
  todate = str(datetime.now().date()).replace('-', '')
  modelpath = './model/model_' + stock_no + '_' + todate + '_weights_base.best.hdf5'
  return modelpath

def train(X_train, y_train, X_test, y_test, 
          x_window_size, y_window_size, 
          split_ratio, batch_size, layer_num, 
          hidden_dim, nb_epoch, filepath):

  model = build_model(x_window_size, y_window_size, layer_num, hidden_dim)

  checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
  earlyStopping = EarlyStopping(monitor="val_loss", mode="min", patience=20)
  reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, epsilon=1e-4, min_lr=1e-5)
  callbacks = [checkpoint, earlyStopping, reduce_lr]

  print('  Start to fit the model......')
  history = model.fit(X_train, y_train, 
                  batch_size=batch_size, 
                  epochs=nb_epoch, 
                  callbacks=callbacks, 
                  validation_data=(X_test, y_test))

  plt.plot(history.history['acc'])
  plt.plot(history.history['val_acc'])
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.ylabel('acc / loss')
  plt.xlabel('epoch')
  plt.title('Accuracy / Loss of model')
  plt.legend(['train acc', 'test acc', 'train loss', 'test loss'], loc='upper right')
  plt.savefig('./pic/model_{0}.png'.format(str(datetime.now()).replace('-', '')))

def read_model(stock_no, x_window_size, y_window_size, layer_num, hidden_dim):
  model_folder = './model/'
  # find all the models which contain the stock number,
  for subdir, dirs, files in os.walk(model_folder):
    matching = [l for l in files if stock_no in l]
  # and then find the newest,
  date_list = []
  for l in matching:
    date_list.append(int(l[-31:-23]))
  maxvalue = max(date_list)
  # concatenate them into model name,
  model_name = './model/model_' + stock_no + '_' + str(maxvalue) + '_weights_base.best.hdf5'
  print('  The model name is: %s..' % model_name)
  print('  Start to load model...')
  model = build_model(x_window_size, y_window_size, layer_num, hidden_dim)
  model.load_weights(model_name)
  print('  Finish Loading.')
  return model

