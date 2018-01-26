import keras
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import BatchNormalization, Dense, Dropout, LeakyReLU, LSTM
from keras.optimizers import Adam

import matplotlib.pyplot as plt, os
from datetime import datetime

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

