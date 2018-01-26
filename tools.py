import numpy as np

def convert_int_to_string(stock_no):
  stock_no = str(stock_no)
  result = ''
  for i in range(4 - len(stock_no)):
    result += '0'
  result += stock_no
  return result

def prediction_mean(y):
  y_pred = []
  for i in range(len(y)):
    if i == 0:
      y_pred.append(y[i][0])
    elif i == 1:
      y_pred.append((y[i][0] + y[i-1][1])/2)
    else:
      y_pred.append((y[i][0] + y[i-1][1] + y[i-2][2])/3)
  return np.array(y_pred)
