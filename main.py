# import
import argparse, numpy as np,  matplotlib.pyplot as plt
import evaluate, run, train, tools
from datetime import datetime, strptime
np.random.seed(12345)

# define console common,
ap = argparse.ArgumentParser()
ap.add_argument('--stock_no', type=int, default=1, help='stock Number on HKEX, e.g. for HK0001, input is 1.')
ap.add_argument('--start_date', type=int, default=20170101, help='start date of learning, format: YYYYMMDD.')
ap.add_argument('--end_date', type=int, default=20171231, help='end date of learning, format: YYYYMMDD.')
ap.add_argument('--x_windows', type=int, default=30, help='sliding Windows for input.')
ap.add_argument('--y_windows', type=int, default=3, help='sliding Windows for output.')
ap.add_argument('--split_ratio', type=float, default=0.8, help='cross Validation Ratio for Learning.')
ap.add_argument('--batch_size', type=int, default=128, help='learning Batch size.')
ap.add_argument('--layer_num', type=int, default=1, help='number of Recurrent Layers.')
ap.add_argument('--hidden_dim', type=int, default=32, help='number of hidden units for each Layers.')
ap.add_argument('--nb_epoch', type=int, default=2, help='times of Iterations.')
mode_help = ('train: download data by yahoo finance, build and save the model.\n' + 
            'run: Do the prediction, return .csv file which contain the output.\n' + 
            'evaluate: Do the Evaluate of the model, return Loss and Accuracy, and plot a prediction curve')
ap.add_argument('--mode', default='train', help=mode_help)
args = vars(ap.parse_args())

stock_no = args['stock_no'] # define a fn to convert to strings,
start = args['start_date']
end = args['end_date']
x_window_size = args['x_windows']
y_window_size = args['y_windows']
split_ratio = args['split_ratio']
batch_size = args['batch_size']
layer_num = args['layer_num']
hidden_dim = args['hidden_dim']
nb_epoch = args['nb_epoch']
MODE = args['mode']

# this should be added to input
stock_no = tools.convert_int_to_string(stock_no)
start = datetime.strptime(str(start), '%Y%d%m')
end = datetime.strptime(str(end), '%Y%d%m')

if __name__ == '__main__':
  if MODE == 'train':
    train.train(stock_no, start, end, x_window_size, y_window_size, 
               split_ratio, batch_size, layer_num, hidden_dim, nb_epoch)

  elif MODE == 'evaluate':
    evaluate.evaluate(stock_no, start, end, x_window_size, y_window_size, 
                     split_ratio, layer_num, hidden_dim)

  elif MODE == 'run':
    run.run(stock_no, start, end, x_window_size, y_window_size, 
           split_ratio, layer_num, hidden_dim)
