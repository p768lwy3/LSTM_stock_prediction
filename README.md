# LSTM_stock_prediction
A python program to predict Hong Kong stock price by Recurrent Neural Network LSTM on Python 3 with keras.
## How to use: </br>
```
Train:
Command: python main.py --mode train (--params)
Return: best weight model in model folder and acc/loss plot in pic folder.
Evaluate:
Command: python main.py --mode evaluate (--params, should be same as above)
Return: Accuracy and Loss values in the terminal, and Price simulation plot in pic folder.
Run:
Command: python main.py --mode run (--params, should be same as above)
Return: csv file of prediction in the future y_windows days in output folder.

Input variable:
'--stock_no', type=int, default=1, 'stock Number on HKEX, e.g. for HK0001, input is 1.'
'--x_windows', type=int, default=30, 'sliding Windows for input.'
'--y_windows', type=int, default=3, 'sliding Windows for output.'
'--split_ratio', type=float, default=0.8, help='cross Validation Ratio for Learning.'
'--batch_size', type=int, default=128, help='learning Batch size.'
'--layer_num', type=int, default=1, 'number of Recurrent Layers.'
'--hidden_dim', type=int, default=32, 'number of hidden units for each Layers.'
'--nb_epoch', type=int, default=2, 'times of Iterations.'
'--mode', default='train',  'train: download data by yahoo finance, build and save the model.
                             run: Do the prediction, return .csv file which contain the output.
                             evaluate: Do the Evaluate of the model, return Loss and Accuracy, 
                             and plot a prediction curve'
```
## Disclaimer: </br>
This is just for learning purposes and reference only. I shall not be responsible or liable for any loss of any kind incurred as a result of the use of the program.
