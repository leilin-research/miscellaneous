# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 15:49:51 2019

@author: 12345678
"""

import pandas as pd
import numpy as np
from models import *
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import LSTM
from keras.layers import RepeatVector
from keras.layers import TimeDistributed
from keras.layers import Dropout, Activation
from keras import backend as K
from keras.layers.core import Lambda
from sklearn.preprocessing import MinMaxScaler

import csv

def PermaDropout(rate):
    return Lambda(lambda x: K.dropout(x, level=rate))

# read training and test dataset
#dataset = pd.read_csv('household_power_consumption_days.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
#dataset = pd.read_csv('ccc.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
with open ('ccc_log.csv','r') as csvfile:
    reader = csv.reader(csvfile)
    rows = [row for row in reader]

dataset = pd.read_csv('ccc_log.csv', header=0, infer_datetime_format=True, parse_dates=['datetime'], index_col=['datetime'])
#dataset = np.array(rows)    
# normalize dataset

scaler = MinMaxScaler()
dataset_norm = scaler.fit_transform(dataset.values)


# split into standard weeks
train, test = dataset_norm[0:630], dataset_norm[631:701]
    
	# restructure into windows of weekly data [weeks,days of a week,dimension]
train = np.array(np.split(train, len(train)/7)) #[159,7,8]
test = np.array(np.split(test, len(test)/7)) #[46,7,8]
    
    # validate train data
print(train.shape)
print(train[0, 0, 0], train[-1, -1, 0])
# validate test
print(test.shape)
print(test[0, 0, 0], test[-1, -1, 0])
    
##################################################################################
###### Encoder-decoder LSTM Model With Univariate Input and Vector Output ########
##################################################################################

#The training data is provided in standard weeks with eight variables, specifically in the shape [159, 7, 8]. 
#The first step is to flatten the data so that we have eight time series sequences.

# flatten data
data = train.reshape((train.shape[0]*train.shape[1], train.shape[2]))

print ('data is', data)
#
#Input, Output
#[d01, d02, d03, d04, d05, d06, d07], [d08, d09, d10, d11, d12, d13, d14]
#[d02, d03, d04, d05, d06, d07, d08], [d09, d10, d11, d12, d13, d14, d15]
#...

n_input = 14  # lagged time steps (in this case, it is days)
n_out = 7

# convert history into inputs and outputs
X, y = list(), list()
in_start = 0
	# step over the entire history one time step at a time
for _ in range(len(data)):
		# define the end of the input sequence
        in_end = in_start + n_input
        out_end = in_end + n_out
		# ensure we have enough data for this instance
        if out_end < len(data):
            ####### for univariate input  ######
            #x_input = data[in_start:in_end, 3]
            #x_input = x_input.reshape((len(x_input), 1))  
            #X.append(x_input)
            ####### for multivariate input  ######
            X.append(data[in_start:in_end,:])
            y.append(data[in_end:out_end,3]) # 3 is the total demand
            # move along one time step
            in_start += 1
    
# prepare data
train_x = np.array(X)
train_y = np.array(y)



## Train the model

# define parameters
verbose = 0 #0
epochs = 80 #20训练周期数
batch_size = 42 #16 训练批次大小（每个批次包含样本数）

n_timesteps, n_features, n_outputs = train_x.shape[1], train_x.shape[2], train_y.shape[1]
# reshape output into [samples, timesteps, features]
train_y = train_y.reshape((train_y.shape[0], train_y.shape[1], 1))

# define model
model = Sequential()
model.add(LSTM(360, activation='linear', input_shape=(n_timesteps, n_features))) #encoder#200
#linear,softmax,softplus,softsign,relu,tanh,sigmoid,hard_sigmoid
model.add(RepeatVector(n_outputs))
#model.add(Dense(100, activation='relu'))
#model.add(Dense(n_outputs))
model.add(LSTM(360, activation='relu', return_sequences=True)) #decoder #200编码空间的潜在维数
model.add(PermaDropout(0.2))
model.add(TimeDistributed(Dense(36, activation='linear'))) #for each time step build a hidden layer, the output is 36
#model.add(PermaDropout(0.2))
model.add(TimeDistributed(Dense(1))) # for each time step build a hidden layer, the output is 1 now
#model.add(PermaDropout(0.2))
model.add(Activation('sigmoid'))
model.compile(loss='mse', optimizer='nadam') #loss= mse, optimizer=adam, 
#'SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam'
#mse，mae，mape，msle，squared_hinge，hinge，binary_crossentropy， categorical_crossentropy
# fit network
model.fit(train_x, train_y, epochs=epochs, batch_size=batch_size, verbose=verbose)

 # make a forecast
all_preds = []
RMSE = []
total = 1000

for i in range(total):
 # history is a list of weekly data
	history = [x for x in train]  # list
	 # walk-forward validation over each week
	predictions = list()
	for i in range(len(test)):
			# predict the week
			#yhat_sequence = forecast(model, history, n_input)
		# flatten data
		data = np.array(history)
		data = data.reshape((data.shape[0]*data.shape[1], data.shape[2]))
		# retrieve last observations for input data
		#input_x = data[-n_input:, 3] # univariate input
		input_x = data[-n_input:, :] # multivariate input
		# reshape into [1, n_input, 1],for univariate input
		#input_x = input_x.reshape((1, len(input_x), 1))  #samples*timesteps*features
		# reshape into [1, n_input, n_features]
		input_x = input_x.reshape((1, input_x.shape[0], input_x.shape[1]))  #samples*timesteps*features
		# forecast the next week
		yhat = model.predict(input_x, verbose=0)   # samples*timesteps
		# we only want the vector forecast
		yhat_seq = yhat[0]
		# store the predictions
		predictions.append(yhat_seq)
		# get real observation and add to history for predicting the next week
		history.append(test[i, :])
	    
	    
	    
	# evaluate predictions days for each week
	predictions = np.array(predictions)  # samples(test weeks)*timesteps*n_features
	
	score, scores = evaluate_forecasts(test[:, :, 3], predictions) #3 预测total

	RMSE.append(score)

	#irenverse scale all predictions and obs
	pred_seq = predictions.reshape(predictions.shape[0]*predictions.shape[1],predictions.shape[2])
	
	all_preds.append(pred_seq)

	#for i in range(len(test)):
	
	#print (pred_seq.shape)

#print (len(all_preds))

all_preds = np.array(all_preds)
aver = np.mean(RMSE)
print (aver)
#print (np.mean(all_preds, axis = 0))
#print (np.std(all_preds, axis = 0))

#test = scaler.inverse_transform(test)
#predictions = scaler.inverse_transform(predictions)
pre_mean = np.mean(all_preds, axis = 0)
pre_std = np.std(all_preds, axis = 0)

np.savetxt("mean"+str(aver)+".csv", pre_mean, delimiter=",")
np.savetxt("std"+str(aver)+".csv", pre_std, delimiter=",")
 
 # summarize scores
#summarize_scores('lstm', score, scores)
# plot scores
#x = np.arange(7)
#days = ['sun', 'mon', 'tue', 'wed', 'thr', 'fri', 'sat']
#plt.plot(x, scores, marker='o')
#plt.ylabel('mape')
#plt.show()
 
 


obs_seq = test[:,:,3].reshape(test.shape[0]*test.shape[1],1)#3 预测total

x = np.arange(len(obs_seq))
plt.plot(x, pred_seq, label = 'one prediction')
plt.plot(x, obs_seq, label = 'real obs')
plt.plot(x, pre_mean, label = 'mean')
plt.legend()
plt.show()































































