# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 16:09:00 2019

@author: 12345678
"""


import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt

# fill missing values with a value at the same time one day ago
def fill_missing(values):
	one_day = 60 * 24
	for row in range(values.shape[0]):
		for col in range(values.shape[1]):
			if np.isnan(values[row, col]):
				values[row, col] = values[row - one_day, col]
                
                
 # split a univariate dataset into train/test sets               
#def split_dataset(data):
#	# split into standard weeks
#	train, test = data[1:-328], data[-328:-6]
#	# restructure into windows of weekly data
#	train = np.array(np.split(train, len(train)/7))
#	test = np.array(np.split(test, len(test)/7))
#	return train, test
 
 


# evaluate one or more weekly forecasts against expected values
def evaluate_forecasts(actual, predicted):
	scores = list()
	# calculate an RMSE score for each day
	for i in range(actual.shape[1]):
		# calculate mse
		mse = mean_squared_error(actual[:, i], predicted[:, i])
		# calculate rmse
		rmse = sqrt(mse)
		# store
		scores.append(rmse)
	# calculate overall RMSE
	s = 0
	for row in range(actual.shape[0]):
		for col in range(actual.shape[1]):
			s += (actual[row, col] - predicted[row, col])**2
	score = sqrt(s / (actual.shape[0] * actual.shape[1]))
	return score, scores


def summarize_scores(name, score, scores):
	s_scores = ', '.join(['%.1f' % s for s in scores])
	print('%s: [%.3f] %s' % (name, score, s_scores))
    
    
    
    
    