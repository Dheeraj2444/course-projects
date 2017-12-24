import sys
import csv 
import math
import heapq
import numpy as np
from nearest import Knn
from nnet import NeuralNetwork
from best import Best
import adaboost

train_or_test, file_name, model_file, model = sys.argv[1:]

if train_or_test == 'train':
	if model == 'nearest':
		knn = Knn()
		knn.train(file_name, model_file)
		
	elif model == 'adaboost':
		trial = adaboost.Adaboost(file_name, None)
		trial.training = file_name
		train_pixels = trial.prepare_data(trial.training)[0]
		weights = trial.train(train_pixels)
		weight_values = weights.values()
		for i in weights:
			weights[i] += 10
		
		with open(model_file, 'w') as myfile:
			for i in weights:
				if i == trial.learner1:
					myfile.write('%s %.9f\n' % ('learner1', weights[i]))
				if i == trial.learner2:
					myfile.write('%s %.9f\n' % ('learner2', weights[i]))
	elif model == 'nnet':
		nnet = NeuralNetwork()
		nnet.train(file_name, model_file, epochs = 10000) 
	elif model == 'best':
		best = Best()
		best.train(file_name, model_file, epochs = 10000) 
	else:
		print 'Specified model not found!!'
else:
	if model == 'nearest' or model == 'nnet' or model == 'best':
		if model_file.endswith('.txt'):
			model_file = model_file+'.npy'

	if model == 'nearest':		
		knn = Knn()
		knn.test(file_name, model_file)
	elif model == 'nnet':
		nnet = NeuralNetwork()
		nnet.test(file_name, model_file)
	elif model == 'best':
		best = Best()
		best.test(file_name, model_file)
	elif model == 'adaboost':
		trial = adaboost.Adaboost(None, file_name)
		trial.testing = file_name
		test_rotation = trial.prepare_data(trial.testing)[1]
		learners = [trial.learner1, trial.learner2]
		print "Accuracy on test set: ", trial.get_accuarcy(test_rotation, trial.test(file_name, model_file)), " %"
	else:
		print 'Specified model not found!!'
