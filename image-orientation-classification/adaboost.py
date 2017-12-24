#!/usr/bin/env python
'''
Model Abstraction: - For the adaboost, we have defined two decision stumps based on 
                     pixel values. In the first stump, we compare blue pixel values
                     (corresponding to sky) on the top row, bottom row, left column, 
                     and right column i.e. if top row has max sum for blue pixels 
                     then it predicts as 0, if bottom row has max value then 180, 
                     if right column has max then 90, and if left column is max then 270.
                     Our second decision stumps compares red pixel values (corresponding
                     to land). If max is at top row, it predicts 180, if bottom row
					 is max, it predicts 0, if right is max then 270, and if 
					 left is max then 90
				   - Initial weight of each observation is assumed as 1/n, where
				     n is the total number of observations
				   - Observation's Weight Update Rule: We have used the weight Update
				     rule as per given in the book: w[i]*(error/(1-error)
				   - Decision Stump's Weight Calculation Rule: For the calculation of 
					 weights of decision stumps, we have used the formulae provided in 
					 the book: log((1-error)/error)

Program Workflow:  - Our program first reads the train file and store all pixel 
					 values corresponding to a image into a list. Then Adaboost trains 
					 on all images in the train file. It first compares the 
					 performance of each classifier on the train set and picks the
					 best perfoming classifier first and starts training. Starting with 1/n 
					 initial weight of each observation, everytime if the classifier 
					 makes an error, it adds value of that observation's weight 
					 to the error (initialized with 0) and keeps on updating the error. 
					 Once the selected classifier makes predictions on overall training 
					 data, weights of all correctly classified observations are updated 
					 as per the weight update rule given above and then weights of all 
					 observations are normalized to maintain a total sum of 1. After 
					 that, the classifier is assigned its weight as per the rule 
					 mentioned above. Once done, the next best performing classifier 
					 is chosen and all above steps repeated again for this classifier. 
					 Once every classifier is has done classifications, each has their 
					 corresponding weight, which we write in the text file and use it 
					 later to make weighted prediction on the test set. To calculate
					 the overall accuracy, it compares the predicted label with actual 
					 label and calculates the accuracy as:
					 (total correctly predicted observations/total observations) * 100
				   - learner1: Decision Stump 1 based on blue pixels
				   - learner2: Decision Stump 2 based on red pixels
				   - comp_learner: Compares the performance of classifiers and returns 
				     the best performing
				   - train: Trains adaboost model and return final weights of each
				     classifier in the form {L1: W1, L2: W2, L3: W3, .....}
				   - As asked in the problem statement, after testing, all test images 
				     are labeled with their corresponding predicted orientation and 
				     written to the file 'adaboost_output.txt' in the format: 
				     test/124567.jpg 180
					 test/8234732.jpg 0

Assumptions/Design Decisions: 
                   - We have made a linear transformation (added a constant 10) to each 
                     classifier's weights in order to make them positive 

Model Performance: - The model results in 70.8377518558'%' accuracy when tested
                     over the test image set 
'''

import math

#adaboost class
class Adaboost():
	def __init__(self, training_data=None, testing_data=None):
		self.training = training_data
		self.testing = testing_data
		self.learners = [self.learner1, self.learner2]

	#function to read data file
	def prepare_data(self, filename):
		self.pixels = []
		self.rotation = []
		self.ids = []
		lines = open(filename).readlines()
		for line in lines:
			line = line.split(" ")
			pic_id, label, pixel = line[0], int(line[1]), map(lambda x: int(x), line[2:])
			self.ids.append(pic_id)
			self.rotation.append(label)
			self.pixels.append(pixel)
		return self.pixels, self.rotation, self.ids

	#blue
	def learner1(self, train):
		out = []
		for data in train:
			top, right, bottom, left = sum(data[2:24:3]), sum(data[23:192:24]), \
			sum(data[170:192:3]), sum(data[2:192:24])
			if max([top, right, bottom, left]) == top:
				out.append(0)
			elif max([top, right, bottom, left]) == bottom:
				out.append(180)
			elif max([top, right, bottom, left]) == right:
				out.append(90)
			elif max([top, right, bottom, left]) == left:
				out.append(270)
		return out

	#brown
	def learner2(self, train):
		out = []
		for data in train:
			top, right, bottom, left = sum(data[0:23:3]), sum(data[21:192:24]), \
			sum(data[168:192:3]), sum(data[0:192:24])
			if max([top, right, bottom, left]) == top:
				out.append(180)
			elif max([top, right, bottom, left]) == bottom:
				out.append(0)
			elif max([top, right, bottom, left]) == right:
				out.append(270)
			elif max([top, right, bottom, left]) == left:
				out.append(90)
		return out

	# function to compare the performance of each classifier on the train set
	def comp_learner(self, train, learners):
		self.train_rotation = self.prepare_data(self.training)[1]
		out = {i:0 for i in learners}
		for learner in learners:
			correct = 0
			pred = learner(train)
			for i in range(len(train)):
				if pred[i] == self.train_rotation[i]:
					correct += self.obs_weights[i]
			out[learner] = correct
		return max(out, key=out.get)

	#training adaboost
	def train(self, train):
		self.obs_weights = [1/float(len(train))]*len(train)
		self.learners = [self.learner1, self.learner2]
		hypothesis_wt = {}
		test = {}
		while len(self.learners) > 0:
			learner = self.comp_learner(train, self.learners)
			error = 0
			pred1 = learner(train)
			for i in range(len(train)):
				if pred1[i] != self.train_rotation[i]:
					error += self.obs_weights[i]
			final_error = error
			for i in range(len(train)):
				if pred1[i] == self.train_rotation[i]:
					self.obs_weights[i] = self.obs_weights[i]*(final_error/(1-final_error))
			self.obs_weights = [float(i)/sum(self.obs_weights) for i in self.obs_weights]
			hypothesis_wt[learner] = math.log((1-final_error)/final_error)
			self.learners.remove(learner)
		return hypothesis_wt

	#function to predict for test file; paramters: test-file name
	def test(self, test_file, model_file):
		pixels, rotation, ids = self.prepare_data(test_file)
		new = {}
		lines = open(model_file).readlines() #reading weights from text file
		for line in lines:
			line = line.split(" ")
			new[line[0]] = float(line[1][:-1])

		new_weight = {}
		for i in new:
			if i == 'learner1':
				new_weight[self.learner1] = new[i]
			if i == 'learner2':
				new_weight[self.learner2] = new[i]

		out = {i: [] for i in range(len(pixels))}
		for learner in self.learners:
			pred = learner(pixels)
			for i, j in enumerate(pred):
				out[i].append((j, new_weight[learner]))
		for key, value in out.items():
			test = {}
			for p,q in value:
				if p in test:
					test[p] += q
				else:
					test[p] = q
			out[key] = max(test, key = test.get)
		with open("adaboost_output.txt", 'w') as output_file:
			for i,j in enumerate(ids):
				output_file.write('%s %d\n' % (j, out[i]))
		return out

	#Calculates overall accuracy
	def get_accuarcy(self, actual, predicted):
		count = 0
		for i in range(len(actual)):
			if actual[i] == predicted[i]:
				count += 1
		return float(count)/len(actual)*100
