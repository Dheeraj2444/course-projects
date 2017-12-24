'''
Model Abstraction:
-The k-nearest neighbours classifier simply takes the raw pixel values as an input and there is no typical training i.e parameter adjustments involved .
-The comparision of image vectors is made while testing phase based on eucledian distance and the correct class is predicted based on the minimum distance
image vectors with the maximum number of class count for respective images.

Workflow:
-Program simply takes the input as training file of 192 dimensional images in the training phase and simply stores the vectors and classes for the same in
the model output file
-While in the testing phase the model file as supplied is used to measure the distances between the test dataset and each vector in the train(i.e model) in order
to come up with the best neighbour using the eucledian distance for a particular testing image.
-The model outputs the predicted values in output file after testing phase

Design Decisions:
-Eucledian distance formula has been modified which is equivalent to noramal sum od distances without squareroot in order to make faster computations.
-k assumed to be 1

Model Performance:
-The model gives an accuracy of 67.9745493107 % when tested over test images data
'''
import sys
import csv 
import numpy as np
class Knn:
	def __init__(self):
		self.vector = []
		self.classes = []
		self.labels = []
		
	def file_read(self, file_name):
		f = open(file_name)
		reader = csv.reader(f, delimiter = " ")
		for line in reader:
			self.labels.append(line[0])
			self.classes.append(int(line[1]))
			self.vector.append(np.asarray(map(int, line[2:])))
	
	def train(self, file_name, model_file):
		self.file_read(file_name)
		data = {'vectors' : np.asarray(self.vector), 'classes': self.classes }
		np.save(model_file, data)
	
	def getMaxNoClass(self, neighbours):
		neighbour_count = {}
		for i in neighbours:
			if neighbour_count.has_key(i[1]):
				neighbour_count[i[1]] += 1
			else:
				neighbour_count[i[1]] = 1
		return max(zip(neighbour_count.keys(), neighbour_count.values()))
		
	def test(self, test_file, model_file, k = 1):
		train_data = np.load(model_file)
		train_data = train_data[()]
		train_vec = train_data['vectors']
		train_class = train_data['classes']
		self.file_read(test_file)
		self.vector = np.asarray(self.vector)
		output_nearest = []
		correct = 0
		for c,vector in enumerate(self.vector):
			dist = self.getEucledianDist(train_vec, vector)
			neighbours = sorted(zip(dist, train_class), key = lambda tup : tup[0] )
			neighbours = neighbours[:k]
			max_class_count = self.getMaxNoClass(neighbours)[0]
			output_nearest.append(str(self.labels[c])+" "+str(max_class_count))
			if max_class_count == self.classes[c]:
				correct += 1
			#print self.labels[c]," predicted:",max_class_count," actual:", self.classes[c]
		
		print "Testing Accuracy: ", (correct/float(len(self.vector)))*100,"%"	
		with open('nearest_output.txt', 'wt') as nearest_file:
			nearest_file.write("\n".join(output_nearest))
		
	def getEucledianDist(self, vec1, vec2):
		return np.sum(abs(vec2-vec1), axis = 1)

