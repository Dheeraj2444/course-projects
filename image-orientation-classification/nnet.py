'''
References:
http://neuralnetworksanddeeplearning.com/chap2.html
http://cs231n.github.io/neural-networks-1/
https://towardsdatascience.com/applied-deep-learning-part-1-artificial-neural-networks-d7834f67a4f6
https://www.youtube.com/watch?v=P02xWy63Q6U

Model Abstraction:
-In case of neural networks, the model comprises of 192 input layer neurons, 10 hidden layer neurons and 4 output layer neurons.
-The intial weights of the whole networks are assigned randomly along with bias neuron and thereafter the weights are changed with respect to error
-The idea of one hot encoding is used to get the correct output in the output layer as we need to multiclassify the images based on 4 different categories
-The model uses rectified linear units as an activation function at the hidden layer and softmax at the output layer
-Model takes the input as the normalized vector of images at each training and testing phase
-The Cross entropy cost function is used in order to calculate and error and its derivative at the output layer

Workflow:
-Prior to training and testing the raw image vector is preprocessed and normalized before feeding in the network
-In training phase:
1. The batch input is given to the network and the activations at each layers are recorded for each epoch
2. After the process of feedforward the error and its derivatives at each output neuron are calculated for each image in a particular batch
3. Thereafter, the error derivative is backpropogated to previous layers with respect to weights between neurons and deltas for each neuron for 
   respective layer are calculated and recorded.
4. After having all deltas for each layer finally the weights and biases are updated by taking an average of product of deltas of layer l and inputs of layer l-1
   for each layer
5. The learning rate decay is introduced to reduce the step size with the minimal change per epoch
6. Finally, the model comprising of weights and biases of hidden and output layer is written to the model file.

-In testing phase:
1. The importing of the model file is carried out and network is fitted with trained weights.
2. Then the test input vectors are simply passed throught the network as feedforward process.

Design Decisions:
-Normalization of image vector using standard normal formula z = (x -min(x))/(max(x) - min(x))
-Use of traditional gradient descent over stochastic gradient

Model Performance:
-The model gives an accuracy of 71.0498409332% when tested over test images data when using relu and softmax at hidden and output layer
'''
import sys
import numpy as np
import csv
class NeuralNetwork:
	def __init__(self, hidden_activation = 'relu', output_activation = 'softmax',learning_rate = 0.01, hidden_neurons = 10, output_neurons = 4):
		#initialize network
		self.learn_rate = learning_rate
		self.imageLabels = []
		self.vector = []
		self.classes = []
		self.hidden_activation = hidden_activation
		self.output_activation = output_activation
		self.hiddenLayer, self.hiddenBias = self.initialize_layer(hidden_neurons,192)
		self.outputLayer, self.outputBias = self.initialize_layer(output_neurons,hidden_neurons)

	def file_read(self, file_name):
		self.vector = []
		self.classes = []
		f = open(file_name)
		reader = csv.reader(f, delimiter = " ")
		for line in reader:
			self.imageLabels.append(line[0])
			self.classes.append(int(line[1]))
			vec = np.asarray(map(int, line[2:]))
			normalized_vector = (vec - min(vec))/ float(max(vec) - min(vec)) #normalization
			self.vector.append(normalized_vector)
			
	#initialize new layer with certain neuron count with respective inputs to neurons
	def initialize_layer(self, neuron_count, input_count):
		np.random.seed(1)
		return np.random.uniform(-0.5, 0.5, size = (neuron_count, input_count)).astype('float128') , np.ones(neuron_count)#row is nodes in layer and columns are inputs
	
	
	
	#activation function	
	
	def activation(self , x, name):
		if name == 'sigmoid':
			return 1 / (1 + np.exp(-x))
		elif name == 'tanh':
			return np.tanh(x)
		elif name == 'softmax':
			return np.exp(x) / np.sum(np.exp(x))
		elif name == 'relu':
			return x * (x > 0)
		else:
			return
	
	def activation_prime(self, x, name):
		if name == 'sigmoid':
			return x * (1 - x)
		elif name == 'tanh':
			return 1 - np.power(x, 2)
		elif name == 'softmax':
			return 1
		elif name == 'relu':
			return 1. * (x > 0)
		else:
			return
			
	def feedForward(self, X, weights, activation, bias):
		return self.activation(np.dot(X, weights.T) + bias, activation)
		
		
	def modactualouts(self, y):
		z = []
		for i in y:
			if(i == 0):
				z.append([1,0,0,0])
			elif(i == 90):
				z.append([0,1,0,0])
			elif(i == 180):
				z.append([0,0,1,0])
			elif(i == 270):
				z.append([0,0,0,1])
		return np.asarray(z)
	#train the network
	def train(self, file_name, model_file, epochs = 10000):
		self.file_read(file_name)
		X = np.asarray(self.vector)
		y = np.asarray(self.classes)
		mody = self.modactualouts(y)
		for e in range(0,epochs):
			hidden_adjust_wts = []
			output_adjust_wts = []
			err = []
			delta_h_layer = []
			delta_o_layer = []
			for counter, vector in enumerate(X):
				#feed forward
				hiddenLayerOut = self.feedForward(vector, self.hiddenLayer, self.hidden_activation, self.hiddenBias)
				outputLayerOut = self.feedForward(hiddenLayerOut, self.outputLayer, self.output_activation, self.outputBias)
				#error calculation
				err.append(0.5 * np.sum((mody[counter] - outputLayerOut)**2))
				error = (outputLayerOut - mody[counter])
				#backpropogation
				delta_output = error * self.activation_prime(outputLayerOut, self.output_activation)
				delta_hidden = self.activation_prime(hiddenLayerOut, self.hidden_activation) * np.dot(delta_output, self.outputLayer)
				
				delta_o_layer.append(delta_output)
				delta_h_layer.append(delta_hidden)
				
				hidden_adjust_wts.append(((delta_hidden * vector[np.newaxis].T) * self.learn_rate).T)
				output_adjust_wts.append(((delta_output * hiddenLayerOut[np.newaxis].T) * self.learn_rate).T)
			#weight adjustments
			len_train = len(X)
			h_wt_avg = np.sum(np.asarray(hidden_adjust_wts), axis = 0)/ len_train
			o_wt_avg = np.sum(np.asarray(output_adjust_wts), axis = 0)/ len_train
			self.hiddenLayer -= h_wt_avg
			self.outputLayer -= o_wt_avg
			#bias adjustments
			self.hiddenBias -= np.sum(np.asarray(delta_h_layer), axis = 0) / len_train
			self.outputBias -= np.sum(np.asarray(delta_o_layer), axis = 0) / len_train
			self.learn_rate = 0.01 / (1 + e / epochs)
		model = {'hiddenLayer' : self.hiddenLayer, 'outputLayer': self.outputLayer, 'hiddenBias' : self.hiddenBias, 'outputBias' : self.outputBias}
		np.save(model_file,model)

	def test(self, file_name, model_file, output_file = 'nnet_output.txt'):
		
		data = np.load(model_file)
		data = data[()]
		hiddenLayer, hiddenBias = data['hiddenLayer'], data['hiddenBias']
		outputLayer, outputBias = data['outputLayer'], data['outputBias']
		
		self.file_read(file_name)
		X = np.asarray(self.vector)
		y = np.asarray(self.classes)
		labels = self.imageLabels
		correct = 0
		output_nnet = []
		for c,i in enumerate(X):
			predicted = 0
			hidden_out = self.feedForward(i, hiddenLayer, self.hidden_activation, hiddenBias)
			output_out = self.feedForward(hidden_out, outputLayer, self.output_activation, outputBias)
			if output_out.argmax() == 0:
				predicted = 0
			elif output_out.argmax() == 1:
				predicted = 90
			elif output_out.argmax() == 2:
				predicted = 180
			elif output_out.argmax() == 3:
				predicted = 270
			output_nnet.append(str(labels[c])+" "+str(predicted))
			if predicted == y[c]:
				correct = correct + 1
		with open(output_file,'wt') as outputfile:
			outputfile.write('\n'.join(output_nnet))
		print "Testing Accuracy: ", (correct/float(len(y)))*100,"%"
