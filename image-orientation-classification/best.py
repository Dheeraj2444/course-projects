'''
The best accuracy for image classification is obtained by nnet
'''
from nnet import NeuralNetwork

class Best:
	def train(self, file_name, model_file, epochs):
		nnet = NeuralNetwork()
		nnet.train(file_name, model_file, epochs)
		
	def test(self, file_name, model_file, output_file = "best_output.txt"):
		nnet = NeuralNetwork()
		nnet.test(file_name, model_file, output_file)
