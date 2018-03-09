import numpy as np
import random
from kmc import k_means

def kernel(x, center, beta):
    return np.exp(-beta * np.linalg.norm(center - x)**2)

class RBFN(object):
    
    def __init__(self, input_size, proto_size):
        self.input_size = input_size
        self.proto_size = proto_size
        self.centers = [[] for e in range(proto_size)]
        self.beta = [0 for e in range(proto_size)]

        #initialize output weights matrix with gaussian values [-1,1]
        self.weights = np.random.randn(1, proto_size)
        self.bias = 0
        
    def gen_centers(self, training_data, stages):
        training_inputs = map(lambda x: x[0], training_data)
        self.centers, self.beta = k_means(training_inputs, self.proto_size, stages)
    
    def activations(self, x):
        return map(lambda c: kernel(x, c[0], c[1]), zip(self.centers, self.beta))
        
    def feedforward(self, x):
        return np.dot(self.weights, self.activations(x)) + self.bias
        
    def pinv_train(self, training_data):
        length = len(training_data)
        training_inputs = map(lambda x: x[0], training_data)
        training_outputs = map(lambda x: x[1], training_data)
        G = []
        
        for i in range(length):
            row = self.activations(training_inputs[i])
            row.append(1)
            G.append(row)
            
        G = np.array(G)
        result = np.dot(np.linalg.pinv(G), training_outputs)
        
        self.weights = np.array(result[:-1])
        self.weights.shape = (1,len(self.weights))
        self.bias = result[-1]
        
    def evaluate(self, training_data):
        length = len(training_data)
        number = 0.0
        
        for example in training_data:
            if round(self.feedforward(example[0])) == float(example[1][0]):
                number += 1
                
        print("Percent correct: " + str(100 * number / length) + "%")
        
        
    
    
    
