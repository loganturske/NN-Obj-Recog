import random
import numpy as np
from convolution_layer import Convolution_Layer

class CNNetwork(object):

    def __init__(self, sizes, convolutional_training_set, n_clusters, patch_size, pool_size, seed=0):
        
        # initilize the network this includeds unsupervied learning of the convolutional kernels
        self.pool_size = pool_size
        print('training convolutional kernels')
        self.convolution = Convolution_Layer(convolutional_training_set, n_clusters, patch_size)
        sizes = [int(self.convolution.feed_forward(convolutional_training_set[0], pool=self.pool_size).shape[0])] + sizes
        self.num_layers = len(sizes)
        self.sizes = sizes
        
        # initalize Random weights for fft portion of network based on Gaussian distribution 
        np.random.seed(seed=seed)
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]

    def feedforward(self, a, convolve=True):
        '''
        this is the implemntation of the cnn it can also be truncated to only 
        peroform hte feed forward section for faster training
        '''
        if convolve:
            a = self.convolution.feed_forward(a, pool=self.pool_size)
        for bias, weight in zip(self.biases, self.weights):
            a = sigmoid(np.dot(weight, a)+bias)
        return a

    def SGD(self, training_data, epochs, mini_batch_size, eta,
            test_data=None, convolve=True, seed=0):
        '''
        implentation of stepenst gradiante decent method for  training of
        feed forward section of network
        '''
        
        # Training data is a list of tuples (inputs, desired outputs)

        # If there is test data to test after each epoch
        if test_data:
            n_test = sum(1 for _ in test_data) 
            # n_test = len(test_data)
        # bathc convolve the input training data unless its already been convolved
        if convolve:
            print('batch convolving training data')
            convolved_traning_data = [(self.convolution.feed_forward(image, pool=self.pool_size), key) for image, key in training_data]
            if test_data:
                print('batch convolving test data')
                convolved_test_data = [(self.convolution.feed_forward(image, pool=self.pool_size), key) for image, key in test_data]
        else:
            convolved_traning_data = training_data
            if test_data:
                convolved_test_data = test_data
        # n = len(training_data)
        self.convolved_traning_data = convolved_traning_data
        self.convolved_test_data = convolved_test_data
        n = sum(1 for _ in convolved_traning_data)
        random.seed(seed)
        for j in range(epochs):
            # Shuffle all of the data
            random.shuffle(convolved_traning_data)
            # Create batches to train on
            mini_batches = [
                convolved_traning_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            # Update all of the batches with using backpropogation
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)
            # Print out results
            if test_data:
                print(("Epoch {0}: {1} / {2}".format(j, self.evaluate(convolved_test_data, convolve=(not convolve)), n_test)))
            else:
                print(("Epoch {0} complete".format(j)))


    def update_mini_batch(self, mini_batch, eta):
        '''
        do ffbp algrothim on small mini batch of training data
        '''        
        # Get the biases and the weights for this batch
        nabla_b = [np.zeros(bias.shape) for bias in self.biases]
        nabla_w = [np.zeros(weight.shape) for weight in self.weights]
        # For each tuple in the batch
        for x, y in mini_batch:
            # Backpropogate and get the changes for the tuple you are on
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            # Change all the bias and weights based on the changes
            nabla_b = [nb+dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw+dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]
        # Formula we learned in class ufor updating the weights and biases
        self.weights = [w-(eta/len(mini_batch))*nw
                        for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b-(eta/len(mini_batch))*nb
                       for b, nb in zip(self.biases, nabla_b)]


    def backprop(self, x, y):
        '''
        implentation of back propgaiton for ff portion of network
        '''
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]
        # feedforward
        activation = x
        activations = [x]
        # Z vectors
        zs = []
        for bias, weight in zip(self.biases, self.weights):
            z = np.dot(weight, activation)+bias
            zs.append(z)
            activation = sigmoid(z)
            activations.append(activation)
        # backward pass
        delta = self.cost_derivative(activations[-1], y) * \
            sigmoid_prime(zs[-1])
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())
        # Go through all the layers
        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = sigmoid_prime(z)
            delta = np.dot(self.weights[-l+1].transpose(), delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].transpose())
        return (nabla_b, nabla_w)

    def evaluate(self, test_data, convolve=True):
        '''
        evalaute the performace of the network given a training set
        '''
        test_results = [(np.argmax(self.feedforward(x, convolve=convolve)), y)
                        for (x, y) in test_data]
        return sum(int(x == y) for (x, y) in test_results)

    def cost_derivative(self, output_activations, y):
        return (output_activations-y)

def sigmoid(z):
    return 1.0/(1.0+np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z)*(1-sigmoid(z))
