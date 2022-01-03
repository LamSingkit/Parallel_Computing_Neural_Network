
# The code is modified base on 
# https://github.com/pradeepsinngh/Parallel-Deep-Learning-in-Python/blob/master/mnist-nn.py

import numpy as np
import math
import scipy.io as sio
import time

# Structure of the 2-layer neural network.
Input_layer_size = 400
Hidden_layer_size = 25
Output_layer_size = 10

def load_training_data(training_file='mnistdata.mat'):
    '''
    Load training data (mnistdata.mat) and return (inputs, labels).
    inputs: numpy array with size (5000, 400).
    labels: numpy array with size (5000, 1).
    '''
    training_data = sio.loadmat(training_file)
    inputs = training_data['X']
    labels = training_data['y']
    return (inputs, labels)

def rand_init_weights(size_in, size_out):
    epsilon_init = 0.12
    return np.random.rand(size_out, 1 + size_in) * 2 * epsilon_init - epsilon_init


def sigmoid(z):
    return 1.0 / (1 + pow(math.e, -z))


def sigmoid_gradient(z):
    return sigmoid(z) * (1 - sigmoid(z))


def cost_function(theta1, theta2, input_layer_size, hidden_layer_size, output_layer_size, inputs, labels, regular=0):
    '''
    Note: theta1, theta2, inputs, labels are numpy arrays:

        theta1: (300, 401)
        theta2: (10, 301)
        inputs: (5000, 400)
        labels: (5000, 1)
    '''
    # construct neural network
    input_layer = np.insert(inputs, 0, 1, axis=1)   

    hidden_layer = np.dot(input_layer, np.transpose(theta1))
    hidden_layer = sigmoid(hidden_layer)
    hidden_layer = np.insert(hidden_layer, 0, 1, axis=1) 

    output_layer = np.dot(hidden_layer, np.transpose(theta2))  
    output_layer = sigmoid(output_layer)
    
    # forward propagation: calculate cost
    cost = 0.0
    for training_index in range(len(inputs)):
        
        outputs = [0] * output_layer_size
        outputs[labels[training_index][0]-1] = 1

        for k in range(output_layer_size):
            cost += -outputs[k] * math.log(output_layer[training_index][k]) - (1 - outputs[k]) * math.log(1 - output_layer[training_index][k])
    cost /= len(inputs)

    # back propagation: calculate gradiants
    theta1_grad = np.zeros_like(theta1)  
    theta2_grad = np.zeros_like(theta2)  
    for index in range(len(inputs)):
        outputs = np.zeros((1, output_layer_size)) 
        outputs[0][labels[index][0]-1] = 1

        # calculate delta3
        delta3 = (output_layer[index] - outputs).T  

        # calculate delta2
        z2 = np.dot(theta1, input_layer[index:index+1].T)  
        z2 = np.insert(z2, 0, 1, axis=0) 
        delta2 = np.multiply(
            np.dot(theta2.T, delta3),  
            sigmoid_gradient(z2)      
        )
        delta2 = delta2[1:]  

        theta1_grad += np.dot(delta2, input_layer[index:index+1])
        theta2_grad += np.dot(delta3, hidden_layer[index:index+1])
    theta1_grad /= len(inputs)
    theta2_grad /= len(inputs)

    return cost, (theta1_grad, theta2_grad)


def gradient_descent(inputs, labels, learning_rate, iteration):
    '''
    return cost and trained model (weights).
    '''
    rand_theta1 = rand_init_weights(Input_layer_size, Hidden_layer_size)
    rand_theta2 = rand_init_weights(Hidden_layer_size, Output_layer_size)
    theta1 = rand_theta1
    theta2 = rand_theta2
    cost = 0.0
    for i in range(iteration):
        cost, (theta1_grad, theta2_grad) = cost_function(theta1, theta2,
            Input_layer_size, Hidden_layer_size, Output_layer_size,
            inputs, labels, regular=0)
        theta1 -= learning_rate * theta1_grad
        theta2 -= learning_rate * theta2_grad
        print('Iteration {0} (learning rate {2}, iteration {3}), cost: {1}'.format(i+1, cost, learning_rate, iteration))
    return cost, (theta1, theta2)


def train(inputs, labels, learning_rate, iteration):
    cost, model = gradient_descent(inputs, labels, learning_rate, iteration)
    return model

if __name__ == '__main__':
    time_start = time.time()

    # Change datasize for experiment 
    datasize_multiplication = 1

    inputs, labels = load_training_data()
    inputs = np.repeat(inputs,datasize_multiplication, axis=0)
    labels = np.repeat(labels,datasize_multiplication, axis=0)
    model = train(inputs, labels, learning_rate = 0.1, iteration = 10)
    time_end = time.time()
    print('\n Total runtime is: {}'.format(time_end - time_start))
