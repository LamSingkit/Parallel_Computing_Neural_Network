# The code is parallelized base on sequential_NN.py

# The code is running on Intel(R) Core(TM) i7-6700K CPU @ 4.00GHz
# with 4 cores and maximum 8 logical processors

import numpy as np
import math
import scipy.io as sio
import time
import functools
from mpi4py import MPI


# Structure of the 2-layer neural network.
Input_layer_size = 400
Hidden_layer_size = 25
Output_layer_size = 10

def load_training_data(training_file = 'mnistdata.mat'):
    '''
    Load training data (mnistdata.mat) and return (inputs, labels).
    inputs: numpy array with size (5000, 400).
    labels: numpy array with size (5000, 1).
    '''
    training_data = sio.loadmat(training_file)
    inputs = training_data['X'].astype('f8')
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
        delta2 = delta2[1:]  # (25,1)

        # calculate gradients of theta1 and theta2
        theta1_grad += np.dot(delta2, input_layer[index:index+1])
        theta2_grad += np.dot(delta3, hidden_layer[index:index+1])
    theta1_grad /= len(inputs)
    theta2_grad /= len(inputs)
    
    return cost, (theta1_grad, theta2_grad)

def gradient_descent_parallelize(X, y, learning_rate, iteration):
    '''
    return cost and trained model (weights).
    '''

    # Rank 0 create theta1, theta2, X_sliced and y_slice, broadcast theta to other ranks
    X_sliced = None
    y_sliced = None
    if comm.rank == 0:
        theta1 = rand_init_weights(Input_layer_size, Hidden_layer_size)
        theta2 = rand_init_weights(Hidden_layer_size, Output_layer_size)
        X_sliced = np.asarray(np.split(X,comm.size))
        y_sliced = np.asarray(np.split(y,comm.size))
    else:
        theta1 = np.zeros((Hidden_layer_size, 1 + Input_layer_size))
        theta2 = np.zeros((Output_layer_size, 1 + Hidden_layer_size))
    comm.Barrier()
    comm.Bcast(theta1, root=0)
    comm.Bcast(theta2, root=0)

    # Scatter data and labels
    X_buffer = np.zeros((int(inputs.shape[0]/comm.size),inputs.shape[1]))
    y_buffer = np.zeros((int(labels.shape[0]/comm.size),labels.shape[1]), dtype='uint8')
    
    comm.Barrier()
    comm.Scatterv(X_sliced, X_buffer, root=0)
    comm.Scatter(y_sliced, y_buffer, root=0)
    comm.Barrier()

    cost = 0.0
    for i in range(iteration):
        time_iteration_start = time.time()
        # Scatter data and labels
        
        # Calculate cost and gradient using sliced data
        cost, (theta1_grad, theta2_grad) = cost_function(theta1, theta2,
            Input_layer_size, Hidden_layer_size, Output_layer_size,
            X_buffer, y_buffer, regular = 0)

        # Gather costs
        cost_buffer = np.zeros((comm.size))
        cost_buffer = comm.gather(cost)
        #comm.Barrier()
        try:
            # Calculate average cost 
            cost = np.sum(cost_buffer)/comm.size
        except TypeError:
            # Only rank 0 will have the value for cost
            pass

        # Gather Gradients
        theta1_grad_buffer = np.asarray([np.zeros_like(theta1_grad)] * comm.size)
        theta2_grad_buffer = np.asarray([np.zeros_like(theta2_grad)] * comm.size)
        comm.Gather(theta1_grad,theta1_grad_buffer)
        comm.Gather(theta2_grad,theta2_grad_buffer)
        # Calculate average gradients
        theta1_grad = np.average(theta1_grad_buffer, 0)
        theta2_grad = np.average(theta2_grad_buffer, 0)
        
        theta1 -= learning_rate * theta1_grad
        theta2 -= learning_rate * theta2_grad
        
        # Broadcast new theta1 theta2
        comm.Bcast([theta1, MPI.DOUBLE])
        comm.Bcast([theta2, MPI.DOUBLE])
        comm.Barrier()
        
        time_iteration_end = time.time()
        if comm.rank == 0:
            print('Iteration {0} cost: {1}, runtime: {2}'.format(i+1, cost, time_iteration_end-time_iteration_start))

    return cost, (theta1, theta2)

def train_parallelize(inputs, labels, learning_rate, iteration):
    cost, model = gradient_descent_parallelize(inputs, labels, learning_rate, iteration)
    return model

if __name__ == '__main__':
    time_start = time.time()

    # Change datasize for experiment 
    datasize_multiplication = 6
    
    inputs, labels = load_training_data()
    inputs = np.repeat(inputs, datasize_multiplication, axis=0)
    labels = np.repeat(labels, datasize_multiplication, axis=0)
    comm = MPI.COMM_WORLD
    model = train_parallelize(inputs, labels, learning_rate = 0.1, iteration = 10)
    time_end = time.time()
    if comm.rank ==0:
        print('\n Total runtime is: {}'.format(time_end - time_start))
