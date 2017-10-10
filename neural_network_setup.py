import tensorflow as tf
import numpy as np
from math import sqrt

def init_weights(shape, layer, nameType, initMethod='normal'):

    if initMethod == 'zeros':
        return tf.Variable(tf.zeros(shape), name=nameType+'%1d' % layer)

    elif initMethod == 'normal':
        return tf.Variable(tf.random_normal(shape), name=nameType+'%1d' % layer)

    else: # The Xavier-method
        fan_in  = shape[0]
        fan_out = shape[1]
        low_val = -4.0*sqrt(6.0/(fan_in + fan_out)) # {sigmoid:4, tanh:1}
        high_val = 4.0*sqrt(6.0/(fan_in + fan_out))
        return tf.Variable(tf.random_uniform(shape, minval=low_val, maxval=high_val), \
                           name=nameType+'%1d' % layer)



def modelSigmoid(data, nNodes=10, hiddenLayers=3, inputs=2, outputs=1,
          wInitMethod='normal', bInitMethod='normal'):

    weights = []
    biases  = []
    neurons = []

    # Setup of first hidden layer
    w1 = init_weights([inputs, nNodes], 1, 'w', wInitMethod)
    b1 = init_weights([nNodes], 1, 'b', bInitMethod)
    h1 = tf.nn.sigmoid(tf.matmul(data, w1) + b1)
    weights.append(w1)
    biases.append(b1)
    neurons.append(h1)

    # Setup all next layers except output
    for layer in range(1, hiddenLayers, 1):
        w = init_weights([nNodes, nNodes], layer+1, 'w', wInitMethod)
        b = init_weights([nNodes], layer+1, 'b', bInitMethod)
        h = tf.nn.sigmoid(tf.matmul(neurons[layer-1], w) + b)
        weights.append(w)
        biases.append(b)
        neurons.append(h)

    # Setup of output layer (single node, energy)
    w_o = init_weights([nNodes, outputs], hiddenLayers+1, 'w', wInitMethod)
    b_o = init_weights([outputs], hiddenLayers+1, 'b', bInitMethod)
    h_o = tf.matmul(neurons[hiddenLayers-1], w_o) + b_o
    weights.append(w_o)
    biases.append(b_o)
    neurons.append(h_o)

    return h_o, weights, biases, neurons



def modelTanh(data, nNodes=10, hiddenLayers=3, inputs=2, outputs=1,
          wInitMethod='normal', bInitMethod='normal'):

    weights = []
    biases  = []
    neurons = []

    # Setup of first hidden layer
    w1 = init_weights([inputs, nNodes], 1, 'w', wInitMethod)
    b1 = init_weights([nNodes], 1, 'b', bInitMethod)
    h1 = tf.nn.tanh(tf.matmul(data, w1) + b1)
    weights.append(w1)
    biases.append(b1)
    neurons.append(h1)

    # Setup all next layers except output
    for layer in range(1, hiddenLayers, 1):
        w = init_weights([nNodes, nNodes], layer+1, 'w', wInitMethod)
        b = init_weights([nNodes], layer+1, 'b', bInitMethod)
        h = tf.nn.tanh(tf.matmul(neurons[layer-1], w) + b)
        weights.append(w)
        biases.append(b)
        neurons.append(h)

    # Setup of output layer (single node, energy)
    w_o = init_weights([nNodes, outputs], hiddenLayers+1, 'w', wInitMethod)
    b_o = init_weights([outputs], hiddenLayers+1, 'b', bInitMethod)
    h_o = tf.matmul(neurons[hiddenLayers-1], w_o) + b_o
    weights.append(w_o)
    biases.append(b_o)
    neurons.append(h_o)

    return h_o, weights, biases, neurons



def modelRelu(data, nNodes=10, hiddenLayers=3, inputs=3, outputs=1,
          wInitMethod='normal', bInitMethod='normal'):
    weights = []
    biases  = []
    neurons = []

    # Setup of first hidden layer
    w1 = init_weights([inputs, nNodes], 1, 'w', wInitMethod)
    b1 = init_weights([nNodes], 1, 'b', bInitMethod)
    h1 = tf.nn.relu(tf.matmul(data, w1) + b1)
    weights.append(w1)
    biases.append(b1)
    neurons.append(h1)

    # Setup all next layers except output
    for layer in range(1, hiddenLayers, 1):
        w = init_weights([nNodes, nNodes], layer+1, 'w', wInitMethod)
        b = init_weights([nNodes], layer+1, 'b', bInitMethod)
        h = tf.nn.relu(tf.matmul(neurons[layer-1], w) + b)
        weights.append(w)
        biases.append(b)
        neurons.append(h)

    # Setup of output layer (single node, energy)
    w_o = init_weights([nNodes, outputs], hiddenLayers+1, 'w', wInitMethod)
    b_o = init_weights([outputs], hiddenLayers+1, 'b', bInitMethod)
    h_o = tf.matmul(neurons[hiddenLayers-1], w_o) + b_o
    weights.append(w_o)
    biases.append(b_o)
    neurons.append(h_o)

    return h_o, weights, biases, neurons


def model(data, activation_function="sigmoid", nNodes=10, hiddenLayers=3, inputs=3, outputs=1,
          wInitMethod='normal', bInitMethod='normal'):
    """
    Takes input specifying what activation_function to use
    """
    if activation_function == "sigmoid":
        neuralNetwork = lambda data: modelSigmoid(data,
                                                nNodes       = nNodes,
                                                hiddenLayers = hiddenLayers,
                                                inputs       = inputs,
                                                outputs      = outputs,
                                                wInitMethod  = wInitMethod,
                                                bInitMethod  = bInitMethod)
        return neuralNetwork(data)
    elif activation_function == "tanh":
        neuralNetwork = lambda data: modelTanh(data,
                                                nNodes       = nNodes,
                                                hiddenLayers = hiddenLayers,
                                                inputs       = inputs,
                                                outputs      = outputs,
                                                wInitMethod  = wInitMethod,
                                                bInitMethod  = bInitMethod)
        return neuralNetwork(data)
    elif activation_function == "relu":
        neuralNetwork = lambda data: modelRelu(data,
                                                nNodes       = nNodes,
                                                hiddenLayers = hiddenLayers,
                                                inputs       = inputs,
                                                outputs      = outputs,
                                                wInitMethod  = wInitMethod,
                                                bInitMethod  = bInitMethod)
        return neuralNetwork(data)
    else:
        print "Please specify what activation function to use! Exiting!"
        from sys import exit
        exit(0)
