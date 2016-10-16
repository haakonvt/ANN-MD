import tensorflow as tf
import numpy as np

def init_weights(shape, layer, nameType, initMethod='normal'):

    if initMethod == 'zeros':
        return tf.Variable(tf.zeros(shape), name=nameType+'%1d' % layer)

    elif initMethod == 'normal':
        return tf.Variable(tf.random_normal(shape), name=nameType+'%1d' % layer)

    else: #xavier
        fanIn  = shape[0]
        fanOut = shape[1]
        low = -4*np.sqrt(6.0/(fanIn + fanOut)) # {sigmoid:4, tanh:1}
        high = 4*np.sqrt(6.0/(fanIn + fanOut))
        return tf.Variable(tf.random_uniform(shape, minval=low, maxval=high), \
                           name=nameType+'%1d' % layer)



def modelSigmoid(data, nNodes=10, hiddenLayers=3, inputs=2, outputs=1,
          wInitMethod='normal', bInitMethod='normal'):

    weights = []
    biases  = []
    neurons = []

    # first hidden layer
    w1 = init_weights([inputs, nNodes], 1, 'w', wInitMethod)
    b1 = init_weights([nNodes], 1, 'b', bInitMethod)
    h1 = tf.nn.sigmoid(tf.matmul(data, w1) + b1)
    weights.append(w1)
    biases.append(b1)
    neurons.append(h1)

    # following layers
    for layer in range(1, hiddenLayers, 1):
        w = init_weights([nNodes, nNodes], layer+1, 'w', wInitMethod)
        b = init_weights([nNodes], layer+1, 'b', bInitMethod)
        h = tf.nn.sigmoid(tf.matmul(neurons[layer-1], w) + b)
        weights.append(w)
        biases.append(b)
        neurons.append(h)

    # output layer
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

    # first hidden layer
    w1 = init_weights([inputs, nNodes], 1, 'w', wInitMethod)
    b1 = init_weights([nNodes], 1, 'b', bInitMethod)
    h1 = tf.nn.tanh(tf.matmul(data, w1) + b1)
    weights.append(w1)
    biases.append(b1)
    neurons.append(h1)

    # following layers
    for layer in range(1, hiddenLayers, 1):
        w = init_weights([nNodes, nNodes], layer+1, 'w', wInitMethod)
        b = init_weights([nNodes], layer+1, 'b', bInitMethod)
        h = tf.nn.tanh(tf.matmul(neurons[layer-1], w) + b)
        weights.append(w)
        biases.append(b)
        neurons.append(h)

    # output layer
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

    # first hidden layer
    w1 = init_weights([inputs, nNodes], 1, 'w', wInitMethod)
    b1 = init_weights([nNodes], 1, 'b', bInitMethod)
    h1 = tf.nn.relu(tf.matmul(data, w1) + b1)
    weights.append(w1)
    biases.append(b1)
    neurons.append(h1)

    # following layers
    for layer in range(1, hiddenLayers, 1):
        w = init_weights([nNodes, nNodes], layer+1, 'w', wInitMethod)
        b = init_weights([nNodes], layer+1, 'b', bInitMethod)
        h = tf.nn.relu(tf.matmul(neurons[layer-1], w) + b)
        weights.append(w)
        biases.append(b)
        neurons.append(h)

    # output layer
    w_o = init_weights([nNodes, outputs], hiddenLayers+1, 'w', wInitMethod)
    b_o = init_weights([outputs], hiddenLayers+1, 'b', bInitMethod)
    h_o = tf.matmul(neurons[hiddenLayers-1], w_o) + b_o
    weights.append(w_o)
    biases.append(b_o)
    neurons.append(h_o)

    return h_o, weights, biases, neurons





def modelReluSigmoid(data, nNodes=10, hiddenLayers=3, inputs=3, outputs=1,
          wInitMethod='normal', bInitMethod='normal'):

    weights = []
    biases  = []
    neurons = []

    # first hidden layer
    w1 = init_weights([inputs, nNodes], 1, 'w', wInitMethod)
    b1 = init_weights([nNodes], 1, 'b', bInitMethod)
    h1 = tf.nn.sigmoid(tf.matmul(data, w1) + b1)
    weights.append(w1)
    biases.append(b1)
    neurons.append(h1)

    # following layers
    for layer in range(1, hiddenLayers, 1):
        w = init_weights([nNodes, nNodes], layer+1, 'w', wInitMethod)
        b = init_weights([nNodes], layer+1, 'b', bInitMethod)
        if layer < hiddenLayers-1:
            h = tf.nn.relu(tf.matmul(neurons[layer-1], w) + b)
        else:
            h = tf.nn.sigmoid(tf.matmul(neurons[layer-1], w) + b)
        weights.append(w)
        biases.append(b)
        neurons.append(h)

    # output layer
    w_o = init_weights([nNodes, outputs], hiddenLayers+1, 'w', wInitMethod)
    b_o = init_weights([outputs], hiddenLayers+1, 'b', bInitMethod)
    h_o = tf.matmul(neurons[hiddenLayers-1], w_o) + b_o
    weights.append(w_o)
    biases.append(b_o)
    neurons.append(h_o)

    return h_o, weights, biases, neurons
