"""
Train a neural network to approximate a 2-variable function
"""
from mpl_toolkits.mplot3d import Axes3D
import neuralNetworkXavier as nnx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
from math import pi
import numpy as np
import warnings
import sys,os
import glob

# Stop matplotlib from giving FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="matplotlib")

def deleteOldData():
    keepData = raw_input('\nDelete all previous plots and run-data? (y/n)')
    if keepData in ['y','yes','']:
        for someFile in glob.glob("SavedRunsNeigh/run*.dat"):
            os.remove(someFile)
        for someFile in glob.glob("SavedPlotsNeigh/fig*.png"):
            os.remove(someFile)


def createData(trainSize,testSize,neighbors=5):
    forceCalc = lambda s: 1.0/s**12 - 1.0/s**6
    low = 0.9;  high = 1.6;  ts = trainSize;  n = neighbors
    # Want random numbers in the discontinous interval [-1.6 --> -0.9 , 0.9 --> 1.6]
    def pureCreation(size,n):
        dim       = (ts,n*3)
        trainData = np.random.uniform(low, high, dim) * np.random.choice([-1,1], dim)
        tDataSol  = np.zeros((ts,3))
        for i in [1,2,3]:# Sum up the neighbors forces
            tDataSol[:,i-1] = np.sum(forceCalc(trainData[:,(i-1)*n:i*n]),axis=1)
        return trainData,tDataSol
    # Create the data
    trainD, trainDS = pureCreation(trainSize,neighbors)
    testD,   testDS = pureCreation(testSize, neighbors)
    return trainD, trainDS, testD, testDS


def train_neural_network(x, epochs, nNodes, hiddenLayers):
    # Begin session
    with tf.Session() as sess:
        # Setup of graph for later computation with tensorflow
        prediction, weights, biases, neurons = neuralNetwork(x)
        cost      = tf.nn.l2_loss(tf.sub(prediction, y))
        optimizer = tf.train.AdamOptimizer().minimize(cost)

        # Number of cycles of feed-forward and backpropagation
        numberOfEpochs    = epochs;
        bestEpochTestLoss = -1
        startEpoch        = 0

        # Initialize variables or restore from file
        saver = tf.train.Saver(weights + biases, max_to_keep=None)
        sess.run(tf.initialize_all_variables())

        bestTrainLoss = 1E100; bestTestLoss = 1E100; bestTestLossSinceLastPlot = 1E100; prevErr = 1E100
        triggerNewLine = False; saveNow = False

        # loop through epocs
        xTrain, yTrain, xTest, yTest = createData(trainSize,testSize)
        for epoch in range(0, numberOfEpochs):
            # loop through batches and cover new data set for each epoch
            _, epochLoss = sess.run([optimizer, cost], feed_dict={x: xTrain, y: yTrain})

            # compute test set loss
            # THIS IS MAYBE USED TO TRAIN?? Gotta check!!
            #_, testCost = sess.run(prediction, feed_dict={x: xTest})
            _, testCost = sess.run([optimizer, cost], feed_dict={x: xTest, y: yTest})

            if epoch%800 == 0:
                triggerNewLine = True
                # Generate new train data each 800th epoch:
                xTrain, yTrain, _, __ = createData(trainSize,testSize)
            if epoch%80  == 0 and numberOfEpochs > epoch+80:
                sys.stdout.write('\r' + ' '*80) # White out line
                sys.stdout.write('\rEpoch %5d out of %5d trainloss/N: %10g, testloss/N: %10g' % \
                      (epoch+1, numberOfEpochs, epochLoss/float(trainSize), testCost/float(testSize)))
                sys.stdout.flush()
            if testCost < bestTestLoss:
                bestTestLoss = testCost
                bestEpochTestLoss = epoch
                if triggerNewLine:
                    sys.stdout.write('\r' + ' '*80) # White out line
                    sys.stdout.write('\rEpoch %5d out of %5d trainloss/N: %10g, testloss/N: %10g\n' % \
                          (epoch+1, numberOfEpochs, epochLoss/float(trainSize), testCost/float(testSize)))
                    sys.stdout.flush()
                triggerNewLine = False

    sys.stdout.write('\r' + ' '*80) # White out line for sake of pretty command line output lol
    return weights, biases, neurons, bestTestLoss/float(testSize)


#--------------#
##### main #####
#--------------#
global trainData

# reset so that variables are not given new names
tf.reset_default_graph()

# number of samples
testSize  = 1000
trainSize = 5000

# number of inputs and outputs
numberOfNeighbors = 5
input_vars  = 3*numberOfNeighbors  # Positions x,y,z of N neighbor atoms
output_vars = 3                    # Force sum x,y,z

x = tf.placeholder('float', [None, input_vars],  name="x")
y = tf.placeholder('float', [None, output_vars], name="y")

#neuralNetwork = lambda data : nnx.modelRelu(data, nNodes=nNodes, hiddenLayers=hiddenLayers, inputs=2, wInitMethod='normal', bInitMethod='normal')
#neuralNetwork = lambda data : nnx.modelTanh(data, nNodes=nNodes, hiddenLayers=hiddenLayers,inputs=2, wInitMethod='normal', bInitMethod='normal')
neuralNetwork = lambda data : nnx.modelSigmoid(data, nNodes=nNodes, hiddenLayers=hiddenLayers,\
                inputs=input_vars, outputs=output_vars,wInitMethod='normal', bInitMethod='normal')

print "---------------------------------------"
epochs = int(sys.argv[1]); nNodes = 20; hiddenLayers = 10
weights, biases, neurons, epochlossPerN = train_neural_network(x, epochs, nNodes, hiddenLayers)
