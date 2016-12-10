"""
Train a neural network to approximate a 2-variable function
"""
from mpl_toolkits.mplot3d import Axes3D
import neuralNetworkXavier as nnx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from math import pi,sqrt
import tensorflow as tf
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

def createDataCFDA(trainSize,testSize,neighbors=5):
    """
    Train with both potential and forces
    # Input:  [x,   y,  z,  r]
    # Output: [Ep, Fx, Fy, Fz]
    """
    if neighbors <= 1:
        print "This code is meant for training a NN with more than one neighbor."
        print "Thus, the number of neighbors will be changed to 20"
        neighbors = 20
        print "...done"

    PES    = lambda s:  1.0/s**12 - 1.0/s**6 # Lennard Jones pot. energy
    FORCES = lambda s: 12.0/s**13 - 6.0/s**7 # Analytic derivative of LJ with minus sign: F = -d/dx Ep

    def trainOrTest(size):# Cube object to be returned
        nInputs = np.zeros((size,neighbors,4))
        xyz     = np.zeros((size,3))
        for i in range(neighbors): # Fill cube slice for each neighbor
            nInputs[:,i,3] = np.random.uniform(0.8, 2.5, size) # this is R
            r2             = nInputs[:,i,3]**2
            xyz[:,0]       = np.random.uniform(0, r2, size)
            xyz[:,1]       = np.random.uniform(0, r2-xyz[:,0], size)
            xyz[:,2]       = r2 - xyz[:,0] - xyz[:,1]
            for row in range(size):
                np.random.shuffle(xyz[row,:]) # This shuffles in-place (so no copying)
            nInputs[:,i,0] = np.sqrt(xyz[:,0]) * np.random.choice([-1,1],size) # 50-50 if position is plus or minus
            nInputs[:,i,1] = np.sqrt(xyz[:,1]) * np.random.choice([-1,1],size)
            nInputs[:,i,2] = np.sqrt(xyz[:,2]) * np.random.choice([-1,1],size)
            # plt.subplot(3,1,1);plt.hist(nInputs[:,:,0].ravel(),bins=70);plt.subplot(3,1,2);plt.hist(nInputs[:,:,1].ravel(),bins=70);plt.subplot(3,1,3);plt.hist(nInputs[:,:,2].ravel(),bins=70);plt.show()
        return nInputs

    def outputGenerator(nInput):
        size    = nInput.shape[0]
        nOutput = np.zeros((size, 4)) # 4: Fx, Fy, Fz and Ep
        r       = nInput[:,:,3]
        # Sum up contribution from all neighbors:
        nOutput[:,0] = np.sum( (FORCES(r) * nInput[:,:,0] / r) ,axis=1) # Fx
        nOutput[:,1] = np.sum( (FORCES(r) * nInput[:,:,1] / r) ,axis=1) # Fy
        nOutput[:,2] = np.sum( (FORCES(r) * nInput[:,:,2] / r) ,axis=1) # Fz
        nOutput[:,3] = np.sum(PES( r ),axis=1) # Ep_tot
        return nOutput

    # Generate train and test data
    input_train  = trainOrTest(trainSize)
    input_test   = trainOrTest(testSize)
    output_train = outputGenerator(input_train)
    output_test  = outputGenerator(input_test)
    return input_train, input_test, output_train, output_test

def train_neural_network(x, epochs, nNodes, hiddenLayers,neighbors=5):
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

        # Loop over epocs
        for epoch in range(0, numberOfEpochs):
            # Generate new data
            xTrain, yTrain, xTest, yTest = createDataCFDA(trainSize,testSize,neighbors)

            # loop through batches and cover new data set for each epoch
            _, epochLoss = sess.run([optimizer, cost], feed_dict={x: xTrain, y: yTrain})

            # compute test set loss
            # THIS IS MAYBE USED TO TRAIN?? Gotta check!!
            #_, testCost = sess.run(prediction, feed_dict={x: xTest})
            _, testCost = sess.run([optimizer, cost], feed_dict={x: xTest, y: yTest})

            if epoch%800 == 0:
                triggerNewLine = True
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

tf.reset_default_graph() # Perhaps unneccessary

# number of samples
testSize  = 100  # Noise free data
trainSize = 100

# number of inputs and outputs
numberOfNeighbors = 5
input_vars  = 4  # Positions x,y,z and r of N neighbor atoms
output_vars = 4                    # Force sum x,y,z and potential energy

x = tf.placeholder('float', [100, numberOfNeighbors, input_vars],  name="x")
y = tf.placeholder('float', [None, output_vars], name="y")

#neuralNetwork = lambda data : nnx.modelRelu(data, nNodes=nNodes, hiddenLayers=hiddenLayers, inputs=2, wInitMethod='normal', bInitMethod='normal')
#neuralNetwork = lambda data : nnx.modelTanh(data, nNodes=nNodes, hiddenLayers=hiddenLayers,inputs=2, wInitMethod='normal', bInitMethod='normal')
neuralNetwork = lambda data : nnx.modelSigmoid(data, nNodes=nNodes, hiddenLayers=hiddenLayers,\
                inputs=input_vars, outputs=output_vars,wInitMethod='normal', bInitMethod='normal')

print "---------------------------------------"
epochs       = 100
nNodes       = 50
hiddenLayers = 2
weights, biases, neurons, epochlossPerN = train_neural_network(x, epochs, nNodes, hiddenLayers, numberOfNeighbors)
