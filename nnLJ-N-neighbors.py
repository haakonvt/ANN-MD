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

# OLD n neighbors x,y,z
# def createData(trainSize,testSize,neighbors=5):
#     forceCalc = lambda s: 1.0/s**12 - 1.0/s**6
#     low = 0.9;  high = 1.6;  ts = trainSize;  n = neighbors
#     # Want random numbers in the discontinous interval [-1.6 --> -0.9 , 0.9 --> 1.6]
#     def pureCreation(size,n):
#         dim       = (ts,n*3) # size istedetfor ts???
#         trainData = np.random.uniform(low, high, dim) * np.random.choice([-1,1], dim)
#         tDataSol  = np.zeros((ts,3)) # size istedetfor ts???
#         for i in [1,2,3]:# Sum up the neighbors forces
#             tDataSol[:,i-1] = np.sum(forceCalc(trainData[:,(i-1)*n:i*n]),axis=1)
#         return trainData,tDataSol
#     # Create the data
#     trainD, trainDS = pureCreation(trainSize,neighbors)
#     testD,   testDS = pureCreation(testSize, neighbors)
#     return trainD, trainDS, testD, testDS

def createData(trainSize,testSize,neighbors=5):
    """ Only coordinate is R """
    PES = lambda s: 1.0/s**12 - 1.0/s**6 # Symmetric function
    low = 0.8;  high = 2.5;  ts = trainSize;  n = neighbors
    def pureCreation(size,n):
        trainData = np.random.uniform(low, high, size)
        tDataSol  = np.zeros(size)
        # Sum up the neighbors forces
        tDataSol[:] = np.sum( PES(trainData),axis=1 )
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

            # Generate new data
            xTrain, yTrain, xTest, yTest = createDataCFDA(trainSize,testSize)

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


def createDataCFDA(trainSize,testSize,neighbors=5):
    if neighbors <= 1:
        print "This code is meant for training a NN with more than one neighbor."
        print "Thus, the number of neighbors will be changed to 20"
        print "...done"
    """
    Train with both potential and forces
    # Input:  [x,   y,  z,  r]
    # Output: [Ep, Fx, Fy, Fz]
    """
    PES    = lambda s:  1.0/s**12 - 1.0/s**6 # Lennard Jones pot. energy
    FORCES = lambda s: 12.0/s**13 - 6.0/s**7 # Analytic derivative of LJ with minus sign: F = -d/dx Ep

    # This gives r in range 0.8 --> 2.5, at least after removing some values
    # low  = -2.5#np.sqrt(0**2/3.) # Set this to lower than 0.8 to get more samples where LJ oscialltes the most, fix later
    # high = 2.5#np.sqrt(2.5**2/3.) # 1.4433756729741

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
            
        plt.subplot(2,1,1)
        plt.hist(nInputs[:,:,3].ravel(),bins=70)
        """ # Suddenly realized that this method was plain stupid... lulz #brain where u @?
        # Remove particles that are too close
        cubeShape = nInputs[:,:,3].shape # save the shape for reshaping later
        lenVec    = np.prod(cubeShape)
        x = nInputs[:,:,0].ravel().reshape(lenVec) # Fixes shape (x,) to (x)
        y = nInputs[:,:,1].ravel().reshape(lenVec)
        z = nInputs[:,:,2].ravel().reshape(lenVec)
        r = nInputs[:,:,3].ravel().reshape(lenVec) # Make into vector
        indicesToRemove  =  list(np.where( r < 0.8 )[0]) # Find indices where particles are too close
        indicesToRemove += (list(np.where( r > 2.5 )[0]))


        for i in indicesToRemove:
            r[i] = np.random.uniform(0.8, 2.5)
            r2 = r[i]**2
            x2 = np.random.uniform(0,r2)
            y2 = np.random.uniform(0,r2-x2)
            z2 = r2 - x2 - y2
            x[i] = sqrt(x2) * np.random.choice([-1,1]) # For single numbers, python inbuilt functions are way quicker than numpy
            y[i] = sqrt(y2) * np.random.choice([-1,1]) # 50-50 if position is plus or minus
            z[i] = sqrt(z2) * np.random.choice([-1,1])
        nInputs[:,:,0] = x.reshape(cubeShape)
        nInputs[:,:,1] = y.reshape(cubeShape)
        nInputs[:,:,2] = z.reshape(cubeShape)
        nInputs[:,:,3] = r.reshape(cubeShape)"""

        plt.subplot(2,1,2)
        plt.hist(nInputs[:,:,3].ravel(),bins=70)
        plt.show()
        return nInputs

    def outputGenerator(nInput):
        size    = nInput.shape[0]
        nOutput = np.zeros((size, neighbors))
        pot     = PES( nInput[:,:,3] )
        # print nInput[:,:,3],"\n"
        # print pot
        # print "#######"


    # Generate train and test data
    input_train  = trainOrTest(trainSize)
    sys.exit()
    input_test   = trainOrTest(testSize)
    output_train = outputGenerator(input_train)
    output_test  = outputGenerator(input_test)
    return input_train, input_test, output_train, output_test
    #print nInputs
    # plt.hist(nInputs[:,:,3],bins=70)
    # plt.show()

#--------------#
##### main #####
#--------------#
createDataCFDA(100000,3,neighbors=5)
sys.exit(0)


"""global trainData

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
"""
