"""
Train a neural network to approximate a 2-variable function
"""
from mpl_toolkits.mplot3d import Axes3D
import neural_network_setup as nns
import matplotlib.pyplot as plt
from math import pi,sqrt,acos
import matplotlib.cm as cm
import tensorflow as tf
import numpy as np
import warnings
import sys,os
import glob

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

def createData_xyz(trainSize, testSize, PES, FORCES, neighbors=5):
    """
    Train with both potential and forces
    # Input:  [x,   y,  z]
    # Output: []
    """
    if neighbors <= 1:
        print "This code is meant for training a NN with more than one neighbor."
        sys.exit()

    # Example
    # PES    = lambda s:  1.0/s**12 - 1.0/s**6 # Lennard Jones pot. energy
    # FORCES = lambda s: 12.0/s**13 - 6.0/s**7 # Analytic derivative of LJ with minus sign: F = -d/dx Ep

    # Generate train and test data
    input_train  = trainOrTest(trainSize)
    input_test   = trainOrTest(testSize)
    output_train = outputGenerator(input_train)
    output_test  = outputGenerator(input_test)


    # # TODO: Not sure the reshape does what I intend (unravel second and last index)
    # return input_train.reshape(trainSize,neighbors*4), output_train, input_test.reshape(testSize,neighbors*4), output_test

def PES_Stillinger_Weber(xyz):
    """
    INPUT
    - xyz: Matrix with columnds containing cartesian coordinates i.e.:
           [[x1 y1 z1]
            [x2 y2 z2]
            [x3 y3 z3]
            [x4 y4 z4]]
    """
    r = np.linalg.norm(xyz,axis=1)
    N = len(r)

    # A lot of definitions first
    A = 7.049556277
    B = 0.6022245584
    p = 4
    q = 0
    a = 1.8
    l = 21  # [lambda]
    g = 1.2 # [gamma]
    cos_tc = -1.0/3.0

    eps = 2.1678  # [eV]
    m   = 28.085  # [amu]
    sig = 0.20951 # [nm]
    Ts  = 25156.73798 # eps/kB [K]

    from math import exp,cos
    rc = (r < a) # Bool array. "False" cast to 0 and "True" to 1
    U2 = eps*A*(B*r**(-p)-r**(-q)) * np.exp(r-a)**-1 * rc
    U3 = lambda rij, rik, theta_ijk: eps*l*exp(g/(rij-a) + g/(rik-a)) * (cos(theta_ijk) - cos_tc)**2 * (rij < a) * (rik < a)

    U = np.sum(U2)
    for j in range(N): # i < j
        for k in range(j,N): # i < j < k
            theta_ijk  = acos(np.dot(xyz[j],xyz[k]) / (r[j]*r[k]))
            U         += U3(r[j], r[k], theta_ijk)



def train_neural_network(x, epochs, nNodes, hiddenLayers,neighbors=5):
    # Begin session
    with tf.Session() as sess:
        # Setup of graph for later computation with tensorflow
        prediction, weights, biases, neurons = neuralNetwork(x)
        cost      = tf.nn.l2_loss(tf.sub(prediction, y))
        optimizer = tf.train.AdamOptimizer(0.005).minimize(cost)

        # Number of cycles of feed-forward and backpropagation
        numberOfEpochs    = epochs;
        bestEpochTestLoss = -1
        startEpoch        = 0

        # Initialize variables or restore from file
        saver = tf.train.Saver(weights + biases, max_to_keep=None)
        sess.run(tf.initialize_all_variables())

        # Save first version of the net
        saveFileName = "SavedRuns/run" + str(0) + ".dat"
        saver.save(sess, saveFileName, write_meta_graph=False)
        saveGraphFunc(sess, weights, biases, 0)

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
                    # If saving is enabled, save the graph variables ('w', 'b')
                    if True:#saveFlag:
                        saveFileName = "SavedRuns/run" + str(epoch+1) + ".dat"
                        saver.save(sess, saveFileName, write_meta_graph=False)
                        saveGraphFunc(sess, weights, biases, epoch+1)
                triggerNewLine = False

    sys.stdout.write('\r' + ' '*80) # White out line for sake of pretty command line output lol
    return weights, biases, neurons, bestTestLoss/float(testSize)

def saveGraphFunc(sess, weights, biases, epoch):
    try:
        os.mkdir("SavedGraphs")
    except:
        pass
    saveGraphName = "SavedGraphs/tf_graph_WB_%d.txt" %(epoch)
    with open(saveGraphName, 'w') as outFile:
        outStr = "%1d %1d %s" % (hiddenLayers, nNodes, "sigmoid")
        outFile.write(outStr + '\n')
        size = len(sess.run(weights))
        for i in range(size):
            i_weights = sess.run(weights[i])
            if i < size-1:
                for j in range(len(i_weights)):
                    for k in range(len(i_weights[0])):
                        outFile.write("%g" % i_weights[j][k])
                        outFile.write(" ")
                    outFile.write("\n")
            else:
                for j in range(len(i_weights[0])):
                    for k in range(len(i_weights)):
                        outFile.write("%g" % i_weights[k][j])
                        outFile.write(" ")
                    outFile.write("\n")
        outFile.write("\n")
        for biasVariable in biases:
            i_biases = sess.run(biasVariable)
            for j in range(len(i_biases)):
                outFile.write("%g" % i_biases[j])
                outFile.write(" ")
            outFile.write("\n")

#--------------#
##### main #####
#--------------#

tf.reset_default_graph() # Perhaps unneccessary

# number of samples
testSize  = 100  # Noise free data
trainSize = 5000

# number of inputs and outputs
numberOfNeighbors = 5
input_vars  = 4*numberOfNeighbors  # Positions x,y,z and r of N neighbor atoms
output_vars = 4                    # Force sum x,y,z and potential energy

x = tf.placeholder('float', shape=(None, input_vars),  name="x")
y = tf.placeholder('float', shape=(None, output_vars), name="y")

#neuralNetwork = lambda data : nns.modelRelu(data, nNodes=nNodes, hiddenLayers=hiddenLayers, inputs=2, wInitMethod='normal', bInitMethod='normal')
#neuralNetwork = lambda data : nns.modelTanh(data, nNodes=nNodes, hiddenLayers=hiddenLayers,inputs=2, wInitMethod='normal', bInitMethod='normal')
neuralNetwork = lambda data : nns.modelSigmoid(data, nNodes=nNodes, hiddenLayers=hiddenLayers,\
                inputs=input_vars, outputs=output_vars,wInitMethod='normal', bInitMethod='zeros')

print "---------------------------------------"
epochs       = 100000
nNodes       = 50
hiddenLayers = 2
weights, biases, neurons, epochlossPerN = train_neural_network(x, epochs, nNodes, hiddenLayers, numberOfNeighbors)
