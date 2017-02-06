"""
Train a neural network to approximate a potential energy surface
with the use of symmetry functions that transform xyz-data.
"""
# from symmetry_functions import G1, G2, G3, G4, G5
from symmetry_transform import symmetryTransform
from mpl_toolkits.mplot3d import Axes3D
from math import pi,sqrt,acos,exp,cos
import neural_network_setup as nns
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
import numpy as np
import sys,os
import glob

def potentialEnergyGenerator(xyz_N, PES):
    size = xyz_N.shape[2]
    Ep   = np.zeros(size)
    # r    = np.linalg.norm(xyz_N, axis=1)

    for i in range(size):
        xyz_i = xyz_N[:,:,i]
        Ep[i] = PES(xyz_i)
    return Ep

def createXYZ(r_min, r_max, size, neighbors=20, histogramPlot=False):
    """
    # Input:  Size of train and test size + number of neighbors
    # Output: xyz-neighbors-matrix of size 'size'
    """
    if neighbors <= 1:
        print "This code is meant for training a NN with more than one neighbor."; sys.exit()
    print "Creating XYZ-neighbor-data for:\n - Neighbors: %d \n - Samples  :%d" %(neighbors,size)
    print "-------------------------------"

    xyz_N = np.zeros((neighbors,3,size))
    xyz   = np.zeros((size,3))
    for i in range(neighbors): # Fill cube slice for each neighbor (quicker than "size")
        r2       = np.random.uniform(r_min, r_max,       size)**2
        xyz[:,0] = np.random.uniform(0,     r2,          size)
        xyz[:,1] = np.random.uniform(0,     r2-xyz[:,0], size)
        xyz[:,2] = r2 - xyz[:,0] - xyz[:,1]
        for row in range(size):
            np.random.shuffle(xyz[row,:]) # This shuffles in-place (so no copying)
        xyz_N[i,0,:] = np.sqrt(xyz[:,0]) * np.random.choice([-1,1],size) # 50-50 if position is plus or minus
        xyz_N[i,1,:] = np.sqrt(xyz[:,1]) * np.random.choice([-1,1],size)
        xyz_N[i,2,:] = np.sqrt(xyz[:,2]) * np.random.choice([-1,1],size)
    if histogramPlot:
        plt.subplot(3,1,1);plt.hist(xyz_N[:,0,:].ravel(),bins=70);plt.subplot(3,1,2);plt.hist(xyz_N[:,1,:].ravel(),bins=70);plt.subplot(3,1,3);plt.hist(xyz_N[:,2,:].ravel(),bins=70);plt.show()
    return xyz_N

def PES_Stillinger_Weber(xyz_i):
    """
    INPUT
    - xyz_i: Matrix with columnds containing cartesian coordinates,
           relative to the current atom i, i.e.:
           [[x1 y1 z1]
            [x2 y2 z2]
            [x3 y3 z3]
            [x4 y4 z4]]
    """
    xyz = xyz_i
    r = np.linalg.norm(xyz,axis=1)
    N = len(r) # Number of neighbors for atom i, which we are currently inspecting

    # A lot of definitions first
    A = 7.049556277
    B = 0.6022245584
    p = 4
    q = 0
    a = 1.8
    l = 21  # lambda
    g = 1.2 # gamma
    cos_tc = -1.0/3.0 # 109.47 deg

    eps = 2.1678  # [eV]
    m   = 28.085  # [amu]
    sig = 0.20951 # [nm]
    Ts  = 25156.73798 # eps/kB [K]

    rc = (r < a) # Bool array. "False" cast to 0 and "True" to 1
    U2 = eps*A*(B*r**(-p)-r**(-q)) * np.exp(r-a)**-1 * rc
    U3 = lambda rij, rik, theta_ijk: eps*l*exp(g/(rij-a) + g/(rik-a)) \
                                     * (cos(theta_ijk) - cos_tc)**2   \
                                     * (rij < a) * (rik < a)

    U = np.sum(U2) # Two body terms
    for j in range(N): # i < j
        for k in range(j,N): # i < j < k
            theta_ijk  = acos(np.dot(xyz[j],xyz[k]) / (r[j]*r[k])) # TODO: Angle sometimes wrong (abs > 1)
            U         += U3(r[j], r[k], theta_ijk) # Three body terms
    return U

def FORCES_Stillinger_Weber(xyz_i):
    """
    To be implemented
    """
    pass

def PES_Lennard_Jones(xyz_i):
    """
    Simple LJ pair potential
    """
    # eps = 1.0318 * 10^(-2) eV
    # sig = 3.405 * 10^(-7) meter
    xyz = xyz_i
    r   = np.linalg.norm(xyz,axis=1)
    U   =  np.sum(4*(1.0/r**12 - 1.0/r**6))
    return U

def FORCES_Lennard_Jones(xyz_i):
    """
    To be implemented:
    FORCES = lambda s: 12.0/s**13 - 6.0/s**7 # Analytic derivative of LJ with minus sign: F = -d/dx Ep
    """
    pass

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

        # Save first version of the net
        saveFileName = "SavedRuns/run" + str(0) + ".dat"
        saver.save(sess, saveFileName, write_meta_graph=False)
        saveGraphFunc(sess, weights, biases, 0)

        bestTrainLoss = 1E100; bestTestLoss = 1E100; bestTestLossSinceLastPlot = 1E100; prevErr = 1E100
        triggerNewLine = False; saveNow = False

        # Loop over epocs
        for epoch in range(0, numberOfEpochs):
            # Generate new data
            xTrain, yTrain, xTest, yTest = createData(trainSize,testSize,neighbors) # TODO: Fix

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

if __name__ == '__main__':
    #--------------#
    ##### main #####
    #--------------#
    r_low     = 2
    r_high    = 5
    size      = 100
    neighbors = 30
    PES   = PES_Stillinger_Weber
    xyz_N = createXYZ(r_low, r_high, size, neighbors)
    Ep    = potentialEnergyGenerator(xyz_N, PES)
    print Ep

    if False:
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
