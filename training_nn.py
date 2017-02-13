"""
Train a neural network to approximate a potential energy surface
with the use of symmetry functions that transform xyz-data.
"""

import neural_network_setup as nns
from create_train_data import *
from symmetry_transform import *
import tensorflow as tf
import numpy as np
import sys,os


def train_neural_network(x, y, epochs, nNodes, hiddenLayers, trainSize, testSize, neighbors=20):
    # Begin session
    with tf.Session() as sess:
        # Setup of graph for later computation with tensorflow
        prediction, weights, biases, neurons = neuralNetwork(x)
        cost      = tf.nn.l2_loss(tf.sub(prediction, y))
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

        # Number of cycles of feed-forward and backpropagation
        numberOfEpochs    = epochs;
        bestEpochTestLoss = -1
        startEpoch        = 0

        # Initialize variables or restore from file
        saver = tf.train.Saver(weights + biases, max_to_keep=None)
        sess.run(tf.global_variables_initializer())

        # Save first version of the net
        saveFileName = "SavedRuns/run" + str(0) + ".dat"
        saver.save(sess, saveFileName, write_meta_graph=False)
        saveGraphFunc(sess, weights, biases, 0, hiddenLayers, nNodes)

        bestTrainLoss = 1E100; bestTestLoss = 1E100; bestTestLossSinceLastPlot = 1E100; prevErr = 1E100
        triggerNewLine = False; saveNow = False; verbose=True

        if readTrainDataFromFile:
            all_data = loadFromFile(testSize, filename, shuffle_rows=False)
            xTest, yTest = all_data(testSize, return_test=True)

        # Loop over epocs
        for epoch in range(0, numberOfEpochs):
            if epoch == 1:
                verbose = False # Print out some info first iteration ONLY
            # Read new data from file
            if readTrainDataFromFile:
                xTrain, yTrain = all_data(trainSize) # Get next batch of data
            # Generate new data
            else: # elif epoch%10 == 0:
                xTrain, yTrain, xTest, yTest = createTrainData(trainSize, testSize, neighbors, PES_Stillinger_Weber, verbose)
            # Loop through batches and cover new data set for each epoch
            _, epochLoss = sess.run([optimizer, cost], feed_dict={x: xTrain, y: yTrain})
            # Compute test set loss
            testCost = sess.run(cost, feed_dict={x: xTest, y: yTest})

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
                        saveGraphFunc(sess, weights, biases, epoch+1, hiddenLayers, nNodes)
                triggerNewLine = False
    sys.stdout.write('\r' + ' '*80) # White out line for sake of pretty command line output lol
    sys.stdout.flush()
    print "\n"
    return weights, biases, neurons, bestTestLoss/float(testSize)

def saveGraphFunc(sess, weights, biases, epoch, hiddenLayers, nNodes):
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

def example_Stillinger_Weber():
    tf.reset_default_graph() # Perhaps unneccessary

    global filename
    global readTrainDataFromFile
    # filename = "stillinger-weber-symmetry-data.txt"
    filename = "SW_train_2e5.txt"
    readTrainDataFromFile = True

    # number of samples
    testSize  = 2000  # Noise free data
    trainSize = 1000

    # IDEA: Perhaps this should be varied under training?
    numberOfNeighbors = 15 # of Si that are part of computation

    # Number of symmetry functions describing local env. of atom i
    _, input_vars = generate_symmfunc_input_Si()
    output_vars = 1 # Potential energy of atom i

    x = tf.placeholder('float', shape=(None, input_vars),  name="x")
    y = tf.placeholder('float', shape=(None, output_vars), name="y")

    global neuralNetwork
    neuralNetwork = lambda data : nns.modelSigmoid(data, nNodes=nNodes, hiddenLayers=hiddenLayers,\
                    inputs=input_vars, outputs=output_vars, wInitMethod='normal', bInitMethod='normal')

    print "---------------------------------------"
    epochs       = 10000
    nNodes       = 50
    hiddenLayers = 3
    weights, biases, neurons, epochlossPerN = train_neural_network(x, y, epochs, nNodes, hiddenLayers, trainSize, testSize, numberOfNeighbors)


if __name__ == '__main__':
    # Run example of SW:
    example_Stillinger_Weber()

    """
    if False:
        np.random.seed(1) # For testing
        sigma     = 1.0
        r_low     = 0.85 * sigma
        r_high    = 1.8  * sigma - 1E8 # SW has a divide by zero at exactly cutoff
        size      = 1
        neighbors = 15
        PES       = PES_Stillinger_Weber
        xyz_N     = createXYZ(r_low, r_high, size, neighbors, histogramPlot=False)
        Ep        = potentialEnergyGenerator(xyz_N, PES)
        G_funcs   = example_generate_G_funcs_input()

        print "Number of symmetry functions used to describe each atom i:", 51
        print "-------------------------------"
        for i in range(size):
            print "Stillinger Weber potential E:", Ep[i],"\n"
            xyz_i = xyz_N[:,:,i]
            G_i   = symmetryTransform(G_funcs, xyz_i)
            print G_i, "\n"
        """
