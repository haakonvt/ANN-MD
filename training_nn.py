"""
Train a neural network to approximate a potential energy surface
with the use of symmetry functions that transform xyz-data.
"""

from file_management import deleteOldData
from timeit import default_timer as timer # Best timer indep. of system
import neural_network_setup as nns
from create_train_data import *
from symmetry_transform import *
import tensorflow as tf
import numpy as np
import sys,os
from math import sqrt


def train_neural_network(x, y, epochs, nNodes, hiddenLayers, trainSize, testSize, learning_rate=0.0002):
    # Number of cycles of feed-forward and backpropagation
    numberOfEpochs = epochs; bestEpochTestLoss = -1; startEpoch = 0
    bestTrainLoss = 1E100; bestTestLoss = 1E100; bestTestLossSinceLastPlot = 1E100; prevErr = 1E100
    triggerNewLine = False; saveNow = False; verbose=True

    saveFlag = False
    list_of_rmse_train = []
    list_of_rmse_test  = []

    # Begin session
    with tf.Session() as sess:
        # Setup of graph for later computation with tensorflow
        prediction, weights, biases, neurons = neuralNetwork(x)
        cost = tf.nn.l2_loss(prediction-y) # Train with RMSE error
        RMSE = tf.sqrt(tf.reduce_mean(tf.square(prediction-y)))    # tf.reduce_mean same as np.mean
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        # Initialize variables or restore from file
        # saver = tf.train.Saver(weights + biases)
        sess.run(tf.global_variables_initializer())
        # if loadFlag:
        #     loadFileName, startEpoch = findLoadFileName()
        #     numberOfEpochs          += startEpoch
        #     saver.restore(sess, "SavedRuns/"+loadFileName)

        # Save first version of the net
        if saveFlag:
            saveFileName = "SavedRuns/run"
            saver.save(sess, saveFileName, global_step=0)
            saveGraphFunc(sess, weights, biases, 0, hiddenLayers, nNodes)

        if readTrainDataFromFile:
            all_data = loadFromFile(testSize, filename, shuffle_rows=False)
            xTest, yTest = all_data(testSize, return_test=True)
        else: # Generate on the fly
            xTest, yTest = createTrainData(testSize, neighbors, PES_Stillinger_Weber, verbose)

        # Loop over epocs
        for epoch in range(0, numberOfEpochs):
            if epoch == 1:
                # print '\nNOTE: "-verbose" turned off now (after first epoch)'
                verbose = False # Print out some info first iteration ONLY
            # Read new data from file
            if readTrainDataFromFile:
                xTrain, yTrain = all_data(trainSize) # Get next batch of data
            # Generate new data
            elif epoch%50 == 0:
                if verbose:
                    print "\nGenerating training data on the fly! Just FYI\n!"
                xTrain, yTrain = createTrainData(trainSize, neighbors, PES_Stillinger_Weber, verbose)

            # Loop through batches of the training set and adjust parameters
            _, trainRMSE = sess.run([optimizer, RMSE], feed_dict={x: xTrain, y: yTrain})
            list_of_rmse_train.append(trainRMSE)

            if epoch%800 == 0 or (epoch%10 == 0 and epoch < 200):
                triggerNewLine = True
            if epoch%80  == 0 and numberOfEpochs > epoch+80:
                # Compute test set loss etc:
                testRMSE  = sess.run(RMSE, feed_dict={x: xTest , y: yTest})
                list_of_rmse_test.append(testRMSE)
                sys.stdout.write('\r' + ' '*80) # White out line
                sys.stdout.write('\r%5d/%5d. RMSE: train: %10g, test: %10g' % \
                                (epoch+1, numberOfEpochs, trainRMSE, testRMSE))
                sys.stdout.flush()
            if trainRMSE < bestTestLoss:
                bestTestLoss = trainRMSE
                bestEpochTestLoss = epoch
                if triggerNewLine:
                    # Compute test set loss etc:
                    testRMSE  = sess.run(RMSE, feed_dict={x: xTest , y: yTest})
                    # trainRMSE = sess.run(RMSE, feed_dict={x: xTrain, y: yTrain})
                    sys.stdout.write('\r' + ' '*80) # White out line
                    sys.stdout.write('\r%5d/%5d. RMSE: train: %10g, test: %10g\n' % \
                                    (epoch+1, numberOfEpochs, trainRMSE, testRMSE))
                    sys.stdout.flush()
                    # If saving is enabled, save the graph variables ('w', 'b')
                    if saveFlag:
                        saveFileName = "SavedRuns/run"
                        saver.save(sess, saveFileName, global_step=(epoch+1))
                        saveGraphFunc(sess, weights, biases, epoch, hiddenLayers, nNodes)
                triggerNewLine = False
    sys.stdout.write('\n' + ' '*105) # White out line for sake of pretty command line output lol
    sys.stdout.flush()
    print "\n"
    np.savetxt("testRMSE.txt", list_of_rmse_test)
    np.savetxt("trainRMSE.txt", list_of_rmse_train)
    import matplotlib.pyplot as plt
    plt.subplot(2,1,1)
    plt.plot(list_of_rmse_test)
    plt.subplot(2,1,2)
    plt.plot(list_of_rmse_train)
    plt.show()
    return weights, biases, neurons, bestTestLoss/float(testSize)

def actualLoss(tf_l2_loss, size):
    # Correcting TF function: l2_loss = sum(t ** 2) / 2
    output = sqrt(tf_l2_loss * 2.0 / float(size))
    return output

def saveGraphFunc(sess, weights, biases, epoch, hiddenLayers, nNodes):
    """
    Saves the neural network weights and biases to file,
    in a format readably by 'humans'
    """
    epoch = epoch + 1
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
    # Variables for Stillinger Weber
    sigma = 2.0951

    tf.reset_default_graph() # Perhaps unneccessary dunno, too lazy to test

    global filename
    global readTrainDataFromFile
    # filename = "stillinger-weber-symmetry-data.txt"
    filename = "SW_train_20000_n10_v1.txt"
    if len(sys.argv) > 1:
        filename = str(sys.argv[1])
        print "Filename gotten from command line:",filename
    if len(sys.argv) > 2:
        epochs = int(sys.argv[2])
    else:
        epochs = 500000
    readTrainDataFromFile = True

    # number of samples
    testSize  = 200  # Noise free data
    trainSize = 100  # Not really trainSize, but rather BATCH SIZE

    # IDEA: Perhaps this should be varied under training?
    # numberOfNeighbors = 12 # of Si that are part of computation
    learning_rate     = 0.001

    # Number of symmetry functions describing local env. of atom i
    _, input_vars = generate_symmfunc_input_Si(sigma)
    output_vars = 1 # Potential energy of atom i

    x = tf.placeholder('float', shape=(None, input_vars),  name="x")
    y = tf.placeholder('float', shape=(None, output_vars), name="y")

    global neuralNetwork
    neuralNetwork = lambda data : nns.modelSigmoid(data, nNodes       = nNodes,
                                                         hiddenLayers = hiddenLayers,
                                                         inputs       = input_vars,
                                                         outputs      = output_vars,
                                                         wInitMethod  = 'normal',
                                                         bInitMethod  = 'normal')

    print "---------------------------------------"
    nNodes       = 40
    hiddenLayers = 4
    weights, biases, neurons, epochlossPerN = train_neural_network(x, y, epochs, \
                nNodes, hiddenLayers, trainSize, testSize, learning_rate)


if __name__ == '__main__':
    # Prepare for new run:
    deleteOldData()

    # Run example of SW:
    t0 = timer()
    example_Stillinger_Weber()
    print "Wall clock time:", timer() - t0
