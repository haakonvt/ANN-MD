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


def train_neural_network(x, y, epochs, nNodes, hiddenLayers, batchSize, testSize, learning_rate=0.0002):
    # Number of cycles of feed-forward and backpropagation
    numberOfEpochs = epochs; bestEpochTestLoss = -1; startEpoch = 0
    bestTrainLoss = 1E100; bestTestLoss = 1E100; bestTestLossSinceLastPlot = 1E100; prevErr = 1E100
    triggerNewLine = False; saveNow = False; verbose=True

    saveFlag = True
    list_of_rmse_train = []
    list_of_rmse_test  = []

    # Begin session
    with tf.Session() as sess:
        # Setup of graph for later computation with tensorflow
        prediction, weights, biases, neurons = neuralNetwork(x)
        cost = tf.nn.l2_loss(prediction-y) # Train with RMSE error
        # cost = tf.reduce_mean(np.abs(prediction-y)) # Train with L1 norm
        RMSE = tf.sqrt(tf.reduce_mean(tf.square(prediction-y)))    # tf.reduce_mean same as np.mean
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

        # Initialize variables or restore from file
        saver = tf.train.Saver(weights + biases)
        sess.run(tf.global_variables_initializer())
        # if loadFlag:
        #     saver.restore(sess, "SavedRuns/"+loadFileName)

        # Save first version of the net
        if saveFlag:
            saveFileName = "SavedRuns/run"
            saver.save(sess, saveFileName, global_step=0)
            saveGraphFunc(sess, weights, biases, 0, hiddenLayers, nNodes)

        if readTrainDataFromFile:
            all_data = loadFromFile(testSize, filename, shuffle_rows=True)
            xTest, yTest = all_data(testSize, return_test=True)
        else: # Generate on the fly
            xTest, yTest = createTrainData(testSize, neighbors, PES_Stillinger_Weber, verbose)
        # Loop over epocs
        for epoch in range(0, numberOfEpochs):
            if epoch == 1:
                verbose     = False # Print out some info first iteration ONLY
            epochIsDone = False
            # Loop over all the train data set in batches
            while not epochIsDone:
                # Read new data from loaded training data file
                xTrain, yTrain, epochIsDone = all_data(batchSize) # Get next batch of data

                # Loop through batches of the training set and adjust parameters
                # 'optimizer' still uses quick cost-function
                _, trainRMSE = sess.run([optimizer, RMSE], feed_dict={x: xTrain, y: yTrain})

                # If all training data has been seen "once" epoch is done
                if epochIsDone:
                    # Compute test set loss etc:
                    testRMSE  = sess.run(RMSE, feed_dict={x: xTest , y: yTest})
                    list_of_rmse_test.append(testRMSE)
                    list_of_rmse_train.append(trainRMSE)
                    sys.stdout.write('\r' + ' '*80) # White out line
                    sys.stdout.write('\r%3d/%3d. RMSE: train: %10g, test: %10g\n' % \
                                    (epoch+1, numberOfEpochs, trainRMSE, testRMSE))
                    sys.stdout.flush()

                    # If saving is enabled, save the graph variables ('w', 'b')
                    if saveFlag:
                        saveFileName = "SavedRuns/run"
                        saver.save(sess, saveFileName, global_step=(epoch+1))
                        saveGraphFunc(sess, weights, biases, epoch, hiddenLayers, nNodes)

    sys.stdout.write('\n' + ' '*105) # White out line for sake of pretty command line output lol
    sys.stdout.flush()
    print "\n"
    np.savetxt("testRMSE.txt", list_of_rmse_test)
    np.savetxt("trainRMSE.txt", list_of_rmse_train)
    import matplotlib.pyplot as plt
    plt.subplot(3,1,1)
    xTest_for_plot = np.linspace(1,10,len(list_of_rmse_test))
    xTrain_for_plot = np.linspace(1,10,len(list_of_rmse_train))
    plt.plot(xTrain_for_plot, list_of_rmse_train, label="train")
    plt.plot(xTest_for_plot, list_of_rmse_test, label="test", lw=2.0)
    plt.subplot(3,1,2)
    plt.semilogy(xTrain_for_plot, list_of_rmse_train, label="train")
    plt.semilogy(xTest_for_plot, list_of_rmse_test, label="test", lw=2.0)
    plt.subplot(3,1,3)
    plt.loglog(xTrain_for_plot, list_of_rmse_train, label="train")
    plt.loglog(xTest_for_plot, list_of_rmse_test, label="test", lw=2.0)
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

def example_Stillinger_Weber(filename, epochs):
    # Variables for Stillinger Weber
    # sigma = 2.0951
    # Number of symmetry functions describing local env. of atom i
    # _, input_vars = generate_symmfunc_input_Si_v1()
    _, input_vars = generate_symmfunc_input_Si_Behler()

    tf.reset_default_graph() # Perhaps unneccessary dunno, too lazy to test

    global readTrainDataFromFile

    readTrainDataFromFile = True

    # number of samples
    testSize  = 10000  # Noise free data
    batchSize = 1000   # Train size is determined by length of loaded file

    # IDEA: Perhaps this should be varied under training?
    # numberOfNeighbors = 12 # of Si that are part of computation
    learning_rate     = 0.001

    # Always just one output = energy
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
    nNodes       = 35
    hiddenLayers = 2
    weights, biases, neurons, epochlossPerN = train_neural_network(x, y, epochs, \
                nNodes, hiddenLayers, batchSize, testSize, learning_rate)

def example_Lennard_Jones():
    # Variables for LJ
    sigma = 1.0
    _, input_vars = generate_symmfunc_input_LJ(sigma)
    #TODO: To be implemented...

def parseCommandLine():
    if len(sys.argv) < 3:
        print "Usage:"
        print ">>> python training_nn.py FILENAME EPOCHS"
        print ">>> python training_nn.py SW_train_data.txt 5000"
        sys.exit(0)
    else:
        filename = str(sys.argv[1])
        print "Filename gotten from command line:",filename
        epochs = int(sys.argv[2])
    return filename, epochs


if __name__ == '__main__':
    # Prepare for new run:
    deleteOldData()

    # Get filename of traindata and number of epochs from command line
    filename, epochs = parseCommandLine()

    # Run example of SW:
    t0 = timer()
    example_Stillinger_Weber(filename, epochs)
    print "Wall clock time:", timer() - t0

    # Run example of LJ:
    # t0 = timer()
    # example_Lennard_Jones(filename, epochs)
    # print "Wall clock time:", timer() - t0
