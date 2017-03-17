"""
Train a neural network to approximate a potential energy surface
with the use of symmetry functions that transform xyz-data.
"""

from plot_tools import plotTestVsTrainLoss
from file_management import deleteOldData
from timeit import default_timer as timer # Best timer indep. of system
import neural_network_setup as nns
from create_train_data import *
from symmetry_transform import *
import tensorflow as tf
import numpy as np
import sys,os
from math import sqrt

def train_neural_network(x, y, epochs, nNodes, hiddenLayers, batchSize, testSize, learning_rate=0.0002, loss_function="L2"):
    # Number of cycles of feed-forward and backpropagation
    numberOfEpochs = epochs
    bestTrainLoss  = 1E100
    list_of_rmse_train = []
    list_of_rmse_test  = []

    # Begin session
    with tf.Session() as sess:
        # Setup of graph for later computation with tensorflow
        prediction, weights, biases, neurons = neuralNetwork(x)
        if   loss_function == "L2": # Train with RMSE error
            cost = tf.nn.l2_loss(prediction-y)
        elif loss_function == "L1": # Train with L1 norm
            cost = tf.reduce_mean(np.abs(prediction-y))
        # Create operation to get the RMSE loss: (not for training, only evaluation)
        RMSE = tf.sqrt(tf.reduce_mean(tf.square(prediction-y)))

        # Create the optimizer, with cost function to minimize
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

        # Load into memory the train/test data, and generate batch for first epoch
        all_data     = loadFromFile(testSize, filename, shuffle_rows=True)
        xTest, yTest = all_data(testSize, return_test=True)

        # Loop over epocs
        for epoch in range(0, numberOfEpochs):
            epochIsDone = False
            # Loop over all the train data set in batches
            while not epochIsDone:
                # Read new data from loaded training data file
                # If you train with batchSize = size of entire train set, use keyword arg:
                # batchIsTrain=True        to skip shuffling each epoch!
                xTrain, yTrain, epochIsDone = all_data(batchSize, shuffle=False) # Get next batch of data

                # Loop through batches of the training set and adjust parameters
                # 'optimizer' still uses quick cost-function
                _, trainRMSE = sess.run([optimizer, RMSE], feed_dict={x: xTrain, y: yTrain})

                # If all training data has been seen "once" epoch is done
                if epochIsDone:
                    # Compute test set loss etc:
                    testRMSE  = sess.run(RMSE, feed_dict={x: xTest , y: yTest})
                    list_of_rmse_test.append(testRMSE)
                    list_of_rmse_train.append(trainRMSE)
                    # Print after performance after loss has decreased 5 % (or last epoch)
                    if bestTrainLoss-1E-14 > trainRMSE*1.05 or epoch == numberOfEpochs-1:
                        bestTrainLoss = trainRMSE
                        sys.stdout.write('\r' + ' '*60) # White out line
                        sys.stdout.write('\r%3d/%3d. RMSE: train: %10g, test: %10g\n' % \
                                        (epoch+1, numberOfEpochs, trainRMSE, testRMSE))
                        sys.stdout.flush()

                        # If saving is enabled, save the graph variables ('w', 'b')
                        if saveFlag:
                            saveFileName = "SavedRuns/run"
                            saver.save(sess, saveFileName, global_step=(epoch+1))
                            saveGraphFunc(sess, weights, biases, epoch, hiddenLayers, nNodes)

    sys.stdout.write('\n' + ' '*60) # White out line for sake of pretty command line output lol
    sys.stdout.flush()
    print "\n"
    np.savetxt("_testRMSE.txt", list_of_rmse_test)
    np.savetxt("_trainRMSE.txt", list_of_rmse_train)
    return weights, biases, neurons, bestTrainLoss/float(batchSize)

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
    # Number of symmetry functions describing local env. of atom i
    # _, input_vars = generate_symmfunc_input_Si_v1()
    _, symm_vec_length = generate_symmfunc_input_Si_Behler()

    # Make sure we start out with a fresh graph
    tf.reset_default_graph()

    # number of samples
    testSize  = 5000  # Should be 20-30 % of total train data
    batchSize = 1000   # Train size is determined by length of loaded file

    # numberOfNeighbors = 12 # of Si that are part of computation
    learning_rate     = 0.001

    # Always just one output = energy
    input_vars  = symm_vec_length   # Number of inputs = number of symm.funcs. used
    output_vars = 1                 # Potential energy of atom i

    # Size of neural network
    activation_function = "sigmoid"
    loss_function       = "L2"
    nNodes              = 5
    hiddenLayers        = 1

    x = tf.placeholder('float', shape=(None, input_vars),  name="x")
    y = tf.placeholder('float', shape=(None, output_vars), name="y")

    global saveFlag, neuralNetwork
    saveFlag      = False
    neuralNetwork = lambda data : nns.model(data,
                                            activation_function = activation_function,
                                            nNodes              = nNodes,
                                            hiddenLayers        = hiddenLayers,
                                            inputs              = input_vars,
                                            outputs             = output_vars,
                                            wInitMethod         = 'normal',
                                            bInitMethod         = 'normal')

    print "---------------------------------------"
    print "Using: learning rate:   %g" %learning_rate
    print "       # hidden layers: %d" %hiddenLayers
    print "       # nodes:         %d" %nNodes
    print "       activation.func: %s" %activation_function
    print "       loss_function:   %s" %loss_function
    print "       batch size:      %d" %batchSize
    print "       test size:       %d" %testSize
    print "---------------------------------------"

    # Start training!
    weights, biases, neurons, RMSE = train_neural_network(x, y, epochs, nNodes, hiddenLayers, batchSize, testSize, learning_rate, loss_function)

    print "---------------------------------------"
    print "Training was done with these settings:"
    print "       learning rate:   %g" %learning_rate
    print "       # hidden layers: %d" %hiddenLayers
    print "       # nodes:         %d" %nNodes
    print "       activation.func: %s" %activation_function
    print "       loss_function:   %s" %loss_function
    print "       batch size:      %d" %batchSize
    print "       test size:       %d" %testSize
    print "---------------------------------------"


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

    # Plot how the RMSE changed over time / epochs
    plotTestVsTrainLoss()

    # Run example of LJ:
    # t0 = timer()
    # example_Lennard_Jones(filename, epochs)
    # print "Wall clock time:", timer() - t0
