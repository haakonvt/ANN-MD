"""
Train a neural network to approximate a potential energy surface
with the use of symmetry functions that transform xyz-data.
"""

from plot_tools import plotTestVsTrainLoss
from file_management import saveGraphFunc, keepData, timeStamp
from timeit import default_timer as timer # Best timer indep. of system
import neural_network_setup as nns
from create_train_data import *
from symmetry_transform import *
import tensorflow as tf
import numpy as np
import sys,os
from math import sqrt

def train_neural_network(x, y, epochs, nNodes, hiddenLayers, batchSize, testSize,
                     learning_rate=0.001, loss_function="L2", activation_function="sigmoid",
                     potential_name=""):
    # Number of cycles of feed-forward and backpropagation
    numberOfEpochs = epochs
    bestTrainLoss  = 1E100
    p_imrove       = 1.2  # Write out how training is going after this improvment in loss
    print_often    = False
    datetime_stamp = timeStamp()
    save_dir       = "Important_data/Trained_networks/" + datetime_stamp + "-" + potential_name
    os.makedirs(save_dir)  # Create folder if not present
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

        # Initialize all graph variables
        sess.run(tf.global_variables_initializer())

        # Create a save-op: (keeps only last iteration, but also one per 30 min trained)
        saver = tf.train.Saver(weights + biases, max_to_keep=1, keep_checkpoint_every_n_hours=0.5)

        # If loadflag specified, load a pre-trained net
        if loadPath: # If loadPath is not length zero
            saver.restore(sess, loadPath)

        # Save first version of the net
        if saveFlag:
            saveFileName = save_dir
            saveFileName += "/run"
            saver.save(sess, saveFileName, global_step=0)
            saveGraphFunc(sess, weights, biases, 0, hiddenLayers, nNodes, save_dir, activation_function)

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
                sess.run(optimizer, feed_dict={x: xTrain, y: yTrain})

                # If all training data has been seen "once" epoch is done
                if epochIsDone:
                    # Compute test set loss etc:
                    testRMSE  = sess.run(RMSE, feed_dict={x: xTest , y: yTest})
                    trainRMSE = sess.run(RMSE, feed_dict={x: xTrain , y: yTrain})
                    list_of_rmse_test.append(testRMSE)
                    list_of_rmse_train.append(trainRMSE)
                    # Print after performance after loss has decreased 5 % (or last epoch)
                    if bestTrainLoss-1E-14 > trainRMSE*p_imrove or epoch == numberOfEpochs-1:
                        if not print_often and trainRMSE < 0.008:
                            p_imrove    = 1.015 # Write out progress more often
                            print_often = True
                        bestTrainLoss = trainRMSE
                        sys.stdout.write('\r' + ' '*60) # White out line
                        sys.stdout.write('\r%3d/%3d. RMSE: train: %10g, test: %10g\n' % \
                                        (epoch+1, numberOfEpochs, trainRMSE, testRMSE))
                        sys.stdout.flush()

                        # If saving is enabled, save the graph variables ('w', 'b')
                        if saveFlag and epoch > 0.75*(numberOfEpochs-1): # When 25 % epochs left, write TF restart file
                            saver.save(sess, saveFileName, global_step=(epoch+1))
                        if epoch == numberOfEpochs-1: # Save last edition of NN (weights & biases)
                            saveGraphFunc(sess, weights, biases, epoch+1, hiddenLayers, nNodes, save_dir, activation_function)
    sys.stdout.write('\n' + ' '*60) # White out line for sake of pretty command line output lol
    sys.stdout.flush(); print "\n"
    np.savetxt(save_dir +  "/testRMSE.txt", list_of_rmse_test)
    np.savetxt(save_dir + "/trainRMSE.txt", list_of_rmse_train)

    # Plot how the RMSE changed over time / epochs
    plotTestVsTrainLoss()

    # Keep data from this simulation/training
    keepData(save_dir)
    return weights, biases, neurons, bestTrainLoss/float(batchSize)

def example_Stillinger_Weber():
    # Get filename of traindata and number of epochs from command line
    global filename, saveFlag,loadPath
    filename, epochs, nodes, hiddenLayers, saveFlag, loadPath = parseCommandLine()

    # Number of symmetry functions describing local env. of atom i
    _, symm_vec_length = generate_symmfunc_input_Si_Behler()

    # Make sure we start out with a fresh graph
    tf.reset_default_graph()

    # number of samples
    testSize  = 5000  # Should be 20-30 % of total train data
    batchSize = 1000   # Train size is determined by length of loaded file

    # Set the learning rate. Standard value: 0.001
    learning_rate = 0.001

    # Always just one output => energy
    input_vars  = symm_vec_length   # Number of inputs = number of symm.funcs. used
    output_vars = 1                 # Potential energy of atom i

    # Choice of loss- and activation function of the neural network
    activation_function = "sigmoid"
    loss_function       = "L2"

    # Create placeholders for the input and output variables
    x = tf.placeholder('float', shape=(None, input_vars),  name="x")
    y = tf.placeholder('float', shape=(None, output_vars), name="y")

    global neuralNetwork
    neuralNetwork = lambda data : nns.model(data,
                                            activation_function = activation_function,
                                            nNodes              = nodes,
                                            hiddenLayers        = hiddenLayers,
                                            inputs              = input_vars,
                                            outputs             = output_vars,
                                            wInitMethod         = 'normal',
                                            bInitMethod         = 'normal')

    print "---------------------------------------"
    print "Using: learning rate:   %g" %learning_rate
    print "       # hidden layers: %d" %hiddenLayers
    print "       # nodes:         %d" %nodes
    print "       activation.func: %s" %activation_function
    print "       loss_function:   %s" %loss_function
    print "       batch size:      %d" %batchSize
    print "       test size:       %d" %testSize
    print "---------------------------------------"

    # Let the training commence!
    weights, biases, neurons, RMSE = train_neural_network(x, y,
                                                        epochs,
                                                         nodes,
                                                  hiddenLayers,
                                                     batchSize,
                                                      testSize,
                                                 learning_rate,
                                                 loss_function,
                                           activation_function,
                                           potential_name="SW")

    print "---------------------------------------"
    print "Training was done with these settings:"
    print "       learning rate:   %g" %learning_rate
    print "       # hidden layers: %d" %hiddenLayers
    print "       # nodes:         %d" %nodes
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
    def error_and_exit():
        print "Usage:"
        print ">>> python training_nn.py FILENAME   EPOCHS NODES HDNLAYER SAVE LOAD"
        print ">>> python training_nn.py SW_dat.txt 5000   30    5        True False"
        sys.exit(0)
    def bool_from_user_input(inp):
        if inp in ['True', 'TRUE', 'true']:
             return True
        elif inp in ['False', 'FALSE', 'false']:
             return False
        else:
            error_and_exit()
    if len(sys.argv) < 7:
        error_and_exit()
    else:
        filename = str(sys.argv[1])
        epochs   = int(sys.argv[2])
        nodes    = int(sys.argv[3]) # Nodes per hidden layer
        hdnlayrs = int(sys.argv[4]) # Number of hidden layers (In addition to input- and output layer)
        saveFlag = bool_from_user_input(str(sys.argv[5]))
        loadPath = bool_from_user_input(str(sys.argv[6])) # Still just a bool
        if loadPath:
            loadPath = raw_input("Please specify location of loadfile (whole path): ")
    return filename, epochs, nodes, hdnlayrs, saveFlag, loadPath

if __name__ == '__main__':
    # Example 1: Argon
    # Potential: Lennard-Jones:
    if False:
        t0 = timer()
        example_Lennard_Jones(filename, epochs)
        print "Wall clock time:", timer() - t0

    # Example 2: Silicon
    # Potential: Stillinger-Weber
    if True:
        t0 = timer()
        save_dir = example_Stillinger_Weber()
        print "Wall clock time:", timer() - t0

    # Example 3: SiC (Silicon Carbide)
    # Potential: Vashista
    if False:
        pass
