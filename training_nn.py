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

    print "#########################\n#########################\n"

    np.random.seed(1) # For testing

    r_low     = 0.
    r_high    = 1.8
    size      = 10
    neighbors = 20
    PES       = PES_Stillinger_Weber
    xyz_N     = createXYZ(r_low, r_high, size, neighbors,histogramPlot=False)
    Ep        = potentialEnergyGenerator(xyz_N, PES)
    G_funcs   = example_generate_G_funcs_input()

    print "Number of symmetry functions used to describe each atom i:", 51
    print "-------------------------------"
    for i in range(size):
        print "Stillinger Weber potential E:", Ep[i],"\n"
        xyz_i = xyz_N[:,:,i]
        G_i   = symmetryTransform(G_funcs, xyz_i)
        # print np.mean(G_i), np.max(G_i), np.min(G_i), np.sum(G_i)
        # print np.mean(xyz_N[i]), np.max(xyz_N[i]), np.min(xyz_N[i])
        print G_i, "\n"
        # if np.sum(G_i) < 1E-10:
        #     print "Input 'break' to quit"
        #     while True:
        #         command = raw_input()
        #         if command == "break":
        #             break
        #         exec(command)
        #     break
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
