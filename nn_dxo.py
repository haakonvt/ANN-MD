"""
Train a neural network to approximate a function from DXOMARK.com
that decides the "Global Score" value.

# Data: https://www.dxomark.com/cameras#hideAdvancedOptions=false&viewMode=list&yDataType=rankDxo
# Extracted data: https://regex101.com/r/vF9pS7/1
"""
import neuralNetworkXavier as nnx
import matplotlib.pyplot as plt
import tensorflow as tf
import datetime as time
import numpy as np
import shutil, sys, os

""" First time run:
x = np.loadtxt("values_dxo.dat"); y = len(x)/4
x = x.reshape((y,4))

totalDataPoints = x.shape[0] #(323, 4)

np.savetxt("values_array_dxo.dat",x,delimiter=',\t',fmt='%.1f')
np.save("raw_data_dxo",x)"""

def functionData(trainSize,testSize):
    """
    Create train input for NN and test data
    """
    x_train = trainData[:,1:]
    x_train = x_train.reshape([trainSize,3])
    y_train = trainData[:,0]
    y_train = y_train.reshape([trainSize,1])

    x_test  = xRand[:,1:]
    x_test = x_test.reshape([testSize,3])
    y_test  = xRand[:,0]
    y_test = y_test.reshape([testSize,1])
    return x_train, y_train, x_test, y_test


def functionDataCreated(trainSize,testSize):
    """Test the neural network with data generated randomly, with approximately the same
    distribution, but with a known function to find the Global Score"""
    score = lambda x,y,z: np.around((98/3.0) * (x/float(26.5) + y/float(14.8) + z/float(3702)), decimals=1)

    xRandArray = np.around(np.random.uniform(17.9, 26.5,   trainSize), decimals=1)
    yRandArray = np.around(np.random.uniform(9.6,  14.8,   trainSize), decimals=1)
    zRandArray = np.around(np.random.uniform(68.0, 3702.0, trainSize), decimals=1)

    x_train = np.column_stack((xRandArray,yRandArray,zRandArray))
    y_train = score(xRandArray, yRandArray, zRandArray)
    y_train = y_train.reshape([trainSize,1])

    xRandArray = np.around(np.random.uniform(17.9, 26.5,   testSize), decimals=1)
    yRandArray = np.around(np.random.uniform(9.6,  14.8,   testSize), decimals=1)
    zRandArray = np.around(np.random.uniform(68.0, 3702.0, testSize), decimals=1)

    x_test = np.column_stack((xRandArray,yRandArray,zRandArray))
    y_test = score(xRandArray, yRandArray, zRandArray)
    y_test = y_test.reshape([testSize,1])
    return x_train, y_train, x_test, y_test


def train_neural_network(x, epochs, nNodes, hiddenLayers, saveFlag, plot=False, learning_rate_choice=0.001):
    # begin session
    with tf.Session() as sess:
        # pass data to network and receive output
        prediction, weights, biases, neurons = neuralNetwork(x)

        cost = tf.nn.l2_loss( tf.sub(prediction, y) )

        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_choice).minimize(cost)

        # number of cycles of feed-forward and backpropagation
        numberOfEpochs    = epochs;
        bestEpochTestLoss = -1
        startEpoch        = 0

        # initialize variables or restore from file
        saver = tf.train.Saver(weights + biases, max_to_keep=None)
        sess.run(tf.initialize_all_variables())
        if loadFlag:
            loadFileName,startEpoch = findLoadFileName()
            numberOfEpochs         += startEpoch
            saver.restore(sess, "SavedRuns/"+loadFileName)

        bestTrainLoss = 1E100; bestTestLoss = 1E100; triggerNewLine = False; saveNow = False
        # loop through epocs
        for epoch in range(startEpoch, numberOfEpochs):
            # loop through batches and cover whole data set for each epoch
            _, epochLoss = sess.run([optimizer, cost], feed_dict={x: xTrain, y: yTrain})

            # compute test set loss
            # THIS IS MAYBE USED TO TRAIN?? Gotta check!!
            _, testCost = sess.run([optimizer, cost], feed_dict={x: xTest, y: yTest})

            if (epoch+1)%int(numberOfEpochs/800.) == 0:
                    triggerNewLine = True
            if epoch%5000 == 0:#saveEveryNepochs: # Save the next version of the neural network that is better than any previous
                saveNow = True
            if testCost < bestTestLoss: # and testCost//float(testSize) < 1.0:
                bestTestLoss = testCost
                bestEpochTestLoss = epoch
                sys.stdout.write('\rEpoch %5d out of %5d trainloss/N: %10g, testloss/N: %10g' % \
                      (epoch+1, numberOfEpochs, epochLoss/float(trainSize), testCost/float(testSize)))
                sys.stdout.flush()
                if triggerNewLine:
                    #sys.stdout.write(' '*80) # White out line
                    sys.stdout.write('\rEpoch %5d out of %5d trainloss/N: %10g, testloss/N: %10g\n' % \
                          (epoch+1, numberOfEpochs, epochLoss/float(trainSize), testCost/float(testSize)))
                    sys.stdout.flush()
                    triggerNewLine = False
                if plot:
                    yy     = sess.run(prediction, feed_dict={x: xTest})
                    error  = yy-yTest
                    yy2    = sess.run(prediction, feed_dict={x: xTrain})
                    error2 = yy2-yTrain
                # If saving is enabled, save the graph variables ('w', 'b')
                if saveNow and saveFlag:
                    saveNow = False
                    saveFileName = "SavedRuns/run" + str(epoch) + ".dat"
                    saver.save(sess, saveFileName)

        if plot:
            print "\nResults for test data:"
            #yy = sess.run(prediction, feed_dict={x: xTest})
            #error = yy-yTest
            accuracy = np.int_(np.round(error)) # 0 means correct guess
            perfResPercent = (len(accuracy)-len(np.nonzero(accuracy)[0]))*100.0/len(accuracy)
            print "Accuracy of perfect predictions: %.1f %%" %perfResPercent
            print np.sort(np.transpose(np.abs(accuracy)))
            plt.hist(error,bins=13)
            plt.xlabel('Test case (number)')
            plt.ylabel('Error: Prediciton - Exact answer')
            np.savetxt("nndxo_err_test.dat",(yy, yTest, error))
            plt.show()

            print "Results for train data:"
            accuracy = np.int_(np.round(error2)) # 0 means correct guess
            perfResPercent = (len(accuracy)-len(np.nonzero(accuracy)[0]))*100.0/len(accuracy)
            print "Accuracy of perfect predictions: %.1f %%" %perfResPercent
            print np.sort(np.transpose(np.abs(accuracy)))
            plt.hist(error2,bins=13)
            plt.xlabel('')
            plt.ylabel('Error: round(Prediciton - Exact answer)')
            np.savetxt("nndxo_err_train.dat",(yy2, yTrain, error2))
            plt.show()

    return weights, biases, neurons, bestTestLoss/float(testSize)


def findLoadFileName():
    if os.path.isdir("SavedRuns"):
        file_list = []
        for a_file in os.listdir("SavedRuns"):
            if a_file[0] == ".": # Ignore hidden files
                continue
            elif a_file[-4:] == "meta" or a_file == "checkpoint":
                continue
            else:
                file_list.append(int(a_file[3:-4]))
        if len(file_list) == 0:
            print "No previous runs exits. Exiting..."; sys.exit()
        newest_file = "run" + str(np.max(file_list)) + ".dat"
        startEpoch  = np.max(file_list)
        print 'Model restored from file:', newest_file
        return newest_file,startEpoch
    else:
        os.makedirs("SavedRuns")
        print "Created 'SavedRuns'-directory. No previous runs exits. Exiting..."; sys.exit()


def isinteger(x):
    try:
        test = np.equal(np.mod(x, 1), 0)
    except:
        try:
            test = np.equal(np.mod(int(x), 1), 0)
        except:
            return False
    return test


#--------------#
##### main #####
#--------------#
raw_data = np.load("raw_data_dxo.npy")
cmdArgs  = len(sys.argv)-1
totalEpochList = [1000]
loadFileName = None; saveFileName = None
global loadFlag; global plotFlag
saveFlag, loadFlag, plotFlag = False, False, False
if cmdArgs < 1: # No input from user, running default
    pass
else: # We need to parse the command line args
    for argIndex in range(1,cmdArgs+1):
        #print "Index:",argIndex, "content:", sys.argv[argIndex]
        arg = sys.argv[argIndex]
        if isinteger(arg):
            if int(arg)<800:
                print "Number of epochs must be greater than 800. Exiting..."; sys.exit()
            totalEpochList = [ (int(arg)) ]
        if arg == "-save":
            saveFlag = True
            print "The neural network (weights & biases) will be saved periodically"
        if arg == "-load":
            loadFlag     = True
            print "Loading latest neural network as starting point"
        if arg == "-plot":
            plotFlag = True
            print "Will plot best prediction after the simulation"
        if arg in ["h","help","-h","-help","--h","--help"]:
            print "Command line arguments possible: -save, -load and -plot."
            print "Also, you can specify the number of epochs (>800). Ex.:"
            print ">>> python nn_dxo.py 5000 -load -plot"
            sys.exit()

# Pick out test data randomly from the data
totalDataPoints = raw_data.shape[0]
randRows = np.random.choice(totalDataPoints, 80, replace=False)
global xRand
xRand = raw_data[randRows,:]

# Pick out the rest for training data
leftoverRows = [i for i in range(totalDataPoints) if i not in randRows]

global trainData
trainData = raw_data[leftoverRows,:]
np.random.shuffle(trainData) # Shuffle rows of the data to minimize impact of ordering of the data

# reset so that variables are not given new names
tf.reset_default_graph()

# number of samples
testSize  = 80 # Out of a hat, we found the number 80
trainSize = totalDataPoints - testSize

# get real world input
#xTrain, yTrain, xTest, yTest = functionData(trainSize,testSize)

# get random input
#xTrain, yTrain, xTest, yTest = functionDataCreated(trainSize,testSize)

# Test with way more data
xTrain, yTrain, xTest, yTest = functionDataCreated(2000,700)

# number of inputs and outputs
inputs  = 3
outputs = 1

x = tf.placeholder('float', [None, inputs],  name="x")
y = tf.placeholder('float', [None, outputs], name="y")

neuralNetwork = lambda data : nnx.modelRelu(data, nNodes=nNodes, hiddenLayers=hiddenLayers, wInitMethod='normal', bInitMethod='normal')
#neuralNetwork = lambda data : nnx.modelSigmoid(data, nNodes=nNodes, hiddenLayers=hiddenLayers,wInitMethod='normal', bInitMethod='normal')

print "---------------------------------------"
learning_rate_choice = 0.001 # Default for AdamOptimizer is 0.001
testCases = 0;
print "Learning rate:", learning_rate_choice
print "(It might take some time before anything meaningful is printed)"

epochlossPerNPrev = 1e100   # "Guaranteed" worse than anything
nNodesBest = 0; hLBest = 0; epochBest = 0
for hiddenLayers in [10]:
    for nNodes in [12]:
        for epochs in totalEpochList:
            testCases += 1
            weights, biases, neurons, epochlossPerN = train_neural_network(x, epochs, \
                    nNodes, hiddenLayers, saveFlag, plot=plotFlag,learning_rate_choice=learning_rate_choice)
            print "\nHid.layers: %2.d, nodes/l: %2.d, epochs: %d, loss/N: %f" %(hiddenLayers,nNodes,epochs,epochlossPerN)
            if epochlossPerN < epochlossPerNPrev:
                epochlossPerNPrev = epochlossPerN
                nNodesBest = nNodes
                hLBest     = hiddenLayers
                epochBest  = epochs
if testCases > 1: # Print out testing different hdn layers and nmbr of nodes
    print "---------------------------------------"
    print "Best combination found after %d epochs:" %epochBest
    print "Layers: %d, nodes/layer: %d, loss/N: %e" %(hLBest,nNodesBest,epochlossPerNPrev)
    print "---------------------------------------"
