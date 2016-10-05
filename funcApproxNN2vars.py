"""
Train a neural network to approximate a 2-variable
"""
import neuralNetworkXavier as nnx
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import sys,os

def scoreFunc1(x,y,return2Darray=False):
    if not return2Darray:
        N   = len(x)
        res = np.zeros(N)
        for i in range(N):
            xi = x[i]; yi = y[i]
            if xi > 0.3 and xi < 0.7 and yi > 0.3 and yi < 0.7:
                res[i] = np.sin(x[i]*4*np.pi)*np.cos(y[i]*4*np.pi)
            else:
                res[i] = np.sin(x[i]*2*np.pi)*np.cos(y[i]*2*np.pi)
        return res
    else:
        X, Y = np.meshgrid(x, y)
        Z    = np.zeros(X.shape)
        N   = len(x)
        for j in range(N):
            for i in range(N):
                if X[i,j] > 0.3 and X[i,j] < 0.7 and Y[i,j] > 0.3 and Y[i,j] < 0.7:
                    Z[i,j] = np.sin(X[i,j]*4*np.pi)*np.cos(Y[i,j]*4*np.pi)
                else:
                    Z[i,j] = np.sin(X[i,j]*2*np.pi)*np.cos(Y[i,j]*2*np.pi)
        return Z.transpose()


def functionNormData(scorefunction,trainSize,testSize):
    score = scorefunction

    xRandArray = np.random.uniform(0, 1, trainSize)
    yRandArray = np.random.uniform(0, 1, trainSize)

    x_train = np.column_stack((xRandArray,yRandArray))
    y_train = score(xRandArray, yRandArray)
    y_train = y_train.reshape([trainSize,1])

    xRandArray = np.random.uniform(0, 1, testSize)
    yRandArray = np.random.uniform(0, 1, testSize)

    x_test = np.column_stack((xRandArray,yRandArray))
    y_test = score(xRandArray, yRandArray)
    y_test = y_test.reshape([testSize,1])
    return x_train, y_train, x_test, y_test


def train_neural_network(x, epochs, nNodes, hiddenLayers, saveFlag, plot=False, learning_rate_choice=0.001):
    # begin session
    with tf.Session() as sess:
        # pass data to network and receive output
        prediction, weights, biases, neurons = neuralNetwork(x)

        cost = tf.nn.l2_loss(tf.sub(prediction, y))

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

            if epoch%800 == 0:
                triggerNewLine = True
            if epoch%80  == 0 and numberOfEpochs > epoch+80:
                sys.stdout.write('\r' + ' '*80) # White out line
                sys.stdout.write('\rEpoch %5d out of %5d trainloss/N: %10g, testloss/N: %10g' % \
                      (epoch+1, numberOfEpochs, epochLoss/float(trainSize), testCost/float(testSize)))
                sys.stdout.flush()
            if epoch%5000 == 0:#saveEveryNepochs: # Save the next version of the neural network that is better than any previous
                saveNow = True if epoch > 0 else False
            if testCost < bestTestLoss: # and testCost//float(testSize) < 1.0:
                bestTestLoss = testCost
                bestEpochTestLoss = epoch
                if triggerNewLine:
                    sys.stdout.write('\r' + ' '*80) # White out line
                    sys.stdout.write('\rEpoch %5d out of %5d trainloss/N: %10g, testloss/N: %10g\n' % \
                          (epoch+1, numberOfEpochs, epochLoss/float(trainSize), testCost/float(testSize)))
                    sys.stdout.flush()
                    triggerNewLine = True if plot else False
                if plot and triggerNewLine:
                    triggerNewLine = False
                    linPoints = np.linspace(0,1,101)
                    zForPlot  = np.zeros((101,101))
                    for row in xrange(101):
                        rowValues = np.ones(101)*linPoints[row]
                        xyForPlot = np.column_stack((rowValues,linPoints))
                        zForPlot[row,:] = sess.run(prediction, feed_dict={x: xyForPlot})[:,0]
                # If saving is enabled, save the graph variables ('w', 'b')
                if saveNow and saveFlag:
                    saveNow = False
                    saveFileName = "SavedRuns/run" + str(epoch) + ".dat"
                    saver.save(sess, saveFileName, write_meta_graph=False)

        if plot:
            print "\n\nResults for test data:"
            Z = scoreFunc1(linPoints,linPoints,return2Darray=True)

            fig = plt.figure()
            a=fig.add_subplot(1,2,1)
            imgplot = plt.imshow(Z)
            a.set_title('Actual function')

            a=fig.add_subplot(1,2,2)
            imgplot = plt.imshow(zForPlot)
            a.set_title('NN approx. func.')
            plt.show()

    sys.stdout.write('\r' + ' '*80) # White out line
    return weights, biases, neurons, bestTestLoss/float(testSize)


def findLoadFileName():
    if os.path.isdir("SavedRuns"):
        file_list = []
        for a_file in os.listdir("SavedRuns"):
            if a_file[0] == ".": # Ignore hidden files
                continue
            elif a_file == "checkpoint":
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
global trainData

# reset so that variables are not given new names
tf.reset_default_graph()

# number of samples
testSize  = 1000
trainSize = 4000

# get real world input
xTrain, yTrain, xTest, yTest = functionNormData(scoreFunc1,trainSize,testSize)

# number of inputs and outputs
inputs  = 2
outputs = 1

x = tf.placeholder('float', [None, inputs],  name="x")
y = tf.placeholder('float', [None, outputs], name="y")

#neuralNetwork = lambda data : nnx.modelRelu(data, nNodes=nNodes, hiddenLayers=hiddenLayers, wInitMethod='normal', bInitMethod='normal')
neuralNetwork = lambda data : nnx.modelSigmoid(data, nNodes=nNodes, hiddenLayers=hiddenLayers,inputs=2, wInitMethod='normal', bInitMethod='normal')

print "---------------------------------------"
learning_rate_choice = 0.001 # Default for AdamOptimizer is 0.001
testCases = 0
print "Learning rate:", learning_rate_choice

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
if testCases > 1: # Print out testing different hidden layers and number of nodes
    print testCases, totalEpochList
    print "---------------------------------------"
    print "Best combination found after %d epochs:" %epochBest
    print "Layers: %d, nodes/layer: %d, loss/N: %e" %(hLBest,nNodesBest,epochlossPerNPrev)
    print "---------------------------------------"
