"""
Train a neural network to approximate a 2-variable function
"""
from mpl_toolkits.mplot3d import Axes3D
import neuralNetworkXavier as nnx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
from math import pi
import numpy as np
import warnings
import sys,os
import glob
import time

# Stop matplotlib from giving FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="matplotlib")

def deleteOldData():
    # Does folders exist?
    foldersCheck = 0
    for filename in ['SavedRuns','SomePlots','SavedGraphs']:
        if not os.path.exists(filename):
            os.makedirs(filename) # Create folders
            foldersCheck += 1
    if not foldersCheck == 3: # No old data, since folders have just been created
        keepData = raw_input('\nDelete all previous plots, run- and graph-data? (y/yes/enter)')
        if keepData in ['y','yes','']:
            for someFile in glob.glob("SavedRuns/run*.dat"):
                os.remove(someFile)
            for someFile in glob.glob("SomePlots/fig*.png"):
                os.remove(someFile)
            for someFile in glob.glob("SavedGraphs/tf_graph_WB*.txt"):
                os.remove(someFile)

def scoreFunc1(x,y,return2Darray=False,returnMaxMin=False):
    z   = np.zeros(x.shape) if return2Darray else np.zeros(x.size)
    # Convert to polar coordinates if need be
    r     = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    #z = np.cos(4*pi*r)*np.exp(-2*r**2) + 0.3*np.cos(16*phi)*r**2
    #z = (0.8-r)**2 * (r<0.8)
    #f = np.sin(10*theta)*np.sin(4*pi*r)/2. * (r<=1)
    #z  = np.tanh(r-1) + np.exp(-8*r**2)
    #z  = np.exp(-0.1*r**2) * ((np.sin(10*theta)+1)/2.)**(1.2+np.cos(6*pi*r))
    z = np.sin(2*theta)
    if returnMaxMin:
        return np.max(z), np.min(z)
    return z # else


def functionNormData(trainSize,testSize,axMin=0,axMax=1,aroundError=[None,None]):
    score = scoreFunc1
    if not aroundError[0]:
        xRandArray = np.random.uniform(axMin, axMax, trainSize)
        yRandArray = np.random.uniform(axMin, axMax, trainSize)

        x_train = np.column_stack((xRandArray,yRandArray))
        y_train = score(xRandArray, yRandArray)
        y_train = y_train.reshape([trainSize,1])
    else:
        # Can train on data outside the interval of interest for precision
        """xLow = aroundError[0]*0.9; xHigh = aroundError[0]*1.1 # for random.uniform
        yLow = aroundError[1]*0.9; yHigh = aroundError[1]*1.1
        xRandArray = np.random.uniform(xLow, xHigh, int(trainSize/10.))
        yRandArray = np.random.uniform(yLow, yHigh, int(trainSize/10.))"""

        stddev = 0.5; trainSize2 = int(trainSize/10.)
        xRandArray = np.random.normal(aroundError[0], stddev, trainSize2)
        yRandArray = np.random.normal(aroundError[1], stddev, trainSize2)

        index = -1
        for x,y in zip(xRandArray,yRandArray):
            index += 1
            while True:
                if x < -1 or x > 1:
                    x                 = np.random.normal(aroundError[0], stddev, 1)
                    xRandArray[index] = x
                    continue
                if y < -1 or y > 1:
                    y                 = np.random.normal(aroundError[0], stddev, 1)
                    yRandArray[index] = y
                    continue
                break

        x_train = np.column_stack((xRandArray,yRandArray))
        y_train = score(xRandArray, yRandArray)
        y_train = y_train.reshape([trainSize2,1])

    xRandArray = np.random.uniform(axMin, axMax, testSize)
    yRandArray = np.random.uniform(axMin, axMax, testSize)

    x_test = np.column_stack((xRandArray,yRandArray))
    y_test = score(xRandArray, yRandArray)
    y_test = y_test.reshape([testSize,1])
    return x_train, y_train, x_test, y_test


def train_neural_network(x, epochs, nNodes, hiddenLayers, saveFlag, noPrint=False, \
                         plot=False, learning_rate_choice=0.001):
    # Begin session
    with tf.Session() as sess:
        # Setup of graph for later computation with tensorflow
        prediction, weights, biases, neurons = neuralNetwork(x)
        cost = tf.nn.l2_loss(tf.sub(prediction, y))
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_choice).minimize(cost)
        #optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate_choice).minimize(cost) # Careful with this (need low learn rate)

        # Number of cycles of feed-forward and backpropagation
        numberOfEpochs    = epochs;
        bestEpochTestLoss = -1
        startEpoch        = 0

        # Initialize variables or restore from file
        saver = tf.train.Saver(weights + biases, max_to_keep=None)
        sess.run(tf.global_variables_initializer())
        if loadFlag:
            loadFileName,startEpoch = findLoadFileName()
            numberOfEpochs         += startEpoch
            saver.restore(sess, "SavedRuns/"+loadFileName)

        bestTrainLoss = 1E100; bestTestLoss = 1E100; bestTestLossSinceLastPlot = 1E100; prevErr = 1E100
        triggerNewLine = False; saveNow = False; aroundError=[None,None]; aSolutionHasBeenSaved = False

        """
        #############################################
        Initiate "some" variables needed for plotting
        #############################################
        """
        plotLinearResolution = 50; axMin = -1; axMax = 1; dpi_choice = 120
        linPoints            = np.linspace(axMin,axMax,plotLinearResolution)
        zForPlot             = np.zeros((plotLinearResolution,plotLinearResolution))
        X, Y                 = np.meshgrid(linPoints, linPoints)
        chosenPercent        = 5
        minPlotDiff          = 2.
        plotEveryPctImprov   = 1.0 - chosenPercent/100
        pIncr                = 0.1
        plotCount            = 0
        zAxisMax, zAxisMin   = scoreFunc1(X,Y,return2Darray=True,returnMaxMin=True)
        if loadFlag:
            plotCount     = int(raw_input("Please input plotcount of last plot [int]: "))
            chosenPercent = 12 - pIncr*plotCount
            if chosenPercent < minPlotDiff:
                chosenPercent = minPlotDiff
        plotNow              = False
        colormap_choice      = cm.BrBG
        for col in xrange(plotLinearResolution):
            colValues = np.ones(plotLinearResolution)*linPoints[col]
            xyForPlot = np.column_stack((colValues,linPoints))
            zForPlot[:,col] = sess.run(prediction, feed_dict={x: xyForPlot})[:,0]
        Z   = scoreFunc1(X,Y,return2Darray=True)
        err = np.sum(np.abs(Z-zForPlot))/float(Z.size) # First average error

        # loop through epocs
        xTrain, yTrain, xTest, yTest = functionNormData(trainSize,testSize,axMin,axMax)

        for epoch in range(startEpoch, numberOfEpochs):
            # loop through batches and cover new data set for each epoch
            _, epochLoss = sess.run([optimizer, cost], feed_dict={x: xTrain, y: yTrain})

            # compute test set loss
            # THIS IS MAYBE USED TO TRAIN?? Gotta check!!
            #_, testCost = sess.run(prediction, feed_dict={x: xTest})
            testCost = sess.run(cost, feed_dict={x: xTest, y: yTest})

            # Create new data for next iteration
            xTrain, yTrain, xTest, yTest = functionNormData(trainSize,testSize,axMin,axMax)
            if epoch%100 == 0:
                triggerNewLine = True
                if True:#testCost/float(testSize) > 1E100:
                    pass
                    # xTrain, yTrain, xTest, yTest = functionNormData(trainSize,testSize,axMin,axMax) # BAD BAD BAD
                else:
                    # New experimental method: Give normal random numbers around largest error:
                    _, __, xTest2, yTest2 = functionNormData(trainSize,testSize)
                    errA = np.abs(sess.run(prediction, feed_dict={x: xTest2, y: yTest2})[:,0])
                    eAI  = np.argmax(errA)
                    #errIndex       = #np.unravel_index(errArray.argmax(), errArray.shape)
                    aroundError[0] = xTest[eAI,0] # (1000, 2)
                    aroundError[1] = xTest[eAI,1]
                    print "\n",aroundError, eAI
                    xTrain, yTrain, xTest, yTest = functionNormData(trainSize,testSize,axMin,axMax,aroundError)
            if epoch%80  == 0 and numberOfEpochs > epoch+80:
                if not noPrint:
                    sys.stdout.write('\r' + ' '*80) # White out line
                    sys.stdout.write('\rEpoch %5d out of %5d trainloss/N: %10g, testloss/N: %10g' % \
                          (epoch+1, numberOfEpochs, epochLoss/float(trainSize), testCost/float(testSize)))
                    sys.stdout.flush()
            if epoch%1000 == 0:#saveEveryNepochs: # Save the next version of the neural network that is better than any previous
                saveNow = True if epoch > 0 else False
            if testCost < bestTestLoss:
                bestTestLoss = testCost
                bestEpochTestLoss = epoch
                if triggerNewLine:
                    if not noPrint:
                        sys.stdout.write('\r' + ' '*80) # White out line
                        sys.stdout.write('\rEpoch %5d out of %5d trainloss/N: %10g, testloss/N: %10g\n' % \
                              (epoch+1, numberOfEpochs, epochLoss/float(trainSize), testCost/float(testSize)))
                        sys.stdout.flush()
                    triggerNewLine = False
            if plot and bestTestLossSinceLastPlot * plotEveryPctImprov > testCost and testCost/float(testSize) < 0.1 or epoch == 0:
                bestTestLossSinceLastPlot = testCost
                for col in xrange(plotLinearResolution):
                    colValues = np.ones(plotLinearResolution)*linPoints[col]
                    xyForPlot = np.column_stack((colValues,linPoints))
                    zForPlot[:,col] = sess.run(prediction, feed_dict={x: xyForPlot})[:,0]
                Z        = scoreFunc1(X,Y,return2Darray=True)
                errArray = np.abs(Z-zForPlot)
                err      = np.sum(errArray)/float(Z.size) # Average error
                if True and epoch > 0:#err < prevErr: # Plot only if error is smaller!
                    prevErr = err
                    chosenPercent     -= pIncr if chosenPercent > minPlotDiff else 0.0
                    plotEveryPctImprov = 1.0 - chosenPercent/100.0 # More plots later in the computation

                    """fig = plt.figure()
                    a = fig.add_subplot(1,2,1, projection='3d')
                    surf = a.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=colormap_choice,
                                           linewidth=0, antialiased=False)
                    a.set_title('Actual function')
                    a.set_xlim3d(0, 1); a.set_ylim3d(0, 1); a.set_zlim3d(zAxisMin*1.2, zAxisMax*1.2)
                    a.set_xlabel('X axis'); a.set_ylabel('Y axis'); a.set_zlabel('Z axis')

                    a = fig.add_subplot(1,2,2, projection='3d')
                    surf = a.plot_surface(X, Y, zForPlot, rstride=1, cstride=1, cmap=colormap_choice,
                                           linewidth=0, antialiased=False)
                    a.set_title('NN approx. Error: %.2e' %err)
                    a.set_xlim3d(0, 1); a.set_ylim3d(0, 1); a.set_zlim3d(zAxisMin*1.2, zAxisMax*1.2)
                    a.set_xlabel('X axis'); a.set_ylabel('Y axis'); a.set_zlabel('Z axis')
                    surf.set_clim(zAxisMin,zAxisMax)

                    plt.savefig("SomePlots/fig%d_epoch%d.png" %(plotCount,epoch), dpi=dpi_choice)
                    plotCount += 1
                    plt.close()"""

                    fig = plt.figure()
                    a=fig.add_subplot(2,3,1)
                    imgplot = plt.imshow(Z, cmap=colormap_choice)
                    a.set_title('Actual function')
                    a=fig.add_subplot(2,3,2)
                    imgplot = plt.imshow(zForPlot, cmap=colormap_choice)
                    imgplot.set_clim(zAxisMin,zAxisMax)
                    a.set_title('NN approx. func.')
                    a=fig.add_subplot(2,3,3)
                    imgplot = plt.imshow(Z-zForPlot, cmap=colormap_choice)
                    a.set_title('Error NN: %.2e' %err)
                    a = fig.add_subplot(2,3,4, projection='3d')
                    surf = a.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=colormap_choice,
                                           linewidth=0, antialiased=False)
                    a = fig.add_subplot(2,3,5, projection='3d')
                    a.set_xlim3d(axMin, axMax); a.set_ylim3d(axMin, axMax); a.set_zlim3d(zAxisMin*1.2, zAxisMax*1.2)
                    surf = a.plot_surface(X, Y, zForPlot, rstride=1, cstride=1, cmap=colormap_choice,
                                           linewidth=0, antialiased=False)
                    a.set_xlim3d(axMin, axMax); a.set_ylim3d(axMin, axMax); a.set_zlim3d(zAxisMin*1.2, zAxisMax*1.2)
                    surf.set_clim(zAxisMin,zAxisMax)
                    a = fig.add_subplot(2,3,6, projection='3d')
                    surf = a.plot_surface(X, Y, Z-zForPlot, rstride=1, cstride=1, cmap=colormap_choice,
                                           linewidth=0, antialiased=False)

                    plt.savefig("SomePlots/fig%d_epoch%d.png" %(plotCount,epoch), dpi=dpi_choice)
                    plotCount += 1
                    plt.close()
                # If saving is enabled, save the graph variables ('w', 'b')
                if saveNow and saveFlag:
                    aSolutionHasBeenSaved = True
                    saveNow = False
                    saveFileName = "SavedRuns/run" + str(epoch) + ".dat"
                    saver.save(sess, saveFileName, write_meta_graph=False)
                    if saveGraph: # Write the current best weights and biases to file
                        saveGraphFunc(sess, weights, biases, epoch)
        # Write weights and biases to file when training is finished, if this is not yet done
        if not aSolutionHasBeenSaved and saveGraph:
            saveGraphFunc(sess, weights, biases, epoch)
    sys.stdout.write('\r' + ' '*80) # White out line for sake of pretty command line output lol
    return weights, biases, neurons, bestTestLoss/float(testSize)

def saveGraphFunc(sess, weights, biases, epoch):
    try:
        os.mkdir("SavedGraphs")
    except:
        pass
    saveGraphName = "SavedGraphs/tf_graph_WB_%d.txt" %(epoch)
    with open(saveGraphName, 'w') as outFile:
        outStr = "%1d %1d %s" % (hl_list[0], node_list[0], "sigmoid")
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
totalEpochs = 5000
loadFileName = None; saveFileName = None
global loadFlag; global plotFlag
saveFlag, loadFlag, plotFlag, saveGraph = False, False, False, False
if cmdArgs < 1: # No input from user, running default
    pass
else: # We need to parse the command line args
    for argIndex in range(1,cmdArgs+1):
        arg = sys.argv[argIndex]
        if isinteger(arg):
            totalEpochs = int(arg)
        if arg in ["-graph", "-savegraph"]:
            saveGraph = True
            print "Best version of the neural net will be saved (weights & biases) as numbers"
            if not saveFlag and "-save" not in sys.argv:
                arg = "-save"
        if arg == "-save":
            saveFlag = True
            print "The neural network will be stored periodically for easy restart of calc."
        if arg == "-load":
            loadFlag = True
            print "Loading latest neural network as starting point"
        if arg == "-plot":
            plotFlag = True
        if arg in ["h","help","-h","-help","--h","--help"]:
            print "Command line arguments possible: -save, -load, -plot"
            print "Also, you can specify the number of epochs. Example:"
            print ">>> python funcApproxNN2vars.py 5000 -load -plot"
            sys.exit()

if not loadFlag:
    deleteOldData()

global trainData

# reset so that variables are not given new names
tf.reset_default_graph()

# number of samples per batch
testSize  = 100
trainSize = 1000

# number of inputs and outputs
inputs  = 2
outputs = 1

x = tf.placeholder('float', [None, inputs],  name="x")
y = tf.placeholder('float', [None, outputs], name="y")

#neuralNetwork = lambda data : nnx.modelRelu(data, nNodes=nNodes, hiddenLayers=hiddenLayers, inputs=2, wInitMethod='normal', bInitMethod='normal')
#neuralNetwork = lambda data : nnx.modelTanh(data, nNodes=nNodes, hiddenLayers=hiddenLayers,inputs=2, wInitMethod='normal', bInitMethod='normal')
neuralNetwork = lambda data : nnx.modelSigmoid(data, nNodes=nNodes, hiddenLayers=hiddenLayers,inputs=2, wInitMethod='normal', bInitMethod='normal')

print "---------------------------------------"
learning_rate_choice = 0.001 # Default for AdamOptimizer is 0.001
testCases = 0
print "Learning rate:", learning_rate_choice

hl_list   = [20]
node_list = [100]

# If saving graph, you cannot test multiple values:
if saveFlag or loadFlag or saveGraph or plotFlag:
    if len(hl_list) > 1 or len(node_list) > 1:
        print "You cannot test multiple versions of HL and neurons AND save graph."
        hl_list   = [int(raw_input("Input the number of hidden layers: "))]
        node_list = [int(raw_input("Input the number of neurons per layer: "))]
noPrint = True if len(node_list)+len(node_list) > 2 else False
epochlossPerNPrev = 1e100   # "Guaranteed" worse than anything
nNodesBest = 0; hLBest = 0; epochBest = 0

start_time = time.time()

for hiddenLayers in hl_list:
    for nNodes in node_list:
        testCases += 1
        weights, biases, neurons, epochlossPerN = train_neural_network(x, totalEpochs, \
                nNodes, hiddenLayers, saveFlag, noPrint, plot=plotFlag,learning_rate_choice=learning_rate_choice)
        print "\rHid.layers: %2.d, nodes/l: %2.d, epochs: %d, loss/N: %e" %(hiddenLayers,nNodes,totalEpochs,epochlossPerN)
        if not noPrint:
            print " "
        if epochlossPerN < epochlossPerNPrev:
            epochlossPerNPrev = epochlossPerN
            nNodesBest = nNodes
            hLBest     = hiddenLayers
            epochBest  = totalEpochs
if testCases > 1: # Print out testing different hidden layers and number of nodes
    print "---------------------------------------"
    print "Best combination found after %d epochs:" %epochBest
    print "Layers: %d, nodes/layer: %d, loss/N: %e" %(hLBest,nNodesBest,epochlossPerNPrev)
    print "---------------------------------------"

# Time taken
print "--- %s seconds ---" % round(time.time() - start_time,2)
