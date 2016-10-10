"""
Train a neural network to approximate a 2-variable function
"""
from mpl_toolkits.mplot3d import Axes3D
import neuralNetworkXavier as nnx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import tensorflow as tf
import numpy as np
import warnings
import sys,os
import glob
#import math

# Stop matplotlib from giving FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning, module="matplotlib")
keepData = raw_input('\nDelete all previous plots and run-data? (y/n)')
if keepData in ['y','yes']:
    for someFile in glob.glob("SavedRuns/run*.dat"):
        os.remove(someFile)
    for someFile in glob.glob("SomePlots/fig*.png"):
        os.remove(someFile)

def scoreFunc1(x,y,return2Darray=False,returnMaxMin=False):
    N = x.size
    if return2Darray:
        x,y = np.meshgrid(x, y)
        z   = np.zeros(x.shape)
        for i in xrange(N):
            for j in xrange(N):
                xi = x[i,j]; yi = y[i,j]
                if   xi < 0.5 and yi < 0.5:
                    z[i,j] = (yi-0.5) if yi >= xi else (xi-0.5)
                elif xi > 0.5 and yi < 0.5:
                    z[i,j] = 0.6*np.sin(xi*4*np.pi)*np.sin(yi*4*np.pi)
                elif xi < 0.5 and yi > 0.5:
                    z[i,j] = np.exp(-80*(yi-0.75)**2)*np.exp(-80*(xi-0.25)**2)
                else: #xi > 0.5 and yi > 0.5:
                    z[i,j] = -6*((xi-0.75)**2 - (yi-0.75)**2)
    else:
        z = np.zeros(N)
        for i in xrange(N):
            xi = x[i]; yi = y[i]
            if   xi < 0.5 and yi < 0.5:
                z[i] = (yi-0.5) if yi >= xi else (xi-0.5)
            elif xi > 0.5 and yi < 0.5:
                z[i] = np.sin(xi*4*np.pi)*np.sin(yi*4*np.pi)
            elif xi < 0.5 and yi > 0.5:
                z[i] = np.exp(-80*(yi-0.75)**2)*np.exp(-80*(xi-0.25)**2)
            else: #xi > 0.5 and yi > 0.5:
                z[i] = -6*((xi-0.75)**2 - (yi-0.75)**2)
    # Formula that takes vectors or 2d-arrays
    #z = np.sin(x*7*np.pi)*np.sin(y*7*np.pi)# * np.sin(x*np.pi)*np.sin(y*np.pi) #* np.exp(-20*(y-0.5)**2)*np.exp(-20*(x-0.5)**2)

    """z  = 0.5*np.exp(-20*(y  )**2)*np.exp(-20*(x  )**2)#np.sin(x*6*np.pi)*np.sin(y*6*np.pi) * np.sin(x*np.pi)*np.sin(y*np.pi) #* np.exp(-20*(y-0.5)**2)*np.exp(-20*(x-0.5)**2)
    z += 0.5*np.exp(-20*(y-1)**2)*np.exp(-20*(x  )**2)
    z += 0.5*np.exp(-20*(y  )**2)*np.exp(-20*(x-1)**2)
    z += 0.5*np.exp(-20*(y-1)**2)*np.exp(-20*(x-1)**2)
    z += np.exp(-20*(y-0.5)**2)*np.exp(-20*(x-0.5)**2)"""
    if not returnMaxMin:
        return z if not return2Darray else z.transpose()
    return np.max(z), np.min(z)


def functionNormData(trainSize,testSize):
    score = scoreFunc1

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


def train_neural_network(x, epochs, nNodes, hiddenLayers, saveFlag, noPrint=False, \
                         plot=False, learning_rate_choice=0.001):
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

        bestTrainLoss = 1E100; bestTestLoss = 1E100; bestTestLossSinceLastPlot = 1E100; prevErr = 1E100
        triggerNewLine = False; saveNow = False

        # Initiate some variables needed for plotting
        plotLinearResolution = 150
        linPoints            = np.linspace(0,1,plotLinearResolution)
        zForPlot             = np.zeros((plotLinearResolution,plotLinearResolution))
        X, Y                 = np.meshgrid(linPoints, linPoints)
        plotEveryNEpochs     = 20
        chosenPercent        = 13.0
        plotEveryPctImprov   = 1.0 - chosenPercent/100
        plotCount            = 0
        zAxisMax, zAxisMin   = scoreFunc1(linPoints,linPoints,return2Darray=True,returnMaxMin=True)
        if loadFlag:
            plotCount = int(raw_input("Please input plotcount of last plot: [int]: "))
        plotNow              = False

        colormap_choice      = cm.BrBG
        # loop through epocs
        xTrain, yTrain, xTest, yTest = functionNormData(trainSize,testSize)
        for epoch in range(startEpoch, numberOfEpochs):
            # loop through batches and cover new data set for each epoch
            _, epochLoss = sess.run([optimizer, cost], feed_dict={x: xTrain, y: yTrain})

            # compute test set loss
            # THIS IS MAYBE USED TO TRAIN?? Gotta check!!
            _, testCost = sess.run([optimizer, cost], feed_dict={x: xTest, y: yTest})

            if epoch%800 == 0:
                triggerNewLine = True
                # Generate new train and test data each 800th epoch:
                xTrain, yTrain, xTest, yTest = functionNormData(trainSize,testSize)
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
            if plot and bestTestLossSinceLastPlot * plotEveryPctImprov > testCost:
                bestTestLossSinceLastPlot = #testCost
                chosenPercent            -= 0.25 if chosenPercent > 3.0 else 0
                plotEveryPctImprov        = 1.0 - chosenPercent/100 # More plots later in the computation
                for row in xrange(plotLinearResolution):
                    rowValues = np.ones(plotLinearResolution)*linPoints[row]
                    xyForPlot = np.column_stack((rowValues,linPoints))
                    zForPlot[row,:] = sess.run(prediction, feed_dict={x: xyForPlot})[:,0]

                Z   = scoreFunc1(linPoints,linPoints,return2Darray=True)
                err = np.sum(np.abs(Z-zForPlot))/float(Z.size) # Average error
                if err < prevErr: # Plot only if error is smaller!
                    prevErr = err
                    fig = plt.figure()
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

                    plt.savefig("SomePlots/fig%d_epoch%d.png" %(plotCount,epoch), dpi=200)
                    plotCount += 1
                    plt.close()


                """fig = plt.figure()
                a=fig.add_subplot(2,3,1)
                imgplot = plt.imshow(Z, cmap=colormap_choice)
                a.set_title('Actual function')
                a=fig.add_subplot(2,3,2)
                imgplot = plt.imshow(zForPlot, cmap=colormap_choice)
                a.set_title('NN approx. func.')
                a=fig.add_subplot(2,3,3)
                imgplot = plt.imshow(Z-zForPlot, cmap=colormap_choice)
                a.set_title('Error NN: %.2e' %err)
                a = fig.add_subplot(2,3,4, projection='3d')
                surf = a.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=colormap_choice,
                                       linewidth=0, antialiased=False)
                a = fig.add_subplot(2,3,5, projection='3d')
                surf = a.plot_surface(X, Y, zForPlot, rstride=1, cstride=1, cmap=colormap_choice,
                                       linewidth=0, antialiased=False)
                a = fig.add_subplot(2,3,6, projection='3d')
                surf = a.plot_surface(X, Y, Z-zForPlot, rstride=1, cstride=1, cmap=colormap_choice,
                                       linewidth=0, antialiased=False)

                plt.savefig("SomePlots/fig%d_epoch%d.png" %(plotCount,epoch), dpi=150)
                plotCount += 1
                plt.close()"""
                # If saving is enabled, save the graph variables ('w', 'b')
                if saveNow and saveFlag:
                    saveNow = False
                    saveFileName = "SavedRuns/run" + str(epoch) + ".dat"
                    saver.save(sess, saveFileName, write_meta_graph=False)

        """if plot:
            print "\n\nResults for test data:"
            Z = scoreFunc1(linPoints,linPoints,return2Darray=True)

            fig = plt.figure()
            a=fig.add_subplot(1,3,1)
            imgplot = plt.imshow(Z)
            a.set_title('Actual function')

            a=fig.add_subplot(1,3,2)
            imgplot = plt.imshow(zForPlot)
            a.set_title('NN approx. func.')

            a=fig.add_subplot(1,3,3)
            imgplot = plt.imshow(Z-zForPlot)
            a.set_title('Error in NN approx.')
            plt.savefig("fig%d.png" %epoch)
            #plt.show()"""

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
            #if int(arg)<800:
            #    print "Number of epochs must be greater than 800. Exiting..."; sys.exit()
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
#xTrain, yTrain, xTest, yTest = functionNormData(trainSize,testSize)

# number of inputs and outputs
inputs  = 2
outputs = 1

x = tf.placeholder('float', [None, inputs],  name="x")
y = tf.placeholder('float', [None, outputs], name="y")

#neuralNetwork = lambda data : nnx.modelRelu(data, nNodes=nNodes, hiddenLayers=hiddenLayers, inputs=2, wInitMethod='normal', bInitMethod='normal')
neuralNetwork = lambda data : nnx.modelSigmoid(data, nNodes=nNodes, hiddenLayers=hiddenLayers,inputs=2, wInitMethod='normal', bInitMethod='normal')

print "---------------------------------------"
learning_rate_choice = 0.001 # Default for AdamOptimizer is 0.001
testCases = 0
print "Learning rate:", learning_rate_choice

hl_list   = [8]
node_list = [15]
noPrint = True if len(node_list)+len(node_list) > 2 else False
epochlossPerNPrev = 1e100   # "Guaranteed" worse than anything
nNodesBest = 0; hLBest = 0; epochBest = 0
for hiddenLayers in hl_list:
    for nNodes in node_list:
        for epochs in totalEpochList:
            testCases += 1
            weights, biases, neurons, epochlossPerN = train_neural_network(x, epochs, \
                    nNodes, hiddenLayers, saveFlag, noPrint, plot=plotFlag,learning_rate_choice=learning_rate_choice)
            print "\rHid.layers: %2.d, nodes/l: %2.d, epochs: %d, loss/N: %e" %(hiddenLayers,nNodes,epochs,epochlossPerN)
            if not noPrint:
                print " "
            if epochlossPerN < epochlossPerNPrev:
                epochlossPerNPrev = epochlossPerN
                nNodesBest = nNodes
                hLBest     = hiddenLayers
                epochBest  = epochs
if testCases > 1: # Print out testing different hidden layers and number of nodes
    #print testCases, totalEpochList
    print "---------------------------------------"
    print "Best combination found after %d epochs:" %epochBest
    print "Layers: %d, nodes/layer: %d, loss/N: %e" %(hLBest,nNodesBest,epochlossPerNPrev)
    print "---------------------------------------"
