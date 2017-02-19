# -*- encoding: UTF8 -*-
"""
Generate XYZ data with corresponding energies from some chosen PES:
- Lennard Jones
- Stillinger Weber etc.
"""
from symmetry_transform import symmetryTransform
from timeit import default_timer as timer # Best timer indep. of system
from math import pi,sqrt,exp,cos
import tensorflow as tf
import numpy as np
import glob
import time
import sys
import os


class loadFromFile:
    """
    Loads file, shuffle rows and keeps it in memory for later use.
    """
    def __init__(self, testSizeSkip, filename, shuffle_rows=False):
        self.skipIndices = testSizeSkip
        self.index       = 0
        self.filename    = filename
        if os.path.isfile(filename): # If file exist, load it
            try:
                self.buffer = np.loadtxt(filename, delimiter=',')
            except Exception as e:
                print "Could not load buffer. Error message follows:\n %s" %s
        else:
            print 'Found no training data called:\n"%s"\“...exiting!' %filename
            sys.exit(0)
        if shuffle_rows:
            np.random.shuffle(self.buffer) # Shuffles rows only (not columns) by default *yey*
        print "Tot. data points loaded from file:", self.buffer.shape[0]
        self.testData  = self.buffer[0:testSizeSkip,:] # Pick out test data from total
        self.buffer    = self.buffer[testSizeSkip:,:]  # Use rest of data for training
        self.totTrainData = self.buffer.shape[0]

    def __call__(self, size, return_test=False, verbose=False):
        """
        Returns the next batch of size 'size' which is a set of rows from the loaded file
        """
        testSize = self.skipIndices
        i        = self.index # Easier to read next couple of lines
        if return_test:
            if size != testSize:
                print "You initiated this class with testSize = %d," %testSize
                print "and now you request trainSize = %d." %size
                print "I will continue with %d (blame the programmer)" %testSize
            symm_vec_test = self.testData[:,1:] # Second column->last
            Ep_test       = self.testData[:,0]  # First column
            Ep_test       = Ep_test.reshape([testSize,1])
            return symm_vec_test, Ep_test
        else:
            if i + size > self.totTrainData:
                if verbose:
                    print "\nWarning: All training data 'used', shuffling & starting over!\n"
                np.random.shuffle(self.buffer) # Must be done, no choice!
                self.index = 0 # Dont use test data for training!
                i          = 0
            if size < self.totTrainData:
                symm_vec_train = self.buffer[i:i+size, 1:] # Second column->last
                Ep_train       = self.buffer[i:i+size, 0]  # First column
                Ep_train       = Ep_train.reshape([size,1])
                self.index += size # Update so that next time class is called, we get the next items
                return symm_vec_train, Ep_train
            else:
                print "Requested batch size %d, is larger than data set %d" %(size, self.totTrainData)
    def return_all_data(self):
        return self.buffer

def potentialEnergyGenerator(xyz_N, PES):
    size = xyz_N.shape[2]
    Ep   = np.zeros(size)
    for i in range(size):
        xyz_i = xyz_N[:,:,i]
        Ep[i] = PES(xyz_i)
    return Ep

def createXYZ(r_min, r_max, size, neighbors=20, histogramPlot=False, verbose=False):
    """
    # Input:  Size of train and test size + number of neighbors
    # Output: xyz-neighbors-matrix of size 'size'

    Generates random numbers with x,y,z that can be [0,r_max] with r in [r_min, r_max]
    """
    if verbose:
        print "Creating XYZ-neighbor-data for:\n - Neighbors: %d \n - Samples  : %d" %(neighbors,size)
        print "-------------------------------"

    xyz_N = np.zeros((neighbors,3,size))
    xyz   = np.zeros((size,3))
    for i in range(neighbors): # Fill cube slice for each neighbor (quicker than "size")
        r2       = np.random.uniform(r_min, r_max,       size)**2
        xyz[:,0] = np.random.uniform(0,     r2,          size)
        xyz[:,1] = np.random.uniform(0,     r2-xyz[:,0], size)
        xyz[:,2] = r2 - xyz[:,0] - xyz[:,1]
        for row in range(size):
            np.random.shuffle(xyz[row,:]) # This shuffles in-place (so no copying)
        xyz_N[i,0,:] = np.sqrt(xyz[:,0]) * np.random.choice([-1,1],size) # 50-50 if position is plus or minus
        xyz_N[i,1,:] = np.sqrt(xyz[:,1]) * np.random.choice([-1,1],size)
        xyz_N[i,2,:] = np.sqrt(xyz[:,2]) * np.random.choice([-1,1],size)
    if histogramPlot:
        import matplotlib.pyplot as plt
        plt.subplot(3,1,1);plt.hist(xyz_N[:,0,:].ravel(),bins=70);plt.subplot(3,1,2);plt.hist(xyz_N[:,1,:].ravel(),bins=70);plt.subplot(3,1,3);plt.hist(xyz_N[:,2,:].ravel(),bins=70);plt.show()
    return xyz_N

def createXYZ_Rjk(r_min, r_max, size, neighbors=20, histogramPlot=False, verbose=False):
    """
    # Input:  Size of train and test size + number of neighbors
    # Output: xyz-neighbors-matrix of size 'size'

    Generates random numbers with x,y,z that can be [0,r_max] with r in [r_min, r_max]

    NOTE:
    Rjk cannot be shorter than r_min. This causes a lot of problems when drawing
    random numbers since (to the programmers knowledge) is no fast way of making sure
    neighbor atom distance is within allowed interval. Think of this problem as dense packing
    of 3D spheres.

    NOTE: NOT FINISHED
    """
    if verbose:
        print "Creating special XYZ-neighbor-data where rjk > r_min."
        print " - Neighbors: %d \n - Samples  : %d" %(neighbors,size)
        print "-------------------------------"

    xyz_N = np.zeros((neighbors,3,size))
    xyz   = np.zeros((size,3))
    for i in range(neighbors): # Fill cube slice for each neighbor (quicker than "size")
        r2       = np.random.uniform(r_min, r_max,       size)**2
        xyz[:,0] = np.random.uniform(0,     r2,          size)
        xyz[:,1] = np.random.uniform(0,     r2-xyz[:,0], size)
        xyz[:,2] = r2 - xyz[:,0] - xyz[:,1]
        # Not we must check the validity of all neighbor distances
        for j in range():
            for k in range(j+1,):
                pass
        for row in range(size):
            np.random.shuffle(xyz[row,:]) # This shuffles in-place (so no copying)
        xyz_N[i,0,:] = np.sqrt(xyz[:,0]) * np.random.choice([-1,1],size) # 50-50 if position is plus or minus
        xyz_N[i,1,:] = np.sqrt(xyz[:,1]) * np.random.choice([-1,1],size)
        xyz_N[i,2,:] = np.sqrt(xyz[:,2]) * np.random.choice([-1,1],size)
    if histogramPlot:
        import matplotlib.pyplot as plt
        plt.subplot(3,1,1);plt.hist(xyz_N[:,0,:].ravel(),bins=70);plt.subplot(3,1,2);plt.hist(xyz_N[:,1,:].ravel(),bins=70);plt.subplot(3,1,3);plt.hist(xyz_N[:,2,:].ravel(),bins=70);plt.show()
    return xyz_N

def PES_Stillinger_Weber(xyz_i):
    """
    INPUT
    - xyz_i: Matrix with columnds containing cartesian coordinates,
           relative to the current atom i, i.e.:
           [[x1 y1 z1]
            [x2 y2 z2]
            [x3 y3 z3]
            [x4 y4 z4]]
    """
    xyz = xyz_i
    r = np.linalg.norm(xyz,axis=1)
    N = len(r) # Number of neighbors for atom i, which we are currently inspecting

    # A lot of definitions first
    A = 7.049556277
    B = 0.6022245584
    p = 4.
    q = 0.
    a = 1.8
    l = 21.  # lambda
    g = 1.2 # gamma
    cos_tc = -1.0/3.0 # 109.47 deg

    eps = 1. #2.1678  # [eV]    # With reduced units = 1
    m   = 28.085  # [amu]
    sig = 1. #2.0951 # [Å]      # With reduced units = 1
    Ts  = 25156.73798 # eps/kB [K]

    # rc = (r < a) # Bool array. "False" cast to 0 and "True" to 1
    U2 = eps*A*(B*r**(-p)-r**(-q)) * np.exp(r-a)**-1 #* rc
    U3 = lambda rij, rik, cos_theta: eps*l*exp(g/(rij-a) + g/(rik-a)) * (cos_theta - cos_tc)**2 #\
                                     #if (rij < a) and (rik < a) else 0.0 # Is this bad practice? lawls
    # Sum up two body terms
    U  = np.sum(U2)
    # Need a double sum to find three body terms
    for j in range(N): # i < j
        for k in range(j+1,N): # i < j < k
            cos_theta = np.dot(xyz[j],xyz[k]) / (r[j]*r[k])
            U        += U3(r[j], r[k], cos_theta)
    return U

def createTrainData(size, neighbors, PES, verbose=False):
    if PES == PES_Stillinger_Weber:
        sigma       = 1.0
        r_low       = 0.85 * sigma
        r_high      = 1.8  * sigma - 1E-8 # SW has a divide by zero at exactly cutoff
        xyz_N = createXYZ(r_low, r_high, size, neighbors, verbose=verbose)
        Ep    = potentialEnergyGenerator(xyz_N, PES)
        Ep    = Ep.reshape([size,1])

        G_funcs, nmbr_G = generate_symmfunc_input_Si()
        nn_input        = np.zeros((size, nmbr_G))

        for i in range(size):
            xyz_i         = xyz_N[:,:,i]
            nn_input[i,:] = symmetryTransform(G_funcs, xyz_i)
            if verbose:
                sys.stdout.write('\r' + ' '*80) # White out line
                percent = round(float(i+1)/size*100., 2)
                sys.stdout.write('\rTransforming xyz with symmetry functions. %.2f %% complete' %(percent))
                sys.stdout.flush()
        if verbose:
            print " "
    else:
        print "To be implemented! For now, use PES = PES_Stillinger_Weber. Exiting..."
        sys.exit(0)
    return nn_input, Ep

def checkAndMaybeLoadPrevTrainData(filename):
    origFilename    = filename
    listOfTrainData = glob.glob("SW_train_*.txt")
    if filename in listOfTrainData: # Filename already exist
        i = 0
        while True:
            i += 1
            filename = origFilename[:-4] + "_v%d" %i + ".txt"
            if filename not in listOfTrainData:
                print "New filename:", filename
                break # Continue changing name until we find one available
    if not listOfTrainData: # No previous files
        return False, None, filename
    else:
        nmbrFiles = len(listOfTrainData)
        yn = raw_input("Found %d file(s). Load them into this file? (y/N) " %nmbrFiles)
        if yn in ["y","Y","yes","Yes","YES"]: # Standard = enter = NO
            loadedData = []
            for file_i in listOfTrainData:
                all_data = loadFromFile(0, file_i, shuffle_rows=False)
                loadedData.append(all_data.return_all_data())
            yn = raw_input("Delete files loaded? (Y/n) ")
            if yn in ["y","Y","yes","Yes","YES",""]: # Standard = enter = YES
                for file_i in listOfTrainData:
                    os.remove(file_i)
                filename = origFilename # Since we delete it here
            # Smash all data into a single file
            if len(loadedData) > 1:
                all_data = np.concatenate(loadedData, axis=0)
            else:
                all_data = loadedData[0]
            return True, all_data, filename
        return False, None, filename

def createTrainDataDump(size, neighbors, PES, filename, only_concatenate=False, verbose=False):
    # Check if file exist and in case, ask if it should be loaded
    filesLoadedBool, prev_data, filename = checkAndMaybeLoadPrevTrainData(filename)
    if only_concatenate:
        if verbose:
            sys.stdout.write('\n\r' + ' '*80) # White out line
            sys.stdout.write('\rSaving all training data to file.')
            sys.stdout.flush()
        np.random.shuffle(prev_data, axis=0) # Shuffle the rows of the data i.e. the symmetry vectors
        np.savetxt(filename, prev_data, delimiter=',')
        if verbose:
            sys.stdout.write('\r' + ' '*80) # White out line
            sys.stdout.write('\rSaving all training data to file. Done!\n')
            sys.stdout.flush()
    else:
        if PES == PES_Stillinger_Weber: # i.e. if not 'only_concatenate'
            sigma       = 1.0
            r_low       = 0.85 * sigma
            r_high      = 1.8  * sigma - 1E-8 # SW has a divide by zero at exactly cutoff
            xyz_N_train = createXYZ(r_low, r_high, size, neighbors, verbose=verbose)
            if verbose:
                sys.stdout.write('\r' + ' '*80) # White out line
                sys.stdout.write('\rComputing potential energy.')
                sys.stdout.flush()
            Ep = potentialEnergyGenerator(xyz_N_train, PES)
            if verbose:
                sys.stdout.write('\r' + ' '*80) # White out line
                sys.stdout.write('\rComputing potential energy. Done!\n')
                sys.stdout.flush()

            G_funcs, nmbr_G = generate_symmfunc_input_Si()
            xTrain          = np.zeros((size, nmbr_G))

            for i in range(size):
                xyz_i       = xyz_N_train[:,:,i]
                xTrain[i,:] = symmetryTransform(G_funcs, xyz_i)
                if verbose and (i+1)%100 == 0:
                    sys.stdout.write('\r' + ' '*80) # White out line
                    percent = round(float(i+1)/size*100., 2)
                    sys.stdout.write('\rTransforming xyz with symmetry functions. %g %% complete' %(percent))
                    sys.stdout.flush()
        else:
            print "To be implemented! For now, use PES = PES_Stillinger_Weber. Exiting..."
            sys.exit(0)
        if verbose:
            sys.stdout.write('\n\r' + ' '*80) # White out line
            sys.stdout.write('\rSaving all training data to file.')
            sys.stdout.flush()
        dump_data = np.zeros((size, nmbr_G + 1))
        dump_data[:,0]  = Ep
        dump_data[:,1:] = xTrain
        if filesLoadedBool:
            dump_data = np.concatenate((dump_data, prev_data), axis=0) # Add loaded files
        np.random.shuffle(dump_data, axis=0) # Shuffle the rows of the data i.e. the symmetry vectors
        np.savetxt(filename, dump_data, delimiter=',')
        if verbose:
            sys.stdout.write('\r' + ' '*80) # White out line
            sys.stdout.write('\rSaving all training data to file. Done!\n')
            sys.stdout.flush()


def FORCES_Stillinger_Weber(xyz_i):
    """
    To be implemented. May not be needed for anything.
    """
    pass

def PES_Lennard_Jones(xyz_i):
    """
    Simple LJ pair potential
    """
    # eps = 1.0318 * 10^(-2) eV
    # sig = 3.405 * 10^(-7) meter
    xyz = xyz_i
    r   = np.linalg.norm(xyz,axis=1)
    U   =  np.sum(4*(1.0/r**12 - 1.0/r**6))
    return U

def FORCES_Lennard_Jones(xyz_i):
    """
    To be implemented. May not be needed for anything.
    FORCES = lambda s: 12.0/s**13 - 6.0/s**7 # Analytic derivative of LJ with minus sign: F = -d/dx Ep
    """
    pass

def generate_symmfunc_input_Si():
    G_funcs = [0,0,0,0,0] # Start out with NO symm.funcs.
    G_vars  = [1,3,2,4,4] # Number of variables symm.func. take as input
    G_args_list = ["rc[i][j]",
                   "rc[i][j], rs[i][j], eta[i][j]",
                   "rc[i][j], kappa[i][j]",
                   "rc[i][j], eta[i][j], zeta[i][j], lambda_c[i][j]",
                   "rc[i][j], eta[i][j], zeta[i][j], lambda_c[i][j]"]
    # Make use of symmetry function G2 and G5: (indicate how many)
    which_symm_funcs = [2, 5] # G5 instead of G4, because SW doesnt care about Rjk
    wsf              = which_symm_funcs
    how_many_funcs   = [9, 43]
    hmf              = how_many_funcs

    # This is where the pain begins -_-
    # Note: [3] * 4 evaluates to [3,3,3,3]
    rc       = [[1.8]*hmf[0], [1.8]*hmf[1]]
    rs       = [[0.85]*hmf[0], [None]]
    #           [0.001, 0.015, 0.035, 0.06, 0.1, 0.2, 0.4, 0.7, 1.8] # Prev values
    eta      = [[0.0  , 0.67 , 1.25 , 2.5 , 5.0, 10.0, 20.0, 40.0, 80.0], \
                [0.001]*4 + [0.06]*4 + [0.12]*4 + [0.24]*8 + [0.4]*8 + [0.68]*8 + [1.15]*7]
    zeta     = [[None], [1,1,2,2]*4 + [1,1,2,2,4,4,16,16]*3 + [1,1,2,2,4,4,16]]
    lambda_c = [[None],[-1,1]*21 + [1]]

    i = 0 # Will be first G-func
    for G,n in zip(wsf, hmf):
        G_funcs[G-1] = [n,  np.zeros((n, G_vars[G-1]))]
        for j in range(n):
            symm_args = eval("np.array([%s])" %(G_args_list[G-1]))
            G_funcs[G-1][1][j] = symm_args
        i += 1
    tot_Gs = np.sum(np.array(hmf))
    return G_funcs, tot_Gs

if __name__ == '__main__':
    dumpToFile       = True
    concatenateFiles = True
    testClass        = False

    if dumpToFile:
        size = 100000
        neighbors = 10
        # filename = "stillinger-weber-symmetry-data.txt"
        filename  = "SW_train_%s.txt" %(str(size))
        print "When run directly (like now), this file dumps training data to file:"
        print '"%s"' %filename
        print "-------------------------------"
        print "Neighbors", neighbors
        print "-------------------------------"
        PES       = PES_Stillinger_Weber
        t0 = timer()
        createTrainDataDump(size, neighbors, PES, filename, \
                            only_concatenate=concatenateFiles, verbose=True)
        t1 = timer() - t0
        print "\nComputation took: %.2f seconds" %t1

    if testClass:
        testSize  = 100 # Remove these from training set
        filename  = "test-class-symmetry-data.txt"
        all_data  = loadFromFile(testSize, filename)
        xTrain, yTrain = all_data(1)
        print xTrain[:,0:5], "\n", yTrain
        xTrain, yTrain = all_data(1)
        print xTrain[:,0:5], "\n", yTrain # Make sure this is different from above print out

    if not dumpToFile and not testClass: # aka no tests run
        print "It looks like this script didnt do anything. Cool :B"
