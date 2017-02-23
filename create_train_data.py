# -*- encoding: UTF8 -*-
"""
Generate XYZ data with corresponding energies from some chosen PES:
- Lennard Jones
- Stillinger Weber etc.
"""
from symmetry_transform import symmetryTransform
from timeit import default_timer as timer # Best timer indep. of system
from math import pi,sqrt,exp,cos,isnan,sin
from warnings import filterwarnings
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
    if len(xyz_N.shape) == 2: # This is just a single neighbor list
        return PES(xyz_N)
    else:
        size = xyz_N.shape[2]
        Ep   = np.zeros(size)
        for i in range(size):
            xyz_i = xyz_N[:,:,i]
            Ep[i] = PES(xyz_i)
        """
        # Plot distribution of potential energy (per particle)
        import matplotlib.pyplot as plt
        plt.hist(Ep,bins=50)
        plt.show()
        """
        return Ep

def potentialEnergyGeneratorSingleNeigList(xyz_i, PES):
    return PES(xyz_i)

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

    eps = 2.1683 # [eV]     # With reduced units = 1
    sig = 2.0951 # [Å]      # With reduced units = 1

    rc = (r < a*sig) # Bool array. "False" cast to 0 and "True" to 1
    filterwarnings("ignore", category=RuntimeWarning) # U2 below can give NaN
    U2 = A*eps*(B*(sig/r)**p-(sig/r)**q) * np.exp(sig/(r-a*sig)) * rc
    filterwarnings("always", category=RuntimeWarning) # Turn warnings back on
    def U2_serial(r_vec, r_cut): # Only use if U2 gives NaN
        U2_E = 0
        for r,rc in zip(r_vec,r_cut):
            if rc:
                U2_E += A*eps*(B*(sig/r)**p-(sig/r)**q) * np.exp(sig/(r-a*sig))
            else:
                pass # Add 0
        return U2_E
    def U3(rij, rik, cos_theta):
        if (rij < a*sig) and (rik < a*sig):
            exp_factor   = exp(g*sig/(rij-a*sig)) * exp(g*sig/(rik-a*sig))
            angle_factor = l*eps*(cos_theta - cos_tc)**2
            return exp_factor * angle_factor
        else:
            return 0.0
    # Sum up two body terms
    U  = np.sum(U2)
    if isnan(U):
        U = U2_serial(r, rc)
        print "\nNaN gotten, re-computing with serial code. U2 = ", U
    # Need a double sum to find three body terms
    for j in range(N): # i < j
        for k in range(j+1,N): # i < j < k
            cos_theta = np.dot(xyz[j],xyz[k]) / (r[j]*r[k])
            U        += U3(r[j], r[k], cos_theta)
    return U

def createTrainData(size, neighbors, PES, verbose=False):
    if PES == PES_Stillinger_Weber:
        sigma       = 2.0951
        r_low       = 0.85 * sigma
        r_high      = 1.8  * sigma - 1E-8 # SW has a divide by zero at exactly cutoff
        xyz_N = createXYZ(r_low, r_high, size, neighbors, verbose=verbose)
        Ep    = potentialEnergyGenerator(xyz_N, PES)
        Ep    = Ep.reshape([size,1])

        G_funcs, nmbr_G = generate_symmfunc_input_Si(sigma)
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

def checkAndMaybeLoadPrevTrainData(filename, no_load=False):
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
    elif not no_load:
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
    else:
        return False, None, filename

def createTrainDataDump(size, neighbors, PES, filename, only_concatenate=False, verbose=False, no_load=False):
    # Check if file exist and in case, ask if it should be loaded
    filesLoadedBool, prev_data, filename = checkAndMaybeLoadPrevTrainData(filename, no_load)
    if only_concatenate:
        if verbose:
            sys.stdout.write('\n\r' + ' '*80) # White out line
            sys.stdout.write('\rSaving all training data to file.')
            sys.stdout.flush()
        np.random.shuffle(prev_data) # Shuffle the rows of the data i.e. the symmetry vectors
        np.savetxt(filename, prev_data, delimiter=',')
        if verbose:
            sys.stdout.write('\r' + ' '*80) # White out line
            sys.stdout.write('\rSaving all training data to file. Done!\n')
            sys.stdout.flush()
    else:
        if PES == PES_Stillinger_Weber: # i.e. if not 'only_concatenate'
            sigma       = 2.0951 # 1.0
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

            G_funcs, nmbr_G = generate_symmfunc_input_Si(sigma)
            xTrain          = np.zeros((size, nmbr_G))

            for i in range(size):
                xyz_i       = xyz_N_train[:,:,i]
                xTrain[i,:] = symmetryTransform(G_funcs, xyz_i)
                if verbose and (i+1)%10 == 0:
                    sys.stdout.write('\r' + ' '*80) # White out line
                    percent = round(float(i+1)/size*100., 2)
                    sys.stdout.write('\rTransforming xyz with symmetry functions. %.2f %% complete' %(percent))
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
        np.random.shuffle(dump_data) # Shuffle the rows of the data i.e. the symmetry vectors
        np.savetxt(filename, dump_data, delimiter=',')
        if verbose:
            sys.stdout.write('\r' + ' '*80) # White out line
            sys.stdout.write('\rSaving all training data to file. Done!\n')
            sys.stdout.flush()

def PES_Lennard_Jones(xyz_i):
    """
    Simple LJ pair potential
    """
    # eps = 1.0318 * 10^(-2) eV
    # sig = 3.405 * 10^(-7) meter
    xyz = xyz_i
    r   = np.linalg.norm(xyz, axis=1)
    U   =  np.sum(4*(1.0/r**12 - 1.0/r**6))
    return U

def generate_symmfunc_input_Si(sigma):
    G_funcs = [0,0,0,0,0] # Start out with NO symm.funcs.
    G_vars  = [1,3,2,4,5] # Number of variables symm.func. take as input
    G_args_list = ["rc[i][j]",
                   "rc[i][j], rs[i][j], eta[i][j]",
                   "rc[i][j], kappa[i][j]",
                   "rc[i][j], eta[i][j], zeta[i][j], lambda_c[i][j]",
                   "rc[i][j], eta[i][j], zeta[i][j], lambda_c[i][j], rs[i][j]"]
    # Make use of symmetry function G2 and G5: (indicate how many)
    which_symm_funcs = [2, 4] # G5 instead of G4, because SW doesnt care about Rjk
    wsf              = which_symm_funcs
    how_many_funcs   = [10, 120]
    hmf              = how_many_funcs

    # This is where the pain begins -_-
    # Note: [3] * 4 evaluates to [3,3,3,3]
    rc       = [[1.8  * sigma]*10, [1.8  * sigma]*120]
    rs       = [[0.85 * sigma]*10, [0.85 * sigma]*120]
    eta      = [[0.0, 0.3, 0.65, 1.25, 2.5, 5.0, 10.0, 20.0, 40.0, 90.0], \
                [0.0]*12 + [0.3]*12 + [0.65]*12 + [1.25]*12 + [2.5]*12  + [5.]*12  + [10.]*12 + [20.]*12 + [40.]*12 + [90.]*12]
    zeta     = [[None], [1,1,2,2,4,4,8,8,16,16,32,32]*10]
    lambda_c = [[None],[-1,1]*60]

    i = 0 # Will be first G-func
    for G,n in zip(wsf, hmf):
        G_funcs[G-1] = [n,  np.zeros((n, G_vars[G-1]))]
        for j in range(n):
            symm_args = eval("np.array([%s])" %(G_args_list[G-1]))
            G_funcs[G-1][1][j] = symm_args
        i += 1
    tot_Gs = np.sum(np.array(hmf))
    return G_funcs, tot_Gs

def testLammpsData(filename):
    Ep = []
    Ep2 = []
    with open(filename, 'r') as lammps_file:
        """
        File looks like this
        x1 y1 z1 r1^2 x2 y2 z2 r2^2 ... xN yN zN rN^2 Ep
        """
        for i,row in enumerate(lammps_file):
            if i < 2000:
                continue # Skip first 2000
            xyzr_i   = np.array(row.split(),dtype=float)
            n_elem = len(xyzr_i-1)/4 # Remove Ep and Compute
            Ep.append(xyzr_i[-1])
            xyzr_i   = xyzr_i[:-1].reshape(n_elem,4)
            xyz_i    = xyzr_i[:,:-1]
            Ep2.append(potentialEnergyGenerator(xyz_i,PES=PES_Stillinger_Weber))
        # import matplotlib.pyplot as plt
        # plt.subplot(2,1,1)
        # plt.hist(Ep,bins=200)
        # plt.subplot(2,1,2)
        # plt.hist(Ep2,bins=30)
        # plt.show()
        # plt.savefig("merkeligEp.pdf")
        # print len(Ep)
        # print np.mean(Ep), np.mean(Ep2), np.mean(Ep2)/np.mean(Ep)

def lammpsDataToSymmToFile(open_filename, save_filename):
    Ep = []
    with open(open_filename, 'r') as lammps_file:
        """
        File looks like this
        x1 y1 z1 r1^2 x2 y2 z2 r2^2 ... xN yN zN rN^2 Ep
        """
        size            = 3000
        sigma           = 2.0951
        G_funcs, nmbr_G = generate_symmfunc_input_Si(sigma)
        xTrain          = np.zeros((size, nmbr_G))
        for i,row in enumerate(lammps_file):
            if i < 2000:
                continue # Skip first 2000
            i -= 2000
            xyzr_i = np.array(row.split(), dtype=float)
            n_elem = len(xyzr_i-1)/4 # Remove Ep and Compute
            # Ep.append(xyzr_i[-1]) # This is broken somehow... compute my own below
            xyzr_i = xyzr_i[:-1].reshape(n_elem,4)
            xyz_i  = xyzr_i[:,:-1]
            Ep.append(potentialEnergyGenerator(xyz_i, PES=PES_Stillinger_Weber))
            xTrain[i,:] = symmetryTransform(G_funcs, xyz_i)
            if (i+1)%10 == 0:
                sys.stdout.write('\r' + ' '*80) # White out line
                percent = round(float(i+1)/size*100., 2)
                sys.stdout.write('\rTransforming xyz with symmetry functions. %.2f %% complete' %(percent))
                sys.stdout.flush()
    dump_data = np.zeros((size, nmbr_G + 1))
    dump_data[:,0]  = Ep
    dump_data[:,1:] = xTrain
    np.random.shuffle(dump_data) # Shuffle the rows of the data i.e. the symmetry vectors
    np.savetxt(save_filename, dump_data, delimiter=',')
    print "\n"

def rotateXYZ(xyz, xr, yr, zr, angle="radians"):
    """
    Rotates around all cartesian axes
    xyz: [x,y,z]
    """
    if angle == "degrees":
        xr = xr/360.*2*pi
        yr = yr/360.*2*pi
        zr = zr/360.*2*pi
        angle = "radians"

    if angle == "radians":
        Rx = np.array([[cos(xr), -sin(xr), 0],
                       [sin(xr),  cos(xr), 0],
                       [0      ,        0, 1]])
        Ry = np.array([[cos(yr) , 0, sin(yr)],
                       [0       , 1,       0],
                       [-sin(yr), 0, cos(yr)]])
        Rz = np.array([[1,       0,        0],
                       [0, cos(zr), -sin(zr)],
                       [0, sin(zr), cos(zr)]])
        R = np.dot(np.dot(Rx, Ry), Rz) # Dot for 2d-arrays does matrix multiplication
        return np.dot(xyz, R)
    else:
        print "Angle must be given in 'radians' or 'degrees'. Exiting."
        sys.exit(0)


if __name__ == '__main__':
    dumpToFile       = False
    concatenateFiles = False
    dumpMultiple     = False
    testAngSymm      = True
    testLammps       = False
    dumpLammpsFile   = False
    testClass        = False

    if testAngSymm:
        sigma     = 2.0951 # 1.0
        r_low     = 0.85 * sigma
        r_high    = 1.8  * sigma - 1E-8 # SW has a divide by zero at exactly cutoff
        size      = 10
        neighbors = 5
        PES       = PES_Stillinger_Weber
        xyz_N     = createXYZ(r_low, r_high, size, neighbors, verbose=False) # size = 10, neigh = 5
        Ep0 = potentialEnergyGenerator(xyz_N, PES)

        for i in range(size):
            xyz_nl = xyz_N[:,:,i] # single nl, (neighbor list)
            xr, yr, zr = np.random.uniform(0,2*np.pi,3) # Rotate all neighbor atoms with same angles (generated randomly)
            for j in range(xyz_nl.shape[0]): # single x,y,z vector
                xyz_nl[j]  = rotateXYZ(xyz_nl[j], xr, yr, zr)
            xyz_N[:,:,i] = xyz_nl
        Ep1 = potentialEnergyGenerator(xyz_N, PES)
        assert np.sqrt(np.mean((Ep0-Ep1)**2)) < 1E-14

    if testLammps:
        filename = "Important_data/lammps_neighbours.txt"
        testLammpsData(filename)

    if dumpLammpsFile:
        open_filename = "Important_data/lammps_neighbours.txt"
        save_filename = "SW_train_lammps_3000.txt"
        lammpsDataToSymmToFile(open_filename, save_filename)

    if dumpMultiple:
        size = 20000
        neigh_list = [4,6,8,10,12,14]
        t0_tot = timer()
        for neighbors in neigh_list:
            filename  = "SW_train_G4_%s_n%s.txt" %(str(size), str(neighbors))
            print "When run directly (like now), this file dumps training data to file:"
            print '"%s"' %filename
            print "-------------------------------"
            print "Neighbors", neighbors
            print "-------------------------------"
            PES       = PES_Stillinger_Weber
            t0 = timer()
            createTrainDataDump(size, neighbors, PES, filename, \
                                only_concatenate=concatenateFiles, verbose=True, \
                                no_load=True)
            t1 = timer() - t0
            print "\nComputation of %d neighbors took: %.2f seconds" %(neighbors,t1)
        t1 = timer() - t0_tot
        if t1 > 1000:
            t1 /= 60.
            print "\nTotal computation took: %.2f minutes" %t1
        else:
            print "\nTotal computation took: %.2f seconds" %t1

    if dumpToFile:
        size = 2000
        neighbors = 12
        # filename = "stillinger-weber-symmetry-data.txt"
        filename  = "SW_train_rs_%s_n%s.txt" %(str(size), str(neighbors))
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
