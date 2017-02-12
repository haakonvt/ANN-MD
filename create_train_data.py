# -*- encoding: UTF8 -*-
"""
Generate XYZ data with corresponding energies from some chosen PES:
- Lennard Jones
- Stillinger Weber etc.
"""
from symmetry_transform import symmetryTransform
from math import pi,sqrt,exp,cos
import tensorflow as tf
import numpy as np
import sys

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

    rc = (r < a) # Bool array. "False" cast to 0 and "True" to 1
    U2 = eps*A*(B*r**(-p)-r**(-q)) * np.exp(r-a)**-1 * rc
    U  = np.sum(U2) # Two body terms
    U3 = lambda rij, rik, cos_theta: eps*l*exp(g/(rij-a) + g/(rik-a)) * (cos_theta - cos_tc)**2 \
                                     if (rij < a) and (rik < a) else 0.0 # Is this bad practice? lawls
    # Need a double sum to find three body terms
    for j in range(N): # i < j
        for k in range(j+1,N): # i < j < k
            cos_theta = np.dot(xyz[j],xyz[k]) / (r[j]*r[k])
            U        += U3(r[j], r[k], cos_theta)
    return U

def createTrainData(trainSize, testSize, neighbors, PES, verbose=False):
    train_ratio = trainSize/ float(testSize+trainSize)
    test_ratio  = testSize / float(testSize+trainSize)
    if PES == PES_Stillinger_Weber:
        sigma       = 1.0
        r_low       = 0.85 * sigma
        r_high      = 1.8  * sigma - 1E-8 # SW has a divide by zero at exactly cutoff
        xyz_N_train = createXYZ(r_low, r_high, trainSize, neighbors, verbose=verbose)
        yTrain      = potentialEnergyGenerator(xyz_N_train, PES)
        yTrain      = yTrain.reshape([trainSize,1])
        xyz_N_test  = createXYZ(r_low, r_high, testSize, neighbors, verbose=verbose)
        yTest       = potentialEnergyGenerator(xyz_N_test, PES)
        yTest       = yTest.reshape([testSize,1])

        G_funcs, nmbr_G = generate_symmfunc_input_Si()
        xTrain          = np.zeros((trainSize, nmbr_G))
        xTest           = np.zeros((testSize , nmbr_G))

        for i in range(trainSize):
            xyz_i       = xyz_N_train[:,:,i]
            xTrain[i,:] = symmetryTransform(G_funcs, xyz_i)
            if verbose:
                sys.stdout.write('\r' + ' '*80) # White out line
                percent = round(float(i+1)/trainSize*100.*train_ratio, 2)
                sys.stdout.write('\rTransforming xyz with symmetry functions. %d %% complete' %(percent))
                sys.stdout.flush()
        for i in range(testSize):
            xyz_i      = xyz_N_test[:,:,i]
            xTest[i,:] = symmetryTransform(G_funcs, xyz_i)
            if verbose:
                sys.stdout.write('\r' + ' '*80) # White out line
                percent = round(train_ratio*100 + float(i+1)/testSize*100.*test_ratio, 2)
                sys.stdout.write('\rTransforming xyz with symmetry functions. %d %% complete' %(percent))
                sys.stdout.flush()
    else:
        print "To be implemented! For now, use PES = PES_Stillinger_Weber. Exiting..."
        sys.exit(0)
    return xTrain, yTrain, xTest, yTest


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
    # Make use of symmetry function G2 and G4: (indicate how many)
    which_symm_funcs = [2, 4]
    wsf              = which_symm_funcs
    how_many_funcs   = [9, 43]
    hmf              = how_many_funcs

    # This is where the pain begins -_-
    # Note: [3] * 4 evaluates to [3,3,3,3]
    rc       = [[1.8]*hmf[0], [1.8]*hmf[1]]
    rs       = [[0.0]*hmf[0], [None]]
    eta      = [[0.001, 0.015, 0.035, 0.06, 0.1, 0.2, 0.4, 0.7, 1.8], \
                [0.001]*4 + [0.003]*4 + [0.008]*4 + [0.015]*8 + [0.025]*8 + [0.045]*8 + [0.08]*7]
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

    """
    if False:
        # Stillinger Weber test
        np.random.seed(1) # For testing
        r_low     = 0.8
        r_high    = 1.9
        size      = 100
        neighbors = 30
        PES       = PES_Stillinger_Weber
        xyz_N     = createXYZ(r_low, r_high, size, neighbors)
        Ep        = potentialEnergyGenerator(xyz_N, PES)
        print Ep
    """
