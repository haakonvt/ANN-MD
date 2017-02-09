"""
Generate XYZ data with corresponding energies from some chosen PES:
- Lennard Jones
- Stillinger Weber etc.
"""
from math import pi,sqrt,exp,cos
import tensorflow as tf
import numpy as np

def potentialEnergyGenerator(xyz_N, PES):
    size = xyz_N.shape[2]
    Ep   = np.zeros(size)

    for i in range(size):
        xyz_i = xyz_N[:,:,i]
        Ep[i] = PES(xyz_i)
    return Ep

def createXYZ(r_min, r_max, size, neighbors=20, histogramPlot=False):
    """
    # Input:  Size of train and test size + number of neighbors
    # Output: xyz-neighbors-matrix of size 'size'

    Generates random numbers with x,y,z that can be [0,r_max] with r in [r_min, r_max]
    """
    # if neighbors <= 1:
    #     print "This code is meant for training a NN with more than one neighbor."; sys.exit()
    print "Creating XYZ-neighbor-data for:\n - Neighbors: %d \n - Samples  :%d" %(neighbors,size)
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
    p = 4
    q = 0
    a = 1.8
    l = 21  # lambda
    g = 1.2 # gamma
    cos_tc = -1.0/3.0 # 109.47 deg

    eps = 2.1678  # [eV]
    m   = 28.085  # [amu]
    sig = 0.20951 # [nm]
    Ts  = 25156.73798 # eps/kB [K]

    rc = (r < a) # Bool array. "False" cast to 0 and "True" to 1
    U2 = eps*A*(B*r**(-p)-r**(-q)) * np.exp(r-a)**-1 * rc
    U  = np.sum(U2) # Two body terms
    U3 = lambda rij, rik, cos_theta: eps*l*exp(g/(rij-a) + g/(rik-a))* (cos_theta - cos_tc)**2 \
                                     if (rij < a) and (rik < a) else 0.0 # Is this bad practice? lawls
    # Need a double sum to find three body terms
    for j in range(N): # i < j
        for k in range(j+1,N): # i < j < k
            cos_theta = np.dot(xyz[j],xyz[k]) / (r[j]*r[k])
            U        += U3(r[j], r[k], cos_theta)
    return U

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


if __name__ == '__main__':

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
