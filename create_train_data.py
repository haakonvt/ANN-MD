# -*- encoding: UTF8 -*-
"""
Generate XYZ data with corresponding energies from some chosen PES:
- Lennard Jones
- Stillinger Weber etc.
"""
from symmetry_transform import symmetryTransform, symmetryTransformBehler
from timeit import default_timer as timer # Best timer indep. of system
from math import pi,sqrt,exp,cos,isnan,sin
from file_management import loadFromFile, readXYZ_Files
from warnings import filterwarnings
import tensorflow as tf
import numpy as np
import glob
import time
import sys
import os

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

def createXYZ(r_min, r_max, size, neighbors=7, histogramPlot=False, verbose=False):
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

def PES_Lennard_Jones(xyz_i):
    """
    Simple LJ pair potential
    """
    eps = 1. # 1.0318 * 10^(-2) eV
    sig = 1. # 3.405 * 10^(-7) meter
    r   = np.linalg.norm(xyz_i, axis=1)
    rc  = 1.6*sig
    LJ0 = abs(4*eps*((sig/rc)**12 - (sig/rc)**6)) # Potential goes to zero at cut
    LJ  = 4*eps*((sig/r)**12 - (sig/r)**6) * (r < rc) + LJ0
    U   =  np.sum( LJ )
    return U

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

    eps = 2.1683 # [eV]
    sig = 2.0951 # [Å]

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
    U2_sum  = np.sum(U2) #/ 2.0 # Each pair share this energy. Contribute half to each
    if isnan(U2_sum):
        U2_sum = U2_serial(r, rc) #/ 2.0
        # print "\nNaN gotten, re-computing with serial code. U2 = ", U2_sum
    # Need a double sum to find three body terms
    U3_sum = 0.0
    for j in range(N): # i < j
        v_rij = xyz[j] # Only depend on j
        rij   = r[j]
        for k in range(j+1,N): # i < j < k
            v_rik = xyz[k]
            v_rjk = xyz[j] - xyz[k]
            rik   = r[k]
            rjk   = np.linalg.norm(v_rjk)
            cos_theta_jik = np.dot(v_rij,  v_rik) / (rij*rik)
            cos_theta_ijk = np.dot(-v_rij, v_rjk) / (rij*rjk) # |rij| = |rji|
            cos_theta_ikj = np.dot(-v_rik,-v_rjk) / (rik*rjk) # rjk_vec = -rkj_vec
            U3_sum += U3(rij, rik, cos_theta_jik) \
                    + U3(rij, rjk, cos_theta_ijk) \
                    + U3(rik, rjk, cos_theta_ikj)
    #U3_sum /= 3 # Each atom in triplet share this energy. Contribute a third to each
    return U3_sum + U2_sum

def createTrainData(size, neighbors, PES, verbose=False):
    if PES == PES_Stillinger_Weber:
        sigma       = 2.0951
        r_low       = 0.85 * sigma
        r_high      = 1.8  * sigma - 1E-8 # SW has a divide by zero at exactly cutoff
        xyz_N = createXYZ(r_low, r_high, size, neighbors, verbose=verbose)
        Ep    = potentialEnergyGenerator(xyz_N, PES)
        Ep    = Ep.reshape([size,1])

        G_funcs, nmbr_G = generate_symmfunc_input_Si_Behler()
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
            r_low       = 0.85 * sigma # Shortest possible dist between atom i and neighbor
            r_high      = 1.8 * sigma - 1E-10 # SW has a divide by zero at exactly cutoff
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

            G_funcs, nmbr_G = generate_symmfunc_input_Si_Behler()
            xTrain          = np.zeros((size, nmbr_G))

            for i in range(size):
                xyz_i       = xyz_N_train[:,:,i]
                # xTrain[i,:] = symmetryTransform(G_funcs, xyz_i)
                xTrain[i,:] = symmetryTransformBehler(G_funcs, xyz_i)
                if verbose and (i+1)%10 == 0:
                    sys.stdout.write('\r' + ' '*80) # White out line
                    percent = round(float(i+1)/size*100., 2)
                    sys.stdout.write('\rTransforming xyz with symmetry functions. %.2f %% complete' %(percent))
                    sys.stdout.flush()
        elif PES == PES_Lennard_Jones:
            sigma       = 1.0
            r_low       = 0.9 * sigma
            r_high      = 1.6 * sigma
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

            G_funcs, nmbr_G = generate_symmfunc_input_LJ(sigma)
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

def createDataDumpBehlerSi():
        PES         = PES_Stillinger_Weber
        size        = 200000
        neighbors   = 10
        sigma       = 2.0951 # 1.0
        r_low       = 0.85 * sigma
        r_high      = 1.8  * sigma - 1E-8 # SW has a divide by zero at exactly cutoff

        xyz_N_train    = createXYZ(r_low, r_high, size, neighbors, verbose=True)
        Ep             = potentialEnergyGenerator(xyz_N_train, PES)
        params, nmbr_G = generate_symmfunc_input_Si_Behler()
        xTrain         = np.zeros((size, nmbr_G))

        for i in range(size):
            xyz_i       = xyz_N_train[:,:,i]
            xTrain[i,:] = symmetryTransformBehler(params, xyz_i)

        dump_data = np.zeros((size, nmbr_G + 1))
        dump_data[:,0]  = Ep
        dump_data[:,1:] = xTrain
        np.random.shuffle(dump_data) # Shuffle the rows of the data i.e. the symmetry vectors
        np.savetxt("SW_Behler_200000_n10.txt", dump_data, delimiter=',')

def generate_symmfunc_input_Si_Behler():
    # Behlers Si-values
    paramsForSymm = []
    with open("Important_data/behler_Si_symm_funcs.txt", "r") as open_file:
        row = -1
        for line in open_file:
            row += 1
            if row == 0:
                continue
            line = line.replace(",", " ")
            linesplit = line.split()
            if row == 1:
                tot_nmbr_symm = int(linesplit[0])
                continue
            if linesplit[0] == "G2":
                """
                'G2', 2.0, 6.0, 0.0 # eta, cut, Rs
                """
                paramsForSymm.append([2] + list(linesplit[1:4]))
            elif linesplit[0] == "G4":
                """
                'G4', 0.01 , 6.0, 1, 1      # eta, cut, zeta, lambda
                """
                paramsForSymm.append([4] + list(linesplit[1:5]))
            else:
                print linesplit[0], "not understood. Should be 'G2' or 'G4'..."
    assert tot_nmbr_symm == len(paramsForSymm)
    return paramsForSymm, tot_nmbr_symm


def generate_symmfunc_input_Si_v1():
    sigma   = 2.0951
    G_funcs = [0,0,0,0,0] # Start out with NO symm.funcs.
    G_vars  = [1,3,2,4,5] # Number of variables symm.func. take as input
    G_args_list = ["rc[i][j]",
                   "rc[i][j], rs[i][j], eta[i][j]",
                   "rc[i][j], kappa[i][j]",
                   "rc[i][j], eta[i][j], zeta[i][j], lambda_c[i][j]",
                   "rc[i][j], eta[i][j], zeta[i][j], lambda_c[i][j]"]
    # Make use of symmetry function G2 and G5: (indicate how many)
    which_symm_funcs = [2, 4] # G5 instead of G4, because SW doesnt care about Rjk
    wsf              = which_symm_funcs
    how_many_funcs   = [10, 120]
    hmf              = how_many_funcs

    # This is where the pain begins -_-
    # Note: [3] * 4 evaluates to [3,3,3,3]
    rc       = [[1.8  * sigma]*10, [1.8  * sigma]*120]
    rs       = [[0.85 * sigma]*10, None]
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

def generate_symmfunc_input_LJ(sigma=1.0):
    """
    Domain:
    a = 0.9 --> b = 1.6 # times sigma
    """
    G_funcs = [0,0,0,0,0] # Start out with NO symm.funcs.
    G_vars  = [1,3,2,4,5] # Number of variables symm.func. take as input
    G_args_list = ["rc[i][j]",
                   "rc[i][j], rs[i][j], eta[i][j]",
                   "rc[i][j], kappa[i][j]",
                   "rc[i][j], eta[i][j], zeta[i][j], lambda_c[i][j]",
                   "rc[i][j], eta[i][j], zeta[i][j], lambda_c[i][j]"]
    # Make use of symmetry function G2 and G5: (indicate how many)
    which_symm_funcs = [2] # G5 instead of G4, because SW doesnt care about Rjk
    wsf              = which_symm_funcs
    how_many_funcs   = [10]
    hmf              = how_many_funcs

    # This is where the pain begins -_-
    # Note: [3] * 4 evaluates to [3,3,3,3]
    rc       = [[1.6*sigma]*10, None]
    rs       = [[0.9*sigma]*10, None]
    eta      = [[0.0, 1.0, 2.5, 5.0, 10.0, 20.0, 40.0, 90.0, 200.0, 500.0], [None]]

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
            xyzr_i   = np.array(row.split(), dtype=float)
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

def NeighListDataToSymmToFile(open_filename, save_filename, size):
    Ep = []
    if size == "all":
        with open(open_filename, 'r') as lammps_file:
            size = 0
            for line in lammps_file:
                size +=1
            save_filename += "%d.txt" %size
    if size == 0:
        print "Length of file is zero! Fix and run again!\nExiting!"
        sys.exit(0)
    with open(open_filename, 'r') as lammps_file:
        """
        File looks like this
        x1 y1 z1 r1^2 x2 y2 z2 r2^2 ... xN yN zN rN^2 Ep
        """
        # G_funcs, nmbr_G = generate_symmfunc_input_Si_v1()
        G_funcs, nmbr_G = generate_symmfunc_input_Si_Behler()
        xTrain          = np.zeros((size, nmbr_G))
        for i,row in enumerate(lammps_file):
            if i >= size:
                continue # Skip to the next row
            xyzr_i = np.array(row.split(), dtype=float)
            n_elem = len(xyzr_i-1)/4 # Remove Ep and compute
            # Ep.append(xyzr_i[-1]) # This is broken by lammps somehow... compute my own below
            xyzr_i = xyzr_i[:-1].reshape(n_elem,4)
            xyz_i  = xyzr_i[:,:-1]
            Ep.append(potentialEnergyGenerator(xyz_i, PES=PES_Stillinger_Weber))
            xTrain[i,:] = symmetryTransformBehler(G_funcs, xyz_i)
            if (i+1)%10 == 0:
                sys.stdout.write('\r' + ' '*80) # White out line
                percent = round(float(i+1)/size*100., 2)
                sys.stdout.write('\rTransforming xyz with symmetry functions. %.2f %% complete' %(percent))
                sys.stdout.flush()
    print "\nNmbr of lines in file", i+1, ", length Ep:", len(Ep), ", size --> file:", size
    dump_data = np.zeros((size, nmbr_G + 1))
    dump_data[:,0]  = Ep
    dump_data[:,1:] = xTrain
    np.random.shuffle(dump_data) # Shuffle the rows of the data i.e. the symmetry vectors
    np.savetxt(save_filename, dump_data, delimiter=',')
    print "Saved symmetry vector training data to file:"
    print '"%s"\n' %save_filename

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

def testAngularInvarianceEpAndSymmFuncs():
    sigma     = 2.0951 # 1.0
    r_low     = 0.85 * sigma
    r_high    = 1.8  * sigma - 1E-8 # SW has a divide by zero at exactly cutoff
    size      = 1
    neighbors = 8
    PES       = PES_Stillinger_Weber
    xyz_N     = createXYZ(r_low, r_high, size, neighbors, verbose=False) # size = 10, neigh = 5
    Ep0       = potentialEnergyGenerator(xyz_N, PES)
    G_funcs, nmbr_G = generate_symmfunc_input_Si_Behler()
    symm_func_vec0  = np.zeros((size, nmbr_G))
    symm_func_vec1  = np.zeros((size, nmbr_G))
    for i in range(size):
        xyz_nl  = xyz_N[:,:,i] # single nl, (neighbor list)
        symm_func_vec0[i,:] = symmetryTransform(G_funcs, xyz_nl) # construct the symmetry vector pre-rotation
        for rotation in range(50): # Do x,y,z rotation a total of 50 times to
            xr, yr, zr = np.random.uniform(0,2*np.pi,3) # Rotate all neighbor atoms with same angles (generated randomly)
            for j in range(xyz_nl.shape[0]): # single x,y,z vector
                xyz_nl[j] = rotateXYZ(xyz_nl[j], xr, yr, zr) # rotate all atoms in neighbor list with same angles
        xyz_N[:,:,i] = xyz_nl
        symm_func_vec1[i,:] = symmetryTransform(G_funcs, xyz_nl) # construct the symmetry vector post-rotation
    Ep1 = potentialEnergyGenerator(xyz_N, PES)
    mae_ep0 = np.mean(np.abs(Ep0))
    mae_ep1 = np.mean(np.abs(Ep1))
    EpDiff  = abs(mae_ep1 - mae_ep0)
    mae_g0  = np.mean(np.abs(symm_func_vec0))
    mae_g1  = np.mean(np.abs(symm_func_vec1))
    GDiff   = abs(mae_g1 - mae_g0)
    print "MAE Ep:", mae_ep0, ", MAE Ep after rotation:", mae_ep1, ", diff:", EpDiff
    print "MAE SymmVec:", mae_g0, ", after rotation:", mae_g1, ", diff:", GDiff


if __name__ == '__main__':
    """
    Suggestion: Only have ONE option set to True at the time.
    """
    # Based on random structures, fixed/variable number of neighbors
    dumpToFile         = False
    dumpMultiple       = True

    # Structures from SW run in lammps
    xyz_to_neigh_lists = False
    dumpXYZ_file       = False # Own algo: "readXYZ_Files"

    # Neig.lists from lammps with John-Anders algo
    dumpLammpsFile     = False

    # Unit tests
    testAngSymm      = False
    testLammps       = False
    testClass        = False

    n_atoms    = 3
    other_info = "" # i.e. "no_3_body"
    if xyz_to_neigh_lists:
        """
        Takes XYZ files from LAMMPS dump and makes neighbor lists
        """
        cutoff         = 3.77118 # Stillinger-Weber
        samples_per_dt = "all"   # Integer value or "all"
        test_boundary  = False   # Just use atoms wherever they are
        file_path = "Important_data/TestNN/enfil_sw_%sp%s.xyz" %(n_atoms,other_info)
        save_file = "Important_data/neigh_list_from_xyz_%sp%s.txt" %(n_atoms,other_info)
        readXYZ_Files(file_path, save_file, samples_per_dt, cutoff, test_boundary)

    if dumpXYZ_file:
        """
        Takes neighbour lists and makes symmetry data / training data
        """
        size          = "all" # should be <= rows in file!!!!
        open_filename = "Important_data/neigh_list_from_xyz_%sp%s.txt" %(n_atoms,other_info)
        save_filename = "SW_train_xyz_%s.txt" %str(size)
        if size == "all":
            save_filename = "SW_train_xyz_%sp%s_" %(n_atoms,other_info)
        NeighListDataToSymmToFile(open_filename, save_filename, size)

    if testLammps:
        filename = "Important_data/neighbours.txt"
        testLammpsData(filename)

    if dumpLammpsFile:
        size          = 10000 # should be <= rows in file!!!!
        open_filename = "Important_data/neigh_100K.txt"
        save_filename = "SW_train_lammps_%d_BEH.txt" %size
        NeighListDataToSymmToFile(open_filename, save_filename, size)

    if dumpToFile:
        if True:
            # This is SW
            size = 10000
            neighbors = 1
            # filename = "stillinger-weber-symmetry-data.txt"
            filename  = "SW_train_rs_%s_n%s.txt" %(str(size), str(neighbors))
            print "When run directly (like now), this file dumps training data to file:"
            print '"%s"' %filename
            print "-------------------------------"
            print "Neighbors", neighbors
            print "-------------------------------"
            PES = PES_Stillinger_Weber
            t0 = timer()
            createTrainDataDump(size, neighbors, PES, filename, \
                                only_concatenate=False, verbose=True)
            t1 = timer() - t0
            print "\nComputation took: %.2f seconds" %t1
        else:
            # This is LJ
            size = 10000
            neighbors = 8
            # filename = "stillinger-weber-symmetry-data.txt"
            filename  = "LJ_train_rs_%s_n%s.txt" %(str(size), str(neighbors))
            print "When run directly (like now), this file dumps training data to file:"
            print '"%s"' %filename
            print "-------------------------------"
            print "Neighbors", neighbors
            print "-------------------------------"
            PES       = PES_Lennard_Jones
            t0 = timer()
            createTrainDataDump(size, neighbors, PES, filename, \
                                only_concatenate=False, verbose=True)
            t1 = timer() - t0
            print "\nComputation took: %.2f seconds" %t1

    if dumpMultiple:
        size = 2000 # PER single value in neigh_list
        # neigh_list = [4]*2 + [5]*6 + [6]*13 + [7]*14 + [8]*9 + [9]*3 + [10] # len: 48
        neigh_list = [1,2,2,2]
        t0_tot = timer()
        PES = PES_Stillinger_Weber
        for ID,neighbors in enumerate(neigh_list):
            filename  = "SW_train_%s_n%s_%s.txt" %(str(size), str(neighbors), str(ID))
            print "When run directly (like now), this file dumps training data to file:"
            print '"%s"' %filename
            print "-------------------------------"
            print "Neighbors", neighbors
            print "-------------------------------"
            t0  = timer()
            createTrainDataDump(size, neighbors, PES, filename, \
                                only_concatenate=False, verbose=True, \
                                no_load=True)
            t1 = timer() - t0
            print "\nComputation of %d neighbors took: %.2f seconds" %(neighbors,t1)
        t1 = timer() - t0_tot
        # Load all files into one master file
        createTrainDataDump(0, 0, PES, "SW_train_manyneigh_%d.txt" %(size*len(neigh_list)), \
                            only_concatenate=True, verbose=True, no_load=False)
        if t1 > 1000:
            t1 /= 60.
            print "\nTotal computation took: %.2f minutes" %t1
        else:
            print "\nTotal computation took: %.2f seconds" %t1

    if testClass:
        testSize  = 100 # Remove these from training set
        filename  = "test-class-symmetry-data.txt"
        all_data  = loadFromFile(testSize, filename)
        xTrain, yTrain = all_data(1)
        print xTrain[:,0:5], "\n", yTrain
        xTrain, yTrain = all_data(1)
        print xTrain[:,0:5], "\n", yTrain # Make sure this is different from above print out

    if testAngSymm:
        testAngularInvarianceEpAndSymmFuncs()
