from symmetry_functions import G1,G2,G3,G4,G5
import numpy as np

def symmetryTransform(G_funcs, xyz_i):
    """
    Input:
    x,y,z-coordinates of N particles that make up
    the neighbor-list of particle i.
        [[x1 y1 z1]
         [x2 y2 z2]
         [x3 y3 z3]
         [x4 y4 z4]]
    G_funcs : List of dictionaries to be used. 'None' means dont use.
        [G1, G2, G3, G4, G5]

    Output:
    G1,G2,...,G5: [g11 g12 g13 ... g1N, g21 g22 ... g2N , ... , g5N]
    Some combination of G's are usually used (not all)

    Definitions of G1,...,G5, see:
        "Atom-centered symmetry functions for constructing high-dimensional neural network potentials"
        by Jorg Behler, The Journal of Chemical Physics 134, 074106 (2011).

    ----------------
    Symm |   Vars
    ----------------
    G1   |   rc
    G2   |   rc, rs, eta
    G3   |   rc, kappa
    G4   |   rc, eta, zeta, lambda_c
    G5   |   rc, eta, zeta, lambda_c
    """

    xyz  = xyz_i
    r    = np.linalg.norm(xyz,axis=1)
    G_output = []

    if G_funcs[0] != 0:
        """
        ### This is G1 ###
        ### Variables: ###
            - rc
        """
        G = 0
        N = G_funcs[G][0]
        for n in range(N):
            values = G_funcs[G][1]
            rc     = float(values[n,0])
            G_output.append( G1(r,rc) )
    if G_funcs[1] != 0:
        """
        ### This is G2 ###
        ### Variables: ###
            - rc, rs, eta
        """
        G = 1
        N = G_funcs[G][0]
        for n in range(N):
            values = G_funcs[G][1]
            rc     = float(values[n,0])
            rs     = float(values[n,1])
            eta    = float(values[n,2])
            G_output.append( G2(r,rc, rs, eta) )
    if G_funcs[2] != 0:
        """
        ### This is G3 ###
        ### Variables:
            -rc, kappa
        """
        G = 2
        N = G_funcs[G][0]
        for n in range(N):
            values = G_funcs[G][1]
            rc     = float(values[n,0])
            kappa  = float(values[n,1])
            G_output.append( G3(r,rc,kappa) )
    if G_funcs[3] != 0:
        """
        ### This is G4 ###
        ### Variables:
            - rc, eta, zeta, lambda_c
        """
        G = 3
        N = G_funcs[G][0]
        for n in range(N):
            values   = G_funcs[G][1]
            rc       = float(values[n,0])
            eta      = float(values[n,1])
            zeta     = float(values[n,2])
            lambda_c = float(values[n,3])
            G_output.append( G4(xyz,rc, eta, zeta, lambda_c) )
            # print rc, eta, zeta, lambda_c
    if G_funcs[4] != 0:
        """
        ### This is G5 ###
        ### Variables:
            - rc, eta, zeta, lambda_c
        """
        G = 4
        N = G_funcs[G][0]
        for n in range(N):
            values   = G_funcs[G][1]
            rc       = float(values[n,0])
            eta      = float(values[n,1])
            zeta     = float(values[n,2])
            lambda_c = float(values[n,3])
            G_output.append( G5(xyz,rc, eta, zeta, lambda_c) )
    return np.array(G_output)

def example_generate_G_funcs_input():
    G_vars  = [1,3,2,4,4] # Number of variables symm.func. take as input
    G_funcs = [0,0,0,0,0] # Choose no symm.funcs.
    G_args_list = ["rc[i][j]",
                   "rc[i][j], rs[i][j], eta[i][j]",
                   "rc[i][j], kappa[i][j]",
                   "rc[i][j], eta[i][j], zeta[i][j], lambda_c[i][j]",
                   "rc[i][j], eta[i][j], zeta[i][j], lambda_c[i][j]"]
    # Make use of symmetry function G2 and G4: (indicate how many)
    which_symm_funcs = [2, 4] ; wsf = which_symm_funcs
    how_many_funcs   = [8, 43]; hmf = how_many_funcs

    # This is where the pain begins -_-
    # Note: [3] * 4 evaluates to [3,3,3,3]
    rc       = [[1.8]*hmf[0], [1.8]*hmf[1]]
    rs       = [[0.0]*hmf[0], [None]]
    eta      = [[0.001, 0.01, 0.02, 0.035, 0.06, 0.1, 0.2, 0.4], \
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
    return G_funcs

def example_generate_G_funcs_input2():
    G_vars  = [1,3,2,4,4] # Number of variables symm.func. take as input
    G_funcs = [0,0,0,0,0] # Choose no symm.funcs.
    G_args_list = ["rc[i][j]",
                   "rc[i][j], rs[i][j], eta[i][j]",
                   "rc[i][j], kappa[i][j]",
                   "rc[i][j], eta[i][j], zeta[i][j], lambda_c[i][j]",
                   "rc[i][j], eta[i][j], zeta[i][j], lambda_c[i][j]"]
    # Make use of symmetry function G2 and G4: (indicate how many)
    which_symm_funcs = [1, 2]; wsf = which_symm_funcs
    how_many_funcs   = [4, 4]; hmf = how_many_funcs

    # This is where the pain begins -_-
    # Note: [3] * 4 evaluates to [3,3,3,3]
    rc       = [[0.45, 0.9, 1.35, 1.8], [1.8]*hmf[1]]
    rs       = [[None], [0.0]*hmf[1]]
    eta      = [[None], [0.01, 0.035, 0.06, 0.3]]

    i = 0 # Will be first G-func
    for G,n in zip(wsf, hmf):
        G_funcs[G-1] = [n,  np.zeros((n, G_vars[G-1]))]
        for j in range(n):
            symm_args = eval("np.array([%s])" %(G_args_list[G-1]))
            G_funcs[G-1][1][j] = symm_args
        i += 1
    return G_funcs

if __name__ == '__main__':
    print "This will perform tests of the Stillinger Weber potential"
    print "-------------------------------"

    r_low     = 0.85
    r_high    = 1.8
    size      = 1       # Note this
    neighbors = 15
    PES       = PES_Stillinger_Weber
    xyz_N     = createXYZ(r_low, r_high, size, neighbors)
    Ep        = potentialEnergyGenerator(xyz_N, PES)
    xyz_i     = xyz_N[:,:,0] # Size is just 1, but anyway..
    G_funcs   = example_generate_G_funcs_input()
    G_vec     = symmetryTransform(G_funcs, xyz_i)

    print "Number of symmetry functions used to describe each atom i:", len(G_vec)
    print "-------------------------------"
    print G_vec
