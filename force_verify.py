from math import *
import numpy as np
from file_management import findPathToData, readXYZ_Files
from symmetry_transform import symmetryTransformBehler
from create_train_data import generate_symmfunc_input_Si_Behler
from plot_tools import plotErrorEvolutionSWvsNN, plotEvolutionSWvsNN_N_diff_epochs, plotForcesSWvsNN, plotLAMMPSforces1atomEvo
import sys
from create_train_data import PES_Stillinger_Weber
import glob
import os
from symmetry_functions import cutoff_cos

def test_structure_N_atom(neigh_cube, nn_eval, plot_single=False, last_timestep=-1):
    """
    Structure:
    xyz = [[0, 0, 0 ], <--- must be origo
           [x2,y2,z2],
           [x3,y3,z3],
           [........],
           [xN,yN,zN]]
    """
    # Will just be a list of numbers Ep
    Ep_SW_list   = []
    Ep_NN_list   = []
    # Will contain tuples (Fx, Fy, Fz)
    Fvec_SW_list = []
    Fvec_NN_list = []

    tot_nmbr_of_atoms = neigh_cube[0].shape[0]

    for t,xyz in enumerate(neigh_cube[0:last_timestep]):
        # Make certain that atoms are centered around (0,0,0):
        if not np.all(xyz[0,:] == 0):
            print "Atoms not properly centered to origo. Exiting!"
            sys.exit(0)
        # Pick out neighbor atoms
        xyz = xyz[1:,:]

        # Potential and forces computed by Stillinger Weber:
        Ep_SW    = PES_Stillinger_Weber(xyz)
        Fvec_SW  = (0,0,0)#evaluate_SW_forces(xyz)

        # Potential and forces computed by trained neural network:
        xyz_symm_0, xyz_0 = nn_eval.create_symvec_from_xyz(xyz, 0)
        xyz_symm_1, xyz_1 = nn_eval.create_symvec_from_xyz(xyz, 1)
        if tot_nmbr_of_atoms > 2:
            xyz_symm_2, xyz_2 = nn_eval.create_symvec_from_xyz(xyz, 2)

        Ep_NN      = nn_eval(xyz_symm_0)

        # This need the actual XYZ coordinates in order to properly differentiate!
        Fvec_NN  = np.zeros(3)#nn_eval.nn_derivative(xyz) # Force from: [G2, G4]
        # print "NN F part 1, Fx, Fy, Fz:", Fvec_NN
        Ep_NN_1  = nn_eval(xyz_symm_1) # Needed to get derivative of NN correct
        Fvec_NN += nn_eval.nn_derivative(xyz_1)
        # # print "NN F part 2, Fx, Fy, Fz:", Fvec_NN
        if tot_nmbr_of_atoms > 2:
            Ep_NN_2  = nn_eval(xyz_symm_2) # Needed to get derivative of NN correct
            Fvec_NN += nn_eval.nn_derivative(xyz_2)
        # print "NN Forces T, Fx, Fy, Fz:", Fvec_NN
        # print " " # New fresh line

        # Append all values to lists:
        Ep_SW_list.append(Ep_SW)
        Ep_NN_list.append(Ep_NN)
        Fvec_SW_list.append(Fvec_SW)
        Fvec_NN_list.append(Fvec_NN)

        # Print out progress
        if t%50 == 0 and t > 500:
            sys.stdout.write("\rTimestep: %d" %t)
            sys.stdout.flush()
    print " "
    # sys.exit(0) # """#########################"""
    nmbr_of_atoms = neigh_cube[0].shape[0]
    if plot_single:
        plotErrorEvolutionSWvsNN(Ep_SW_list, Ep_NN_list, nmbr_of_atoms)
    # Return values for more plotting
    return Ep_SW_list, Ep_NN_list, nmbr_of_atoms, Fvec_SW_list, Fvec_NN_list


if __name__ == '__main__':
    try:
        N = int(sys.argv[1])
        M = int(sys.argv[2])
        last_timestep = M
    except:
        print "Usage:\n>>> python force_verify.py N M"
        print "- N is the different NN-versions to visualize"
        print "- M is the last timestep"
        sys.exit(0)
    path_to_file = "Important_data/TestNN/enfil_sw_3p.xyz"
    neigh_cube   = readXYZ_Files(path_to_file, "no-save-file.txt", return_array=True)
    loadPath     = findPathToData(find_tf_savefile=True)
    master_list  = []

    # Activation functions with derivatives:
    sigmoid  = lambda x: 1.0/(1+np.exp(-x))
    ddx_sig  = lambda x: sigmoid(x)*(1-sigmoid(x))
    relu     = lambda x: np.maximum(x, 0, x) # (in-place of x --> quick!!)
    ddx_relu = lambda x: np.array((x >= 0), dtype=float)
    act_tanh = lambda x: np.tanh(x)
    ddx_tanh = lambda x: 1.0 - np.tanh(x)**2 #1.0/np.cosh(x)**2
    if N == 1:
        plot_single = True
    else:
        plot_single = False
    for i in range(N):
        nn_eval = load_NN_from_file(loadPath, sigmoid, ddx_sig)
        Ep_SW, Ep_NN, N_atoms, F_SW, F_NN = test_structure_N_atom(neigh_cube,
                                                                  nn_eval,
                                                                  plot_single,
                                                                  last_timestep=last_timestep)
        master_list.append([Ep_SW, Ep_NN, N_atoms, nn_eval.what_epoch])

    # Plot each epoch in a new subplot:
    if N > 1:
        plotEvolutionSWvsNN_N_diff_epochs(N, master_list)

    F_LAMMPS = plotLAMMPSforces1atomEvo()[:len(F_NN)]
    # Plot comparison of forces
    plotForcesSWvsNN(F_LAMMPS, F_NN, show=True)



'''def evaluate_SW_forces(xyz_orig):
    """
    Stillinger and Weber,  Phys. Rev. B, v. 31, p. 5262, (1985)

    epsilon, sigma, a, lambda, gamma, costheta0, A, B, p, q
    2.1683 , 2.0951,  1.80,  21.0, 1.20,  -1.0/, 7.049556277,  0.6022245584,  4.0,  0.0
    """
    eps   = 2.1683
    sig   = 2.0951
    a     = 1.80
    lamb  = 21.0
    gam   = 1.2
    A     = 7.049556277
    B     = 0.6022245584
    p     = 4.0
    q     = 0.0
    cos0 = -1.0/3.0

    xyz   = np.copy(xyz_orig)
    f_vec = np.zeros(3)
    r     = np.linalg.norm(xyz, axis=1) # XYZ is only neighbor coordinates
    N     = len(r)

    # Force calculation from 2-body terms:
    dU2dR = A*eps*(-r*sig*(B*(sig/r)**p - (sig/r)**q) + \
            (a*sig - r)**2*(-B*p*(sig/r)**p + q*(sig/r)**q))*np.exp(-sig/(a*sig - r))/(r*(a*sig - r)**2)

    for j in range(N):
        f_vec += xyz[j,:] * dU2dR[j] / r[j] # Minus sign comes from F = -d/dx V

    if len(r) > 1:
        rij   = r[0]
        rik   = r[1]
        rjk   = np.linalg.norm(xyz[0,:]-xyz[1,:])
        xyz_0 = np.zeros(3) # Just for clarity, (0,0,0)

        # Part of code written for sigma = 1
        xyz /= sig
        rij /= sig
        rik /= sig
        rjk /= sig
        xyz_jk = xyz[1]-xyz[0]

        cos_theta_jik = cos_theta_check(np.dot(xyz[0],xyz[1])               / (rij*rik))
        cos_theta_ijk = cos_theta_check(np.dot(xyz_0-xyz[0], xyz[1]-xyz[0]) / (rij*rjk))
        cos_theta_ikj = cos_theta_check(np.dot(xyz_0-xyz[1], xyz[0]-xyz[1]) / (rik*rjk))

        def h(r1, r2, cos_theta): # Serial code!
            if (r1 < a) and (r2 < a):
                term = eps * lamb * exp(gam/(r1-a)) * exp(gam/(r2-a)) * (cos_theta - cos0)**2
            else:
                term = 0.0
            return term

        h_jik = h(rij, rik, cos_theta_jik) # Energy of triplet with angle jik
        h_ijk = h(rij, rjk, cos_theta_ijk) # Energy of triplet with angle ijk
        h_ikj = h(rik, rjk, cos_theta_ikj) # Energy of triplet with angle ikj

        def dhjik_dri():
            if (rij < a) and (rik < a):
                term = -gam*h_jik*((xyz[0]/rij)*1/(rij-a)**2 + (xyz[1]/rik)*1/(rik-a)**2) \
                        + 2*lamb*np.exp(gam/(rij-a) + gam/(rik-a))*(cos_theta_jik - cos0)  \
                        * ((xyz[0]/rij)*(1/rik) + (xyz[1]/rik)*(1/rij) - (xyz[0]/(rij*rik)+xyz[1]/(rik*rij))*cos_theta_jik)
                return term
            else:
                return 0.0

        def dhijk_dri():
            if (rij < a) and (rjk < a):
                term = -gam*h_ijk*((xyz[0]/rij)*1/(rij-a)**2) \
                        + 2*lamb*np.exp(gam/(rij-a) + gam/(rjk-a))*(cos_theta_ijk - cos0) \
                        * ((xyz_jk/rjk)*(1/rij) + (xyz[0]/rij)*(1/rij)*cos_theta_ijk)
                return term
            else:
                return 0.0

        def dhikj_dri():
            if (rik < a) and (rjk < a):
                term = -gam*h_ikj*((xyz[1]/rik)*1/(rik-a)**2) \
                        + 2*lamb*np.exp(gam/(rik-a) + gam/(rjk-a))*(cos_theta_ikj - cos0) \
                        *((-xyz_jk/rjk)*(1/rik) + (xyz[1]/rik)*(1/rik)*cos_theta_ikj)
                return term
            else:
                return 0.0

        # Sum up 2- and 3-body terms
        f_vec_3body = -1.0*(dhjik_dri() + dhijk_dri() + dhikj_dri())

    # Sum up forces
    tot_f_vec = f_vec # 2-body
    # if len(r) > 1:
        # tot_f_vec += f_vec_3body
    tot_f_vec += f_vec_3body #+ f_vec

    # print "SW Forces 2, Fx, Fy, Fz:", f_vec
    # print "SW Forces 3, Fx, Fy, Fz:", f_vec_3body
    # print "SW Forces T, Fx, Fy, Fz:", tot_f_vec
    # print " "

    return tot_f_vec
'''
