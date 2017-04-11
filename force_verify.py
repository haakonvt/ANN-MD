from math import *
import numpy as np
from file_management import findPathToData, readXYZ_Files
from symmetry_transform import symmetryTransformBehler
from create_train_data import generate_symmfunc_input_Si_Behler
from plot_tools import plotErrorEvolutionSWvsNN, plotEvolutionSWvsNN_N_diff_epochs, plotForcesSWvsNN
import sys
from create_train_data import PES_Stillinger_Weber
import glob
import os
from symmetry_functions import cutoff_cos

def test_structure_N_atom(neigh_cube, nn_eval, plot_single=False):
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

    for t,xyz in enumerate(neigh_cube[0:3500]):
        # Make certain that atoms are centered around (0,0,0):
        if not np.all(xyz[0,:] == 0):
            print "Atoms not properly centered to origo. Exiting!"
            sys.exit(0)
        # Pick out neighbor atoms
        xyz = xyz[1:,:]
        # Potential and forces computed by Stillinger Weber:
        Ep_SW    = PES_Stillinger_Weber(xyz)
        Fvec_SW  = evaluate_SW_forces(xyz)
        Ep_SW_list.append(Ep_SW)
        Fvec_SW_list.append(Fvec_SW)
        # Potential and forces computed by trained neural network:
        xyz_symm = nn_eval.create_symvec_from_xyz(xyz)
        Ep_NN    = nn_eval(xyz_symm)
        Fvec_NN  = nn_eval.nn_derivative(xyz) # This need the actual XYZ coordinates in order to properly differentiate!
        Ep_NN_list.append(Ep_NN)
        Fvec_NN_list.append(Fvec_NN)
        if t%50 == 0: # Print out progress
            sys.stdout.write("\rTimestep: %d" %t)
            sys.stdout.flush()
    print " "
    nmbr_of_atoms = neigh_cube[0].shape[0]
    if plot_single:
        plotErrorEvolutionSWvsNN(Ep_SW_list, Ep_NN_list, nmbr_of_atoms)
    # Return values for more plotting
    return Ep_SW_list, Ep_NN_list, nmbr_of_atoms, Fvec_SW_list, Fvec_NN_list


class load_NN_from_file():
    """
    Loads and stores the neural network of choice
    """
    def __init__(self, loadPath, act_func, ddx_act_f):
        node_w_list, node_biases, what_epoch = read_NN_from_file(loadPath)
        G_funcs, nmbr_G = generate_symmfunc_input_Si_Behler()
        self.what_epoch  = what_epoch
        self.all_layers  = len(node_w_list) + 1
        self.hdn_layers  = self.all_layers - 2
        self.node_biases = node_biases
        self.G_funcs     = G_funcs
        self.nmbr_G      = nmbr_G
        self.act_func    = act_func
        self.ddx_act_f   = ddx_act_f
        self.node_w_list = node_w_list
        # Force last weight vector to be Nx1 matrix
        self.node_w_list[-1] = node_w_list[-1].reshape(node_w_list[-1].shape[0],1)
    def __call__(self, sym_vec):
        """
        Evaluates the neural network and returns the energy
        """
        self.node_sum  = []
        vec_prev_layer = sym_vec # First input
        self.node_sum.append(vec_prev_layer)
        # Evaluate the neural network:
        for i,w_mat in enumerate(self.node_w_list):
            out_layer  = np.dot(vec_prev_layer, w_mat)
            out_layer += self.node_biases[i]
            if i != len(self.node_w_list)-1: # We dont use act_func on output layer
                self.node_sum.append(out_layer) # Dont care about last sum because its the same as the final output (energy)!
                vec_prev_layer = self.act_func(out_layer)
        return float(out_layer)
    def nn_derivative(self, xyz):
        """
        Essentially what is done during backpropagation, except we also need
        to differentiate symmetry functions with respect to cartesian coordinates.

        NB: xyz need to contain neighbor coordinates only!
        """
        tot_layers = self.all_layers
        hdn_layers = self.hdn_layers
        ddx_act_f  = self.ddx_act_f
        f_vec      = np.zeros(3)   # Will contain the forces (Fx, Fy, Fz)
        output     = np.array(1.0) # Derivative of output neruon is 1 since its f(x) = x
        deriv_list = [0] * tot_layers
        deriv_list[-1] = output
        # Loop backwards through layers of NN (from output to the input)
        for i in reversed(range(1,hdn_layers+1)):
            weights_trans = np.transpose(self.node_w_list[i])
            deriv_list[i] = np.dot(deriv_list[i+1], weights_trans) \
                          * ddx_act_f(self.node_sum[i])
        # Assume linear activation function used on input nodes:
        weights_trans = np.transpose(self.node_w_list[0])
        deriv_list[0] = np.dot(deriv_list[1], weights_trans)
        dNNdG         = deriv_list[0].transpose()
        # Loop over all values of the symmetry vector
        for i in range(self.nmbr_G):
            which_symm = self.G_funcs[i][0]
            dEdG_i     = float(-dNNdG[i])
            if which_symm == 2: # This is G2
                # print "G2", dNNdG[i]
                _, eta, rc, rs = self.G_funcs[i]
                dG2dXYZ_values = dG2dXYZ(xyz, float(eta), float(rc), float(rs))
                f_vec += dEdG_i * dG2dXYZ_values
            elif which_symm == 4: # This is G4
                # print "G4", dNNdG[i]
                _, eta, rc, zeta, lamb = self.G_funcs[i]
                dG4dXYZ_values = dG4dXYZ(xyz, float(eta), float(rc), float(zeta), float(lamb))
                f_vec += dEdG_i * dG4dXYZ_values
            else:
                print "Only use symmetry functions G2 or G4! Exiting!"
                sys.exit(0)
        return f_vec

    def create_symvec_from_xyz(self, xyz):
        symm_vec = symmetryTransformBehler(self.G_funcs, xyz)
        return symm_vec
    def what_epoch(self):
        return self.what_epoch

def dG4dXYZ(xyz, eta, rc, zeta, lamb):
    """
    Derivative of symmetry function G4 w.r.t xij, yij, zij.
    """
    # Vectors of x,y,z-coordinates for neighbours
    xij = xyz[0,0]
    yij = xyz[0,1]
    zij = xyz[0,2]
    xik = xyz[1,0]
    yik = xyz[1,1]
    zik = xyz[1,2]
    r   = np.linalg.norm(xyz, axis=1) # Neighbour distances
    rij = r[0]
    rik = r[1]
    rjk = np.linalg.norm(xyz[0,:]-xyz[1,:])
    cosTheta = (xij*xik + yij*yik + zij*zik) / (rij*rik)

    # ...and finally the derivaties from sympy: (hold on to something!!)
    dG4_dxij = -2**(-zeta)*(cos(pi*rik/rc) + 1)*(cos(pi*rjk/rc) + 1)\
               *(0.5*eta*rc*rij**2*rik*xij*(cosTheta*lamb + 1)**(zeta + 1)\
               *(cos(pi*rij/rc) + 1) + 0.25*lamb*rc*zeta*(cosTheta*lamb + 1)**zeta\
               *(cosTheta*rik*xij - rij*xik)*(cos(pi*rij/rc) + 1) + 0.25*pi*rij*rik*xij\
               *(cosTheta*lamb + 1)**(zeta + 1)*sin(pi*rij/rc))*exp(-eta*(rij**2 + rik**2 + rjk**2))\
               /(rc*rij**2*rik*(cosTheta*lamb + 1))

    dG4_dyij = -2**(-zeta)*(cos(pi*rik/rc) + 1)*(cos(pi*rjk/rc) + 1)\
               *(0.5*eta*rc*rij**2*rik*yij*(cosTheta*lamb + 1)**(zeta + 1)\
               *(cos(pi*rij/rc) + 1) + 0.25*lamb*rc*zeta*(cosTheta*lamb + 1)**zeta\
               *(cosTheta*rik*yij - rij*yik)*(cos(pi*rij/rc) + 1) + 0.25*pi*rij*rik*yij\
               *(cosTheta*lamb + 1)**(zeta + 1)*sin(pi*rij/rc))*exp(-eta*(rij**2 + rik**2 + rjk**2))\
               /(rc*rij**2*rik*(cosTheta*lamb + 1))

    dG4_dzij = -2**(-zeta)*(cos(pi*rik/rc) + 1)*(cos(pi*rjk/rc) + 1)\
               *(0.5*eta*rc*rij**2*rik*zij*(cosTheta*lamb + 1)**(zeta + 1)\
               *(cos(pi*rij/rc) + 1) + 0.25*lamb*rc*zeta*(cosTheta*lamb + 1)**zeta\
               *(cosTheta*rik*zij - rij*zik)*(cos(pi*rij/rc) + 1) + 0.25*pi*rij*rik*zij\
               *(cosTheta*lamb + 1)**(zeta + 1)*sin(pi*rij/rc))*exp(-eta*(rij**2 + rik**2 + rjk**2))\
               /(rc*rij**2*rik*(cosTheta*lamb + 1))
    return -1.0*np.array([dG4_dxij, dG4_dyij, dG4_dzij])

def dG2dXYZ(xyz, eta, rc, rs):
    """
    Derivative of symmetry function G2 w.r.t xij, yij, zij.
    """
    # Vectors of x,y,z-coordinates for neighbours
    xij = xyz[:,0]
    yij = xyz[:,1]
    zij = xyz[:,2]
    r   = np.linalg.norm(xyz, axis=1) # Neighbour distances
    term = (eta*rc*(r - rs)*(np.cos(pi*r/rc) + 1.0) + 0.5*pi*np.sin(pi*r/rc)) \
           * np.exp(-eta*(r - rs)**2)/(rc*r)
    Fx = -1.0*np.sum(-xij * term) # Minus sign comes from F = -d/dx V
    Fy = -1.0*np.sum(-yij * term)
    Fz = -1.0*np.sum(-zij * term)
    return np.array([Fx, Fy, Fz])

def evaluate_SW_forces(xyz_orig):
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

    def U2_serial(r):
        if (r < a*sig):
            return A*eps*(B*(sig/r)**p-(sig/r)**q) * np.exp(sig/(r-a*sig))
        else:
            return 0.0
    # U2 = np.array([U2_serial(r[0])])#, U2_serial(r[1])]) # TODO: ONLY WORKS WITH 2 neighbors!!

    # Force calculation from 2-body terms:
    dU2dR = A*eps*(-r*sig*(B*(sig/r)**p - (sig/r)**q) + \
            (a*sig - r)**2*(-B*p*(sig/r)**p + q*(sig/r)**q))*np.exp(-sig/(a*sig - r))/(r*(a*sig - r)**2)
    # -A*eps*sig*(B*(sig/r)**p - (sig/r)**q)*np.exp(sig/(-a*sig + r))/(-a*sig + r)**2 + A*eps*(-B*p*(sig/r)**p/r + q*(sig/r)**q/r)*np.exp(sig/(-a*sig + r))
    # -A*eps*sig*(B*(sig/r)**p - (sig/r)**q)*np.exp(sig/(-a*sig + r))/(-a*sig + r)**2 + A*eps*(-B*p*(sig/r)**p/r + q*(sig/r)**q/r)*np.exp(sig/(-a*sig + r))

    for j in range(N):
        f_vec += xyz[j,:] * dU2dR[j] / r[j]

    rij   = r[0]
    rik   = r[1]
    rjk   = np.linalg.norm(xyz[0,:]-xyz[1,:])
    xyz_0 = np.zeros(3) # Just for clarity, (0,0,0)

    def cos_theta_check(cos_theta):
        if cos_theta > 1:
            print "CosTheta value fixed to 1:", cos_theta
            cos_theta = 1.0
        elif cos_theta < -1:
            print "CosTheta value fixed to -1:", cos_theta
            cos_theta = -1.0
        return cos_theta

    cos_theta_jik = cos_theta_check(np.dot(xyz[0],xyz[1])               / (rij*rik))
    cos_theta_ijk = cos_theta_check(np.dot(xyz_0-xyz[0], xyz[1]-xyz[0]) / (rij*rjk))
    cos_theta_ikj = cos_theta_check(np.dot(xyz_0-xyz[1], xyz[0]-xyz[1]) / (rik*rjk))

    xyz /= sig
    rij /= sig
    rik /= sig
    rjk /= sig
    xyz_jk = xyz[1]-xyz[0]

    def h(r1, r2, cos_theta): # Serial code!
        if (r1 < a) and (r2 < a):
            term = eps * lamb * exp(gam/(r1-a)) * exp(gam/(r2-a)) * (cos_theta - cos0)**2
        else:
            term = 0.0
        return term

    h_jik = h(rij, rik, cos_theta_jik)
    h_ijk = h(rij, rjk, cos_theta_ijk)
    h_ikj = h(rik, rjk, cos_theta_ikj)

    dhjik_dri = -gam*h_jik*((xyz[0]/rij)*1/(rij-a)**2 + (xyz[1]/rik)*1/(rik-a)**2) \
                + 2*lamb*np.exp(gam/(rij-a) + gam/(rik-a))*(cos_theta_jik - cos0)  \
                * ((xyz[0]/rij)*(1/rik) + (xyz[1]/rik)*(1/rij) - (xyz[0]/(rij*rik)+xyz[1]/(rik*rij))*cos_theta_jik)

    dhijk_dri = -gam*h_ijk*((xyz[0]/rij)*1/(rij-a)**2) \
                + 2*lamb*np.exp(gam/(rij-a) + gam/(rjk-a))*(cos_theta_ijk - cos0) \
                * ((xyz_jk/rjk)*(1/rij) + (xyz[0]/rij)*(1/rij)*cos_theta_ijk)

    dhikj_dri = -gam*h_ikj*((xyz[1]/rik)*1/(rik-a)**2) \
                + 2*lamb*np.exp(gam/(rik-a) + gam/(rjk-a))*(cos_theta_ikj - cos0) \
                *((-xyz_jk/rjk)*(1/rik) + (xyz[1]/rik)*(1/rik)*cos_theta_ikj)

    # Sum up 2- and 3-body terms
    tot_f_vec = f_vec #+ (dhjik_dri + dhijk_dri + dhikj_dri)/3.

    # print "Tot 2-body force:", f_vec
    # print "Tot 3-body force:", dhjik_dri + dhijk_dri + dhikj_dri
    # print "Total force     :", tot_f_vec"""
    return tot_f_vec


def read_NN_from_file(loadPath):
    which_graph_file = loadPath + "graph.dat"
    # Remove filename at the end (variable length)
    for i,letter in enumerate(reversed(loadPath)):
        if letter == "/": # Find index i of last letter "/" in path
            break
    loadPath = loadPath[:-i] # Now points to the folder, not the file
    folder_list = glob.glob(loadPath + "graph*")
    folder_list = sorted(folder_list, key=os.path.getmtime)# Sort by time created
    if len(folder_list) == 1:
        which_graph_file = folder_list[0]
    else:
        print "\nFound multiple graph_EPOCHS.dat-files. Choose one:"
        for i,graph_file in enumerate(folder_list):
            print " %d)"%i, graph_file[61:]
        i = int(raw_input("Input an integer: "))
        which_graph_file = folder_list[i] # Dont use loadPath, since glob-list has entire (relative) path
    what_epoch       = int(which_graph_file[66:-4])
    with open(which_graph_file, "r") as nn_file:
        hdn_layers, nodes, act_func, nn_inp, nn_out = nn_file.readline().strip().split()
        hdn_layers   = int(hdn_layers); nodes = int(nodes); nn_inp = int(nn_inp); nn_out = int(nn_out)
        tot_w_lines  = (hdn_layers-1)*nodes + nn_inp + 1 # Last one from output layer
        node_weights = np.zeros((tot_w_lines,nodes))
        node_biases  = []
        line_index = 0 # We dont care about first line, already taken care of above
        for line in nn_file:
            line = line.strip().split() # Remove newline and spaces. Then split into list of numbers
            if not line:
                continue # Skip line between W and B
            line = np.array(line, dtype=float) # Convert list of strings to array of floats
            if line_index < tot_w_lines: # Reading node weights
                # print "W",line_index,line[0] # Error checking
                node_weights[line_index,:] = line
            else:
                # print "B",line_index,line[0] # Error checking
                node_biases.append(line)
            line_index += 1
    """
    Convert to proper matrices
    """
    node_w_list = []
    node_w_list.append(node_weights[0:nn_inp,:])
    # print node_weights[nn_inp:nn_inp+nodes,:]
    # raw_input("ASDF")
    for i in range(nn_inp, tot_w_lines-1, nodes): # Loop over hidden layer 2 -->
        node_w_list.append(node_weights[i:i+nodes,:])
    node_w_list.append(node_weights[-1,:]) # This is output node
    return node_w_list, node_biases, what_epoch


if __name__ == '__main__':
    try:
        N = int(sys.argv[1])
    except:
        print "Usage:\n>>> python force_verify.py N"
        print "...where N is the different NN-versions to visualize"
        sys.exit(0)
    path_to_file = "Important_data/TestNN/enfil_sw_3p.xyz"
    neigh_cube   = readXYZ_Files(path_to_file, "no-save-file.txt", return_array=True)
    loadPath     = findPathToData(find_tf_savefile=True)
    master_list  = []

    # Activation functions with derivatives:
    sigmoid  = lambda x: 1.0/(1+np.exp(-x))
    ddx_sig  = lambda x: sigmoid(x)*(1-sigmoid(x))
    relu     = lambda x: np.maximum(x, 0, x) # relu (in-place, quick!)
    # ddx_relu    = lambda x: # TODO: Not implemented
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
                                                                  plot_single=False)
        master_list.append([Ep_SW, Ep_NN, N_atoms, nn_eval.what_epoch])

    # Plot each epoch on new subplot:
    if N > 1:
        plotEvolutionSWvsNN_N_diff_epochs(N, master_list)

    # Plot comparison of forces
    plotForcesSWvsNN(F_SW, F_NN)
    # print "F SW 0:", F_SW[0]
    # print "F NN 0:", F_NN[0]
    # print "diff 0:", F_NN[0] / F_SW[0], "\n"
    # print "F SW 1:", F_SW[1]
    # print "F NN 1:", F_NN[1]
    # print "diff 1:", F_NN[1] / F_SW[1]
