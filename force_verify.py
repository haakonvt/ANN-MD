from math import *
import numpy as np
from file_management import findPathToData
from symmetry_transform import symmetryTransformBehler
from create_train_data import generate_symmfunc_input_Si_Behler

def test_structure_3atom():
    """
    Structure:
    xyz = [[0, 0, 0 ], <--- must be origo
           [x2,y2,z2],
           [x3,y3,z3]]
    """
    h = sqrt(3)
    xyz = np.array([[0,0,0],
                    [2,0,0],
                    [1,h,0]], dtype=float)

    f_vec_nn = evaluate_NN_forces(xyz)
    # print "Tot NN force:    ",f_vec_nn
    f_vec_sw = evaluate_SW_forces(xyz)
    # print "Tot SW force:    ",f_vec_sw


def evaluate_SW_forces(xyz):
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

    f_vec = np.zeros(3)

    r  = np.linalg.norm(xyz[1:,:], axis=1) # Only j != i
    U2 = A*eps*(B*(sig/r)**p-(sig/r)**q) * np.exp(sig/(r-a*sig)) * (r < a*sig)
    U2_sum = np.sum(U2)

    dU2dR = -A*eps*sig*(B*(sig/r)**p - (sig/r)**q)*np.exp(sig/(-a*sig + r))/(-a*sig + r)**2 + A*eps*(-B*p*(sig/r)**p/r + q*(sig/r)**q/r)*np.exp(sig/(-a*sig + r))
    for j in [1,2]:
        # print xyz[j,:]*dU2dR[j-1]/r[j-1]
        f_vec += xyz[j,:]*dU2dR[j-1]/r[j-1]
    # print dU2dR, sum(dU2dR)

    """
    i = 0, j = 1, k = 2
    """
    rij           = r[0]
    rik           = r[1]
    rjk           = np.linalg.norm(xyz[1,:]-xyz[2,:])
    cos_theta_jik = np.dot(xyz[1],xyz[2])                / (rij*rik)
    cos_theta_ijk = np.dot(xyz[0]-xyz[1], xyz[2]-xyz[1]) / (rij*rjk)
    cos_theta_ikj = np.dot(xyz[0]-xyz[2], xyz[1]-xyz[2]) / (rik*rjk)
    # print "cos Theta:",cos_theta_jik, cos_theta_ijk, cos_theta_ikj

    def h(r1, r2, cos_theta):
        term = exp(gam*sig/(r1-a*sig)) * exp(gam*sig/(r2-a*sig)) * lamb*eps*(cos_theta - cos0)**2
        return term

    h_jik = h(rij, rik, cos_theta_jik)
    h_ijk = h(rij, rjk, cos_theta_ijk)
    h_ikj = h(rik, rjk, cos_theta_ikj)
    U3 = h_jik + h_ijk + h_ikj

    xyz /= sig
    rij /= sig
    rik /= sig
    rjk /= sig
    xyz_jk = xyz[2]-xyz[1]
    dhjik_dri = -gam*h_jik*((xyz[1]/rij)*1/(rij-a)**2 + (xyz[2]/rik)*1/(rik-a)**2) \
                + 2*lamb*np.exp(gam/(rij-a) + gam/(rik-a))*(cos_theta_jik - cos0)  \
                * ((xyz[1]/rij)*(1/rik) + (xyz[2]/rik)*(1/rij) - (xyz[1]/(rij*rik)+xyz[2]/(rik*rij))*cos_theta_jik)

    dhijk_dri = -gam*h_ijk*((xyz[1]/rij)*1/(rij-a)**2) \
                + 2*lamb*np.exp(gam/(rij-a) + gam/(rjk-a))*(cos_theta_ijk - cos0) \
                * ((xyz_jk/rjk)*(1/rij) + (xyz[1]/rij)*(1/rij)*cos_theta_ijk)

    dhikj_dri = -gam*h_ikj*((xyz[2]/rik)*1/(rik-a)**2) \
                + 2*lamb*np.exp(gam/(rik-a) + gam/(rjk-a))*(cos_theta_ikj - cos0) \
                *((-xyz_jk/rjk)*(1/rik) + (xyz[2]/rik)*(1/rik)*cos_theta_ikj)
    # print "Tot 2-body force:", f_vec
    # print "Tot 3-body force:", dhjik_dri + dhijk_dri + dhikj_dri
    print "U_SW:", U3+U2_sum
    f_vec += dhjik_dri + dhijk_dri + dhikj_dri
    return f_vec


def evaluate_NN_forces(xyz):
    """
    Evaluation checked and fund bug free
    """
    node_w_list, node_biases = read_NN_from_file()
    G_funcs, nmbr_G = generate_symmfunc_input_Si_Behler()
    symm_vec = symmetryTransformBehler(G_funcs, xyz[1:,:]) # Only give neighbors!!

    act_function   = lambda x: 1.0/(1+np.exp(-x))
    vec_prev_layer = symm_vec # First input
    node_w_list[0] = node_w_list[0]
    for i,w_mat in enumerate(node_w_list):
        out_layer  = np.dot(vec_prev_layer, w_mat)
        # print i,"A", out_layer
        out_layer += node_biases[i]
        # print i,"B", out_layer
        if i != len(node_w_list)-1: # We dont use act_func on output layer
            vec_prev_layer = act_function(out_layer)
            # print i,"C", vec_prev_layer
    print "U_NN:", float(out_layer)
    return 0,0,0


def read_NN_from_file():
    loadPath = findPathToData(find_tf_savefile=True)
    # Remove filename at the end (variable length)
    for i,letter in enumerate(reversed(loadPath)):
        if letter == "/": # Find index i of last letter "/" in path
            break
    loadPath = loadPath[:-i] # Now points to the folder, not the file
    with open(loadPath+"graph.dat", "r") as nn_file:
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
    return node_w_list, node_biases








if __name__ == '__main__':
    test_structure_3atom()
    # read_NN_from_file()
