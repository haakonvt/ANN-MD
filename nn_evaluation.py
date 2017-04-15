"""
NOT FINISHED, DONT RUN!
"""

import numpy as np

class neural_network():
    """
    Loads and stores the neural network of choice
    """
    def __init__(self, loadPath, act_func, ddx_act_f):
        node_w_list, node_biases, what_epoch = read_NN_from_file(loadPath)
        G_funcs, nmbr_G  = generate_symmfunc_input_Si_Behler()
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
        f_vec_G2   = np.zeros(3)   # Will contain the forces (Fx, Fy, Fz)
        f_vec_G4   = np.zeros(3)
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
        return dNNdG
    def create_symvec_from_xyz(self, xyz, index):
        """
        XYZ is neighbor-coordinates only!
        """
        symm_vec = symmetryTransformBehler(self.G_funcs)
        return symm_vec
    def what_epoch(self):
        return self.what_epoch


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
    # Convert to proper matrices
    node_w_list = []
    node_w_list.append(node_weights[0:nn_inp,:])
    # print node_weights[nn_inp:nn_inp+nodes,:]
    # raw_input("ASDF")
    for i in range(nn_inp, tot_w_lines-1, nodes): # Loop over hidden layer 2 -->
        node_w_list.append(node_weights[i:i+nodes,:])
    node_w_list.append(node_weights[-1,:]) # This is output node
    return node_w_list, node_biases, what_epoch
