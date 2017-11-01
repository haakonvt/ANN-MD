import numpy as np
import sys

def load_symm_vector(filename,max_to_load=1000):
    all_data = []
    counter  = 0
    with open(filename, 'r') as symm_vec:
        for line in symm_vec:
            all_data.append(np.fromstring(line, dtype=float, sep=','))
            counter += 1
            if counter > max_to_load:
                break
    all_data = np.asarray(all_data) # Convert to matrix
    pot_E    = all_data[:,0]
    all_data = all_data[:,1:] # Remove Ep from symm.vec.s
    return all_data,pot_E

def compare_symm_funcs(all_data, Ep, pst_rem=0.2, random_sample=True, random_sample_pst=0.75, verbose=False):
    N = all_data.shape[0]
    all_MSEs = np.zeros((N,2))
    all_MAEs = np.zeros((N,2))
    for i in range(N):
        if N > 500 and i%50==0:
            sys.stdout.write('\rAtom %g / %g done!' %(i,N)); sys.stdout.flush()
        sf = all_data[i,:]
        MSE_low  = 1E100    # Start out with guaranteed worst case
        j_close  = -1
        MSE = np.mean((sf-all_data)**2)#,axis=1) # Measure of similarity
        MAE = np.mean(np.abs(sf-all_data))#,axis=1) # Measure of similarity
        all_MSEs[i,:] = MSE,i
        all_MAEs[i,:] = MAE,i
        # print "atom %g:"%i, MSE
    sys.stdout.write('\rAtom %g / %g done!\n' %(N,N)); sys.stdout.flush()
    ind_MSE = all_MSEs[:,0].argsort() # Sort from least to most unique
    ind_MAE = all_MAEs[:,0].argsort() # Sort from least to most unique
    ind_list = list(ind_MSE[:int(N*pst_rem)]) + list(ind_MAE[:int(N*pst_rem)])
    if verbose:
        print "Indices to remove [before]:",len(ind_list)
    if random_sample:
        """ Let X pst. of the doomed training examples live, randomly """
        ind_to_survive = int(random_sample_pst*len(ind_list))
        ind_list = np.random.choice(ind_list, ind_to_survive, replace=False)
    ind_rem = np.array(list(set(ind_list))) # 'set' removes duplicates
    if verbose:
        print "...After:",len(ind_rem)
    pruned_data = np.delete(all_data, ind_rem, axis=0)
    pruned_Ep   = np.delete(Ep, ind_rem)
    return pruned_data, pruned_Ep

def save_pruned_data_to_file(filename,dump_data,dump_Ep):
    new_shape = (dump_Ep.shape[0],1+dump_data.shape[1])
    all_data       = np.zeros(new_shape)
    all_data[:,0]  = dump_Ep
    all_data[:,1:] = dump_data
    # np.random.shuffle(all_data) # Optional reshuffle
    filename += "_" + str(new_shape[0]) + "_pruned.txt"
    np.savetxt(filename, all_data, delimiter=',')
    print "\nPruned symmetry vector training data saved to file:"
    print '"%s"\n' %(filename)

if __name__ == '__main__':
    """
    Removes ~15 percent of training data with similarity measures L1 and L2.

    Note: The pruned files can be combined with "dumpMultiple" in "create_train_data.py"
    OBS:  Filename and specific settings MUST be set before running.
    """
    N_max        = 20010
    filename     = "SW_train_xyz_2048p_20010"
    symm_vecs,Ep = load_symm_vector(filename+".txt",max_to_load=N_max)#int(sys.argv[1]))
    pruned_data, pruned_Ep = compare_symm_funcs(symm_vecs, Ep)

    filename     = "SW_train_xyz_many"
    save_pruned_data_to_file(filename, pruned_data, pruned_Ep)

    # if False:
    #     # Similarity measures scales as N^2, but we dont need to compute over
    #     # the entire training set. We split up for efficiency:
    #     filename     = "SW_train_xyz_"
    #     split        = 20
    #     factor       = N_max/split
    #     for i in range(split):
    #         pruned_data, pruned_Ep = compare_symm_funcs(symm_vecs[i*factor:(i+1)*factor,:],\
    #                                                     Ep[i*factor:(i+1)*factor])
    #         save_pruned_data_to_file(filename+"pruned_%g.txt"%i,pruned_data, pruned_Ep)
    #         # print symm_vecs[:,0],"-----\n",pruned_data[:,0]
