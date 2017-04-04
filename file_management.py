import os
import sys
import glob
import shutil
import datetime
import numpy as np

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
            print 'Found no training data called:\n"%s"\"...exiting!' %filename
            sys.exit(0)
        if shuffle_rows:
            np.random.shuffle(self.buffer) # Shuffles rows only (not columns) by default *yey*
        # print "Tot. data points loaded from file:", self.buffer.shape[0]
        self.testData  = self.buffer[0:testSizeSkip,:] # Pick out test data from total
        self.buffer    = self.buffer[testSizeSkip:,:]  # Use rest of data for training
        self.totTrainData = self.buffer.shape[0]

    def __call__(self, size, return_test=False, verbose=False, shuffle=False):
        """
        Returns the next batch of size 'size' which is a set of rows from the loaded file
        """
        epochIsDone = False
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
                epochIsDone = True # Move to next epoch, all data has been seen
                if verbose:
                    print "\nWarning: All training data 'used', shuffling (most likely) & starting over!\n"
                if shuffle:
                    np.random.shuffle(self.buffer)
                self.index = 0 # Dont use test data for training!
                i          = 0
            if size <= self.totTrainData:
                symm_vec_train = self.buffer[i:i+size, 1:] # Second column->last
                Ep_train       = self.buffer[i:i+size, 0]  # First column
                Ep_train       = Ep_train.reshape([size,1])
                self.index += size # Update so that next time class is called, we get the next items
                return symm_vec_train, Ep_train, epochIsDone
            else:
                print "Requested batch size %d, is larger than data set %d" %(size, self.totTrainData)
    def number_of_train_data(self):
        """
        Returns the total number of data points after test size has been removed
        """
        return len(self.buffer[:,0])
    def return_all_data(self):
        """
        Assumes testSize = 0 or else this will just return train data
        """
        return self.buffer

def findPathToData(find_tf_savefile=False):
    folder      = "Important_data/Trained_networks/"
    folder_list = glob.glob(folder + "KEEP*")
    if not folder_list:
        print " ERROR: No directories found! Exiting!"
        sys.exit(0)
    print "\nFolders with training data found (marked 'keep'):"
    for i,ifold in enumerate(folder_list):
        print " %g) %s" %(i,ifold[len(folder):])
    imax = i
    sentence = "Please specify which folder to load from, (0-%d): " %imax
    while True:
        try:
            inp = int(raw_input(sentence))
        except:
            print "Try again, enter the number corresponding to the folder!"
        if inp >= 0 and inp <= imax:
            break
        else:
            print "Try again, enter the number corresponding to the folder!"
    if not find_tf_savefile:
        return folder_list[inp] + "/"
    else:
        folder = folder_list[inp]
        for a_file in os.listdir(folder):
            if a_file[:3] == "run":
                epoch = ""
                for letter in a_file[3:]:
                    if letter == ".":
                        break
                    epoch += letter
        folder += "/run" + epoch
        return folder

def keepData(save_dir):
    yes_or_no = raw_input("Do you want to keep this run? (Mark as 'keep') [Y/n] ")
    if yes_or_no in ["yes","YES","Yes","y","Y",""]:
        new_save_dir = save_dir[0:32] + "KEEP-" + save_dir[32:]
        os.rename(save_dir, new_save_dir)
        shutil.copy2("Important_data/parameters.dat", new_save_dir)
        print "Data from this run kept safely on disk:\n'%s'" %(new_save_dir)

def deleteOldData():
    directory = "Important_data/Trained_networks/"
    for something in os.listdir(directory):
        if something[:5] == "KEEP-":
            continue
        else:
            file_path = directory + something
            if os.path.isdir(file_path): # Leave files
                shutil.rmtree(file_path)
            else:
                os.remove(file_path)     # ...or not

def saveGraphFunc(sess, weights, biases, epoch, hiddenLayers, nNodes, save_dir, activation_function):
    """
    Saves the neural network weights and biases to file,
    in a format readably by 'humans'
    """
    saveFileName = save_dir + "/graph.dat"
    with open(saveFileName, 'w') as outFile:
        outStr = "%1d %1d %s 48 1" % (hiddenLayers, nNodes, activation_function) #TODO: 48 = nmbr of inputs
        outFile.write(outStr + '\n')
        size = len(sess.run(weights))
        for i in range(size):
            i_weights = sess.run(weights[i])
            if i < size-1:
                for j in range(len(i_weights)):
                    for k in range(len(i_weights[0])):
                        outFile.write("%s" % repr(i_weights[j][k]))
                        outFile.write(" ")
                    outFile.write("\n")
            else:
                for j in range(len(i_weights[0])):
                    for k in range(len(i_weights)):
                        outFile.write("%s" % repr(i_weights[k][j]))
                        outFile.write(" ")
                    outFile.write("\n")
        outFile.write("\n")
        for biasVariable in biases:
            i_biases = sess.run(biasVariable)
            for j in range(len(i_biases)):
                outFile.write("%s" % repr(i_biases[j]))
                outFile.write(" ")
            outFile.write("\n")

def timeStamp():
    return datetime.datetime.now().strftime("%H.%M.%S--%d.%m.%Y")

def readXYZ_Files(path_to_file, save_name, cutoff=3.77118):
    """
    Create the master list.
    Neighbouring atoms will vary, so I use a nested list
    """
    print "NB: This might take some minutes, depending on the number of time steps!"
    master_neigh_list = []
    samples_per_dt    = 30
    tot_nmbr_of_atoms = 0
    time_step         = 0
    with open(path_to_file, 'r') as xyzFile:
        row = -1
        for line in xyzFile:
            row += 1
            if time_step == 0:
                if row == 0:
                    tot_nmbr_of_atoms = int(line)
                    xyz_ti = np.zeros((tot_nmbr_of_atoms, 3))
                    print "Number of atoms:", tot_nmbr_of_atoms
                    continue
                elif row == 1:
                    print 'Comment line said: "%s"' %line[:-1]
                    continue
            elif row == 0 or row == 1:
                continue
            index = row - 2
            xyz_ti[index,:] = line.split()[1:]
            if index == tot_nmbr_of_atoms-1:
                sys.stdout.write("\rTime step %d done!" %time_step)
                sys.stdout.flush()
                row = -1
                time_step += 1
                compute_neigh_lists(xyz_ti, master_neigh_list, samples_per_dt, cutoff)
    with open(save_name, 'w') as xyzFile:
        print "\nWriting neighbourlists to file:"
        print save_name
        for single_list in master_neigh_list:
            out_string = ""
            for number in single_list:
                out_string += str(number) + " "
            xyzFile.write(out_string[:-1] + "\n")

def compute_neigh_lists(xyz, master_neigh_list, samples_per_dt, cutoff):
    """
    Tip: Use at least 20x20x20 unit cells with i.e. Stillinger-Weber!
    """
    i = 0; i_tot_checked = 0
    tot_neig = samples_per_dt
    N = xyz.shape[0]
    # Set limit for distance to any wall x,y,z-direction
    r_max = xyz.max() - cutoff*1.1
    r_min = xyz.min() + cutoff*1.1
    while True or i_tot_checked < tot_neig*100: # Stop computation eventually if too small system is given as input
        i_tot_checked += 1
        rand_atom = np.random.randint(N)
        x,y,z = xyz[rand_atom,:]
        if x < r_min or x > r_max or y < r_min or y > r_max or z < r_min or z > r_max:
            # print "Too close",x,y,z
            continue # Atom too close to wall
        # Center coordinate system around chosen atom:
        i += 1 # Found one to use
        xyz_copy = np.copy(xyz)
        xyz_copy[:,0] -= x
        xyz_copy[:,1] -= y
        xyz_copy[:,2] -= z
        # Chosen coordinates are now: x,y,z = 0,0,0
        r = np.linalg.norm(xyz_copy, axis=1)
        # Find neighbours within cutoff
        r_less  = (r < cutoff)
        nn_list = []
        for j,neigh_bool in enumerate(r_less):
            if neigh_bool and j != rand_atom: # Inside cutoff and "not itself"
                nn_list.append(xyz_copy[j,0])
                nn_list.append(xyz_copy[j,1])
                nn_list.append(xyz_copy[j,2])
                nn_list.append(r[j])
        nn_list.append("nan") # This file does not containt pot. energy. So if wrongly read, give NAN
        master_neigh_list.append(nn_list)
        if i == tot_neig:
            break


if __name__ == '__main__':
    print "By running this, you will delete all files and folders"
    print "related to trained nets that are NOT marked to keep!"
    raw_input("Hit enter to continue... ")
    deleteOldData()
