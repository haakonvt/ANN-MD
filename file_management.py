import shutil
import os,glob
import datetime

def keepData(save_dir):
    yes_or_no = raw_input("Do you want to keep this run? (If not -> auto-delete) [Y/n] ")
    if yes_or_no in ["yes","YES","Yes","y","Y",""]:
        new_save_dir = save_dir[0:32] + "KEEP-" + save_dir[32:]
        os.rename(save_dir, new_save_dir)
        print "Data from this run kept safely on disk:\n'%s'" %(new_save_dir)
    else:
        shutil.rmtree(save_dir)
        print "Data from this run deleted!"

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
    saveFileName = save_dir + "/NN_params_%d.txt" %(epoch)
    with open(saveFileName, 'w') as outFile:
        outStr = "%1d %1d %s" % (hiddenLayers, nNodes, activation_function)
        outFile.write(outStr + '\n')
        size = len(sess.run(weights))
        for i in range(size):
            i_weights = sess.run(weights[i])
            if i < size-1:
                for j in range(len(i_weights)):
                    for k in range(len(i_weights[0])):
                        outFile.write("%g" % i_weights[j][k])
                        outFile.write(" ")
                    outFile.write("\n")
            else:
                for j in range(len(i_weights[0])):
                    for k in range(len(i_weights)):
                        outFile.write("%g" % i_weights[k][j])
                        outFile.write(" ")
                    outFile.write("\n")
        outFile.write("\n")
        for biasVariable in biases:
            i_biases = sess.run(biasVariable)
            for j in range(len(i_biases)):
                outFile.write("%g" % i_biases[j])
                outFile.write(" ")
            outFile.write("\n")

def timeStamp():
    return datetime.datetime.now().strftime("%H.%M.%S--%d.%m.%Y")

if __name__ == '__main__':
    deleteOldData()
