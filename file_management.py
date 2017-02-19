import os,glob

def foldersCheck():
    """
    Do folders exist?
    """
    foldersMade = 0
    for filename in ['SavedRuns','SomePlots','SavedGraphs']:
        if not os.path.exists(filename):
            os.makedirs(filename) # Create folders
            foldersMade += 1
    return foldersMade

def deleteOldData():
    foldersMade = foldersCheck()
    if not foldersMade == 3: # No old data, since folders have just been created
        keepData = raw_input('\nDelete all previous plots, run- and graph-data? (y/yes/enter)')
        if keepData in ['y','yes','']:
            for someFile in glob.glob("SavedRuns/run*"):
                os.remove(someFile)
            for someFile in glob.glob("SomePlots/fig*.png"):
                os.remove(someFile)
            for someFile in glob.glob("SavedGraphs/tf_graph_WB*.txt"):
                os.remove(someFile)

def findLoadFileName():
    """
    Not updated for TF v1.0
    """
    print "Not updated for TF v1.0. Exiting!"
    import sys
    sys.exit(0)
    if os.path.isdir("SavedRuns"):
        file_list = []
        for a_file in os.listdir("SavedRuns"):
            if a_file[0] == ".": # Ignore hidden files
                continue
            elif a_file == "checkpoint":
                continue
            else:
                file_list.append(int(a_file[3:-4]))
        if len(file_list) == 0:
            print "No previous runs exits. Exiting..."; sys.exit()
        newest_file = "run" + str(np.max(file_list)) + ".dat"
        startEpoch  = np.max(file_list)
        print 'Model restored from file:', newest_file
        return newest_file,startEpoch
    else:
        os.makedirs("SavedRuns")
        print "Created 'SavedRuns'-directory. No previous runs exits. Exiting..."; sys.exit()

if __name__ == '__main__':
    deleteOldData()
