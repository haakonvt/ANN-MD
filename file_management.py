import os,glob

def foldersCheck():
    """
    Do folders exist?
    """
    foldersMade = 0
    for filename in ['SavedRuns','SomePlots','SavedGraphs']:
        if not os.path.exists(filename):
            os.makedirs(filename) # Create folders
            foldersCheck += 1
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

if __name__ == '__main__':
    deleteOldData()
