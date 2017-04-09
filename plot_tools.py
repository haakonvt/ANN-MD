import matplotlib.pyplot as plt
import numpy as np

def plotTestVsTrainLoss(save_dir, list_of_rmse_train, list_of_rmse_test):
    if not list_of_rmse_test or not list_of_rmse_train:
        """
        No input gotten (or not enough), must read from file
        """
        list_of_rmse_test  = np.loadtxt(save_dir + "/testRMSE.txt")
        list_of_rmse_train = np.loadtxt(save_dir + "/trainRMSE.txt")

    plt.subplot(3,1,1)
    xTest_for_plot = np.linspace(0,1,len(list_of_rmse_test))
    xTrain_for_plot = np.linspace(0,1,len(list_of_rmse_train))
    plt.plot(xTrain_for_plot, list_of_rmse_train, label="train")
    plt.plot(xTest_for_plot, list_of_rmse_test, label="test") #, lw=2.0)
    plt.subplot(3,1,2)
    plt.semilogy(xTrain_for_plot, list_of_rmse_train, label="train")
    plt.semilogy(xTest_for_plot, list_of_rmse_test, label="test") #, lw=2.0)
    plt.subplot(3,1,3)
    plt.semilogy(xTrain_for_plot, list_of_rmse_train, label="train")
    plt.loglog(xTest_for_plot, list_of_rmse_test, label="test") #, lw=2.0)
    plt.savefig(save_dir+"/RMSE_evo.pdf")
    plt.show()

def plotErrorEvolutionSWvsNN(EP_SW_list, EP_NN_list, nmbr_of_atoms):
    EP_SW_list  = np.array(EP_SW_list, dtype=float)
    EP_NN_list  = np.array(EP_NN_list, dtype=float)
    arr_diff    =  EP_SW_list - EP_NN_list
    arr_rel_err = (EP_SW_list - EP_NN_list)/EP_SW_list

    plt.suptitle("Tot. atoms: %d" %nmbr_of_atoms)
    plt.subplot(3,1,1)
    plt.plot(EP_SW_list, label="SW")
    plt.plot(EP_NN_list, label="NN")#, markevery=5) #, lw=2.0)
    plt.ylabel("Potential Energy")
    plt.legend()
    plt.subplot(3,1,2)
    plt.plot(arr_diff, label="SW - NN")
    plt.legend()
    plt.subplot(3,1,3)
    plt.plot(arr_rel_err, label="(SW - NN) / SW")
    plt.xlabel("Timestep")
    plt.legend()
    plt.show()

def plotEvolutionSWvsNN_N_diff_epochs(N, master_list):
    plt.suptitle("Tot. atoms: %d" %(master_list[0][2]))
    for i in range(1,N+1):
        plt.subplot(N,1,i) # Stack vertically
        EP_SW = np.array(master_list[i-1][0], dtype=float)
        EP_NN = np.array(master_list[i-1][1], dtype=float)
        plt.title("NN after %d epochs" %(master_list[i-1][3]))
        plt.plot(EP_SW, label="SW")
        plt.plot(EP_NN, label="NN")#, markevery=5) #, lw=2.0)
        plt.ylabel("Potential Energy")
    plt.show()

if __name__ == '__main__':
    plotTestVsTrainLoss()
