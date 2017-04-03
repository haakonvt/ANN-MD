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
    # plt.show()
    plt.savefig(save_dir+"/RMSE_evo.pdf")

if __name__ == '__main__':
    plotTestVsTrainLoss()
