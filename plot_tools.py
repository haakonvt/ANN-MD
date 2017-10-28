# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from matplotlib import rc
rc('text.latex', unicode=True)

import matplotlib.pyplot as plt
import numpy as np
import datetime # Making unique names for plots
import glob

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
    arr_diff    =  EP_NN_list - EP_SW_list
    arr_rel_err = (EP_SW_list - EP_NN_list)/EP_SW_list

    plt.suptitle("%d-atom system" %nmbr_of_atoms, fontsize=18)
    plt.subplot(3,1,1)

    plt.plot(EP_SW_list, label="SW")
    plt.plot(EP_NN_list, label="NNP")#, markevery=5) #, lw=2.0)
    plt.ylabel("Potential Energy [eV]")
    plt.legend(loc="lower right")
    plt.subplot(3,1,2)
    plt.plot(arr_diff, label="NNP - SW")
    plt.ylabel("Absolute Error")
    plt.legend(loc="upper right")
    plt.subplot(3,1,3)
    plt.plot(arr_rel_err, label="(SW - NNP) / SW")
    plt.ylabel("Relative Error")
    plt.xlabel("Timestep")
    plt.legend(loc="upper right")

    plt.subplots_adjust(left=0.14,bottom=0.11,right=0.97,top=0.91,wspace=0.3,hspace=0.22)
    plt.show()

def plotEvolutionSWvsNN_N_diff_epochs(N, master_list):
    # plt.suptitle("Tot. atoms: %d" %(master_list[0][2]))
    plt.suptitle("Evolution of NNP", fontsize=18)
    for i in range(1,N+1):
        plt.subplot(N,1,i) # Stack vertically
        EP_SW = np.array(master_list[i-1][0], dtype=float)
        EP_NN = np.array(master_list[i-1][1], dtype=float)
        plt.title("NNP after %d epochs" %(master_list[i-1][3]))
        # plt.semilogy(np.abs(EP_SW-EP_NN), label="|SW-NNP|")
        plt.plot(EP_SW, label="SW")
        plt.plot(EP_NN, label="NNP")#, markevery=5) #, lw=2.0)
        plt.ylabel("Potential Energy")
        plt.legend()
    plt.subplots_adjust(left=0.09,bottom=0.05,right=0.98,top=0.93,hspace=0.33)
    plt.show()

def plotForcesSWvsNN(F_SW, F_NN, show=True):
    F_SW = np.array(F_SW, dtype=float)
    F_NN = np.array(F_NN, dtype=float)

    F_SW_tot = np.linalg.norm(F_SW, axis=1)
    F_NN_tot = np.linalg.norm(F_NN, axis=1)

    plt.suptitle("Forces SW vs NNP", fontsize=18)

    plt.subplot(4,2,1)
    plt.plot(F_SW_tot, label="SW")
    plt.plot(F_NN_tot, label="NNP")
    plt.ylabel(r"Tot. force [eV/\u00C5]")
    plt.legend()

    plt.subplot(4,2,2)
    plt.plot((F_SW_tot-F_NN_tot), label="SW-NNP")
    plt.ylabel("Abs. error: Tot. force")
    plt.legend(loc="upper right")

    plt.subplot(4,2,3)
    plt.plot(F_SW[:,0])#, label="SW")
    plt.plot(F_NN[:,0])#, label="NNP")
    plt.ylabel("Forces X")
    plt.legend(loc="upper right")

    plt.subplot(4,2,4)
    plt.plot((F_SW[:,0]-F_NN[:,0]))#, label="SW-NNP")
    plt.ylabel("Abs. error: Forces X")
    plt.legend(loc="upper right")

    plt.subplot(4,2,5)
    plt.plot(F_SW[:,1])#, label="SW")
    plt.plot(F_NN[:,1])#, label="NNP")
    plt.ylabel("Forces Y")
    plt.legend(loc="upper right")

    plt.subplot(4,2,6)
    plt.plot((F_SW[:,1]-F_NN[:,1]))#, label="SW-NNP")
    plt.ylabel("Abs. error: Forces Y")
    plt.legend(loc="upper right")

    plt.subplot(4,2,7)
    plt.plot(F_SW[:,2])#, label="SW")
    plt.plot(F_NN[:,2])#, label="NNP")
    plt.ylabel("Forces Z")
    plt.xlabel("Timestep")
    plt.legend(loc="upper right")

    plt.subplot(4,2,8)
    plt.plot((F_SW[:,2]-F_NN[:,2]))#, label="SW-NNP")
    plt.ylabel("Abs. error: Forces Z")
    plt.xlabel("Timestep")
    plt.legend(loc="upper right")
    # name = datetime.datetime.now().strftime("%H-%M-%S-%d-%m-%Y") + ".pdf"
    # plt.savefig(name)
    if show:
        plt.show()

def plotLAMMPSforces1atomEvo(show=False):
    file_dir  = "Important_data/Test_nn/Forces/"
    file_list = glob.glob(file_dir+"dump_forces*")
    count = 0
    force_list = []
    while True:
        one_file = file_dir + "dump_forces%s" %count
        if one_file in file_list:
            with open(one_file, "r") as force_file:
                for i,line in enumerate(force_file):
                    if i == 9:
                        line = line.split()
                        force_list.append(line)
                        break
            count += 1 # Go to next file
        else:
            break
    if show:
        plotForcesSWvsNN(force_list, [])
    else:
        return force_list[:-1]

if __name__ == '__main__':
    # plotLAMMPSforces1atomEvo()
    plotTestVsTrainLoss()
