# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
rc('lines', linewidth=2)
rc('font', family='serif')
rc('legend',**{'fontsize':8}) # Font size for legend
rc('text.latex', unicode=True)

import os,sys
sys.path.append("/Users/haakonvt/Documents/Python-progs/master/LTF")
from symmetry_transform import * # From above directory
import matplotlib.pyplot as plt
import numpy as np

# General plot settings:
plt.figure(1)
plt.suptitle("Response curve,\nradial symmetry functions",fontsize=18)
plt.subplots_adjust(top=0.855, hspace=0.395, wspace=0.3) # Prev 0.85 , 0.885
plot_rows = 3
plot_cols = 2

# Add text in figure coordinates
# plt.figtext(0.5, 0.61, 'First derivative', ha='center', va='center',fontsize=14)
# plt.figtext(0.5, 0.327, 'Second derivative', ha='center', va='center',fontsize=14)

plot_resolution = 1001
r = np.linspace(0,7,plot_resolution)
N = 6                      # Should be an even number because middle color is nearly invisible
colormap = plt.cm.Spectral #nipy_spectral # Other possible colormaps: Set1, Accent, nipy_spectral, Paired
colors   = [colormap(i) for i in np.linspace(0, 1, N)]

ax1 = plt.subplot(plot_rows,plot_cols,1)
rc_range = range(1,7)
for i,rc in enumerate(rc_range):
    plt.plot(r,G1_single_neighbor(r,rc),color=colors[i])
plt.legend([r"$R_c = %d$" %rc for rc in rc_range], fancybox=True)
ax1.set_ylabel(r"$G^1(r)$", fontsize=14)
plt.title(r"$R_c = 1,..,6$")


ax2 = plt.subplot(plot_rows,plot_cols,2)
rc = 6; rs = 0
eta_range = [1.79, 0.71, 0.32, 0.14, 0.04, 0.00]
for i,eta in enumerate(eta_range):
    plt.plot(r, G2_single_neighbor(r,rc,rs,eta), color=colors[i])
plt.legend([r"$\eta = %s$" %(str(round(eta,2))) for eta in eta_range], fancybox=True)
ax2.set_ylabel(r"$G^2(r)$", fontsize=14)
plt.title(r"$R_c = 6, \;R_s = 0$")
# plt.ylim([-0.1,1.1])
# plt.xlim([0,3])
# ax1.legend(legend_list,loc="best", ncol=2, columnspacing=0.6, labelspacing=0.25, fancybox=True)#, title=r"Value of $R_c$:") # title="Legend", fontsize=12

ax3 = plt.subplot(plot_rows,plot_cols,3)
rc       = 6
eta      = 1.2
rs_range = np.linspace(1.5, 5, 6)
N        = len(rs_range)
colors   = [colormap(i) for i in np.linspace(0, 1, N)]
for i,rs in enumerate(rs_range):
    plt.plot(r, G2_single_neighbor(r,rc,rs,eta), color=colors[i])
plt.legend([r"$R_s = %s$" %str(round(rs,1)) for rs in rs_range], fancybox=True)
plt.title(r"$R_c = 6, \;\eta = %g, \;R_s > 0$" %eta)
ax3.set_ylabel(r"$G^2(r)$", fontsize=14)


ax4      = plt.subplot(plot_rows,plot_cols,4)
eta      = 10
rs_range = np.linspace(1.5, 5, 6)
for i,rs in enumerate(rs_range):
    plt.plot(r, G2_single_neighbor(r,rc,rs,eta), color=colors[i])
plt.legend([r"$R_s = %s$" %str(round(rs,1)) for rs in rs_range], fancybox=True)
plt.title(r"$R_c = 6, \;\eta = %d, \;R_s > 0$" %eta)
ax4.set_ylabel(r"$G^2(r)$", fontsize=14)


ax5 = plt.subplot(plot_rows,plot_cols,5)
rc  = 6
N   = 6
kappa_list = np.linspace(0.5,1.8,N)
colors   = [colormap(i) for i in np.linspace(0, 1, N)]
for i, kappa in enumerate(kappa_list):
    G3_s_n = G3_single_neighbor(r, rc, kappa)
    plt.plot(r, G3_s_n, color=colors[i])
ax5.set_ylabel(r"$G^3(r)$", fontsize=14)
ax5.yaxis.labelpad = -5.5 # Move ylabel a little to the right
plt.legend([r"$\kappa = %s$" %str(round(kappa,1)) for kappa in kappa_list], ncol=2, \
            fancybox=True, loc="upper right", columnspacing=0.6, labelspacing=0.25)
plt.title(r"$R_c = 6$")
plt.ylim([-1.,1.5])
ax5.set_xlabel(r"$R_{ij}$ [\u00C5ngstr\o{}m]") # OMG this was difficult to get right
# plt.setp(ax5, yticks=[-1, 0, 1])
# plt.xlim([-2,7])


ax6 = plt.subplot(plot_rows,plot_cols,6)
zeta     = 1 # ?!
lambda_c = 1
eta_range = [2, 0.5, 0.2, 0.1, 0.05, 0.025]
for i, eta in enumerate(eta_range):
    G4_s_n = G4_single_neighbor_radial(r, zeta, lambda_c, eta)
    G5_s_n = G5_single_neighbor_radial(r, zeta, lambda_c, eta)
    plt.plot(r, G4_s_n, color=colors[i], label=r"$\eta = %g$" %eta)
    plt.plot(r, G5_s_n, color=colors[i], ls="--")
ax6.set_ylabel(r"$G^4(r), \;G^5(r)$", fontsize=14)
plt.legend(fancybox=True) # Show only labels for G4
plt.title(r"$\lambda=1, \;\zeta=1, \;\theta = \frac{\pi}{3}$")
ax6.set_xlabel(r"$R_{ij}$ [\u00C5ngstr\o{}m]") # OMG this was difficult to get right


"""
###########################
END OF PLOT COMMANDS
###########################
"""

if len(sys.argv) > 1:
    if sys.argv[1] == "-replace":
        replace = True
        autoSave = True
else:
    replace = False
    autoSave = False

maybe = ""
if not autoSave:
    maybe = raw_input("\nEnter 'YES' to save figure as a copy: ")
if maybe == "YES" or autoSave:
    filename         = "radial_symm_functions.pdf"
    directory        = "/Users/haakonvt/Dropbox/uio/master/latex-master/Illustrations/"
    filenameWithPath = directory + filename
    i = 1; file_test = filename
    while True:
        if file_test in os.listdir(directory):
            file_test = filename[:-4] + str(i) + filename[-4:]
            i += 1
            continue
        else:
            newFilenameWithPath = directory + file_test
            break
    if replace:
        plt.savefig(filenameWithPath, bbox_inches='tight')
        print '\nFigure replaced previous with filename:\n"%s"\n' %filename
    else:
        plt.savefig(newFilenameWithPath, bbox_inches='tight')
        if i != 1:
            print '\nFigure saved as a copy with filename:\n"%s"\n' %file_test
else:
    if replace:
        print "Argument '-replace' has no purpose when fig not saved, just FYI."
    plt.show()
