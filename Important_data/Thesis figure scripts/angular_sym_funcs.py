# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
rc('lines', linewidth=2)
rc('font', family='serif')
rc('legend',**{'fontsize':7.5}) # Font size for legend EXCEPT its title
legend_title_fontsize = 9.5   # Font size for legend title. Hackz below
rc('text.latex', unicode=True)

import os,sys
sys.path.append("/Users/haakonvt/Documents/Python-progs/master/LTF")
from symmetry_transform import * # From above directory
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
# from matplotlib.patches import FancyBboxPatch
# from matplotlib.font_manager import FontProperties
import numpy as np

# General plot settings:
plt.figure(1)
plt.suptitle("Response curve,\nangular symmetry functions",fontsize=18)
plt.subplots_adjust(top=0.85, hspace=0.3, wspace=0.3) # Prev 0.885
plot_rows = 2
plot_cols = 2

# Add text in figure coordinates
# plt.figtext(0.5, 0.61, 'First derivative', ha='center', va='center',fontsize=14)
# plt.figtext(0.5, 0.327, 'Second derivative', ha='center', va='center',fontsize=14)

plot_resolution = 137
N = 6                      # Should be an even number because middle color is nearly invisible
colormap = plt.cm.Spectral #nipy_spectral # Other possible colormaps: Set1, Accent, nipy_spectral, Paired
colors   = [colormap(i) for i in np.linspace(0, 1, N)]


ax1 = plt.subplot(plot_rows,plot_cols,1)
theta    = np.linspace(0,2*pi,plot_resolution)
lambda_c = 1
eta      = 0.025
zeta     = 1
r_cut    = 1
plt.title(r"$R = 0.8\times Rc$")
plt.plot(theta/2./pi*360, G4_single_neighbor_rjk(theta, r_cut, zeta, lambda_c, eta), color="k")
plt.plot(theta/2./pi*360, G5_single_neighbor_rjk(theta, r_cut, zeta, lambda_c, eta), color="k", ls="--")
legend = plt.legend([r"$G^4$", r"$G^5$"], title=r"$\lambda = \zeta = %g,\\ \hphantom{asd}\eta=%g$" %(lambda_c,eta), loc="best", borderpad=0.6, handlelength=5, fancybox=True, fontsize=9.5)
legend.get_title().set_fontsize(legend_title_fontsize)
plt.xlim([0,360])
ax1.set_ylabel(r"$G^4(\theta), \;G^5(\theta)$", fontsize=14)
plt.setp(ax1, xticks=range(0,361,60))
plt.setp(ax1, yticks=np.linspace(0,0.02,5))


ax2 = plt.subplot(plot_rows,plot_cols,2)
p_rc_range = np.linspace(0.45, 0.65, 6)
plt.title(r"$R = R_p\times Rc$") # r"$\lambda=%g, \eta=%g, \zeta=%g$" %(lambda_c,eta,zeta)
for i,p_rc in enumerate(p_rc_range):
    G4_rjk = G4_single_neighbor_rjk(theta, r_cut, zeta, lambda_c, eta, percent_of_rc=p_rc)
    plt.plot(theta/2./pi*360, G4_rjk, color=colors[i], label=r"$R_p = %g$" %(p_rc), lw=1.3)
    G5_rjk = G5_single_neighbor_rjk(theta, r_cut, zeta, lambda_c, eta, percent_of_rc=p_rc)
    plt.plot(theta/2./pi*360, G5_rjk, color=colors[i], ls="--")
legend = plt.legend(fancybox=True, loc="best", borderpad=0.5, handlelength=3.5)
plt.xlim([0,360])
ax2.set_ylabel(r"$G^4(\theta), \;G^5(\theta)$", fontsize=14)
plt.setp(ax2, xticks=range(0,361,60))


ax3     = plt.subplot(plot_rows,plot_cols,3)
r_ij_ik = 2
r_cut   = 6
eta     = 0.005
lambda_c  = 1
zeta_list = [1,2,4,8,16,32]
N        = len(zeta_list)
colors   = [colormap(i) for i in np.linspace(0, 1, N)]
for i in range(N):
    zeta   = zeta_list[i]
    G4_s_n = G4_single_neighbor(theta, r_ij_ik, r_cut, zeta, lambda_c, eta)
    G5_s_n = G5_single_neighbor(theta, r_ij_ik, r_cut, zeta, lambda_c, eta)
    plt.plot(theta/2./pi*360, G4_s_n,       color=colors[i], label=r"$\zeta = %d$" %zeta, lw=1.3)
    plt.plot(theta/2./pi*360, G5_s_n, "--", color=colors[i])
ax3.set_xlabel(r"$\theta_{ijk}$ [degree]")
plt.xlim([0,360])
plt.title(r"$\lambda = 1,\;R_{ij} = R_{ik} = 1$", fontsize=14)
plt.legend(fancybox=True, loc="best", ncol=1, handlelength=3.5)#, columnspacing=0.6, labelspacing=0.25)
ax3.set_ylabel(r"$G^4(\theta), \;G^5(\theta)$", fontsize=14)
plt.setp(ax3, xticks=range(0,361,60))

r_ij_ik = 2
r_cut   = 15
eta     = 0.01
ax4 = plt.subplot(plot_rows,plot_cols,4)
lambda_c = -1
for i in range(N):
    zeta     = zeta_list[i]
    G5_s_n   = G5_single_neighbor(theta, r_ij_ik, r_cut, zeta, lambda_c, eta)
    G4_s_n   = G4_single_neighbor(theta, r_ij_ik, r_cut, zeta, lambda_c, eta)
    plt.plot(theta/2./pi*360, G4_s_n,       color=colors[i], label=r"$\zeta = %d$" %zeta, lw=1.3)
    plt.plot(theta/2./pi*360, G5_s_n, "--", color=colors[i])
plt.xlim([0,360])
plt.title(r"$\lambda = -1,\;R_{ij} = R_{ik} = 1$", fontsize=14)
plt.legend(fancybox=True, loc="best", ncol=1, handlelength=3.5)#, columnspacing=0.6, labelspacing=0.25)
ax4.set_ylabel(r"$G^4(\theta), \;G^5(\theta)$", fontsize=14)
plt.setp(ax4, xticks=range(0,361,60))
ax4.set_xlabel(r"$\theta_{ijk}$ [degree]")

# # Need higher plot resolution for last two plots:
# theta  = np.linspace(0,2*pi,10*plot_resolution-1)
# ax5    = plt.subplot(plot_rows,plot_cols,5)
# lambda_c  = 1
# zeta_list = [1,2,4,8,16,32]
# N        = len(zeta_list)
# colors   = [colormap(i) for i in np.linspace(0, 1, N)]
# for i in range(N):
#     zeta     = zeta_list[i]
#     G4_s_n   = G4_single_neighbor(theta, r_ij_ik, r_cut, zeta, lambda_c, eta)
#     plt.plot(theta/2./pi*360, G4_s_n, color=colors[i])
# plt.xlim([0,360])
# plt.title(r"$\lambda = 1,\;R_{ij} = R_{ik} = 1$", fontsize=14)
# plt.legend([r"$\zeta = %d$" %zeta_list[i] for i in range(N)],fancybox=True, loc="upper center")
# ax5.set_xlabel(r"$\theta_{ijk}$ [degree]")
# ax5.set_ylabel(r"$G^4(\theta)$")
# plt.setp(ax5, xticks=range(0,361,60))



r_ij_ik = 0.5
r_cut   = 1
eta     = 1.2
lambda_c = -1

# ax6 = plt.subplot(plot_rows,plot_cols,6)
# lambda_c = -1
# for i in range(N):
#     zeta     = zeta_list[i]
#     G4_s_n   = G4_single_neighbor(theta, r_ij_ik, r_cut, zeta, lambda_c, eta)
#     plt.plot(theta/2./pi*360, G4_s_n, color=colors[i])
# plt.xlim([0,360])
# plt.title(r"$\lambda = -1,\;R_{ij} = R_{ik} = 1$", fontsize=14)
# plt.legend([r"$\zeta = %d$" %zeta_list[i] for i in range(N)],fancybox=True, loc="best")
# ax6.set_xlabel(r"$\theta_{ijk}$ [degree]")
# ax6.set_ylabel(r"$G^4(\theta)$")
# plt.setp(ax6, xticks=range(0,361,60))

# ax6 = plt.subplot(plot_rows,plot_cols,6, projection='3d')
# ax6.patch.set_alpha(0.0)
# res                 = 100
# theta               = np.linspace(0,2*pi,res)
# r_c                 = np.linspace(r_cut*0.8,r_cut*8,res)
# theta_grid, Rc_grid = np.meshgrid(theta, r_c)
#
# zeta = zeta_list[0]
# Z = G4_single_neighbor_2D(theta_grid, Rc_grid, r_ij_ik, zeta, lambda_c, eta)
#
# ax6.plot_surface(theta_grid, Rc_grid, Z, rstride=5, cstride=3, cmap=colormap,
#                        linewidth=0.5, antialiased=False)


"""
###########################
END OF PLOT COMMANDS
###########################
"""

if len(sys.argv) > 1:
    if sys.argv[1] == "-replace":
        replace  = True
        autoSave = True
    if sys.argv[1] == "-save":
        replace  = False
        autoSave = True
else:
    replace = False
    autoSave = False

if not autoSave:
    print "\nTo save a copy, append '-save' and re-run.\nShowing to SCREEN only!"
if autoSave:
    filename         = "angular_symm_functions.pdf"
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
