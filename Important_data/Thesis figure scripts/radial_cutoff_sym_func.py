from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
rc('lines', linewidth=2)
rc('font', family='serif')
rc('legend',**{'fontsize':10}) # Font size for legend

import os,sys
sys.path.append("/Users/haakonvt/Documents/Python-progs/master/LTF")
from symmetry_transform import * # From above directory
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

def the_differ(x,dx,deg=1):
    """
    Returns the same number of points as input vector, in
    contrast to np.diff(x) / dx. Also, second order for
    points not on boundary
    """
    if deg == 1: # First derivative
        return np.gradient(x, dx)
    if deg == 2: # Second derivative
        return np.gradient(np.gradient(x, dx), dx)
    print "\n\nMust give arg 'deg', (1 or 2).\n\n";

# General plot settings:
plt.figure(1)
plt.suptitle(r"Cutoff functions",fontsize=18)
plt.subplots_adjust(top=0.865, hspace=0.4)

# Add text in figure coordinates
plt.figtext(0.5, 0.61, 'First derivative', ha='center', va='center',fontsize=14)
plt.figtext(0.5, 0.327, 'Second derivative', ha='center', va='center',fontsize=14)

N  = 8
r  = np.linspace(-0.01,3,2001) # Not from 0 because of some edge artifacts after d/dx'ing
dr = r[1] - r[0]
rc_array = np.linspace(0.4,2.8,N)
colormap = plt.cm.Spectral #nipy_spectral # Other possible colormaps: Set1, Accent, nipy_spectral, Paired
colors   = [colormap(i) for i in np.linspace(0, 1, N)]

legend_list = [r"$R_c$ = " + str(round(rc,1)) for rc in rc_array]

ax1 = plt.subplot(3,2,1)
y_max = 0 # For normalization of graphs
plot_list = []
for i,rc in enumerate(rc_array):
    r_cut = cutoff_cos(r,rc)
    if y_max < np.max(np.abs(r_cut)):
        y_max = np.max(np.abs(r_cut))
    plot_list.append(["plt.plot(r,r_cut/y_max,color=colors[%d])" %i, r_cut])
    plt.title(r"$\frac{1}{2}\left(\cos{\left(\frac{\pi R_{ij}}{R_c}\right)}+1 \right)$", fontsize=16)
for plt_command,r_cut in plot_list:
    eval(plt_command)
# plt.text(0.95,0.9,r"$\frac{1}{2}\left(\cos{\left(\frac{\pi R_{ij}}{R_c}\right)}+1 \right)$", horizontalalignment='right',
#          verticalalignment='top',transform=ax1.transAxes)
# ax1.add_patch(FancyBboxPatch((1.5, 0.8),1.3, 0.15,boxstyle="round,pad=0.1",ec="k", fc="none"))
plt.ylim([-0.1,1.1])
plt.xlim([0,3])


ax2 = plt.subplot(3,2,2)
y_max = 0 # For normalization of graphs
plot_list = []
for i,rc in enumerate(rc_array):
    r_cut = cutoff_tanh(r,rc)
    if y_max < np.max(np.abs(r_cut)):
        y_max = np.max(np.abs(r_cut))
    plot_list.append(["plt.plot(r,r_cut/y_max,color=colors[%d])" %i, r_cut])
    plt.title(r"$\tanh^3\left(1-\frac{R_{ij}}{R_c}\right)$", fontsize=16)
for plt_command,r_cut in plot_list:
    eval(plt_command)
# plt.text(0.95,0.9,r"$\tanh^3\left(1-\frac{R_{ij}}{R_c}\right)$", horizontalalignment='right',
#          verticalalignment='top',transform=ax2.transAxes)
# ax2.add_patch(FancyBboxPatch((1.65, 0.8),1.15, 0.15,boxstyle="round,pad=0.1",ec="k", fc="none"))
plt.ylim([-0.1,1.1])
plt.xlim([0,3])
ax2.legend(legend_list,loc="best", ncol=2, columnspacing=0.6, labelspacing=0.25, fancybox=True)#, title=r"Value of $R_c$:") # title="Legend", fontsize=12

# Change some parameters a bit for clearer plots
N  = 6
rc_array    = np.linspace(1,2.5,N)
colors      = [colormap(i) for i in np.linspace(0, 1, N)]
legend_list = [r"$R_c$ = " + str(round(rc,1)) for rc in rc_array]

ax3 = plt.subplot(3,2,3)
y_max = 0 # For normalization of graphs
plot_list = []
for i,rc in enumerate(rc_array):
    r_cut = the_differ(cutoff_cos(r,rc), dr ,deg=1)
    if y_max < np.max(np.abs(r_cut)):
        y_max = np.max(np.abs(r_cut))
    plot_list.append(["plt.plot(r,r_cut/y_max,color=colors[%d])" %i, r_cut])
for plt_command,r_cut in plot_list:
    eval(plt_command)
plt.ylim([-1.1,0.1])
plt.xlim([0,3])


ax4 = plt.subplot(3,2,4)
y_max = 0 # For normalization of graphs
plot_list = []
for i,rc in enumerate(rc_array):
    r_cut = the_differ(cutoff_tanh(r,rc), dr, deg=1)
    if y_max < np.max(np.abs(r_cut)):
        y_max = np.max(np.abs(r_cut))
    plot_list.append(["plt.plot(r,r_cut/y_max,color=colors[%d])" %i, r_cut])
for plt_command,r_cut in plot_list:
    eval(plt_command)
plt.ylim([-1.1,0.1])
plt.xlim([0,3])
ax4.legend(legend_list,loc="best", ncol=2, columnspacing=0.6, labelspacing=0.25, fancybox=True) # title="Legend", fontsize=12

ax5 = plt.subplot(3,2,5)
y_max = 0 # For normalization of graphs
plot_list = []
for i,rc in enumerate(rc_array):
    r_cut = the_differ(cutoff_cos(r,rc), dr, deg=2)
    if y_max < np.max(np.abs(r_cut)):
        y_max = np.max(np.abs(r_cut))
    plot_list.append(["plt.plot(r_none,r_cut/y_max,color=colors[%d])" %i, r_cut])
counter = 0
for plt_command,r_cut in plot_list:
    r_none = np.array(r, dtype=object)   # Allowing for None in array
    for i,r_c in enumerate(r_cut[::-1]): # Array in reverse
        r_cut_max = np.max(r_cut)/y_max
        if r_c != 0.0:
            # r_break_top_x = r[-i-5]
            # r_break_top_y = r_cut[-i-5]
            for j in [-1,-2,-3,-4]:
                r_cut[-i+j]  = None
                r_none[-i+j] = None
            break                        # None added, quitting
    plt.plot([rc_array[counter]]*2,[r_cut_max,0],"--",linewidth=1, color=colors[counter])
    eval(plt_command)
    plt.plot(rc_array[counter], r_cut_max, ">", mew=0,ms=7,color=colors[counter])
    plt.plot(rc_array[counter], 0, "<", mew=0,ms=7,color=colors[counter])
    counter += 1
plt.ylim([-1.1,1.25])
plt.xlim([0,3])
ax5.set_xlabel(r"$R_{ij}")

ax6 = plt.subplot(3,2,6)
y_max = 0 # For normalization of graphs
plot_list = []
for i,rc in enumerate(rc_array):
    r_cut = the_differ(cutoff_tanh(r,rc), dr, deg=2)
    if y_max < np.max(np.abs(r_cut)):
        y_max = np.max(np.abs(r_cut))
    plot_list.append(["plt.plot(r,r_cut/y_max,color=colors[%d])" %i, r_cut])
for plt_command,r_cut in plot_list:
    eval(plt_command)
plt.ylim([-0.3,1.1])
plt.xlim([0,3])
ax6.set_xlabel(r"$R_{ij}")

if len(sys.argv) > 1:
    if sys.argv[1] == "-replace":
        replace = True
else:
    replace = False

maybe = raw_input("\nEnter 'YES' to save figure as a copy: ")
if maybe == "YES":
    filename         = "radial_cutoff_symm_functions.pdf"
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
