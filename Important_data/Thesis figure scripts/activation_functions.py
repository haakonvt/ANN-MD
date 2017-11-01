from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)

import numpy as np
import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 1.75
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import math
import os,sys

xmin = -10; xmax = 10
x = np.linspace(xmin,xmax,1001)

def y1(x):
    return 1./(1+np.exp(-x)) # sigmoid
def y2(x):
    return np.maximum(x,0)  # ReLU


fig = plt.figure()
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
colormap = plt.cm.Spectral #nipy_spectral # Other possible colormaps: Set1, Accent, nipy_spectral, Paired
colors   = [colormap(i) for i in np.linspace(0, 1, 6)]

plt.suptitle(r"Activation functions", fontsize=18)

# Adjust figure spacing
plt.subplots_adjust(left  = 0.175, right = 0.85)

# Add text in figure coordinates
plt.figtext(0.5, 0.69, 'Effect of bias addition', ha='center', va='center',fontsize=14)
plt.figtext(0.5, 0.34, 'Effect of altering weights', ha='center', va='center',fontsize=14)

gs1 = GridSpec(1, 2)
gs1.update(top=0.9, bottom=0.74, wspace = 0.1, hspace = 0.1)

gs2 = GridSpec(2, 2)
gs2.update(top=0.665, bottom=0.39, wspace = 0.1, hspace = 0.1)

gs3 = GridSpec(2, 2)
gs3.update(top=0.315, bottom=0.04, wspace = 0.1, hspace = 0.1)


ax1 = plt.subplot(gs1[0,0])
plt.plot(x,y1(x),"k")
plt.ylim([-0.1,1.1])
plt.title("Sigmoid")


ax2 = plt.subplot(gs1[0,1])
plt.plot(x,y2(x),"k")
plt.ylim([-1,10])
plt.title("ReLU")

ax3 = plt.subplot(gs2[0,0])
for i,bias in enumerate(np.linspace(2,-2,6)):
    plt.plot(x,y1(x+bias),color=colors[i])
plt.ylim([-0.1,1.1])
plt.setp(ax3, xticks=[])

ax4 = plt.subplot(gs2[0,1])
for i,bias in enumerate(np.linspace(2,-2,6)):
    plt.plot(x,y2(x+bias),color=colors[i])
plt.ylim([-1,10])
plt.setp(ax4, xticks=[])

ax5 = plt.subplot(gs2[1,0])
for i,bias in enumerate(np.linspace(0,1,6)):
    plt.plot(x,bias+y1(x),color=colors[i])
plt.ylim([-0.1,2.1])

ax6 = plt.subplot(gs2[1,1])
for i,bias in enumerate(np.linspace(2,-2,6)):
    plt.plot(x,bias+y2(x),color=colors[i])
plt.ylim([-1,10])


o = .87
f = 1.6
fan_array = [o*f**3,o*f**2,o*f, o ,o/f,o/f**2]

ax7  = plt.subplot(gs3[0,0])
for i,w in enumerate(fan_array):
    plt.plot(x,y1(x*w),color=colors[i])
plt.ylim([-0.1,1.1])
plt.setp(ax7, xticks=[])

ax8  = plt.subplot(gs3[0,1])
for i,w in enumerate(fan_array):
    plt.plot(x,y2(x*w),color=colors[i])
plt.ylim([-1,10])
plt.setp(ax8, xticks=[])

ax9  = plt.subplot(gs3[1,0])
for i,w in enumerate(np.linspace(1,2,6)):
    plt.plot(x,w*y1(x),color=colors[i])
plt.ylim([-0.1,2.1])

ax10 = plt.subplot(gs3[1,1])
for i,w in enumerate(fan_array):
    plt.plot(x,w*y2(x),color=colors[i])
plt.ylim([-1,10])


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
    filename         = "activation_functions.pdf"
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
