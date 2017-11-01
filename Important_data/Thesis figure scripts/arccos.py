# -*- coding: utf-8 -*-
from __future__ import unicode_literals
from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
rc('lines', linewidth=2)
rc('font', family='serif')
rc('legend',**{'fontsize':14}) # Font size for legend
# rc('text.latex', unicode=True)

import os,sys
import matplotlib.pyplot as plt
import numpy as np
from math import pi

# General plot settings:
plot_resolution = 2373
x = np.linspace(0,1,plot_resolution)
N = 3                      # Should be an even number because middle color is nearly invisible
colormap = plt.cm.Spectral #nipy_spectral # Other possible colormaps: Set1, Accent, nipy_spectral, Paired
colors   = [colormap(i) for i in np.linspace(0, 1, N)]

plt.figure(figsize=(8,3))
# plt.subplots_adjust(left  = 0.175, right = 0.85)

plt.plot(x, np.arccos(1-2*x), color=colors[0], label=r"$\arccos(1-2v)$")

plt.xlim([-0.05,1.05])
plt.ylim([-0.15,pi+0.15])
plt.xlabel(r"$v$", fontsize=16)
plt.ylabel(r"$\theta(v)$", fontsize=16)
plt.legend(loc="best", fancybox=True) # Show only labels for G4
plt.title(r"Mapping of $v$ to $\theta$", fontsize=18)


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
    filename         = "arccos.pdf"
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
