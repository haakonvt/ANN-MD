from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
rc('text', usetex=True)
rc('legend',**{'fontsize':11}) # Font size for legend

from mpl_toolkits.axes_grid.axislines import SubplotZero
import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 2.5
import matplotlib.pyplot as plt
from math import erf,sqrt
import numpy as np

xmin = -4; xmax = 4
x = np.linspace(xmin,xmax,1001)

y1 = lambda x: np.array([erf(0.5*i*sqrt(np.pi)) for i in x])
y2 = lambda x: np.tanh(x)
y3 = lambda x: 4./np.pi*np.arctan(np.tanh(np.pi*x/4.))
y4 = lambda x: x/np.sqrt(1.+x**2)
y5 = lambda x: 2.0/np.pi*np.arctan(np.pi/2.0 * x)
y6 = lambda x: x/(1+np.abs(x))

fig = plt.figure(1)
ax = SubplotZero(fig, 111)
fig.add_subplot(ax)
plt.subplots_adjust(left  = 0.125,  # the left side of the subplots of the figure
                    right = 0.9,    # the right side of the subplots of the figure
                    bottom = 0.1,   # the bottom of the subplots of the figure
                    top = 0.9,      # the top of the subplots of the figure
                    wspace = 0.,   # the amount of width reserved for blank space between subplots
                    hspace = 0.)   # the amount of height reserved for white space between subplots

plt.setp(ax, xticks=[-3,-2,-1,1,2,3], xticklabels=[" "," "," "," "," "," ",], yticks=[-1,1], yticklabels=[" "," ",])

# Make coordinate axes with "arrows"
for direction in ["xzero", "yzero"]:
    ax.axis[direction].set_visible(True)

# Coordinate axes with arrow (guess what, these are the arrows)
plt.arrow(2.65, 0.0, 0.5, 0.0, color="k", clip_on=False, head_length=0.06, head_width=0.08)
plt.arrow(0.0, 1.03, 0.0, 0.1, color="k", clip_on=False, head_length=0.06, head_width=0.08)

# Remove edge around the entire plot
for direction in ["left", "right", "bottom", "top"]:
    ax.axis[direction].set_visible(False)

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
colormap = plt.cm.Spectral #nipy_spectral # Other possible colormaps: Set1, Accent, nipy_spectral, Paired
colors   = [colormap(i) for i in np.linspace(0, 1, 6)]

plt.title("Six sigmoid functions", fontsize=18, y=1.08)

leg_list = [r"$\mathrm{erf}\left(\frac{\sqrt{\pi}}{2}x \right)$",
            r"$\tanh(x)$",
            r"$\frac{2}{\pi}\mathrm{gd}\left( \frac{\pi}{2}x \right)$",
            r"$x\left(1+x^2\right)^{-\frac{1}{2}}$",
            r"$\frac{2}{\pi}\mathrm{arctan}\left( \frac{\pi}{2}x \right)$",
            r"$x\left(1+|x|\right)^{-1}$"]

for i in range(1,7):
    s = "ax.plot(x,y%s(x),color=colors[i-1])" %(str(i))
    eval(s)
ax.legend(leg_list,loc="best", ncol=2, fancybox=True) # title="Legend", fontsize=12
# ax.grid(True, which='both')
ax.set_aspect('equal')
ax.set_xlim([-3.1,3.1])
ax.set_ylim([-1.1,1.1])

ax.annotate('1', xy=(0.08, 1-0.02))
ax.annotate('0', xy=(0.08, -0.2))
ax.annotate('-1', xy=(0.08, -1-0.03))

for i in [-3,-2,-1,1,2,3]:
    ax.annotate('%s' %str(i), xy=(i-0.03, -0.2))

maybe = raw_input("\nUpdate figure directly in master thesis?\nEnter 'YES' (anything else = ONLY show to screen) ")
if maybe == "YES": # Only save to disc if need to be updated
    filenameWithPath = "/Users/haakonvt/Dropbox/uio/master/latex-master/Illustrations/six_sigmoids.pdf"
    plt.savefig(filenameWithPath, bbox_inches='tight') #, pad_inches=0.2)
    print 'Saved over previous file in location:\n "%s"' %filenameWithPath
else:
    print 'Figure was only shown on screen.'
    plt.show()
