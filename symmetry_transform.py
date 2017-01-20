from matplotlib import rc
rc('font',**{'family':'sans-serif','sans-serif':['Helvetica']})
# rc('text', usetex=True)

import matplotlib as mpl
mpl.rcParams['lines.linewidth'] = 2.5

import matplotlib.pyplot as plt
from math import cos,pi,tanh,acos # Faster than numpy for scalars
import numpy as np

def symmetryTransform(particleCoordinates):
    """
    Input:
    x,y,z-coordinates of N particles
    [[x1 y1 z1]
     [x2 y2 z2]
     [x3 y3 z3]
     [x4 y4 z4]]

    Output:
    G1 [g1 g2 g3 ... gN]
    G2 [etc.]
    etc...
    """
    xyz = particleCoordinates
    N   = xyz.shape[0]


def cutoff_tanh(r,rc):
    """
    Can take scalar and vector input of r and evaluate the cutoff function
    """
    if type(r) == int:
        if r <= rc:
            return tanh(1-r/rc)**3
        else:
            return 0.
    else:
        return np.tanh(1-r/rc)**3 * (r <= rc)


def cutoff_cos(r,rc):
    """
    Can take scalar and vector input of r and evaluate the cutoff function
    """
    if type(r) == int:
        if r <= rc:
            return 0.5*(cos(pi*r/rc)+1)
        else:
            return 0.
    else:
        return 0.5*(np.cos(pi*r/rc)+1) * (r <= rc)


def G1(r, rc, cutoff):
    N         = len(r)
    r_cut     = cutoff(r)
    summation = np.sum( r_cut )
    return summation


def G2(r, rc, rs, eta, cutoff):
    N         = len(r)
    r_cut     = cutoff(r)
    summation = np.sum( np.exp(-eta*(r-rs)**2)*r_cut )
    return summation


def G3(r, rc, kappa, cutoff):
    N         = len(r)
    r_cut     = cutoff(r)
    summation = np.sum( np.cos(kappa*r)*r_cut )
    return summation


def G4(xyz, rc, eta, zeta, lambda_c, cutoff):
    """ xyz:
    [[x1 y1 z1]
     [x2 y2 z2]
     [x3 y3 z3]
     [x4 y4 z4]]
     """
    r         = np.linalg.norm(xyz,axis=1)
    N         = len(r)
    r_cut     = cutoff(r)
    summation = 0
    for j in range(N):
        for k in range(N):
            r_jk       = np.linalg.norm(xyz[j] - xyz[k])
            theta_ijk  = acos( np.dot(xyz[j],xyz[k]) / (r[j]*r[k]) )
            cutoff_ijk = r_cut[j]*r_cut[k]*cutoff(r_jk)
            part_sum   = (1+lambda_c * cos(theta_ijk))**zeta * exp(-eta*(r[j]**2+r[k]**2+r_jk**2))
            summation += part_sum*cutoff_ijk
    summation *= 2**(1-zeta)
    return


def G5(r, rc, cutoff):
    N         = len(r)
    r_cut     = cutoff(r)
    summation = np.sum(r_cut)
    for j in range(N):
        pass
    return


def the_differ(x):
    h = (x[1]-x[0])
    x = np.diff(x) / h
    return np.diff(x)/h

if __name__ == '__main__':
    """
    Mainly for testing purpose
    """
    # General plot settings:
    colormap = plt.cm.Spectral #nipy_spectral # Other possible colormaps: Set1, Accent, nipy_spectral, Paired

    if True: # Case one
        N      = 6
        colors = [colormap(i) for i in np.linspace(0, 1, N)]
        r    = np.linspace(0,3,10001)
        for i,rc in enumerate(np.linspace(0.75,2.75,N)):
            r_cut = cutoff_cos(r,rc)
            plt.plot(r,r_cut,color=colors[i])
            plt.title("Case 1")
        plt.show()

    if True: # Case one
        N      = 6
        colors = [colormap(i) for i in np.linspace(0, 1, N)]
        r    = np.linspace(0,3,10001)
        for i,rc in enumerate(np.linspace(0.75,2.75,N)):
            r_cut = the_differ(cutoff_cos(r,rc))
            plt.plot(r[:-2],r_cut,color=colors[i])
            plt.title("Case 2")
        plt.show()

    if True: # Case two
        N      = 6
        colors = [colormap(i) for i in np.linspace(0, 1, N)]
        r    = np.linspace(0,3,10001)
        for i,rc in enumerate(np.linspace(0.75,2.75,N)):
            r_cut = cutoff_tanh(r,rc)
            plt.plot(r,r_cut,color=colors[i])
            plt.title("Case 3")
        plt.show()

    if True: # Case two
        N      = 6
        colors = [colormap(i) for i in np.linspace(0, 1, N)]
        r    = np.linspace(0,3,10001)
        for i,rc in enumerate(np.linspace(0.75,2.75,N)):
            r_cut = the_differ(cutoff_tanh(r,rc))
            plt.plot(r[:-2],r_cut,color=colors[i])
            plt.title("Case 4")
        plt.show()
