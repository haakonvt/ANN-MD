from math import cos,pi

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


def cutoff(rij,rc):
    """
    Can take scalar and vector input of rij and evaluate the cutoff function
    """
    if type(rij) == int:
        if rij <= rc:
            0.5*(cos(pi*rij/rc)+1)
        else:
            return 0
    elif type(rij) in [list,np.ndarray]:
        



def G1(rij, rc):

    return


def G2(rij, rc):

    return


def G3(rij, rc):

    return

def G4(rij, rc):

    return

def G5(rij, rc):

    return
