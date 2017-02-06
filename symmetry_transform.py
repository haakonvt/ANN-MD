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

# TODO: Almost everything
