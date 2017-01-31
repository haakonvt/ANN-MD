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


def G1(r, rc, cutoff=cutoff_cos):
    r_cut     = cutoff(r,rc)
    summation = np.sum( r_cut )
    return summation

def G1_single_neighbor(r, rc, cutoff=cutoff_cos):
    return cutoff(r,rc)


def G2(r, rc, rs, eta, cutoff=cutoff_cos):
    r_cut     = cutoff(r,rc)
    summation = np.sum( np.exp(-eta*(r-rs)**2)*r_cut )
    return summation

def G2_single_neighbor(r, rc, rs, eta, cutoff=cutoff_cos):
    r_cut     = cutoff(r,rc)
    return np.exp(-eta*(r-rs)**2)*r_cut

def G3(r, rc, kappa, cutoff=cutoff_cos):
    r_cut     = cutoff(r,rc)
    summation = np.sum( np.cos(kappa*r)*r_cut )
    return summation

def G3_single_neighbor(r, rc, kappa, cutoff=cutoff_cos):
    r_cut     = cutoff(r,rc)
    return np.cos(kappa*r)*r_cut

def G4(xyz, rc, eta, zeta, lambda_c, cutoff=cutoff_cos):
    """ xyz:
    [[x1 y1 z1]
     [x2 y2 z2]
     [x3 y3 z3]
     [x4 y4 z4]]
     """
    r         = np.linalg.norm(xyz,axis=1)
    N         = len(r)
    r_cut     = cutoff(r,rc)
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

def G4_single_neighbor(theta, r_all, zeta, lambda_c, eta):
    """
    Assumes cutoffs to be normalized to 1 and is removed from eqs
    """
    rij,rik,rjk = [r_all]*3 # All equal
    N = len(theta)
    exp_factor   = np.exp(-eta*(rij**2 + rik**2 + rjk**2))
    angle_factor = 2**(1-zeta) * (1 + lambda_c * np.cos(theta))**zeta
    return angle_factor * exp_factor

def G4_single_neighbor_rjk(theta, rc, zeta, lambda_c, eta, cutoff=cutoff_cos):
    """
    rij   = rik = 0.8 Rc
    theta = 0,..,360
    """
    N   = len(theta)
    rij = 0.8 * rc
    rjk = np.sqrt(2*rij**2 - 4*rij*np.cos(theta))
    exp_factor    = np.exp(-eta*(2*rij**2 + rjk**2))
    angle_factor  = 2**(1-zeta) * (1 + lambda_c * np.cos(theta))**zeta
    cutoff_factor = cutoff(rij)**2 * cutoff(rjk)
    return angle_factor * exp_factor * cutoff_factor

def G4_single_neighbor_radial(r, zeta, lambda_c, eta):
    """
    adfds
    """
    theta = pi/3. # Constant at 60 degrees aka pi/3
    exp_factor   = np.exp(-eta*3*r**2)
    angle_factor = 2**(1-zeta) * (1 + lambda_c * np.cos(theta))**zeta
    return angle_factor * exp_factor

def G5(xyz, rc, eta, zeta, lambda_c, cutoff=cutoff_cos):
    """ xyz:
    [[x1 y1 z1]
     [x2 y2 z2]
     [x3 y3 z3]
     [x4 y4 z4]]
     """
    r         = np.linalg.norm(xyz,axis=1)
    N         = len(r)
    r_cut     = cutoff(r,rc)
    summation = 0
    for j in range(N):
        for k in range(N):
            theta_ijk  = acos( np.dot(xyz[j],xyz[k]) / (r[j]*r[k]) )
            cutoff_ijk = r_cut[j]*r_cut[k]
            part_sum   = (1+lambda_c * cos(theta_ijk))**zeta * exp(-eta*(r[j]**2+r[k]**2))
            summation += part_sum*cutoff_ijk
    summation *= 2**(1-zeta)
    return

def G5_single_neighbor_radial(r, zeta, lambda_c, eta):
    """
    Radial part of G5 when rij = rik
    """
    theta = pi/3. # Constant at 60 degrees aka pi/3
    exp_factor   = np.exp(-eta*2*r**2) # rij = rik
    angle_factor = 2**(1-zeta) * (1 + lambda_c * np.cos(theta))**zeta
    return angle_factor * exp_factor

if __name__ == '__main__':
    """
    Mainly for testing purpose
    """

    print "This does absolutely nothing, I'm afraid dear!"
