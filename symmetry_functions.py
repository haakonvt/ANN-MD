from math import exp,cos,pi,tanh,sqrt # Faster than numpy for scalars
import numpy as np

"""
#################
The cutoff functions
#################
"""

def cutoff_tanh(r,rc):
    """
    Can take scalar and vector input of r and evaluate the cutoff function
    """
    # rc = float(rc)
    if type(r) == int:
        if r <= rc:
            return tanh(1-r/rc)**3
        else:
            return 0.
    else:
        return np.tanh(1-r/rc)**3 * (r <= rc)

def cutoff_cos(r,rc):
    """
    Can take scalar, vector or matrix input of r and evaluate the cutoff function
    """
    # rc = float(rc)
    if type(r) == int:
        if r <= rc:
            return 0.5*(cos(pi*r/rc)+1)
        else:
            return 0.
    else:
        return 0.5*(np.cos(pi*r/rc)+1) * (r <= rc)

"""
#################
Single particle symmetry functions
#################
"""

def G1(r, rc, cutoff=cutoff_cos):
    r_cut     = cutoff(r,rc)
    summation = np.sum( r_cut )
    return summation

def G2(r, rc, rs, eta, cutoff=cutoff_cos):
    r_cut     = cutoff(r,rc)
    summation = np.sum( np.exp(-eta*(r-rs)**2)*r_cut )
    return summation

def G3(r, rc, kappa, cutoff=cutoff_cos):
    r_cut     = cutoff(r,rc)
    summation = np.sum( np.cos(kappa*r)*r_cut )
    return summation

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
    # IDEA: Speed up double loop with numpy arrays (if possible)
    # r_jk_array = np.zeros((N,N))
    # r_jk_array = np.linalg.norm(xyz - xyz) ??? nowhere near finished
    for j in range(N):
        for k in range(N): # This double counts angles... as in the litterature
            if j == k:
                continue # Skip j=k
            r_jk       = np.linalg.norm(xyz[j] - xyz[k])
            cos_theta  = np.dot(xyz[j],xyz[k]) / (r[j]*r[k])
            cutoff_ijk = r_cut[j] * r_cut[k] * cutoff(r_jk, rc)
            part_sum   = (1+lambda_c * cos_theta)**zeta * exp(-eta*(r[j]**2+r[k]**2+r_jk**2))
            summation += part_sum*cutoff_ijk
    summation *= 2**(1-zeta) # Normalization factor
    return summation

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
        for k in range(N): # This double counts angles... as in the litterature
            if j == k:
                continue # Skip j=k
            cos_theta  = np.dot(xyz[j],xyz[k]) / (r[j]*r[k])
            cutoff_ijk = r_cut[j] * r_cut[k]
            part_sum   = (1+lambda_c * cos_theta)**zeta * exp(-eta*(r[j]**2+r[k]**2))
            summation += part_sum*cutoff_ijk
    summation *= 2**(1-zeta)
    return summation

"""
#################
N particle symmetry functions
#################
"""

def G1_N(r, type, rc, cutoff=cutoff_cos):
    """
    r    = [r1        , r2      , ...,        rN]
    type = ["Hydrogen", "Oxygen", ...,  "Carbon"]
    """
    # TODO: May move this to symmetry_transform.py in stead...

"""
#################
Next functions are mainly for testing purposes
i.e. plotting response-curves etc.
#################
"""

def G1_single_neighbor(r, rc, cutoff=cutoff_cos):
    return cutoff(r,rc)

def G2_single_neighbor(r, rc, rs, eta, cutoff=cutoff_cos):
    r_cut     = cutoff(r,rc)
    return np.exp(-eta*(r-rs)**2)*r_cut

def G3_single_neighbor(r, rc, kappa, cutoff=cutoff_cos):
    r_cut     = cutoff(r,rc)
    return np.cos(kappa*r)*r_cut

def G4_single_neighbor_rjk(theta, rc, zeta, lambda_c, eta, cutoff=cutoff_cos, percent_of_rc=0.8):
    """
    rij   = rik = 0.8 Rc
    theta = 0,..,360
    """
    rij           = percent_of_rc * rc
    cos_theta     = np.cos(theta)
    rjk           = sqrt(2) * rij * np.sqrt(1 - cos_theta) # Simplified law of cosines
    exp_factor    = np.exp(-eta*(2*rij**2 + rjk**2))
    angle_factor  = 2**(1-zeta) * (1 + lambda_c * cos_theta)**zeta
    cutoff_factor = cutoff(rij, rc)**2 * cutoff(rjk, rc)
    return angle_factor * exp_factor * cutoff_factor

def G4_single_neighbor_radial(r, zeta, lambda_c, eta):
    """
    adfds
    """
    theta = pi/3. # Constant at 60 degrees aka pi/3
    exp_factor   = np.exp(-eta*3*r**2)
    angle_factor = 2**(1-zeta) * (1 + lambda_c * np.cos(theta))**zeta
    return angle_factor * exp_factor

def G4_single_neighbor_radial_cut(r, rc, zeta, lambda_c, eta, cutoff=cutoff_cos):
    """
    With cutoff
    """
    theta = pi/3. # Constant at 60 degrees aka pi/3
    exp_factor   = np.exp(-eta*3*r**2)
    angle_factor = 2**(1-zeta) * (1 + lambda_c * np.cos(theta))**zeta
    return angle_factor * exp_factor * cutoff(r, rc)**3

def G5_single_neighbor_radial_cut(r, rc, zeta, lambda_c, eta, rs, cutoff=cutoff_cos):
    """
    With cutoff
    """
    theta = pi/3. # Constant at 60 degrees aka pi/3
    exp_factor   = np.exp(-eta * 2*(r-rs)**2)
    angle_factor = 2**(1-zeta) * (1 + lambda_c * cos(theta))**zeta
    return angle_factor * exp_factor * cutoff(r, rc)**2

def G4_single_neighbor_2D(theta_grid, rc_grid, r_all, zeta, lambda_c, eta):
    cutoff        = cutoff_cos
    rij           = r_all # rij = rik
    cos_theta     = np.cos(theta_grid)
    rjk           = sqrt(2) * rij * np.sqrt(1 - cos_theta) # Simplified law of cosines
    exp_factor    = np.exp(-eta*(2*rij**2 + rjk**2))
    angle_factor  = 2**(1-zeta) * (1 + lambda_c * cos_theta)**zeta
    cutoff_factor = cutoff(rij, rc_grid)**2 * cutoff(rjk, rc_grid)
    return angle_factor * exp_factor * cutoff_factor

def G4_single_neighbor(theta, r_all, rc, zeta, lambda_c, eta):
    """
    NB: Number 4, not 5
    """
    cutoff        = cutoff_cos
    rij           = r_all # rij = rik
    cos_theta     = np.cos(theta)
    rjk           = sqrt(2) * rij * np.sqrt(1 - cos_theta) # Simplified law of cosines
    exp_factor    = np.exp(-eta*(2*rij**2 + rjk**2))
    angle_factor  = 2**(1-zeta) * (1 + lambda_c * cos_theta)**zeta
    cutoff_factor = cutoff(rij, rc)**2 * cutoff(rjk, rc)
    return angle_factor * exp_factor * cutoff_factor

def G5_single_neighbor(theta, r_all, rc, zeta, lambda_c, eta):
    """
    Assumes cutoffs to be normalized to 1 and is removed from eqs
    """
    cutoff        = cutoff_cos
    rij           =  r_all # Both equal
    exp_factor    = np.exp(-eta*2*rij**2)
    angle_factor  = 2**(1-zeta) * (1 + lambda_c * np.cos(theta))**zeta
    cutoff_factor = cutoff(rij, rc)**2
    return angle_factor * exp_factor * cutoff_factor

def G5_single_neighbor_radial(r, zeta, lambda_c, eta):
    """
    Radial part of G5 when rij = rik
    """
    theta = pi/3. # Constant at 60 degrees aka pi/3
    exp_factor   = np.exp(-eta*2*r**2) # rij = rik
    angle_factor = 2**(1-zeta) * (1 + lambda_c * np.cos(theta))**zeta
    return angle_factor * exp_factor

def G5_single_neighbor_rjk(theta, rc, zeta, lambda_c, eta, cutoff=cutoff_cos, percent_of_rc=0.8):
    """
    rij   = rik = 0.8 Rc
    theta = 0,..,360
    """
    rij           = percent_of_rc * rc
    cos_theta     = np.cos(theta)
    exp_factor    = np.exp(-eta*2*rij**2)
    angle_factor  = 2**(1-zeta) * (1 + lambda_c * cos_theta)**zeta
    cutoff_factor = cutoff(rij, rc)**2
    return angle_factor * exp_factor * cutoff_factor

if __name__ == '__main__':
    """
    Mainly for testing purpose
    """
    print "This does absolutely nothing, I'm afraid dear!"
