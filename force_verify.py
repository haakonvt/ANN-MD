from math import *
import numpy as np

def test_structure_3atom():
    """
    Structure:
    xyz = [[0, 0, 0 ], <--- must be origo
           [x2,y2,z2],
           [x3,y3,z3]]
    """
    h = sqrt(3)
    xyz = np.array([[0,0,0],
                    [2,0,0],
                    [1,h,0]], dtype=float)

    f_vec_sw = evaluate_SW_forces(xyz)
    print "Tot SW force:    ",f_vec_sw
    f_vec_nn = evaluate_NN_forces(xyz)
    print "Tot NN force:    ",f_vec_nn


def evaluate_SW_forces(xyz):
    """
    Stillinger and Weber,  Phys. Rev. B, v. 31, p. 5262, (1985)

    epsilon, sigma, a, lambda, gamma, costheta0, A, B, p, q
    2.1683 , 2.0951,  1.80,  21.0, 1.20,  -1.0/, 7.049556277,  0.6022245584,  4.0,  0.0
    """
    eps   = 2.1683
    sig   = 2.0951
    a     = 1.80
    lamb  = 21.0
    gam   = 1.2
    A     = 7.049556277
    B     = 0.6022245584
    p     = 4.0
    q     = 0.0
    cos0 = -1.0/3.0

    f_vec = np.zeros(3)

    r  = np.linalg.norm(xyz[1:,:], axis=1) # Only j != i
    U2 = A*eps*(B*(sig/r)**p-(sig/r)**q) * np.exp(sig/(r-a*sig)) * (r < a*sig)
    # print U2, sum(U2)
    dU2dR = -A*eps*sig*(B*(sig/r)**p - (sig/r)**q)*np.exp(sig/(-a*sig + r))/(-a*sig + r)**2 + A*eps*(-B*p*(sig/r)**p/r + q*(sig/r)**q/r)*np.exp(sig/(-a*sig + r))
    for j in [1,2]:
        # print xyz[j,:]*dU2dR[j-1]/r[j-1]
        f_vec += xyz[j,:]*dU2dR[j-1]/r[j-1]
    # print dU2dR, sum(dU2dR)

    """
    i = 0, j = 1, k = 2
    """
    rij           = r[0]
    rik           = r[1]
    rjk           = np.linalg.norm(xyz[1,:]-xyz[2,:])
    cos_theta_jik = np.dot(xyz[1],xyz[2])                / (rij*rik)
    cos_theta_ijk = np.dot(xyz[0]-xyz[1], xyz[2]-xyz[1]) / (rij*rjk)
    cos_theta_ikj = np.dot(xyz[0]-xyz[2], xyz[1]-xyz[2]) / (rik*rjk)
    # print "cos Theta:",cos_theta_jik, cos_theta_ijk, cos_theta_ikj

    def h(r1, r2, cos_theta):
        term = exp(gam*sig/(r1-a*sig)) * exp(gam*sig/(r2-a*sig)) * lamb*eps*(cos_theta - cos0)**2
        return term

    h_jik = h(rij, rik, cos_theta_jik)
    h_ijk = h(rij, rjk, cos_theta_ijk)
    h_ikj = h(rik, rjk, cos_theta_ikj)
    # U3 = h_jik + h_ijk + h_ikj
    # print U3

    xyz /= sig
    rij /= sig
    rik /= sig
    rjk /= sig
    xyz_jk = xyz[2]-xyz[1]
    dhjik_dri = -gam*h_jik*((xyz[1]/rij)*1/(rij-a)**2 + (xyz[2]/rik)*1/(rik-a)**2) \
                + 2*lamb*np.exp(gam/(rij-a) + gam/(rik-a))*(cos_theta_jik - cos0)  \
                * ((xyz[1]/rij)*(1/rik) + (xyz[2]/rik)*(1/rij) - (xyz[1]/(rij*rik)+xyz[2]/(rik*rij))*cos_theta_jik)

    dhijk_dri = -gam*h_ijk*((xyz[1]/rij)*1/(rij-a)**2) \
                + 2*lamb*np.exp(gam/(rij-a) + gam/(rjk-a))*(cos_theta_ijk - cos0) \
                * ((xyz_jk/rjk)*(1/rij) + (xyz[1]/rij)*(1/rij)*cos_theta_ijk)

    dhikj_dri = -gam*h_ikj*((xyz[2]/rik)*1/(rik-a)**2) \
                + 2*lamb*np.exp(gam/(rik-a) + gam/(rjk-a))*(cos_theta_ikj - cos0) \
                *((-xyz_jk/rjk)*(1/rik) + (xyz[2]/rik)*(1/rik)*cos_theta_ikj)
    print "Tot 2-body force:", f_vec
    print "Tot 3-body force:", dhjik_dri + dhijk_dri + dhikj_dri
    f_vec += dhjik_dri + dhijk_dri + dhikj_dri
    return f_vec


def evaluate_NN_forces(xyz):
    pass
    
if __name__ == '__main__':
    test_structure_3atom()
    # evaluate_SW_forces("asdf")
