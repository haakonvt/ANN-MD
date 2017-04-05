from math import *
import numpy as np

# from sympy.utilities.codegen import codegen
# from sympy import *
# from sympy import lambdify

def test_structure_3atom():
    """
    Structure:
    xyz = [[x1,y1,z1],
           [x2,y2,z2],
           [x3,y3,z3]]
    """
    h = sqrt(3)
    xyz = np.array([[0,0,0],
                    [2,0,0],
                    [1,h,0]], dtype=float)
    # print rij, rik, xij, yij, zij, xik, yik, zik

    fx,fy,fz = evaluate_SW_forces(xyz)#rij, rik, xij, yij, zij, xik, yik, zik)
    # print "SW, xyz:",fx,fy,fz
    # fx,fy,fz = evaluate_NN_forces(rij, rik, xij, yij, zij, xik, yik, zik)
    # print "SW, xyz:",fx,fy,fz


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
    for j in [0,1]:
        f_vec += xyz[j,:]*dU2dR[j]/r[j]
    # print dU2dR, sum(dU2dR)

    """
    i = 0, j = 1, k = 2
    """
    rij = r[0]
    rik = r[1]
    rjk = np.linalg.norm(xyz[1,:]-xyz[2,:])
    cos_theta_jik = np.dot(xyz[1],xyz[2])                / (rij*rik)
    cos_theta_ijk = np.dot(xyz[0]-xyz[1], xyz[2]-xyz[1]) / (rij*rjk)
    cos_theta_ikj = np.dot(xyz[0]-xyz[2], xyz[1]-xyz[2]) / (rik*rjk)
    def h(r1, r2, cos_theta):
        term = exp(gam*sig/(r1-a*sig)) * exp(gam*sig/(r2-a*sig)) * lamb*eps*(cos_theta - cos0)**2
        return term
    U3 = h(rij, rik, cos_theta_jik) + h(rij, rjk, cos_theta_ijk) + h(rik, rjk, cos_theta_ikj)
    print U3


    return 0,0,0
    """
    a     = 1.80
    sigma = 2.0951
    B     = 0.6022245584
    cosTheta = (xij*xik + yij*yik + zij*zik)/(rij*rik)
    cosTheta0 = -1.0/3.0
    r0 = a*sigma

    V3 = B*exp(sigma/(rij-r0) + sigma/(rik-r0)) * (cosTheta - cosTheta0)**2

    dVdXij = -B*(cosTheta - cosTheta0)*(sigma*rij*rik*xij*(cosTheta - cosTheta0) \
            + 2*(r0 - rij)**2*(cosTheta*rik*xij - rij*xik))*exp(sigma*(-2*r0 + rij \
            + rik)/((r0 - rij)*(r0 - rik)))/(rij**2*rik*(r0 - rij)**2)

    dVdYij = -B*(cosTheta - cosTheta0)*(sigma*rij*rik*yij*(cosTheta - cosTheta0) \
            + 2*(r0 - rij)**2*(cosTheta*rik*yij - rij*yik))*exp(sigma*(-2*r0 + rij \
            + rik)/((r0 - rij)*(r0 - rik)))/(rij**2*rik*(r0 - rij)**2)

    dVdZij = -B*(cosTheta - cosTheta0)*(sigma*rij*rik*zij*(cosTheta - cosTheta0) \
            + 2*(r0 - rij)**2*(cosTheta*rik*zij - rij*zik))*exp(sigma*(-2*r0 + rij \
            + rik)/((r0 - rij)*(r0 - rik)))/(rij**2*rik*(r0 - rij)**2)

    dVdXik = -B*(cosTheta - cosTheta0)*(sigma*rij*rik*xik*(cosTheta - cosTheta0) \
            + 2*(r0 - rik)**2*(cosTheta*rij*xik - rik*xij))*exp(sigma*(-2*r0 + rij \
            + rik)/((r0 - rij)*(r0 - rik)))/(rij*rik**2*(r0 - rik)**2)

    dVdYik = -B*(cosTheta - cosTheta0)*(sigma*rij*rik*yik*(cosTheta - cosTheta0) \
            + 2*(r0 - rik)**2*(cosTheta*rij*yik - rik*yij))*exp(sigma*(-2*r0 + rij \
            + rik)/((r0 - rij)*(r0 - rik)))/(rij*rik**2*(r0 - rik)**2)

    dVdZik = -B*(cosTheta - cosTheta0)*(sigma*rij*rik*zik*(cosTheta - cosTheta0) \
            + 2*(r0 - rik)**2*(cosTheta*rij*zik - rik*zij))*exp(sigma*(-2*r0 + rij \
            + rik)/((r0 - rij)*(r0 - rik)))/(rij*rik**2*(r0 - rik)**2)

    print "V", V
    print "dVdXij", dVdXij
    print "dVdYij", dVdYij
    print "dVdZij", dVdZij

    print "dVdXik", dVdXik
    print "dVdYik", dVdYik
    print "dVdZik", dVdZik

    xij, yij, zij, xik, yik, zik = symbols('xij, yij, zij, xik, yik, zik')
    B, costheta0, r0, ksi = symbols('B, cosTheta0, r0, ksi')
    rij, rik, rijDotRik, cosTheta = symbols('rij, rik, rijDotRik, cosTheta')

    V = B*exp(ksi / (sqrt(xij**2 + yij**2 + zij**2) - r0) + ksi/( sqrt(xik**2 + yik**2 + zik**2) - r0))*( (xij*xik + yij*yik + zij*zik) / ( sqrt(xij**2 + yij**2 + zij**2)*sqrt(xik**2 + yik**2 + zik**2)) - costheta0)**2

    dVdXij = diff(V, xij)
    dVdXij = dVdXij.subs(sqrt(xij**2 + yij**2 + zij**2), rij)
    dVdXij = dVdXij.subs(sqrt(xik**2 + yik**2 + zik**2), rik)
    dVdXij = dVdXij.subs(xij*xik + yij*yik + zij*zik, rijDotRik)
    dVdXij = dVdXij.subs(rijDotRik/(rij*rik), cosTheta)

    dVdYij = diff(V, yij)
    dVdYij = dVdYij.subs(sqrt(xij**2 + yij**2 + zij**2), rij)
    dVdYij = dVdYij.subs(sqrt(xik**2 + yik**2 + zik**2), rik)
    dVdYij = dVdYij.subs(xij*xik + yij*yik + zij*zik, rijDotRik)
    dVdYij = dVdYij.subs(rijDotRik/(rij*rik), cosTheta)

    dVdZij = diff(V, zij)
    dVdZij = dVdZij.subs(sqrt(xij**2 + yij**2 + zij**2), rij)
    dVdZij = dVdZij.subs(sqrt(xik**2 + yik**2 + zik**2), rik)
    dVdZij = dVdZij.subs(xij*xik + yij*yik + zij*zik, rijDotRik)
    dVdZij = dVdZij.subs(rijDotRik/(rij*rik), cosTheta)

    dVdXik = diff(V, xik)
    dVdXik = dVdXik.subs(sqrt(xij**2 + yij**2 + zij**2), rij)
    dVdXik = dVdXik.subs(sqrt(xik**2 + yik**2 + zik**2), rik)
    dVdXik = dVdXik.subs(xij*xik + yij*yik + zij*zik, rijDotRik)
    dVdXik = dVdXik.subs(rijDotRik/(rij*rik), cosTheta)

    dVdYik = diff(V, yik)
    dVdYik = dVdYik.subs(sqrt(xij**2 + yij**2 + zij**2), rij)
    dVdYik = dVdYik.subs(sqrt(xik**2 + yik**2 + zik**2), rik)
    dVdYik = dVdYik.subs(xij*xik + yij*yik + zij*zik, rijDotRik)
    dVdYik = dVdYik.subs(rijDotRik/(rij*rik), cosTheta)

    dVdZik = diff(V, zik)
    dVdZik = dVdZik.subs(sqrt(xij**2 + yij**2 + zij**2), rij)
    dVdZik = dVdZik.subs(sqrt(xik**2 + yik**2 + zik**2), rik)
    dVdZik = dVdZik.subs(xij*xik + yij*yik + zij*zik, rijDotRik)
    dVdZik = dVdZik.subs(rijDotRik/(rij*rik), cosTheta)

    print "dVdXij", "\n", simplify(dVdXij),"\n"
    print "dVdYij", "\n", simplify(dVdYij),"\n"
    print "dVdZij", "\n", simplify(dVdZij),"\n"

    print "dVdXik", "\n", simplify(dVdXik),"\n"
    print "dVdYik", "\n", simplify(dVdYik),"\n"
    print "dVdZik", "\n", simplify(dVdZik)

    # lambdify(simplify(dVdXij))
    # lambdify(simplify(dVdYij))
    # lambdify(simplify(dVdZij))
    # lambdify(simplify(dVdXik))
    # lambdify(simplify(dVdYik))
    # lambdify(simplify(dVdZik))"""


if __name__ == '__main__':
    test_structure_3atom()
    # evaluate_SW_forces("asdf")
