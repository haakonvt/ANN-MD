from sympy.utilities.codegen import codegen
from sympy import *
import sys

# Decide form of output from command line
try:
    output_choice = sys.argv[1]
except:
    print "Usage:"
    print ">>> python sympy_diff.py latex"
    print ">>> python sympy_diff.py c"
    print ">>> python sympy_diff.py python"
    sys.exit()

if output_choice == "latex":
    # Symbols are defined in a way to make latex writing pretty
    out_c_code = False
    out_latex  = True
    xij, yij, zij             = symbols(r'x_{ij}, y_{ij}, z_{ij}')
    xik, yik, zik             = symbols(r'x_{ik}, y_{ik}, z_{ik}')
    xjk, yjk, zjk             = symbols(r'x_{jk}, y_{jk}, z_{jk}')
    rc, rs, eta, lamb, zeta   = symbols(r'r_c, r_s, \eta, \lambda, \zeta', constant=True)
    rij, rik, rjk, rij_dot_ik = symbols(r'r_{ij}, r_{ik}, r_{jk}, rij_dot_ik', positive=True)
    cosTheta                  = symbols(r'\cos(\theta)')
    kappa                     = symbols(r'\kappa', constant=True)
else:
    # Symbols are defined in a way to make c-code understandable
    if output_choice == "c":
        out_c_code = True
    else:
        out_c_code = False
    out_latex  = False
    xij, yij, zij             = symbols('xij, yij, zij')
    xik, yik, zik             = symbols('xik, yik, zik')
    xjk, yjk, zjk             = symbols('xjk, yjk, zjk')
    rc, rs, eta, lamb, zeta   = symbols('rc, rs, eta, lambda, zeta', constant=True)
    rij, rik, rjk, rij_dot_ik = symbols('rij, rik, rjk, rij_dot_ik', positive=True)
    cosTheta                  = symbols('cosTheta')
    kappa                     = symbols(r'kappa', constant=True)

# Cutoff functions: assuming ONLY legal input
fc_rij = 0.5*(cos(pi*sqrt(xij**2 + yij**2 + zij**2)/rc)+1)
fc_rik = 0.5*(cos(pi*sqrt(xik**2 + yik**2 + zik**2)/rc)+1)
fc_rjk = 0.5*(cos(pi*sqrt(xjk**2 + yjk**2 + zjk**2)/rc)+1)

# Symmetry functions
G1    = 1.0*fc_rij
G2    = exp(-eta*(sqrt(xij**2 + yij**2 + zij**2)-rs)**2) * fc_rij
G3    = cos(kappa*sqrt(xij**2 + yij**2 + zij**2))        * fc_rij
cos_t = ( xij*xik + yij*yik + zij*zik )/ \
        ( sqrt(xij**2 + yij**2 + zij**2) * sqrt(xik**2 + yik**2 + zik**2) )
G4    = 2**(1-zeta) \
        * (1+lamb*cos_t)**zeta \
        * exp(-eta*(xij**2+yij**2+zij**2+ \
                    xik**2+yik**2+zik**2+ \
                    xjk**2+yjk**2+zjk**2))\
        * fc_rij * fc_rik * fc_rjk
G5    = 2**(1-zeta) \
        * (1+lamb*cos_t)**zeta \
        * exp(-eta*(xij**2+yij**2+zij**2+ \
                    xik**2+yik**2+zik**2))\
        * fc_rij * fc_rik

def print_G1_derivative(dG1d_var, var, out_c_code=False, out_latex=False):
    print "\n--------------------"
    print "dG1/d%s " %str(var)
    print "--------------------"
    dG1d_var = dG1d_var.subs(sqrt(xij**2 + yij**2 + zij**2), rij)
    dG1d_var = dG1d_var.simplify()
    if out_c_code:
        print codegen(("dG1d"+str(var), dG1d_var), "C", "file")[0][1]
    elif out_latex:
        print latex(dG1d_var)
    else:
        print dG1d_var

def print_G2_derivative(dG2d_var, var, out_c_code=False, out_latex=False):
    print "\n--------------------"
    print "dG2/d%s " %str(var)
    print "--------------------"
    dG2d_var = dG2d_var.subs(sqrt(xij**2 + yij**2 + zij**2), rij)
    dG2d_var = dG2d_var.simplify()
    if out_c_code:
        print codegen(("dG2d"+str(var), dG2d_var), "C", "file")[0][1]
    elif out_latex:
        print latex(dG2d_var)
    else:
        print dG2d_var

def print_G3_derivative(dG3d_var, var, out_c_code=False, out_latex=False):
    print "\n--------------------"
    print "dG3/d%s " %str(var)
    print "--------------------"
    dG3d_var = dG3d_var.subs(sqrt(xij**2 + yij**2 + zij**2), rij)
    dG3d_var = dG3d_var.simplify()
    if out_c_code:
        print codegen(("dG3d"+str(var), dG3d_var), "C", "file")[0][1]
    elif out_latex:
        print latex(dG3d_var)
    else:
        print dG3d_var

def print_G4_derivative(dG4d_var, var, out_c_code=False, out_latex=False):
    print "\n--------------------"
    print "dG4/d%s " %str(var)
    print "--------------------"
    dG4d_var = dG4d_var.subs( sqrt(xij**2 + yij**2 + zij**2), rij        )
    dG4d_var = dG4d_var.subs( sqrt(xik**2 + yik**2 + zik**2), rik        )
    dG4d_var = dG4d_var.subs( sqrt(xjk**2 + yjk**2 + zjk**2), rjk        )
    dG4d_var = dG4d_var.subs( xij*xik + yij*yik + zij*zik   , rij_dot_ik )
    dG4d_var = dG4d_var.subs( rij_dot_ik/(rij*rik)          , cosTheta   )
    dG4d_var = dG4d_var.subs( xij**2 + yij**2 + zij**2      , rij**2     )
    dG4d_var = dG4d_var.subs( xik**2 + yik**2 + zik**2      , rik**2     )
    dG4d_var = dG4d_var.subs( xjk**2 + yjk**2 + zjk**2      , rjk**2     )
    dG4d_var = dG4d_var.simplify()
    if out_c_code:
        print codegen(("dG4d"+str(var), dG4d_var), "C", "file")[0][1]
    elif out_latex:
        print latex(dG4d_var)
    else:
        print dG4d_var

def print_G5_derivative(dG5d_var, var, out_c_code=False, out_latex=False):
    print "\n--------------------"
    print "dG5/d%s " %str(var)
    print "--------------------"
    dG5d_var = dG5d_var.subs( sqrt(xij**2 + yij**2 + zij**2), rij        )
    dG5d_var = dG5d_var.subs( sqrt(xik**2 + yik**2 + zik**2), rik        )
    dG5d_var = dG5d_var.subs( xij*xik + yij*yik + zij*zik   , rij_dot_ik )
    dG5d_var = dG5d_var.subs( rij_dot_ik/(rij*rik)          , cosTheta   )
    dG5d_var = dG5d_var.subs( xij**2 + yij**2 + zij**2      , rij**2     )
    dG5d_var = dG5d_var.subs( xik**2 + yik**2 + zik**2      , rik**2     )
    dG5d_var = dG5d_var.simplify()
    if out_c_code:
        print codegen(("dG5d"+str(var), dG5d_var), "C", "file")[0][1]
    elif out_latex:
        print latex(dG5d_var)
    else:
        print dG5d_var

# Symbolic differentiation of G1
if raw_input("\nDifferentiate G1? (y/yes/enter)") in ["y","yes",""]:
    dG1dXij = diff(G1, xij)
    dG1dYij = diff(G1, yij)
    dG1dZij = diff(G1, zij)

    print_G1_derivative(dG1dXij, xij, out_c_code, out_latex)
    print_G1_derivative(dG1dYij, yij, out_c_code, out_latex)
    print_G1_derivative(dG1dZij, zij, out_c_code, out_latex)

# Symbolic differentiation of G2
if raw_input("\nDifferentiate G2? (y/yes/enter)") in ["y","yes",""]:
    dG2dXij = diff(G2, xij)
    dG2dYij = diff(G2, yij)
    dG2dZij = diff(G2, zij)

    print_G2_derivative(dG2dXij, xij, out_c_code, out_latex)
    print_G2_derivative(dG2dYij, yij, out_c_code, out_latex)
    print_G2_derivative(dG2dZij, zij, out_c_code, out_latex)

# Symbolic differentiation of G3
if raw_input("\nDifferentiate G3? (y/yes/enter)") in ["y","yes",""]:
    dG3dXij = diff(G3, xij)
    dG3dYij = diff(G3, yij)
    dG3dZij = diff(G3, zij)

    print_G3_derivative(dG3dXij, xij, out_c_code, out_latex)
    print_G3_derivative(dG3dYij, yij, out_c_code, out_latex)
    print_G3_derivative(dG3dZij, zij, out_c_code, out_latex)

# Symbolic differentiation of G4
if raw_input("\nDifferentiate G4? (y/yes/enter)") in ["y","yes",""]:
    dG4dXij = diff(G4, xij)
    dG4dYij = diff(G4, yij)
    dG4dZij = diff(G4, zij)
    dG4dXik = diff(G4, xik)
    dG4dYik = diff(G4, yik)
    dG4dZik = diff(G4, zik)

    print_G4_derivative(dG4dXij, xij, out_c_code, out_latex)
    print_G4_derivative(dG4dYij, yij, out_c_code, out_latex)
    print_G4_derivative(dG4dZij, zij, out_c_code, out_latex)
    print_G4_derivative(dG4dXik, xik, out_c_code, out_latex)
    print_G4_derivative(dG4dYik, yik, out_c_code, out_latex)
    print_G4_derivative(dG4dZik, zik, out_c_code, out_latex)

# Symbolic differentiation of G5
if raw_input("\nDifferentiate G5? (y/yes/enter)") in ["y","yes",""]:
    dG5dXij = diff(G5, xij)
    dG5dYij = diff(G5, yij)
    dG5dZij = diff(G5, zij)
    dG5dXik = diff(G5, xik)
    dG5dYik = diff(G5, yik)
    dG5dZik = diff(G5, zik)

    print_G5_derivative(dG5dXij, xij, out_c_code, out_latex)
    print_G5_derivative(dG5dYij, yij, out_c_code, out_latex)
    print_G5_derivative(dG5dZij, zij, out_c_code, out_latex)
    print_G5_derivative(dG5dXik, xik, out_c_code, out_latex)
    print_G5_derivative(dG5dYik, yik, out_c_code, out_latex)
    print_G5_derivative(dG5dZik, zik, out_c_code, out_latex)
