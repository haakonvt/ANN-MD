from sympy.utilities.codegen import codegen
from sympy import *
import sys

# Decide form of output from command line
try:
    c_or_latex = sys.argv[1]
except:
    print "Usage:"
    print ">>> python sympy_diff.py latex"
    print ">>> python sympy_diff.py c"
    sys.exit()

if c_or_latex == "latex":
    # Symbols are defined in a way to make latex writing pretty
    out_c_code = False
    out_latex  = True
    xij, yij, zij             = symbols(r'x_{ij}, y_{ij}, z_{ij}')
    xik, yik, zik             = symbols(r'x_{ik}, y_{ik}, z_{ik}')
    xjk, yjk, zjk             = symbols(r'x_{jk}, y_{jk}, z_{jk}')
    rc, rs, eta, lamb, zeta   = symbols(r'r_c, r_s, \eta, \lambda, \zeta', constant=True)
    rij, rik, rjk, rij_dot_ik = symbols(r'r_{ij}, r_{ik}, r_{jk}, rij_dot_ik', positive=True)
    cosTheta                  = symbols(r'\cos(\theta)')
elif c_or_latex == "c":
    # Symbols are defined in a way to make c-code understandable
    out_c_code = True
    out_latex  = False
    xij, yij, zij             = symbols('xij, yij, zij')
    xik, yik, zik             = symbols('xik, yik, zik')
    xjk, yjk, zjk             = symbols('xjk, yjk, zjk')
    rc, rs, eta, lamb, zeta   = symbols('rc, rs, eta, lambda, zeta', constant=True)
    rij, rik, rjk, rij_dot_ik = symbols('rij, rik, rjk, rij_dot_ik', positive=True)
    cosTheta                  = symbols('cosTheta')

# Cutoff functions
fc_rij = 0.5*(cos(pi*sqrt(xij**2 + yij**2 + zij**2)/rc)+1)
fc_rik = 0.5*(cos(pi*sqrt(xik**2 + yik**2 + zik**2)/rc)+1)
fc_rjk = 0.5*(cos(pi*sqrt(xjk**2 + yjk**2 + zjk**2)/rc)+1)

# Symmetry functions
G2    = exp(-eta*(sqrt(xij**2 + yij**2 + zij**2)-rs)**2) * fc_rij
cos_t = ( xij*xik + yij*yik + zij*zik )/ \
        ( sqrt(xij**2 + yij**2 + zij**2) * sqrt(xik**2 + yik**2 + zik**2) )
G4    = 2**(1-zeta) \
        * (1+lamb*cos_t)**zeta \
        * exp(-eta*(xij**2+yij**2+zij**2+ \
                   xik**2+yik**2+zik**2+ \
                   xjk**2+yjk**2+zjk**2)) * fc_rij * fc_rik * fc_rjk

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
    if out_latex:
        print latex(dG4d_var)

def print_G2_derivative(dG2d_var, var, out_c_code=False, out_latex=False):
    print "\n--------------------"
    print "dG2/d%s " %str(var)
    print "--------------------"
    dG2d_var = dG2d_var.subs(sqrt(xij**2 + yij**2 + zij**2), rij)
    dG2d_var = dG2d_var.simplify()
    if out_c_code:
        print codegen(("dG2d"+str(var), dG2d_var), "C", "file")[0][1]
    if out_latex:
        print latex(dG2d_var)

# Symbolic differentiation of G2
dG2dXij = diff(G2, xij)
dG2dYij = diff(G2, yij)
dG2dZij = diff(G2, zij)

print_G2_derivative(dG2dXij, xij, out_c_code, out_latex)
print_G2_derivative(dG2dYij, xij, out_c_code, out_latex)
print_G2_derivative(dG2dZij, xij, out_c_code, out_latex)

# Symbolic differentiation of G4
dG4dXij = diff(G4, xij)
dG4dYij = diff(G4, yij)
dG4dZij = diff(G4, zij)

print_G4_derivative(dG4dXij, xij, out_c_code, out_latex)
print_G4_derivative(dG4dYij, xij, out_c_code, out_latex)
print_G4_derivative(dG4dZij, xij, out_c_code, out_latex)
