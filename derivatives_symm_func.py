"""
NOT FINISHED, DONT RUN!
"""

# Loop over all values of the symmetry vector
    for i in range(self.nmbr_G):
        which_symm = self.G_funcs[i][0]
        dEdG_i     = float(-dNNdG[i])
        if which_symm == 2:
            """
            # This is G2
            """
            _, eta, rc, rs = self.G_funcs[i]
            dG2dXYZ_values = dG2dXYZ(xyz, float(eta), float(rc), float(rs))
            f_vec_G2 += dEdG_i * dG2dXYZ_values
        elif which_symm == 4:
            """
            # This is G4
            """
            _, eta, rc, zeta, lamb = self.G_funcs[i]
            dG4dXYZ_values = dG4dXYZ(xyz, float(eta), float(rc), float(zeta), float(lamb))
            f_vec_G4 += dEdG_i * dG4dXYZ_values
        else:
            print "Only use symmetry functions G2 or G4! Exiting!"
            sys.exit(0)
    # print "NN tot G2, Fx, Fy, Fz:", f_vec_G2,"\n\n"
    # print "NN Force G4, Fx, Fy, Fz:", f_vec_G4
    return  f_vec_G4 + f_vec_G2



def dG4dXYZ(xyz, eta, rc, zeta, lamb):
    """
    Derivative of symmetry function G4 w.r.t xij, yij, zij.
    """
    # Vectors of x,y,z-coordinates for neighbours
    if len(xyz[:,0]) != 2:
        # print "Warning: dG4 d x,y,z only works on single unique triplet"
        return np.zeros(3)
    r = np.linalg.norm(xyz, axis=1) # Neighbour distances
    xij = xyz[0,0]; yij = xyz[0,1]; zij = xyz[0,2]
    xik = xyz[1,0]; yik = xyz[1,1]; zik = xyz[1,2]
    rij = r[0]    ; rik = r[1]    ; rjk = np.linalg.norm(xyz[0,:]-xyz[1,:])
    cosTheta = (xij*xik + yij*yik + zij*zik) / (rij*rik)

    # ...and finally the derivaties from sympy: (hold on to something!!)
    dG4_dxij = -2**(-zeta)*(cos(pi*rik/rc) + 1)*(cos(pi*rjk/rc) + 1)\
               *(0.5*eta*rc*rij**2*rik*xij*(cosTheta*lamb + 1)**(zeta + 1)\
               *(cos(pi*rij/rc) + 1) + 0.25*lamb*rc*zeta*(cosTheta*lamb + 1)**zeta\
               *(cosTheta*rik*xij - rij*xik)*(cos(pi*rij/rc) + 1) + 0.25*pi*rij*rik*xij\
               *(cosTheta*lamb + 1)**(zeta + 1)*sin(pi*rij/rc))*exp(-eta*(rij**2 + rik**2 + rjk**2))\
               /(rc*rij**2*rik*(cosTheta*lamb + 1))
    dG4_dyij = -2**(-zeta)*(cos(pi*rik/rc) + 1)*(cos(pi*rjk/rc) + 1)\
               *(0.5*eta*rc*rij**2*rik*yij*(cosTheta*lamb + 1)**(zeta + 1)\
               *(cos(pi*rij/rc) + 1) + 0.25*lamb*rc*zeta*(cosTheta*lamb + 1)**zeta\
               *(cosTheta*rik*yij - rij*yik)*(cos(pi*rij/rc) + 1) + 0.25*pi*rij*rik*yij\
               *(cosTheta*lamb + 1)**(zeta + 1)*sin(pi*rij/rc))*exp(-eta*(rij**2 + rik**2 + rjk**2))\
               /(rc*rij**2*rik*(cosTheta*lamb + 1))
    dG4_dzij = -2**(-zeta)*(cos(pi*rik/rc) + 1)*(cos(pi*rjk/rc) + 1)\
               *(0.5*eta*rc*rij**2*rik*zij*(cosTheta*lamb + 1)**(zeta + 1)\
               *(cos(pi*rij/rc) + 1) + 0.25*lamb*rc*zeta*(cosTheta*lamb + 1)**zeta\
               *(cosTheta*rik*zij - rij*zik)*(cos(pi*rij/rc) + 1) + 0.25*pi*rij*rik*zij\
               *(cosTheta*lamb + 1)**(zeta + 1)*sin(pi*rij/rc))*exp(-eta*(rij**2 + rik**2 + rjk**2))\
               /(rc*rij**2*rik*(cosTheta*lamb + 1))
    """dG4_dxik = -2**(-zeta)*(cos(pi*rij/rc) + 1)*(cos(pi*rjk/rc) + 1)\
               *(0.5*eta*rc*rij*rik**2*xik*(cosTheta*lamb + 1)**(zeta + 1)\
               *(cos(pi*rik/rc) + 1) + 0.25*lamb*rc*zeta*(cosTheta*lamb + 1)**zeta\
               *(cosTheta*rij*xik - rik*xij)*(cos(pi*rik/rc) + 1) + 0.25*pi*rij*rik*xik\
               *(cosTheta*lamb + 1)**(zeta + 1)*sin(pi*rik/rc))*exp(-eta*(rij**2 + rik**2 + rjk**2))\
               /(rc*rij*rik**2*(cosTheta*lamb + 1))
    dG4_dyik = -2**(-zeta)*(cos(pi*rij/rc) + 1)*(cos(pi*rjk/rc) + 1)\
               *(0.5*eta*rc*rij*rik**2*yik*(cosTheta*lamb + 1)**(zeta + 1)\
               *(cos(pi*rik/rc) + 1) + 0.25*lamb*rc*zeta*(cosTheta*lamb + 1)**zeta\
               *(cosTheta*rij*yik - rik*yij)*(cos(pi*rik/rc) + 1) + 0.25*pi*rij*rik*yik\
               *(cosTheta*lamb + 1)**(zeta + 1)*sin(pi*rik/rc))*exp(-eta*(rij**2 + rik**2 + rjk**2))\
               /(rc*rij*rik**2*(cosTheta*lamb + 1))
    dG4_dzik = -2**(-zeta)*(cos(pi*rij/rc) + 1)*(cos(pi*rjk/rc) + 1)\
               *(0.5*eta*rc*rij*rik**2*zik*(cosTheta*lamb + 1)**(zeta + 1)\
               *(cos(pi*rik/rc) + 1) + 0.25*lamb*rc*zeta*(cosTheta*lamb + 1)**zeta\
               *(cosTheta*rij*zik - rik*zij)*(cos(pi*rik/rc) + 1) + 0.25*pi*rij*rik*zik\
               *(cosTheta*lamb + 1)**(zeta + 1)*sin(pi*rik/rc))*exp(-eta*(rij**2 + rik**2 + rjk**2))\
               /(rc*rij*rik**2*(cosTheta*lamb + 1))"""
    # tot_force = -1.0*np.array([dG4_dxij+dG4_dxik, dG4_dyij+dG4_dyik, dG4_dzij+dG4_dzik]) # Minus sign comes from F = -d/dx V
    tot_force = np.array([dG4_dxij, dG4_dyij, dG4_dzij]) # Minus sign comes from F = -d/dx V
    return tot_force

def dG2dXYZ(xyz, eta, rc, rs):
    """
    Derivative of symmetry function G2 w.r.t xij, yij, zij.
    """
    # Vectors of x,y,z-coordinates for neighbours
    xij = xyz[0,0]
    yij = xyz[0,1]
    zij = xyz[0,2]
    rij   = np.linalg.norm((xij, yij, zij)) # Neighbour distance
    term = (eta*rc*(rij - rs)*(cos(pi*rij/rc) + 1.0) + 0.5*pi*sin(pi*rij/rc)) \
           * exp(-eta*(rij - rs)**2)/(rc*rij)
    Fx = -1.0*(xij * term)
    Fy = -1.0*(yij * term)
    Fz = -1.0*(zij * term)
    # print "X Y Z, FX FY FZ", xij, yij, zij, Fx, Fy, Fz
    # raw_input("NEXT")
    return np.array([Fx, Fy, Fz]) # Minus sign comes from F = -d/dx V

# def dG2dXYZ(xyz, eta, rc, rs):
#     """
#     Derivative of symmetry function G2 w.r.t xij, yij, zij.
#     """
#     # Vectors of x,y,z-coordinates for neighbours
#     xij = xyz[:,0]
#     yij = xyz[:,1]
#     zij = xyz[:,2]
#     r   = np.linalg.norm(xyz, axis=1) # Neighbour distances
#     term = (eta*rc*(r - rs)*(np.cos(pi*r/rc) + 1.0) + 0.5*pi*np.sin(pi*r/rc)) \
#            * np.exp(-eta*(r - rs)**2)/(rc*r)
#     Fx = -1.0*np.sum(-xij * term) # Minus sign comes from F = -d/dx V
#     Fy = -1.0*np.sum(-yij * term)
#     Fz = -1.0*np.sum(-zij * term)
#     return np.array([Fx, Fy, Fz])


def cos_theta_check(cos_theta):
    if cos_theta > 1:
        print "CosTheta value fixed to 1:", cos_theta
        cos_theta = 1.0
    elif cos_theta < -1:
        print "CosTheta value fixed to -1:", cos_theta
        cos_theta = -1.0
    return cos_theta
