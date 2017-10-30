from symmetry_functions import cutoff_cos
from create_train_data import generate_symmfunc_input_Si_Behler
from math import * # Much quicker for _single_ floats than numpy equvivalent
import numpy as np
import sys

def create_neighbour_list(all_atoms, selfindex, return_self=True):
    """
    Puts atom with index 'selfindex' in origo (0,0,0).

    Method:
    Subtracts x0,y0,z0 if selfindex is 0 from the rest of the atoms
    castersian coordinates.

    Below is illustration if selfindex = 1:
    BEFORE          -->  AFTER
    [[ 0,  1,  2],  -->  [[-3, -3, -1],
     [ 3,  4,  5],  -->   [ 0,  0,  0],
     [ 6,  7,  8]]  -->   [ 3,  3,  3]]
    """
    xyz  = np.copy(all_atoms) # Dont change all_atoms array!
    xyz -= xyz[selfindex,:]
    # Remove atoms outside SW cutoff! ...by moving them far away
    for i, pos in enumerate(xyz):
        if i == selfindex:
            continue
        else:
            if np.linalg.norm(pos) > 3.77118:
                xyz[i] = np.array([100.,100.,100.]) # Should be far enough away ;)
    if not return_self:
        # Delete row including self atom
        xyz = np.delete(xyz, (selfindex), axis=0)
    return xyz

def force_calculation(dNNdG_matrix, all_atoms):
    """
    Inputs: dNNdg, all_atoms
    - dNNdG_matrix is the matrix composed of vectors of same length as the symmetry vectors.
    It contains the derivative of the neural network with respect to its inputs, i.e.
    the symmetry vector of atom i.
    - all_atoms is a matrix (numpy array) where atom i is:
    x,y,z = all_atoms[i,:]

    This function further differentiates the symmetry
    vector with respect to actual atomic coordinates in order to find
    analytic forces.

    Returns:
    Numpy array with total forces on all particles in all xyz-directions.
    """
    def i2xyz(i):
        """ For easy reading of error checks """
        if i == 0:
            return "x"
        elif i == 1:
            return "y"
        else:
            return "z"
    tot_atoms   = all_atoms.shape[0]
    all_indices = range(tot_atoms)

    # Loop over all atoms
    total_forces = np.zeros((tot_atoms, 3)) # Fx, Fy, Fz on all atoms
    for selfindex, atom in enumerate(all_atoms):
        selfneighborpositions = create_neighbour_list(all_atoms, selfindex)
        selfneighborindices   = [ind for ind in all_indices if ind != selfindex]
        self_dNNdG            = dNNdG_matrix[selfindex,:]

        # Loop over directions X,Y,Z: (0 = x, 1 = y, 2 = z)
        for i in range(3):
            # Calculating derivative of fingerprints of self atom w.r.t. coordinates of itself.
            sym_f_der = symmetry_func_derivative(selfindex, selfneighborindices, selfneighborpositions, selfindex, i)
            total_forces[selfindex, i] += -np.dot(sym_f_der, self_dNNdG) # Minus sign since F = -d/dx V
            # if selfindex == 0:
            #     print i2xyz(i),-np.dot(sym_f_der, self_dNNdG)

            # Calculating derivative of fingerprints of neighbor atom w.r.t. coordinates of self atom.
            for neigh_index in selfneighborindices:
                # for calculating forces, summation runs over neighbor atoms
                neighborpositions = create_neighbour_list(all_atoms, neigh_index)
                nneighborindices  = [ind for ind in all_indices if ind != neigh_index]
                neigh_dNNdG       = dNNdG_matrix[neigh_index,:]

                # for calculating derivatives of fingerprints, summation runs over neighboring atoms
                sym_f_der = symmetry_func_derivative(neigh_index, nneighborindices, neighborpositions, selfindex, i)
                total_forces[selfindex, i] += -np.dot(sym_f_der, neigh_dNNdG)
                if selfindex == 0:
                    print i2xyz(i),-np.dot(sym_f_der, neigh_dNNdG)
        # print "F sum: X,Y,Z:", total_forces[selfindex]
        # raw_input("one atom done!")
    return total_forces


def symmetry_func_derivative(index, neighborindices, neighborpositions, m, l):
    """
    Returns the value of the derivative of G for atom with index 'index',
    with respect to coordinate x_l of atom index m.

    Parameters
    ----------
    index : int
        Index of the center atom.
    neighborindices : list of int
        List of neighbors' indices.
    neighborpositions : list of list of float
        List of Cartesian atomic positions.
    m : int
        Index of the pair atom.
    l : int
        Direction of the derivative; is an integer 0, 1, 2
    """
    G_funcs, nmbr_G = generate_symmfunc_input_Si_Behler()
    ddx_symm_vec    = np.zeros(nmbr_G)

    # Loop over all values of the symmetry vector
    for i in range(nmbr_G):
        which_symm = G_funcs[i][0]
        if which_symm == 2:
            _, eta, rc, rs  = G_funcs[i]
            eta, rc, rs     = float(eta), float(rc), float(rs)
            ddx_symm_vec[i] = calculate_ddx_G2(neighborindices, neighborpositions, eta, rc, rs, index, m, l)
        elif which_symm == 4:
            _, eta, rc, zeta, lamb = G_funcs[i]
            eta, rc, zeta, lamb    = float(eta), float(rc), float(zeta), float(lamb)
            ddx_symm_vec[i]        = calculate_ddx_G4(neighborindices, neighborpositions, lamb, zeta, eta, rc, index, m, l)
            # ddx_symm_vec[i]   = calculate_ddx_G4(neighborindices, neighborpositions, lamb, zeta, eta, rc, index, m, l)
            # raw_input("skip ahead!")
        else:
            print "Only use symmetry functions G2 or G4! Exiting!"
            sys.exit(0)
    return ddx_symm_vec


def calculate_ddx_G2(neighborindices, neighborpositions, eta, Rc, Rs, i, m, l):
    """
    Calculates coordinate derivative of G2 symmetry function for atom at
    index i and position Ri with respect to coordinate x_{l} of atom index
    m.

    Parameters
    ---------
    neighborindices : list of int
        List of int of neighboring atoms.
    neighborsymbols : list of str
        List of symbols of neighboring atoms.
    neighborpositions : list of list of float
        List of Cartesian atomic positions of neighboring atoms.
    """
    Ri    = neighborpositions[i] # This also contains xyz of the atom we are inspecting
    value = 0.0                  # Single value of the symmetry vector derivative
    cutoff_func     = cutoff_cos
    ddx_cutoff_func = ddx_cutoff_cos
    num_neighbors   = len(neighborindices)
    if np.sum(np.abs(Ri)) > 1E-15: # Make sure we have the right atom in center
        print "You may have wrong atom in center, pos:", Ri, ". Exiting"
        sys.exit(0)
    for j in neighborindices:    # Loop over everyone EXCEPT atom with position Ri
        Rj       = neighborpositions[j,:] # x,y,z of atom j
        dRijdRml = dRij_dRml(i, j, Ri, Rj, m, l)
        if dRijdRml != 0:
            Rij    = np.linalg.norm(Rj - Ri) # Ri = 0,0,0, but still written for clarity
            term   = -2.0*eta*(Rij-Rs) * cutoff_func(Rij, Rc) + ddx_cutoff_func(Rij, Rc)
            value += exp(-eta*(Rij-Rs)**2) * term * dRijdRml
    return value

def calculate_ddx_G4(neighborindices, neighborpositions, lamb, zeta, eta, rc, index, m, l):
    """
    Derivative of symmetry function G4 w.r.t xij, yij, zij.
    """
    # xyz, eta, rc, zeta, lamb
    xyz = np.array([neighborpositions[ind] for ind in neighborindices])
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
    dG4_dxik = -2**(-zeta)*(cos(pi*rij/rc) + 1)*(cos(pi*rjk/rc) + 1)\
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
               /(rc*rij*rik**2*(cosTheta*lamb + 1))
    tot_force = -1.0*np.array([dG4_dxij+dG4_dxik, dG4_dyij+dG4_dyik, dG4_dzij+dG4_dzik])[l] # Minus sign comes from F = -d/dx V
    return tot_force

def ddx_cutoff_cos(Rij, Rc):
    """
    Derivative of the cosine cutoff function.
    """
    if Rij > Rc:# or Rij > 3.77118:
        return 0.
    else:
        return (-0.5 * pi / Rc) * sin(pi * Rij / Rc)



if __name__ == '__main__':
    """
    Checking that neigh. list correctly removes atoms outside cutoff at 3.77118
    """
    test = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]])
    for selfindex in range(4):
        print selfindex
        print create_neighbour_list(test, selfindex), "\n"
