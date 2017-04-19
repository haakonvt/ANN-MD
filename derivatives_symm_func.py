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
    # TODO: Implement the cutoff
    xyz  = np.copy(all_atoms) # Dont change all_atoms array!
    xyz -= xyz[selfindex,:]
    if not return_self:
        # Delete row including self atom
        xyz = np.delete(xyz, (selfindex), axis=0)
    return xyz

def force_calculation(dNNdG_matrix, all_atoms):
    """
    Inputs: network, dNNdg, all_atoms
    - network is an instance of class 'neural network'.
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

            # Calculating derivative of fingerprints of neighbor atom w.r.t. coordinates of self atom.
            for neigh_index in selfneighborindices:
                # for calculating forces, summation runs over neighbor atoms
                neighborpositions = create_neighbour_list(all_atoms, neigh_index)
                nneighborindices  = [ind for ind in all_indices if ind != neigh_index]
                neigh_dNNdG       = dNNdG_matrix[neigh_index,:]

                # for calculating derivatives of fingerprints, summation runs over neighboring atoms
                sym_f_der = symmetry_func_derivative(neigh_index, nneighborindices, neighborpositions, selfindex, i)
                total_forces[selfindex, i] += -np.dot(sym_f_der, neigh_dNNdG)
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

    Returns
    -------
    fingerprintprime : list of float
        The value of the derivative of the fingerprints for atom with index
        and symbol with respect to coordinate x_{l} of atom index m.
    """
    G_funcs, nmbr_G = generate_symmfunc_input_Si_Behler()
    ddx_symm_vec    = np.zeros(nmbr_G)

    # Loop over all values of the symmetry vector
    for i in range(nmbr_G):
        which_symm = G_funcs[i][0]
        if which_symm == 2:
            _, eta, rc, rs  = G_funcs[i]
            eta, rc, rs     = float(eta), float(rc), float(rs)
            ddx_symm_vec[i] = calculate_ddx_G2(neighborindices,
                                              neighborpositions,
                                              eta, rc, rs, index, m, l)
        elif which_symm == 4:
            _, eta, rc, zeta, lamb = G_funcs[i]
            eta, rc, zeta, lamb    = float(eta), float(rc), float(zeta), float(lamb)
            ddx_symm_vec[i]        = calculate_ddx_G4(neighborindices,
                                                      neighborpositions,
                                                      lamb, zeta, eta, rc, index, m, l)
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
    G_element : dict
        Symmetry functions of the center atom.
    eta : float
        Parameter of Behler symmetry functions.
    cutoff : dict
        Cutoff function, typically from amp.descriptor.cutoffs. Should be also
        formatted as a dictionary by todict method, e.g.
        cutoff=Cosine(6.5).todict()
    i : int
        Index of the center atom.
    Ri : list
        Position of the center atom. Should be fed as a list of three floats.
    m : int
        Index of the atom force is acting on.
    l : int
        Direction of force.

    Returns
    -------
    value : float
        Coordinate derivative of G2 symmetry function for atom at index a and
        position Ri with respect to coordinate x_{l} of atom index m.
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
        """print "AMP value:", value
        # Vectors of x,y,z-coordinates for neighbours
        xij = neighborpositions[j,l]
        # yij = neighborpositions[j,1]
        # zij = neighborpositions[j,2]
        rij    = np.linalg.norm(Rj - Ri) # Ri = 0,0,0, but still written for clarity
        # rij   = np.linalg.norm((xij, yij, zij)) # Neighbour distance
        term = (eta*Rc*(rij - Rs)*(cos(pi*rij/Rc) + 1.0) + 0.5*pi*sin(pi*rij/Rc)) \
               * exp(-eta*(rij - Rs)**2)/(Rc*rij)
        Fx = -1.0*(xij * term)
        # Fy = -1.0*(yij * term)
        # Fz = -1.0*(zij * term)
        print "OWN Fxyz :", -Fx
        # print "OWN Fy   :", -Fy
        # print "OWN Fz   :", -Fz
        raw_input("\n\n########\n")
        # return np.array([Fx, Fy, Fz]) # Minus sign comes from F = -d/dx V"""
    return value


def calculate_ddx_G4(neighborindices, neighborpositions, lamb, zeta, eta, cutoff, i, m, l):
    """Calculates coordinate derivative of G4 symmetry function for atom at
    index i and position Ri with respect to coordinate x_{l} of atom index m.

    See Eq. 13d of the supplementary information of Khorshidi, Peterson,
    CPC(2016).

    Parameters
    ----------
    neighborindices : list of int
        List of int of neighboring atoms.
    neighborpositions : list of list of float
        List of Cartesian atomic positions of neighboring atoms.
    lamb : float
        Parameter of Behler symmetry functions.
    zeta : float
        Parameter of Behler symmetry functions.
    eta : float
        Parameter of Behler symmetry functions.
    cutoff : dict
        Cutoff function, typically from amp.descriptor.cutoffs. Should be also
        formatted as a dictionary by todict method, e.g.
        cutoff=Cosine(6.5).todict()
    i : int
        Index of the center atom.
    Ri : numpy array
        Position of the center atom.
    m : int
        Index of the atom force is acting on.
    l : int
        Direction of force.

    Returns
    -------
    value : float
        Coordinate derivative of G4 symmetry function for atom at index i and
        position Ri with respect to coordinate x_{l} of atom index m.
    """
    Ri              = neighborpositions[i] # This also contains xyz of the atom we are inspecting
    Rc              = cutoff
    cutoff_func     = cutoff_cos
    ddx_cutoff_func = ddx_cutoff_cos
    value           = 0.0
    # number of neighboring atoms
    # counts = range(len(neighborpositions))
    for jj,j in enumerate(neighborindices): # J = index of atom j, jj = index of j in neigh.indices
        for k in neighborindices:#[jj+1:]:
            if j == k:
                continue
            Rj = neighborpositions[j]
            Rk = neighborpositions[k]
            Rij_vector = neighborpositions[j] - Ri
            Rij = np.linalg.norm(Rij_vector)
            Rik_vector = neighborpositions[k] - Ri
            Rik = np.linalg.norm(Rik_vector)
            Rjk_vector = neighborpositions[k] - neighborpositions[j]
            Rjk = np.linalg.norm(Rjk_vector)
            cos_theta_ijk = np.dot(Rij_vector, Rik_vector) / (Rij * Rik)
            c1 = (1.0 + lamb * cos_theta_ijk)

            fcRij = cutoff_func(Rij, Rc)
            fcRik = cutoff_func(Rik, Rc)
            fcRjk = cutoff_func(Rjk, Rc)
            if zeta == 1:
                term1 = exp(- eta * (Rij ** 2 + Rik ** 2 + Rjk ** 2))
            else:
                term1 = c1 ** (zeta - 1.0) * \
                    exp(- eta * (Rij ** 2 + Rik ** 2 + Rjk ** 2))
            term2 = 0.0
            fcRijfcRikfcRjk = fcRij * fcRik * fcRjk
            dCosthetadRml   = dCos_theta_ijk_dR_ml(i, j, k, Ri, Rj, Rk, m, l)
            if dCosthetadRml != 0:
                term2 += lamb * zeta * dCosthetadRml
            dRijdRml = dRij_dRml(i, j, Ri, Rj, m, l)
            if dRijdRml != 0:
                term2 += -2. * c1 * eta * Rij * dRijdRml
            dRikdRml = dRij_dRml(i, k, Ri, Rk, m, l)
            if dRikdRml != 0:
                term2 += -2. * c1 * eta * Rik * dRikdRml
            dRjkdRml = dRij_dRml(j, k, Rj, Rk, m, l)
            if dRjkdRml != 0:
                term2 += -2. * c1 * eta * Rjk * dRjkdRml
            term3 = fcRijfcRikfcRjk * term2
            term4 = ddx_cutoff_func(Rij, Rc) * dRijdRml * fcRik * fcRjk
            term5 = fcRij * ddx_cutoff_func(Rik, Rc) * dRikdRml * fcRjk
            term6 = fcRij * fcRik * ddx_cutoff_func(Rjk, Rc) * dRjkdRml

            value += term1 * (term3 + c1 * (term4 + term5 + term6))
        value *= 2. ** (1. - zeta)
        return value


def dCos_theta_ijk_dR_ml(i, j, k, Ri, Rj, Rk, m, l):
    """Returns the derivative of Cos(theta_{ijk}) with respect to
    x_{l} of atomic index m.

    See Eq. 14f of the supplementary information of Khorshidi, Peterson,
    CPC(2016).

    Parameters
    ----------
    i : int
        Index of the center atom.
    j : int
        Index of the first atom.
    k : int
        Index of the second atom.
    Ri : float
        Position of the center atom.
    Rj : float
        Position of the first atom.
    Rk : float
        Position of the second atom.
    m : int
        Index of the atom force is acting on.
    l : int
        Direction of force.

    Returns
    -------
    dCos_theta_ijk_dR_ml : float
        Derivative of Cos(theta_{ijk}) with respect to x_{l} of atomic index m.
    """
    Rij_vector = Rj - Ri
    Rij = np.linalg.norm(Rij_vector)
    Rik_vector = Rk - Ri
    Rik = np.linalg.norm(Rik_vector)
    dCos_theta_ijk_dR_ml = 0

    dRijdRml = dRij_dRml_vector(i, j, m, l)
    if np.array(dRijdRml).any() != 0:
        dCos_theta_ijk_dR_ml += np.dot(dRijdRml, Rik_vector) / (Rij * Rik)

    dRikdRml = dRij_dRml_vector(i, k, m, l)
    if np.array(dRikdRml).any() != 0:
        dCos_theta_ijk_dR_ml += np.dot(Rij_vector, dRikdRml) / (Rij * Rik)

    dRijdRml = dRij_dRml(i, j, Ri, Rj, m, l)
    if dRijdRml != 0:
        dCos_theta_ijk_dR_ml += - np.dot(Rij_vector, Rik_vector) * dRijdRml / ((Rij ** 2.) * Rik)

    dRikdRml = dRij_dRml(i, k, Ri, Rk, m, l)
    if dRikdRml != 0:
        dCos_theta_ijk_dR_ml += - np.dot(Rij_vector, Rik_vector) * dRikdRml / (Rij * (Rik ** 2.))
    return dCos_theta_ijk_dR_ml


def dRij_dRml_vector(i, j, m, l):
    """Returns the derivative of the position vector R_{ij} with respect to
    x_{l} of itomic index m.

    See Eq. 14d of the supplementary information of Khorshidi, Peterson,
    CPC(2016).

    Parameters
    ----------
    i : int
        Index of the first atom.
    j : int
        Index of the second atom.
    m : int
        Index of the atom force is acting on.
    l : int
        Direction of force.

    Returns
    -------
    list of float
        The derivative of the position vector R_{ij} with respect to x_{l} of
        atomic index m.
    """
    if (m != i) and (m != j):
        return [0, 0, 0]
    else:
        dRij_dRml_vector = [None, None, None]
        c1 = kronecker_delta(m, j) - kronecker_delta(m, i)
        dRij_dRml_vector[0] = c1 * kronecker_delta(0, l)
        dRij_dRml_vector[1] = c1 * kronecker_delta(1, l)
        dRij_dRml_vector[2] = c1 * kronecker_delta(2, l)
        return dRij_dRml_vector


def kronecker_delta(i, j):
    """Kronecker delta function.

    Parameters
    ----------
    i : int
        First index of Kronecker delta.
    j : int
        Second index of Kronecker delta.

    Returns
    -------
    int
        The value of the Kronecker delta.
    """
    if i == j:
        return 1
    else:
        return 0


def ddx_cutoff_cos(Rij, Rc):
    """
    Derivative of the cosine cutoff function.

    Parameters
    ----------
    Rij : float
        Distance between pair atoms.

    Returns
    -------
    float
        The vaule of derivative of the cutoff function.
    """
    if Rij > Rc:
        return 0.
    else:
        return (-0.5 * pi / Rc) * sin(pi * Rij / Rc)

def dRij_dRml(i, j, Ri, Rj, m, l):
    """Returns the derivative of the norm of position vector R_{ij} with
    respect to coordinate x_{l} of atomic index m.

    Parameters
    ----------
    i : int
        Index of the first atom.
    j : int
        Index of the second atom.
    Ri : float
        Position of the first atom.
    Rj : float
        Position of the second atom.
    m : int
        Index of the atom force is acting on.
    l : int
        Direction of force.

    Returns
    -------
    dRij_dRml : list of float
        The derivative of the noRi of position vector R_{ij} with respect to
        x_{l} of atomic index m.
    """
    Rij = np.linalg.norm(Rj - Ri)
    if m == i:
        dRij_dRml = -(Rj[l] - Ri[l]) / Rij
    elif m == j:
        dRij_dRml = (Rj[l] - Ri[l]) / Rij
    else:
        dRij_dRml = 0
    return dRij_dRml


def cos_theta_check(cos_theta):
    if cos_theta > 1:
        print "CosTheta value fixed to 1:", cos_theta
        cos_theta = 1.0
    elif cos_theta < -1:
        print "CosTheta value fixed to -1:", cos_theta
        cos_theta = -1.0
    return cos_theta


    ''' Loop over all values of the symmetry vector
    for i in range(nmbr_G):
        which_symm = G_funcs[i][0]
        dEdG_i     = float(-dNNdG[i])
        for cur_atom_index, cur_atom in enumerate(tot_atoms):
            xi, yi, zi = cur_atom


            if which_symm == 2:
                """
                # This is G2
                """
                _, eta, rc, rs = G_funcs[i]
                dG2dXYZ_values = dG2dXYZ(xyz, float(eta), float(rc), float(rs))
                f_vec_G2 += dEdG_i * dG2dXYZ_values
            elif which_symm == 4:
                """
                # This is G4
                """
                _, eta, rc, zeta, lamb = G_funcs[i]
                dG4dXYZ_values = dG4dXYZ(xyz, float(eta), float(rc), float(zeta), float(lamb))
                f_vec_G4 += dEdG_i * dG4dXYZ_values
            else:
                print "Only use symmetry functions G2 or G4! Exiting!"
                sys.exit(0)
        # print "NN tot G2, Fx, Fy, Fz:", f_vec_G2,"\n\n"
        # print "NN Force G4, Fx, Fy, Fz:", f_vec_G4
        # TODO return  f_vec_G4 + f_vec_G2'''



# def dG4dXYZ(xyz, eta, rc, zeta, lamb):
#     """
#     Derivative of symmetry function G4 w.r.t xij, yij, zij.
#     """
#     # Vectors of x,y,z-coordinates for neighbours
#     if len(xyz[:,0]) != 2:
#         # print "Warning: dG4 d x,y,z only works on single unique triplet"
#         return np.zeros(3)
#     r = np.linalg.norm(xyz, axis=1) # Neighbour distances
#     xij = xyz[0,0]; yij = xyz[0,1]; zij = xyz[0,2]
#     xik = xyz[1,0]; yik = xyz[1,1]; zik = xyz[1,2]
#     rij = r[0]    ; rik = r[1]    ; rjk = np.linalg.norm(xyz[0,:]-xyz[1,:])
#     cosTheta = (xij*xik + yij*yik + zij*zik) / (rij*rik)
#
#     # ...and finally the derivaties from sympy: (hold on to something!!)
#     dG4_dxij = -2**(-zeta)*(cos(pi*rik/rc) + 1)*(cos(pi*rjk/rc) + 1)\
#                *(0.5*eta*rc*rij**2*rik*xij*(cosTheta*lamb + 1)**(zeta + 1)\
#                *(cos(pi*rij/rc) + 1) + 0.25*lamb*rc*zeta*(cosTheta*lamb + 1)**zeta\
#                *(cosTheta*rik*xij - rij*xik)*(cos(pi*rij/rc) + 1) + 0.25*pi*rij*rik*xij\
#                *(cosTheta*lamb + 1)**(zeta + 1)*sin(pi*rij/rc))*exp(-eta*(rij**2 + rik**2 + rjk**2))\
#                /(rc*rij**2*rik*(cosTheta*lamb + 1))
#     dG4_dyij = -2**(-zeta)*(cos(pi*rik/rc) + 1)*(cos(pi*rjk/rc) + 1)\
#                *(0.5*eta*rc*rij**2*rik*yij*(cosTheta*lamb + 1)**(zeta + 1)\
#                *(cos(pi*rij/rc) + 1) + 0.25*lamb*rc*zeta*(cosTheta*lamb + 1)**zeta\
#                *(cosTheta*rik*yij - rij*yik)*(cos(pi*rij/rc) + 1) + 0.25*pi*rij*rik*yij\
#                *(cosTheta*lamb + 1)**(zeta + 1)*sin(pi*rij/rc))*exp(-eta*(rij**2 + rik**2 + rjk**2))\
#                /(rc*rij**2*rik*(cosTheta*lamb + 1))
#     dG4_dzij = -2**(-zeta)*(cos(pi*rik/rc) + 1)*(cos(pi*rjk/rc) + 1)\
#                *(0.5*eta*rc*rij**2*rik*zij*(cosTheta*lamb + 1)**(zeta + 1)\
#                *(cos(pi*rij/rc) + 1) + 0.25*lamb*rc*zeta*(cosTheta*lamb + 1)**zeta\
#                *(cosTheta*rik*zij - rij*zik)*(cos(pi*rij/rc) + 1) + 0.25*pi*rij*rik*zij\
#                *(cosTheta*lamb + 1)**(zeta + 1)*sin(pi*rij/rc))*exp(-eta*(rij**2 + rik**2 + rjk**2))\
#                /(rc*rij**2*rik*(cosTheta*lamb + 1))
#     """dG4_dxik = -2**(-zeta)*(cos(pi*rij/rc) + 1)*(cos(pi*rjk/rc) + 1)\
#                *(0.5*eta*rc*rij*rik**2*xik*(cosTheta*lamb + 1)**(zeta + 1)\
#                *(cos(pi*rik/rc) + 1) + 0.25*lamb*rc*zeta*(cosTheta*lamb + 1)**zeta\
#                *(cosTheta*rij*xik - rik*xij)*(cos(pi*rik/rc) + 1) + 0.25*pi*rij*rik*xik\
#                *(cosTheta*lamb + 1)**(zeta + 1)*sin(pi*rik/rc))*exp(-eta*(rij**2 + rik**2 + rjk**2))\
#                /(rc*rij*rik**2*(cosTheta*lamb + 1))
#     dG4_dyik = -2**(-zeta)*(cos(pi*rij/rc) + 1)*(cos(pi*rjk/rc) + 1)\
#                *(0.5*eta*rc*rij*rik**2*yik*(cosTheta*lamb + 1)**(zeta + 1)\
#                *(cos(pi*rik/rc) + 1) + 0.25*lamb*rc*zeta*(cosTheta*lamb + 1)**zeta\
#                *(cosTheta*rij*yik - rik*yij)*(cos(pi*rik/rc) + 1) + 0.25*pi*rij*rik*yik\
#                *(cosTheta*lamb + 1)**(zeta + 1)*sin(pi*rik/rc))*exp(-eta*(rij**2 + rik**2 + rjk**2))\
#                /(rc*rij*rik**2*(cosTheta*lamb + 1))
#     dG4_dzik = -2**(-zeta)*(cos(pi*rij/rc) + 1)*(cos(pi*rjk/rc) + 1)\
#                *(0.5*eta*rc*rij*rik**2*zik*(cosTheta*lamb + 1)**(zeta + 1)\
#                *(cos(pi*rik/rc) + 1) + 0.25*lamb*rc*zeta*(cosTheta*lamb + 1)**zeta\
#                *(cosTheta*rij*zik - rik*zij)*(cos(pi*rik/rc) + 1) + 0.25*pi*rij*rik*zik\
#                *(cosTheta*lamb + 1)**(zeta + 1)*sin(pi*rik/rc))*exp(-eta*(rij**2 + rik**2 + rjk**2))\
#                /(rc*rij*rik**2*(cosTheta*lamb + 1))"""
#     # tot_force = -1.0*np.array([dG4_dxij+dG4_dxik, dG4_dyij+dG4_dyik, dG4_dzij+dG4_dzik]) # Minus sign comes from F = -d/dx V
#     tot_force = np.array([dG4_dxij, dG4_dyij, dG4_dzij]) # Minus sign comes from F = -d/dx V
#     return tot_force

# def dG2dXYZ(xyz, eta, rc, rs):
#     """
#     Derivative of symmetry function G2 w.r.t xij, yij, zij.
#     """
#     # Vectors of x,y,z-coordinates for neighbours
#     xij = xyz[0,0]
#     yij = xyz[0,1]
#     zij = xyz[0,2]
#     rij   = np.linalg.norm((xij, yij, zij)) # Neighbour distance
#     term = (eta*rc*(rij - rs)*(cos(pi*rij/rc) + 1.0) + 0.5*pi*sin(pi*rij/rc)) \
#            * exp(-eta*(rij - rs)**2)/(rc*rij)
#     Fx = -1.0*(xij * term)
#     Fy = -1.0*(yij * term)
#     Fz = -1.0*(zij * term)
#     # print "X Y Z, FX FY FZ", xij, yij, zij, Fx, Fy, Fz
#     # raw_input("NEXT")
#     return np.array([Fx, Fy, Fz]) # Minus sign comes from F = -d/dx V

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
