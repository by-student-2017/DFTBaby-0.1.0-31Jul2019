"""
Try to decompose the potential which depends on all atom positions V(r1,r2,...,rN)
into a sum of pairwise interactions  V = sum_(i<j) V^(ij)(|ri-rj|)  that depend only on the relative
distances between pairs of atoms. Then V^(ij) is the repulsive potential we are looking for.
Of course such a decomposition is only approximately correct, so we need to find the best one. 
"""
from numpy import array, zeros, reshape, ones
from numpy.linalg import norm, solve, det, lstsq
import numpy as np
from scipy.optimize import fmin

from DFTB import XYZ
from DFTB.AtomicData import hartree_to_eV, bohr_to_angs, atom_names
from DFTB import utils
#from DFTB.RepulsivePotential import lapack

def symmat_index(i,j, N):
    if i == j:
        return None
    if i > j:
        tmp = i
        i = j
        j = tmp
    return sum([N-2-a for a in range(0,i)])+(j-1)

#
class SymmatIndeces:
    def __init__(self, N):
        self.N = N
        self.pair2index = {}
        self.index2pair = {}
        for a in range(0, N):
            for b in range(0, N):
                if a == b:
                    self.pair2index[(a,b)] = None
                else:
                    i = symmat_index(a,b,self.N)
                    self.pair2index[(a,b)] = i
                    if a < b:
                        self.index2pair[i] = (a,b)
    def p2i(self,a,b):
        return self.pair2index[(a,b)]
    def i2p(self,i):
        return self.index2pair[i]
#

def pairwise_decomposition_numerical(atomlist, forcelist):
    Nat = len(atomlist)
    nx = zeros((Nat,Nat))
    ny = zeros((Nat,Nat))
    nz = zeros((Nat,Nat))
    Fx = zeros(Nat)
    Fy = zeros(Nat)
    Fz = zeros(Nat)
    for l,(Zl,posl) in enumerate(atomlist):
        # unit vectors along bond lengths
        for i,(Zi,posi) in enumerate(atomlist):
            if i == l:
                continue
            r_li = array(posl) - array(posi)
            n_li = r_li/norm(r_li)
            nx[l,i] = n_li[0]
            ny[l,i] = n_li[1]
            nz[l,i] = n_li[2]
        # cartesian forces
        Fx[l] = forcelist[l][1][0]
        Fy[l] = forcelist[l][1][1]
        Fz[l] = forcelist[l][1][2]
    def to_minimize(pair_force):
        dVpair = zeros((Nat,Nat))
        for i in range(0,Nat):
            for j in range(i,Nat):
                if i == j:
                    # dVpair[i,i] = 0
                    continue
                # V^(1)_ij(r) = d/dr V_ij(r) = - F_ij(r)
                dVpair[i,j] = -pair_force[symmat_index(i,j,Nat)]
                dVpair[j,i] = dVpair[i,j]
        Sx = 0.0
        Sy = 0.0
        Sz = 0.0
        for l in range(0, Nat):
            dSx = Fx[l]
            dSy = Fy[l]
            dSz = Fz[l]
            for i in range(0, Nat):
                if i == l:
                    continue
                dSx += dVpair[l,i] * nx[l,i]
                dSy += dVpair[l,i] * ny[l,i]
                dSz += dVpair[l,i] * nz[l,i]
            Sx += abs(dSx)
            Sy += abs(dSy)
            Sz += abs(dSz)
        S = Sx + Sy + Sz
#        print "S = %s" % S
        return S
    pair_force_guess = ones(int((pow(Nat,2)-Nat)/2))
    pair_force_opt, Sopt,dummy1,dummy2,dummy3 = fmin(to_minimize, pair_force_guess, full_output=True)
    print "pair_force_opt = %s" % pair_force_opt
    # error per atom
    error = Sopt/float(Nat)
    print "ERROR PER ATOM = %s" % error
#    assert Sopt < 1.0e-3
    
    bond_lengths = {}
    pair_forces = {}
    for i,(Zi,posi) in enumerate(atomlist):
        for j,(Zj,posj) in enumerate(atomlist):
            if i < j:
                bond_lengths[(i,j)] = norm(array(posi)-array(posj))
                pair_forces[(i,j)] = pair_force_opt[symmat_index(i,j,Nat)]

    import string
    print "             Pairwise Forces:"
    print "             ================"
    print "        atom i - atom j       bond length [bohr]   bond length [AA]      force F_ij [hartree/bohr]             force F_ij [eV/AA]"
    for i,j in pair_forces:
            print "           %s-  %s   %s %s %s %s" % \
                (string.ljust("%s%d" % (atom_names[atomlist[i][0]-1], i+1), 4), \
                 string.ljust("%s%d" % (atom_names[atomlist[j][0]-1], j+1), 4), \
                 string.rjust("%.7f bohr" % bond_lengths[(i,j)], 20), \
                 string.rjust("%.7f AA" % (bond_lengths[(i,j)]*bohr_to_angs), 20), \
                 string.rjust("%.7f hartree/bohr" % pair_forces[(i,j)], 30), \
                 string.rjust("%.7f eV/AA" % (pair_forces[(i,j)]*hartree_to_eV/bohr_to_angs), 30))
    return bond_lengths, pair_forces, error

"""
def pairwise_decomposition_analytical(atomlist, forcelist):
    Nat = len(atomlist)
    nx = zeros((Nat,Nat))
    ny = zeros((Nat,Nat))
    nz = zeros((Nat,Nat))
    Fx = zeros(Nat)
    Fy = zeros(Nat)
    Fz = zeros(Nat)
    for l,(Zl,posl) in enumerate(atomlist):
        # unit vectors along bond lengths
        for i,(Zi,posi) in enumerate(atomlist):
            if i == l:
                continue
            r_li = array(posl) - array(posi)
            n_li = r_li/norm(r_li)
            nx[l,i] = n_li[0]
            ny[l,i] = n_li[1]
            nz[l,i] = n_li[2]
        # cartesian forces
        Fx[l] = forcelist[l][1][0]
        Fy[l] = forcelist[l][1][1]
        Fz[l] = forcelist[l][1][2]
    dim = (pow(Nat,2) - Nat)/2
    M = zeros( (dim,dim) )
    for alpha in range(0, len(atomlist)):
        for beta in range(alpha+1, len(atomlist)):
            for l in range(0, len(atomlist)):
                if l == beta:
                    continue
                for i in range(l+1, len(atomlist)):
                    M[symmat_index(alpha,beta,Nat), symmat_index(l,i, Nat)] \
                        = nx[alpha,beta] * nx[l,i] \
                        + ny[alpha,beta] * ny[l,i] \
                        + nz[alpha,beta] * nz[l,i]
    b = zeros(dim)
    for alpha in range(0, len(atomlist)):
        for beta in range(alpha+1, len(atomlist)):
            for l in range(0, len(atomlist)):
                if l == beta:
                    continue
                b[symmat_index(alpha,beta,Nat)] -= \
                    + Fx[l]*nx[alpha,beta] \
                    + Fy[l]*ny[alpha,beta] \
                    + Fz[l]*nz[alpha,beta]
    print "M = \n"
    print M
    print "b = \n"
    print b

    pair_force_opt = solve(M,b)
    print "x = \n"
    print pair_force_opt

    bond_lengths = {}
    pair_forces = {}
    for i,(Zi,posi) in enumerate(atomlist):
        for j,(Zj,posj) in enumerate(atomlist):
            if i < j:
                bond_lengths[(i,j)] = norm(array(posi)-array(posj))
                pair_forces[(i,j)] = pair_force_opt[symmat_index(i,j,Nat)]

    import string
    print "             Pairwise Forces:"
    print "             ================"
    print "        atom i - atom j       bond length [bohr]   bond length [AA]      force F_ij [hartree/bohr]             force F_ij [eV/AA]"
    for i,j in pair_forces:
            print "           %s-  %s   %s %s %s %s" % \
                (string.ljust("%s%d" % (atom_names[atomlist[i][0]-1], i+1), 4), \
                 string.ljust("%s%d" % (atom_names[atomlist[j][0]-1], j+1), 4), \
                 string.rjust("%.7f bohr" % bond_lengths[(i,j)], 20), \
                 string.rjust("%.7f AA" % (bond_lengths[(i,j)]*bohr_to_angs), 20), \
                 string.rjust("%.7f hartree/bohr" % pair_forces[(i,j)], 30), \
                 string.rjust("%.7f eV/AA" % (pair_forces[(i,j)]*hartree_to_eV/bohr_to_angs), 30))
    return bond_lengths, pair_forces
"""

def pairwise_decomposition_analytical(atomlist, forcelist):
    Nat = len(atomlist)
    active_atoms = [i for i in range(0, Nat)]
    print "Nat = %s" % Nat
    nx = zeros((Nat,Nat))
    ny = zeros((Nat,Nat))
    nz = zeros((Nat,Nat))
    Fx = zeros(Nat)
    Fy = zeros(Nat)
    Fz = zeros(Nat)
    distances = zeros((Nat,Nat))
    for l,active_l in enumerate(active_atoms):
        (Zl,posl) = atomlist[active_l]
        # unit vectors along bond lengths
        for i,active_i in enumerate(active_atoms):
            (Zi,posi) = atomlist[active_i]
            if i == l:
                continue
            r_li = array(posl) - array(posi)
            nrm = norm(r_li)
            distances[l,i] = nrm
            n_li = r_li/nrm
            nx[l,i] = n_li[0]
            ny[l,i] = n_li[1]
            nz[l,i] = n_li[2]
        # cartesian forces
        Fx[l] = forcelist[l][1][0]
        Fy[l] = forcelist[l][1][1]
        Fz[l] = forcelist[l][1][2]

    dim = (pow(Nat,2) - Nat)/2
    M = zeros( (dim,dim) )
    Msym = [["" for i in range(0, dim)] for j in range(0, dim)]
    visited = zeros( (dim,dim), dtype=int)

    R = zeros((dim,dim))
    I = SymmatIndeces(Nat)
    for i in range(0, dim):
        ai,bi = I.i2p(i)
        assert ai < bi
        M[i,i] = 2.0
        Msym[i][i] = "     2     "
        for j in range(i+1, dim):
            aj,bj = I.i2p(j)
            assert aj < bj
            ridrj = nx[ai,bi] * nx[aj,bj] \
                    + ny[ai,bi] * ny[aj,bj] \
                    + nz[ai,bi] * nz[aj,bj]
            if ai != aj and bi != aj and bi == bj:
                M[i,j] += ridrj
                Msym[i][j] += " r_%d%d*r_%d%d " % (ai+1,bi+1,aj+1,bj+1)
                visited[i,j] += 1
                visited[j,i] += 1
            if ai == aj and ai != bj and bi != bj:
                M[i,j] += ridrj
                Msym[i][j] += " r_%d%d*r_%d%d " % (ai+1,bi+1,aj+1,bj+1)
                visited[i,j] += 1
                visited[j,i] += 1
            if ai == bj and aj != bi and ai != aj:
                M[i,j] -= ridrj
                Msym[i][j] += "-r_%d%d*r_%d%d " % (ai+1,bi+1,aj+1,bj+1)
                visited[i,j] += 1
                visited[j,i] += 1
            if bi == aj and ai != bj and ai != aj:
                M[i,j] -= ridrj
                Msym[i][j] += "-r_%d%d*r_%d%d " % (ai+1,bi+1,aj+1,bj+1)
                visited[i,j] += 1
                visited[j,i] += 1

            M[j,i] = M[i,j]
            Msym[j][i] = Msym[i][j]
    print "Msym"
    for i in range(0, dim):
        for j in range(0, dim):
            if Msym[i][j] == "":
                print "     0     ",
            else:
                print Msym[i][j],
        print ""

    """
    for alpha in range(0, Nat):
        for beta in range(alpha+1, Nat):
            M[symmat_index(alpha,beta,Nat),symmat_index(alpha,beta,Nat)] = 2.0
            visited[symmat_index(alpha,beta,Nat),symmat_index(alpha,beta,Nat)] = 1
            for i in range(0, Nat):
                if i == alpha or i == beta:
                    continue

#                M[symmat_index(alpha,beta,Nat), symmat_index(i,beta, Nat)] \
#                        += nx[alpha,beta] * nx[i,beta] \
#                        + ny[alpha,beta] * ny[i,beta] \
#                        + nz[alpha,beta] * nz[i,beta]
#                M[symmat_index(alpha,beta,Nat), symmat_index(i,alpha, Nat)] \
#                        -= nx[alpha,beta] * nx[i,alpha] \
#                        + ny[alpha,beta] * ny[i,alpha] \
#                        + nz[alpha,beta] * nz[i,alpha]

                M[symmat_index(alpha,beta,Nat), symmat_index(i,beta, Nat)] \
                        = nx[alpha,beta] * nx[i,beta] \
                        + ny[alpha,beta] * ny[i,beta] \
                        + nz[alpha,beta] * nz[i,beta]
                visited[symmat_index(alpha,beta,Nat), symmat_index(i,beta, Nat)] = 1
                M[symmat_index(alpha,beta,Nat), symmat_index(i,alpha, Nat)] \
                        = -nx[alpha,beta] * nx[i,alpha] \
                        - ny[alpha,beta] * ny[i,alpha] \
                        - nz[alpha,beta] * nz[i,alpha]
                visited[symmat_index(alpha,beta,Nat), symmat_index(i,alpha, Nat)] = 1
    """
    print "Which matrix elements were not set?"
    txt = ""
    for i in range(0, dim):
        for j in range(0, dim):
            if visited[i,j] != 0:
                txt += "%d" % visited[i,j]
            else:
                txt += "0"
        txt += "\n"
    print txt
    #
    """
    b = zeros(dim)
    for alpha in range(0, len(atomlist)):
        for beta in range(alpha+1, len(atomlist)):
            b[symmat_index(alpha,beta,Nat)] -= \
                    + (Fx[beta] - Fx[alpha])*nx[alpha,beta] \
                    + (Fy[beta] - Fy[alpha])*ny[alpha,beta] \
                    + (Fz[beta] - Fz[alpha])*nz[alpha,beta]
    """
    b = zeros(dim)
    for i in range(0, dim):
        ai,bi = I.i2p(i)
        b[i] = -(                       \
            (Fx[bi] - Fx[ai])*nx[ai,bi] \
           +(Fy[bi] - Fy[ai])*ny[ai,bi] \
           +(Fz[bi] - Fz[ai])*nz[ai,bi])

    print "M = \n"
    labels = ["r_%d%d" % I.i2p(i) for i in range(0, dim)]
    print utils.annotated_matrix(M, labels, labels)
    print "det(M) = %s" % det(M)
    print "b = \n"
    print b

    # check that M is symmetric
    for i in range(0,dim):
        for j in range(0, dim):
            assert M[i,j] == M[j,i]
    #

    """
    if det(M) != 0.0:
        print "solve"
        pair_force_opt = lapack.solve(M,b)
    else:
        print "lstsq"
        pair_force_opt = lstsq(M,b)[0]
    """
    pair_force_opt, residuals, rank, singvals = lstsq(M,b)
    print "residuals"
    print residuals
    print "rank"
    print rank
    print "singular values"
    print singvals

    print "x = \n"
    print pair_force_opt

    bond_lengths = {}
    pair_forces = {}
    for i,active_i in enumerate(active_atoms):
        (Zi,posi) = atomlist[active_i]
        for j,active_j in enumerate(active_atoms):
            (Zj,posj) = atomlist[active_j]
            if active_i < active_j:
                bond_lengths[(active_i,active_j)] = norm(array(posi)-array(posj))
                pair_forces[(active_i,active_j)] = pair_force_opt[symmat_index(i,j,Nat)]

    import string
    print "             Pairwise Forces:"
    print "             ================"
    print "        atom i - atom j       bond length [bohr]   bond length [AA]      force F_ij [hartree/bohr]             force F_ij [eV/AA]"
    for active_i,active_j in pair_forces:
        print "           %s-  %s   %s %s %s %s" % \
                (string.ljust("%s%d" % (atom_names[atomlist[active_i][0]-1], active_i+1), 4), \
                 string.ljust("%s%d" % (atom_names[atomlist[active_j][0]-1], active_j+1), 4), \
                 string.rjust("%.7f bohr" % bond_lengths[(active_i,active_j)], 20), \
                 string.rjust("%.7f AA" % (bond_lengths[(active_i,active_j)]*bohr_to_angs), 20), \
                 string.rjust("%.7f hartree/bohr" % pair_forces[(active_i,active_j)], 30), \
                 string.rjust("%.7f eV/AA" % (pair_forces[(active_i,active_j)]*hartree_to_eV/bohr_to_angs), 30))

    def error(pair_force):
        """
        Not every potential can be decomposed into a sum a pair potentials.
        This function determines the error incurred by making the decomposition.
        """
        dVpair = zeros((Nat,Nat))
        for i in range(0,Nat):
            for j in range(i,Nat):
                if i == j:
                    # dVpair[i,i] = 0
                    continue
                # F_ij = - d/dr V_ij(r)
                dVpair[i,j] = -pair_force[symmat_index(i,j,Nat)]
                dVpair[j,i] = dVpair[i,j]
        Sx = 0.0
        Sy = 0.0
        Sz = 0.0
        for l in range(0, Nat):
            dSx = Fx[l]
            dSy = Fy[l]
            dSz = Fz[l]
            for i in range(0, Nat):
                if i == l:
                    continue
                dSx += dVpair[l,i] * nx[l,i]
                dSy += dVpair[l,i] * ny[l,i]
                dSz += dVpair[l,i] * nz[l,i]
            Sx += abs(dSx)
            Sy += abs(dSy)
            Sz += abs(dSz)
        S = Sx + Sy + Sz
        return S
    e = error(pair_force_opt)
    error = e/float(Nat)
    print "ERROR PER ATOM = %s" % error
    return bond_lengths, pair_forces, error


def pairwise_decomposition_3at_analytical(atomlist, forcelist):
    Nat = len(atomlist)
    assert Nat == 3
    nx = zeros((Nat,Nat))
    ny = zeros((Nat,Nat))
    nz = zeros((Nat,Nat))
    Fx = zeros(Nat)
    Fy = zeros(Nat)
    Fz = zeros(Nat)
    for l,(Zl,posl) in enumerate(atomlist):
        # unit vectors along bond lengths
        for i,(Zi,posi) in enumerate(atomlist):
            if i == l:
                continue
            r_li = array(posl) - array(posi)
            n_li = r_li/norm(r_li)
            nx[l,i] = n_li[0]
            ny[l,i] = n_li[1]
            nz[l,i] = n_li[2]
        # cartesian forces
        Fx[l] = forcelist[l][1][0]
        Fy[l] = forcelist[l][1][1]
        Fz[l] = forcelist[l][1][2]
    dim = (pow(Nat,2) - Nat)/2
    assert dim == 3
    M = zeros( (dim,dim) )
    M[0,0] = 2.0
    M[1,1] = 2.0
    M[2,2] = 2.0
    M[0,1]  = nx[0,1]*nx[0,2] + ny[0,1]*ny[0,2] + nz[0,1]*nz[0,2]
    M[0,2] -= nx[0,1]*nx[1,2] + ny[0,1]*ny[1,2] + nz[0,1]*nz[1,2]
    M[1,2] = nx[0,2]*nx[1,2] + ny[0,2]*ny[1,2] + nz[0,2]*nz[1,2]
    M[1,0] = M[0,1]
    M[2,0] = M[0,2]
    M[2,1] = M[1,2]
    b = zeros(dim)
    b[0] -= (Fx[1] - Fx[0])*nx[0,1] + (Fy[1] - Fy[0])*ny[0,1] + (Fz[1] - Fz[0])*nz[0,1]  
    b[1] -= (Fx[2] - Fx[0])*nx[0,2] + (Fy[2] - Fy[0])*ny[0,2] + (Fz[2] - Fz[0])*nz[0,2]  
    b[2] -= (Fx[2] - Fx[1])*nx[1,2] + (Fy[2] - Fy[1])*ny[1,2] + (Fz[2] - Fz[1])*nz[1,2]  

    print "nx = %s" % nx
    print "ny = %s" % ny
    print "nz = %s" % nz
    print "M = \n"
    print M
    print "b = \n"
    print b

    pair_force_opt = solve(M,b)
    print "x = \n"
    print pair_force_opt
    ####
    def error(pair_force):
        """
        Not every potential can be decomposed into a sum a pair potentials.
        This function determines the error incurred by making the decomposition.
        """
        dVpair = zeros((Nat,Nat))
        for i in range(0,Nat):
            for j in range(i,Nat):
                if i == j:
                    # dVpair[i,i] = 0
                    continue
                dVpair[i,j] = -pair_force[symmat_index(i,j,Nat)]
                dVpair[j,i] = dVpair[i,j]
        Sx = 0.0
        Sy = 0.0
        Sz = 0.0
        for l in range(0, Nat):
            dSx = Fx[l]
            dSy = Fy[l]
            dSz = Fz[l]
            for i in range(0, Nat):
                if i == l:
                    continue
                dSx += dVpair[l,i] * nx[l,i]
                dSy += dVpair[l,i] * ny[l,i]
                dSz += dVpair[l,i] * nz[l,i]
            Sx += abs(dSx)
            Sy += abs(dSy)
            Sz += abs(dSz)
        S = Sx + Sy + Sz
        return S
    e = error(pair_force_opt)
    error = e/float(Nat)
    print "ERROR PER ATOM = %s" % error
    #####
    bond_lengths = {}
    pair_forces = {}
    for i,(Zi,posi) in enumerate(atomlist):
        for j,(Zj,posj) in enumerate(atomlist):
            if i < j:
                bond_lengths[(i,j)] = norm(array(posi)-array(posj))
                pair_forces[(i,j)] = pair_force_opt[symmat_index(i,j,Nat)]

    import string
    print "             Pairwise Forces:"
    print "             ================"
    print "        atom i - atom j       bond length [bohr]   bond length [AA]      force F_ij [hartree/bohr]             force F_ij [eV/AA]"
    for i,j in pair_forces:
            print "           %s-  %s   %s %s %s %s" % \
                (string.ljust("%s%d" % (atom_names[atomlist[i][0]-1], i+1), 4), \
                 string.ljust("%s%d" % (atom_names[atomlist[j][0]-1], j+1), 4), \
                 string.rjust("%.7f bohr" % bond_lengths[(i,j)], 20), \
                 string.rjust("%.7f AA" % (bond_lengths[(i,j)]*bohr_to_angs), 20), \
                 string.rjust("%.7f hartree/bohr" % pair_forces[(i,j)], 30), \
                 string.rjust("%.7f eV/AA" % (pair_forces[(i,j)]*hartree_to_eV/bohr_to_angs), 30))
    return bond_lengths, pair_forces, error


if __name__ == "__main__":
    import Gaussian
    import sys

    log_file = sys.argv[1]
    atoms = Gaussian.read_geometry(log_file)
    atomlist = []
    for k in atoms.keys():
        atomlist.append(atoms[k])
    forces = Gaussian.read_forces(log_file)
    forcelist = []
    for k in atoms.keys():
        forcelist.append(forces[k])

#    pairwise_decomposition_3at_analytical(atomlist, forcelist)
    pairwise_decomposition_analytical(atomlist, forcelist)
    """
    N = 4
    for i in range(0, N):
        for j in range(0, N):
            print "i=%s  j=%s   index=%s" % (i,j, symmat_index(i,j,N))
    """
