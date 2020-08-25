#!/usr/bin/env python
"""
The coefficients of the scattering orbital in the basis of atomic scattering orbitals
is determined from the Hulthen/Kohn variational principle
"""

import numpy as np
import numpy.linalg as la
from numpy.polynomial import legendre
from scipy.special import gamma

from DFTB.BasisSets import import_pseudo_atom
from DFTB.SlaterKoster.SKIntegrals import spline_effective_potential
from DFTB.Scattering.SlakoScattering import AtomicScatteringBasisSet
from DFTB.MolecularIntegrals.MulticenterIntegration import multicenter_integration, select_angular_grid, outerN
from DFTB.XYZ import atomlist2vector
from DFTB.AtomicData import atomlist2masses
from DFTB.Modeling.MolecularCoords import center_of_mass
from DFTB.BasisSets import AtomicBasisFunction
from DFTB.Scattering.PAD import so3_quadrature, rotate_sphere

class AtomicPotentialSuperposition:
    """
    superposition of atomic effective potentials

     V_eff[molecule](r) = sum_A V_eff[A](r)
    """
    def __init__(self, atomlist, confined=True):
        atomtypes = list(set([Zi for (Zi,posi) in atomlist]))
        atomtypes.sort()
        self.atomlist = atomlist
        self.atomic_potentials = {}
        for Zi in atomtypes:
            # load radial density of pseudo atoms
            confined_atom, free_atom = import_pseudo_atom(Zi)
            if confined == True:
                atom = confined_atom
            else:
                atom = free_atom
            # The effective potential at at r=0 is not correct, so leave it out
            r = atom.r
            potI_spline = spline_effective_potential(atom.r[r > 0], atom.effective_potential[r > 0], atom.r0)
            self.atomic_potentials[Zi] = potI_spline
    def __call__(self, x,y,z, exclude_centers=[]):
        """ 
        evaluate the superposition of atomic effective potentials on a grid 
        """
        pot0 = 0*x
        for i,(Zi,posi) in enumerate(self.atomlist):
            if i in exclude_centers:
                continue
            #print "  potential due to atom %d" % i
            potI_spline = self.atomic_potentials[Zi]
            xI,yI,zI = x-posi[0], y-posi[1], z-posi[2]
            # distance to atomic center I
            rI = np.sqrt(xI**2+yI**2+zI**2)
            # add unperturbed effective potential due to atom I
            pot0 += potI_spline(rI)
        return pot0


    
class AsymptoticSolutionsBasisSet:
    def __init__(self, atomlist, E, lmax=2):
        """
        solutions of the free scattering problem H0=T

          (T-E)|i> = 0

        """
        k = np.sqrt(2*E)
        # find center of mass as the origin for the free solutions 
        masses = atomlist2masses(atomlist)
        cm = center_of_mass(masses, atomlist2vector(atomlist))
        #
        self.bfsS = []
        self.bfsC = []
        # switching function
        #  g(0) = 0    g(oo) = 1                                         
        def g(r):
	    gr = 0*r
            gr[r > 0] = 1.0/(1.0+np.exp(-(1-1.0/r[r > 0])))
            return gr

        def asymptotic_coulomb(k,l,Z=0.0):
            """ asymptotic radial functions """
            eta = -Z/k
            gm = gamma(l+1+1.0j*eta)
            sigma_l = np.arctan2(gm.imag, gm.real)
            def Sl(r):
                rho = k*r[r > 0]
                theta_l = 0*r
                theta_l[r > 0] = rho - eta*np.log(2*rho) - l*np.pi/2.0 + sigma_l
                return g(r)*np.sin(theta_l)/r
            def Cl(r):
                rho = k*r[r > 0]
                theta_l = 0*r
                theta_l[r > 0] = rho - eta*np.log(2*rho) - l*np.pi/2.0 + sigma_l
                return g(r) * np.cos(theta_l)/r
            return Sl, Cl
            
        for l in range(0, lmax+1):
            Sl,Cl = asymptotic_coulomb(k,l)
            for m in range(-l,l+1):
                Slm = AtomicBasisFunction(None, cm, 0, l,m, Sl, None)
                Clm = AtomicBasisFunction(None, cm, 0, l,m, Cl, None)
                self.bfsS.append(Slm)
                self.bfsC.append(Clm)

class LinearCombinationWavefunction:
    def __init__(self, bfs, coeffs):
        """
        wavefunction which is a linear combination of basis functions
        
          |Psi> = sum_i c_i |i>

        Parameters
        ----------
        bfs         : list of basis functions, each should be an 
                  instance of AtomicBasisFunction
        coeffs      : coefficients of the basis functions, coeffs[i]
                  is the coefficient belonging to bfs[i]
        """
        # number of basis functions
        self.nbfs = len(bfs)
        self.bfs = bfs
        self.coeffs = coeffs
        assert len(bfs) == len(coeffs), "Number of basis functions and length of coefficient vector must agree!"
    def amp(self, x,y,z):
        """
        evaluate Psi(x,y,z) on a grid of cartesian coordinates
        """
        psi = 0*x
        for i in range(0, self.nbfs):
            psi += self.coeffs[i] * self.bfs[i].amp(x,y,z)
        return psi
    def __str__(self):
        txt = ""
        for i in range(0, self.nbfs):
            txt += " %+7.5f * %s" % (self.coeffs[i], self.bfs[i])
        return txt
    
class GaussianMolecularOrbital(LinearCombinationWavefunction):
    def __init__(self, ucontr_basis, orb):
        """
        linear combination of uncontracted Gaussian basis functions.

        This is just a wrapper class to give orbitals defined with respect to
        `UncontractedBasisSet` the same interface as `LinearCombinationWavefunction`

        Parameters
        ----------
        ucontr_basis  :  instance of UncontractedBasisSet
        orb  : MO coefficients, orb[i] is the coefficient of the i-th basis function
        """
        self.ucontr_basis = ucontr_basis
        self.orb = orb
        assert len(self.orb) == self.ucontr_basis.nbfs
    def amp(self, x,y,z):
        return self.ucontr_basis.wavefunction(self.orb, x,y,z)

        
def scattering_integrals(atomlist, bfs, pot0,
                         radial_grid_factor=3,
                         lebedev_order=23):
    """
    compute the integrals
      L_ij = <i|(H-E)|j> = <i|(T + Vj + sum_{k!=j} Vk - E)|j>
           = <i|(T+Vj-E)|j> + <i|sum_{k!=j} Vk|j>
           =      0         + <i|sum_{k!=j} Vk|j>
    numerically on a multicenter grid, |i> are atomic scattering states

    Parameters
    ----------
    atomlist  :  list of tuples (Zi,[xi,yi,zi]) with atomic positions which
                 define the multicenter grid for numerical integration
    bfs       :  list of instances of AtomicBasisFunction, |i> = bfs[i]
    pot0      :  instance of AtomicPotentialSuperposition, 
                 pos0(x,y,z, exclude_centers=[j]) evaluates the sum of atomic
                 effective potentials excluding atom j, i.e. sum_{k!=j} Vk(x,y,z)

    Optional
    --------
    radial_grid_factor  : factor by which the number of radial grid points is increased 
                          for integration on the interval [0,+inf]
    lebedev_order       : order of Lebedev grid for angular integrals

    Returns
    -------
    L         :  matrix L[i,j]
    """
    print "computing scattering integrals between continuum orbitals ..."
    # bring date into a form understood by the module MolecularIntegrals
    Nat = len(atomlist)
    atomic_numbers = np.zeros(Nat, dtype=int)
    atomic_coordinates = np.zeros((3,Nat))
    for i in range(0, Nat):
        Z,pos = atomlist[i]
        atomic_numbers[i] = Z
        atomic_coordinates[:,i] = pos
    # number of basis functions
    nbfs = len(bfs)
        
    L = np.zeros((nbfs,nbfs))
    for i in range(0, nbfs):
        for j in range(0, nbfs):
            #print "i= %d    atom=%s l=%d m=%d" % (i, bfs[i].atom_index, bfs[i].l, bfs[i].m)
            #print "  j= %d    atom=%s l=%d m=%d" % (j, bfs[j].atom_index, bfs[j].l, bfs[j].m)
            # exclude the effective potential due to atom j
            center_j = bfs[j].atom_index
            def integrand(x,y,z):
                return bfs[i].amp(x,y,z) * \
                    pot0(x,y,z, exclude_centers=[center_j]) * bfs[j].amp(x,y,z)

            L[i,j] = multicenter_integration(integrand, atomic_coordinates, atomic_numbers,
                                             radial_grid_factor=radial_grid_factor,
                                             lebedev_order=lebedev_order)
            #print "L[%d,%d]= %s" % (i,j, L[i,j])
    return L


def transition_dipole_integrals(atomlist, wfsA, wfsB,
                                radial_grid_factor=3,
                                lebedev_order=23):
    """
    transition dipole integrals 
     
        D_ij = <i|r|j>              |i> in A, |j> in B

    between two sets of orbitals A and B are computed numerically on a multicenter grid. 
    At least one of sets of orbitals should be square integrable, otherwise the integrals 
    do not exist.

    Parameters
    ----------
    atomlist  :  list of tuples (Zi,[xi,yi,zi]) with atomic positions which
                 define the multicenter grid for numerical integration
    wfsA       : list of `norbA` bra wavefunctions |i>, instances of LinearCombinationWavefunction 
    wfsB       : list of `norbB` ket wavefunctions |j>, instances of LinearCombinationWavefunction 

    Optional
    --------
    radial_grid_factor  : factor by which the number of radial grid points is increased 
    lebedev_order       : order of Lebedev grid for angular integrals

    Returns
    -------
    dipoles: matrix with shape (norbA,norbB,3)
             dipoles[i,j,0] for instance would be <i|x|j>
    """
    print "computing dipole integrals between bound and continuum orbitals ..."
    # bring date into a form understood by the module MolecularIntegrals
    Nat = len(atomlist)
    atomic_numbers = np.zeros(Nat, dtype=int)
    atomic_coordinates = np.zeros((3,Nat))
    for i in range(0, Nat):
        Z,pos = atomlist[i]
        atomic_numbers[i] = Z
        atomic_coordinates[:,i] = pos
        
    # number of wavefunctions in each set
    norbA = len(wfsA)
    norbB = len(wfsB)

    dipoles = np.zeros((norbA,norbB,3), dtype=complex)

    for i in range(0, norbA):
        for j in range(0, norbB):
            # define integrands for x-,y- and z-component of dipole matrix elements
            def integrand_X(x,y,z):
                return wfsA[i].amp(x,y,z).conjugate() * x * wfsB[j].amp(x,y,z)
            def integrand_Y(x,y,z):
                return wfsA[i].amp(x,y,z).conjugate() * y * wfsB[j].amp(x,y,z)
            def integrand_Z(x,y,z):
                return wfsA[i].amp(x,y,z).conjugate() * z * wfsB[j].amp(x,y,z)
            # perform numerical integrations using Becke's scheme
            # ... for <i|x|j>
            dipoles[i,j,0] = multicenter_integration(integrand_X, atomic_coordinates, atomic_numbers,
                                                     radial_grid_factor=radial_grid_factor,
                                                     lebedev_order=lebedev_order)
            # ... for <i|y|j>
            dipoles[i,j,1] = multicenter_integration(integrand_Y, atomic_coordinates, atomic_numbers,
                                                     radial_grid_factor=radial_grid_factor,
                                                     lebedev_order=lebedev_order)
            # ... for <i|z|j>
            dipoles[i,j,2] = multicenter_integration(integrand_Z, atomic_coordinates, atomic_numbers,
                                                     radial_grid_factor=radial_grid_factor,
                                                     lebedev_order=lebedev_order)
            """
            ### DEBUG
            def integrand_Olap(x,y,z):
                return wfsA[i].amp(x,y,z).conjugate() * wfsB[j].amp(x,y,z)
            olap = multicenter_integration(integrand_Olap, atomic_coordinates, atomic_numbers,
                                           radial_grid_factor=radial_grid_factor,
                                           lebedev_order=lebedev_order)
            print "overlap <%d|%d>= %s" % (i,j,olap)
            ###
            """

    # Since all wavefunctions are real, the dipole matrix elements are real, too
    return dipoles.real

def analyze_dipoles_angmom(dipoles, lms):
    """
    partition modulus squared of transition dipoles by the angular momentum
    of the final continuum orbital.

    This is needed to check whether the sum
                              2
        sum_j |<E,lj,mj|r|i>|

    which is proportional to the total photoionization cross section sigma(E)
    converges, as lmax is increased.

    Parameters
    ----------
    dipoles    : numpy array with transition dipoles between bound and continuum orbitals
                 dipoles[i,j,:] = <bound i| r |continuum j>
    lms        : list of tuples (lj,mj) with angular momentum quantum numbers for each
                 continuum orbital j
    """
    nb, nc, nxyz = dipoles.shape
    assert nxyz == 3
    assert nc == len(lms)
    for i in range(0, nb):
        print ""
        print "transition dipoles for orbital %d to continuum" % i
        print "                                  2  "
        print "  l         sum_j |<E,lj,mj|r|i>|   "
        print "-------------------------------------"
        # sum over m of |<E,l,m|r|i>|^2
        sum_t2m = 0.0
        # (lj,mj) are the asymptotic angular momentum quantum numbers of a continuum
        # orbital with energy E
        for j,(l,m) in enumerate(lms):
            # add   |<E,lj,mj|x|i>|^2 + |<E,lj,mj|y|i>|^2 + |<E,lj,mj|z|i>|^2
            sum_t2m += np.sum(abs(dipoles[i,j,:])**2)   
            if m == l:
                print "  %d               %e" % (l,sum_t2m)
                # reset 
                sum_t2m = 0.0
                
def continuum_normalization(continuum_orbitals, energy, rmax=300.0, npts_r=60, lebedev_order=23):
    """
    continuum orbitals are normalized such that asymptotically their radial part behaves
    as

            S(r) = 1/r sin(k*r + delta)

    To find the factor by which the unnormalized continuum orbitals have to be multiplied,
    we integrate the modulus squared over one wavelength 2*pi/k starting at a large radius
    R:
                 /pi           /2 pi  /R+2pi/k   2                2
       I(wfn) =  | sin(th) dth | dph  |   dr    r   |wfn(r,th,ph)|        
                 /0            /0     /R        
    
    and compare with the corresponding integral for the asymptotic form S(r)

               /R+2pi/k     2
       I(S) =  |  dr     sin (k*r+delta)  = pi / k
               /R

    The wavefunction is then normalized as 

       wfn(normalized) = sqrt(I(S)/I(wfn)) * wfn(unnormalized)

    Parameters
    ----------
    continuum_orbitals: list of unnormalized continuum orbitals, instances of 
                        LinearCombinationWavefunction or AtomicBasisFunction
    energy            : energy of continuum orbital (in Hartree), E=1/2 k^2
    
    Optional
    --------
    rmax              : radius (in bohr) at which the integration states
    npts_r            : number of points in quadrature rule for radial integration on interval [rmax,rmax+2pi/k]
    lebedev_order     : order of Lebedev grid for angular integral

    Returns
    -------
    scaling_factors   : list of scaling factors sqrt(I(S)/I(wfn)) 
    """
    # wavenumber
    k = np.sqrt(2*energy)
    # angular grid
    Lmax, (th,ph,angular_weights) = select_angular_grid(lebedev_order)
    sc = np.sin(th)*np.cos(ph)
    ss = np.sin(th)*np.sin(ph)
    c  = np.cos(th)
    # radial grid
    # sample points and weights for Gauss-Legendre quadrature on the interval [-1,1]
    leggauss_pts, leggauss_weights = legendre.leggauss(npts_r)
    # For Gaussian quadrature the integral [-1,1] has to be changed into [rmax,rmax+2pi/k].
    # new endpoints of interval
    a = rmax
    b = rmax+2.0*np.pi/k
    # transformed sampling points and weights
    r = 0.5*(b-a)*leggauss_pts + 0.5*(a+b)
    radial_weights = 0.5*(b-a)*leggauss_weights
    # cartesian coordinates of grid
    x = outerN(r, sc).flatten()
    y = outerN(r, ss).flatten()
    z = outerN(r, c ).flatten()
    r2 = x*x+y*y+z*z
    
    weights = outerN(radial_weights, 4.0*np.pi * angular_weights).flatten()
    
    # desired integral I(S) for asymptotic orbital
    Is = np.pi / k
    
    # For each continuum orbital, we compute I(wfn) and the scaling factor
    scaling_factors = np.zeros(len(continuum_orbitals))
    for iw,wfn in enumerate(continuum_orbitals):
        # integral I(wfn)
        I = np.sum(r2 * abs(wfn.amp(x,y,z))**2 * weights)
        # scaling factor
        scaling_factors[iw] = np.sqrt(Is/I)

    return scaling_factors


def variational_kohn(atomlist, energy, lmax=2, rmax=300.0, npts_r=60,
                     radial_grid_factor=3, lebedev_order=23):
    """
    find the scattering states for a molecular Hamiltonian
     
      H = T + sum_k Vk

    where the molecular potential consists of a superposition of atomic
    effective potentials Vk

    Asymptotically the scattering orbital can be classified by its angular momentum l,m.
    The scattering orbital is assumed to have the form:

      |Psi_{l,m}> = cos(delta_{l,m}) |S_{l,m}> + sin(delta_{l,m}) |C_{l,m}> + sum_i C_{l,m,i} |Ai,li,mi>

    where |S_{l,m}> and |C_{l,m}> are sine- and cosine-like free solutions of (T-E)|Psi> = 0
    with angular quantum numbers l,m and |Ai,li,mi> are atomic continuum orbitals, 
    which are solutions of (T+Vi-E)|Psi> = 0. delta_{l,m} is the phase shift
    
    The coefficients C and the phase shift delta for each partial wave 
    are determined from a variational principle.

    Parameters
    ----------
    atomlist      :  list of tuples (Z,[x,y,z]) with atom number and nuclear position
    energy        :  energy E=1/2 k^2 (in Hartree)

    Optional
    --------
    lmax              :  highest angular momentum of atomic continuum orbitals
    rmax              : radius (in bohr) at which the integration states
    npts_r            : number of points in quadrature rule for radial integration on interval [rmax,rmax+2pi/k]
    radial_grid_factor: factor by which the number of radial grid points is increased 
                        for integration on the interval [0,+inf]
    lebedev_order     : order of Lebedev grid for angular integral


    Returns
    -------
    continuum_orbitals  :  list of continuum orbitals |Psi_{l,m}>, which are instances of
                           LinearCombinationWavefunction
    phase_shifts        :  list of phase shift delta_{l,m}
    lms                 :  list of tuples (l,m) with the asymptotic angular momentum quantum
                           numbers for each continuum orbital
    """
    # molecular potential V = sum_k Vk
    pot0 = AtomicPotentialSuperposition(atomlist, confined=False)
    # basis of atomic scattering states
    # The atomic basis functions |i> = |A,l,m> belonging to atom A
    # are solution of
    #    (T + V_A - E)|A,l,m> = 0
    # 
    basis = AtomicScatteringBasisSet(atomlist, energy, lmax=lmax)
    # sets of linearly independent solutions for each partial wave l,m
    #    (T - E)|S_{l,m}> = 0
    #    (T - E)|C_{l,m}> = 0
    basis0 = AsymptoticSolutionsBasisSet(atomlist, energy, lmax=lmax)
    # The basis functions can thus be grouped into three categories
    #  * atomic scattering functions centered on each atom  :  |A,l,m>
    #    for each atom A
    #    and for l=0,...,lmax   m=-lmax,...,0,...,lmax
    #  * sine-like spherical waves at the center of mass    :  |S_{l,m}>
    #    for l=0,...,lmax   m=-lmax,...,0,...,lmax
    #  * cosine-like spherical waves at the center of mass  :  |C_{l,m}>
    #    for l=0,...,lmax   m=-lmax,...,0,...,lmax

    # count basis functions
    #  * on atomic centers
    nb = len(basis.bfs)
    #  * sine-like basis functions
    ns = len(basis0.bfsS)
    #  * cosine-like basis functions
    nc = len(basis0.bfsC)
    
    # Now we need to compute the matrix elements of the operator L=H-E
    # between all basis functions
    #     |A,l,m>       |S_{l,m}>     |C_{l,m}>
    bfs = basis.bfs + basis0.bfsS + basis0.bfsC
    L = scattering_integrals(atomlist, bfs, pot0,
                             radial_grid_factor=radial_grid_factor,
                             lebedev_order=lebedev_order)
    # extract blocks from L-matrix, which is not Hermitian
    #
    #      Lbb Lbs Lbc
    # L =  Lsb Lss Lsc
    #      Lcb Lcs Lcc
    #
    # Lbb[i,j] = <Ai,li,mi|(H-E)|Aj,lj,mj>
    Lbb = L[:nb,:nb]
    # Lbs[i,j] = <Ai,li,mi|(H-E)|S_{lj,mj}>
    Lbs = L[:nb,nb:nb+ns]
    # Lbc[i,j] = <Ai,li,mi|(H-E)|C_{lj,mj}>
    Lbc = L[:nb,nb+ns:nb+ns+nc]
    # Lss[i,j] = <S_{li,mi}|(H-E)|S_{lj,mj}>
    Lss = L[nb:nb+ns,nb:nb+ns]
    # Lsb[i,j] = <S_{li,mi}|(H-E)|Aj,lj,mj>
    Lsb = L[nb:nb+ns,:nb]
    # Lsc[i,j] = <S_{li,mi}|(H-E)|C_{lj,mj}>
    Lsc = L[nb:nb+ns,nb+ns:nb+ns+nc]
    # Lcs[i,j] = <C_{li,mi}|(H-E)|S_{lj,mj}>
    Lcs = L[nb+ns:nb+ns+nc,nb:nb+ns]
    # Lcb[i,j] = <C_{li,mi}|(H-E)|Aj,lj,mj>
    Lcb = L[nb+ns:nb+ns+nc,:nb]
    # Lcc[i,j] = <C_{li,mi}|(H-E)|C_{lj,mj}>
    Lcc = L[nb+ns:nb+ns+nc,nb+ns:nb+ns+nc]

    # Determine coefficients cS and cC in the expansion
    #  |uS_{l,m}> = |S_{l,m}> + sum_k cS_{l,m;Ak,lk,mk} |Ak,lk,mk>
    #  |uC_{l,m}> = |C_{l,m}> + sum_k cC_{l,m;Ak,lk,mk} |Ak,lk,mk>
    # from the conditions
    #  (1)  <Ai,li,mi|(H-E)|uS_{l,m}> = 0
    #  (2)  <Ai,li,mi|(H-E)|uC_{l,m}> = 0
    cS = - la.solve(Lbb, Lbs)    # solution of eqn. (1)  Lbs + Lbb.cS = 0
    cC = - la.solve(Lbb, Lbc)    # solution of eqn. (2)  Lbc + Lbb.cC = 0

    # The partial waves are assumed to have the form
    #  |Psi_{l,m}> = |uS_{l,m}> + t_{l,m} |uC_{l,m}>
    # The tangent of the phase shift t_{l,m} = tan(delta_{l,m}) is
    # determined from the conditions
    #  (3)  <uS_{l,m}|(H-E)|Psi_{l,m}> = 0
    #  (4)  <uC_{l,m}|(H-E)|Psi_{l,m}> = 0
    # This leads to the following 2 x 2 system of equations
    #   (Mss  Msc)   (1)
    #   (        ) * ( ) = 0
    #   (Mcs  Mcc)   (t)
    # Both equations cannot be satisfied at the same time, so we
    # have to choose one of them
    #

    # coefficients are real, so we do not really need complex conjugation
    cSt = cS.conjugate().transpose()
    cCt = cC.conjugate().transpose()
    
    Mss = Lss + np.dot(cSt,Lbs) + np.dot(Lsb,cS) + np.dot(cSt,np.dot(Lbb,cS))
    Msc = Lsc + np.dot(cSt,Lbc) + np.dot(Lsb,cC) + np.dot(cSt,np.dot(Lbb,cC))
    Mcs = Lcs + np.dot(cCt,Lbs) + np.dot(Lcb,cS) + np.dot(cCt,np.dot(Lbb,cS))
    Mcc = Lcc + np.dot(cCt,Lbc) + np.dot(Lcb,cC) + np.dot(cCt,np.dot(Lbb,cC))

    continuum_orbitals = []
    phase_shifts = []
    lms = []
    for i in range(0, ns):
        l,m = basis0.bfsS[i].l, basis0.bfsS[i].m
        #  solve  Mss + t Msc = 0
        t = -Mss[i,i]/Msc[i,i]
        # the other possibility would be to solve  Mcs + t Mcc = 0
        # t = -Mcs[i,i]/Mcc[i,i]

        # compute phase shift
        delta = np.arctan(t)
        # Because a global phase is irrelevant, the phase shift is only
        # determined module pi. sin(pi+delta) = -sin(delta)
        while delta < 0.0:
            delta += np.pi
        
        # and normalization factor
        # sin(kr+delta) = sin(kr) cos(delta) + sin(delta) cos(kr)
        #               = cos(delta) [ sin(kr) + tan(delta) cos(kr)]

        # normalized
        # |Psi_{l,m}> = cos(delta) |uS_{l,m}> + sin(delta) |uC_{l,m}>
        #             = cos(delta) |S_{l,m}> + sin(delta) |C_{l,m}>
        #                 + sum_k [ cos(delta) cS_{l,m;Ak,lk,mk} + sin(delta) cC_{l,m;Ak,lk,mk} ] |Ak,lk,mk>
        
        # basis functions and coefficients
        bfs = [basis0.bfsS[i], basis0.bfsC[i]] + basis.bfs
        coeffs = np.zeros(2+nb)
        coeffs[0] = np.cos(delta)
        coeffs[1] = np.sin(delta)
        coeffs[2:] = np.cos(delta) * cS[:,i] + np.sin(delta) * cC[:,i]

        # create continuum wavefunction 
        psi = LinearCombinationWavefunction(bfs, coeffs)
        continuum_orbitals.append(psi)
        phase_shifts.append(delta)
        lms.append( (l,m) )

    # All continuum orbitals are rescaled / normalized, such that asymptotically
    # their radial parts tend to  1/r sin(k*r + delta)
    scaling_factors = continuum_normalization(continuum_orbitals, energy,
                                              rmax=rmax, npts_r=npts_r,
                                              lebedev_order=lebedev_order)
    for iw,wfn in enumerate(continuum_orbitals):
        # scale coefficients of linear combination of basis functions by sqrt(Is/I)
        wfn.coeffs *= scaling_factors[iw]
    
    return continuum_orbitals, phase_shifts, lms
    

def orientation_averaged_pad(dipoles, continuum_orbitals, energy,
                             rmax=300.0, npts_euler=10, npts_r=30, npts_theta=50):
    """
    average the photoelectron angular distribution (PAD) numerically over all molecular
    orientations. For a particular orientation of the electric field polarization the
    PAD in the molecular frame is given by the angular probability distribution of the 
    dipole prepared continuum orbital

        |psi(polarization)> = sqrt(2*E) sum_k (polarization).(dipoles_k) |continuum orbital k>

    for large distances, integrated over one wavelength

                       /rmax + 2 pi / k   2               2
        MFPAD(th,ph) = | dr              r  |psi(r,th,ph)|
                       /rmax 


    The MFPAD is then averaged over all molecular orientations represented by the rotation
    matrix R(a,b,c) (which depends on 3 Euler angles). In each orientation the field 
    polarization is rotated to R.polarization and the appropriately weighted MFPAD(a,b,c;th,ph)
    is added to the averaged PAD(th,ph). 

    Finally the total cross section sigma and the anisotropy parameters are extracted from
    the PAD(th,ph) by projection onto Legendre polynomials. Any dependence of PAD on phi
    is averaged out, a remaining dependence can be used to judge the quality of the grid
    for numerical integration over Euler angles.
    

    Parameters
    ----------
    dipoles            : transition dipoles between the initial bound orbital and the 
                         continuum orbital, dipoles[k,:] = <bound|r|k>
    continuum_orbitals : list of continuum orbitals, instances of LinearCombinationWavefunction
    energy             : photokinetic energy of continuum orbitals

    Optional
    --------
    rmax               : radius at which the molecular potential is indistinguishable from the asymptotic potential
    npts_euler         : number of grid points for numerical averaging over Euler angles
    npts_r             : number of points for integrating over radial interval [rmax,rmax+2pi/k]
    npts_theta         : number of points in theta grid, the anisotropy parameter beta
                         is extracted from PAD(th) = sigma/4 pi [1 + beta * P2(cos(th))]

    Returns
    -------
    betas              : list of floats, [sigma, beta1, beta2, beta3, beta4]
                         PAD(th) = sigma/(4 pi) (1 + sum_k=1^4 beta_k Pk(cos(th)))
    """
    # wave number for kinetic energy E=1/2 k^2
    k = np.sqrt(2*energy)
    # wavelength
    wavelength = 2.0 * np.pi / k
    # spherical grid on which the MFPAD(th,ph) is computed, the radial dimension
    # is integrated over
    # PAD(th) has no dependence on phi, so one point is enough
    npts_phi = 1
    # The anisotropy parameters are extracted from PAD(th) by projecting
    # onto Legendre polynomials.The integration
    #                           /pi
    #  <PAD(th),P_L(cos(th))> = |   sin(th) PAD(th) P_L(cos(th))  dt
    #                           /0
    #                         = sum_k w_k PAD(th_k) P_L(cos(th_k))
    #
    # is performed by Gauss-Legendre quadrature.
    # get sample points x = cos(th) and weights
    assert npts_theta <= 100, "Legendre-Gauss quadrature points only tested up to degree 100!"
    x,weights_theta = legendre.leggauss(npts_theta)
    thetas_leggauss = np.arccos(x)
    # radial grid
    # sample points and weights for Gauss-Legendre quadrature on the interval [-1,1]
    leggauss_pts, leggauss_weights = legendre.leggauss(npts_r)
    # For Gaussian quadrature the integral [-1,1] has to be changed into [rmax,rmax+2pi/k].
    # new endpoints of interval
    a = rmax
    b = rmax+wavelength
    # transformed sampling points and weights
    r_leggauss = 0.5*(b-a)*leggauss_pts + 0.5*(a+b)
    weights_r = 0.5*(b-a)*leggauss_weights

    # create coordinate arrays
    rs, thetas, phis = np.meshgrid( r_leggauss,
                                    thetas_leggauss,
                                    np.linspace(0.0 , 2.0*np.pi      , npts_phi),
                                    indexing='ij')
    # weights for radial integration
    weights_rs, dummy, dummy = np.meshgrid(weights_r,
                                           np.ones(npts_theta),
                                           np.ones(npts_phi),
                                           indexing='ij')
    shape = rs.shape
    # grid of Euler angles for numerical integration on SO3
    weights_rot, rotations = so3_quadrature(npts_euler)

    # number of orbitals in the continuum basis
    nmo = len(continuum_orbitals)
    
    # In the laboratory frame the field polarization points along the z-axis
    epol = np.array([0.0, 0.0, 1.0])

    # averaged PAD, sum of rotated MFPADs
    pad = np.zeros((npts_theta, npts_phi))
    
    # compute MFPAD for each orientation and average
    for i,(w,R) in enumerate(zip(weights_rot, rotations)):
        # R is a rotation matrix, w its weight in the quadrature formula
        
        # print some regular progress report so nobody gets impatient
        if i % min(100, len(weights_rot)/2) == 0:
            print "%d of %d" % (i,len(weights_rot))

        # rotate the field polarization direction into the molecular frame
        epol_rot = np.dot(R, epol)

        # compute the coefficients of the dipole prepared continuum orbitals
        # in the basis of the molecular continuum orbitals
        mo_dipprep = np.zeros(nmo)
        for xyz in [0,1,2]:
            mo_dipprep += dipoles[:,xyz] * epol_rot[xyz]
        mo_dipprep *= np.sqrt(2*energy)

        # Now we evaluate MFPAD on the rotated (th,ph) grid.
        # First we need to rotate the grid
        thetas_rot, phis_rot = rotate_sphere(R, thetas, phis)
        # convert rotated spherical grid into cartesian coordinates
        xs = rs * np.sin(thetas_rot) * np.cos(phis_rot)
        ys = rs * np.sin(thetas_rot) * np.sin(phis_rot)
        zs = rs * np.cos(thetas_rot)
        grid = (xs,ys,zs)

        # evaluate the amplitude of the dipole prepared continuum orbital
        # on the grid
        wfn_dipprep = np.zeros(shape, dtype=complex)
        for mo in range(0, nmo):
            wfn_dipprep += mo_dipprep[mo] * continuum_orbitals[mo].amp(*grid)

        # integrate the probability density over one wavelength along the
        # radial axis to obtain the angular distribution in the molecular frame
        # (MFPAD)
        mfpad = np.sum( abs(wfn_dipprep)**2 * rs**2 * weights_rs, axis=0)

        # add MFPAD for orientation R to the averaged PAD with weight from quadrature rule
        pad += w * mfpad

    # project PAD(th) onto Legendre polynomials P_L(th)
    # argument of Legendre polynomials is x = cos(th)
    # Legendre polynomials of degrees L=0,1,...,6
    P0 = np.ones(x.shape)
    P1 = x
    P2 = (3*x**2 - 1.0)/2.0
    P3 = (5*x**3 - 3*x)/2.0
    P4 = (35*x**4 - 30*x**2 + 3)/8.0
    P5 = (63*x**5 -70*x**3 + 15*x)/8.0
    P6 = (231*x**6 - 315*x**4 + 105*x**2 - 5)/16.0
    P = [P0,P1,P2,P3,P4,P5,P6]
    """
    ### DEBUG
    # check that Legendre polynomials are indeed orthonormal
    for i in range(0, len(P)):
        for j in range(0, len(P)):
            olap_ij = (2.0*i+1.0)/2.0 * np.sum(weights_theta * P[i]*P[j])
            print "<P(%d)|P(%d)>= %s" % (i,j,olap_ij)
    ###
    """
    # extract `betas` by projecting PAD(th) onto Legendre polynomials
    _betas = np.zeros((5,npts_phi))
    # In principle PAD(th,ph) depends on both `th` and `ph`. However, the `ph` dependence
    # is completely averaged out, so that in reality we have PAD(th). To check that the
    # PAD(th,ph) does not depend on ph, we compute the projection of PAD(th) onto the
    # Legendre polynomials for each angle ph and verify that we get always approximately the
    # same coefficients. If this is not the case the grid of Euler angles is not dense enough.
    for i in range(0, npts_phi):
        # coefficients of PAD in basis of Legendre polynomials
        c = np.zeros(len(P))
        for n in range(0, len(P)):
            # integration by Gauss-Legendre quadrature
            c[n] = (2.0*n+1.0)/2.0 * np.sum(weights_theta * pad[:,i] * P[n])
        # convert coefficients to the form
        # PAD(th) = sigma/(4 pi) * [1 + sum_n=1^4 betaL P_n(th) ]
        sigma = c[0] * (4.0*np.pi)
        beta1 = c[1] / c[0]
        beta2 = c[2] / c[0]
        beta3 = c[3] / c[0]
        beta4 = c[4] / c[0]
        # combine parameters for PAD(th,ph=phi_i) 
        _betas[:,i] = np.array([sigma,beta1,beta2,beta3,beta4])
    # average over `phi` angle
    betas = np.mean(_betas, axis=1)
    # compute standard deviation, if the numerical integration over orientations is complete
    # the deviation should be very small, i.e. no variation with `phi`
    betas_std = np.std(_betas, axis=1)

    print "averages over angle phi:"
    print "  sigma = %s  beta1 = %s   beta2 = %s  beta3 = %s   beta4 = %s" % tuple(betas)
    print "standard deviations"
    print "  sigma = %s  beta1 = %s   beta2 = %s  beta3 = %s   beta4 = %s" % tuple(betas_std)
    
    return betas
        
############### Testing ############

def test_mesh_size():
    """
    check that the integrals <i|(H-E)|j> between scattering orbitals do not depend
    on the mesh size, i.e. are converged for a the grid sizes used
    """
    import matplotlib.pyplot as plt
    # H2
    atomlist = [(1,[0,0,-0.7]), (1,[0,0,+0.7])]
    E = 0.1
    basis = AtomicScatteringBasisSet(atomlist, E)
    pot0 = AtomicPotentialSuperposition(atomlist, confined=False)

    for radial_grid_factor in [1,2,3,4]:
        for lebedev_order in [11, 17, 23]:
            L = scattering_integrals(atomlist, basis.bfs, pot0,
                                     radial_grid_factor=radial_grid_factor,
                                     lebedev_order=lebedev_order)
            
            import matplotlib.pyplot as plt
            plt.imshow(L)
            plt.show()
    

def plot_phase_shifts_h2():
    # H2
    atomlist = [(1,[0,0,-0.7]), (1,[0,0,+0.7])]

    n = 200
    energies = np.linspace(0.01, 1.0, n)
    phase_shifts_energy = []
    for i,E in enumerate(energies):
        print " %d of %d   energy= %s" % (i+1,len(energies), E)
        continuum_orbitals, phase_shifts, lms = variational_kohn(atomlist, E)
        phase_shifts_energy.append( phase_shifts )

    import matplotlib.pyplot as plt
    plt.xlabel(r"energy $E=\frac{1}{2} k^2$ / Hartree", fontsize=16)
    plt.xlabel(r"phase shifts $\delta_{l,m}$", fontsize=16)

    phase_shifts_energy = np.array(phase_shifts_energy)
    for i,(l,m) in enumerate(lms):
        plt.plot(energies, phase_shifts_energy[:,i], lw=2, label=r"$\delta_{%d,%d}$" % (l,m))

    plt.legend()
    plt.plot()
    plt.show()
    
def test_transition_dipole_integrals(fchk_file):
    from DFTB.MolecularIntegrals.fchkfile import G09ResultsDFT
    res = G09ResultsDFT(fchk_file)
    # compute transition dipole between HOMO and LUMO
    assert res.nelec_alpha == res.nelec_beta
    orb_HOMO = res.orbs_alpha[:,res.nelec_alpha-1]
    orb_LUMO = res.orbs_alpha[:,res.nelec_alpha]
    # define wavefunctions/orbitals
    wfs = [GaussianMolecularOrbital(res.basis, orb_HOMO),
           GaussianMolecularOrbital(res.basis, orb_LUMO)]

    # convert to atomlist format
    atomlist = []
    for i in range(0, res.nat):
        atomlist.append( (res.atomic_numbers[i], res.coordinates[:,i]) )
        
    dipoles = transition_dipole_integrals(atomlist, wfs, wfs)

    # <HOMO|r|HOMO>  <HOMO|r|LUMO>
    # <LUMO|r|HOMO>  <LUMO|r|LUMO>
    
    print dipoles[0,0,:]
    print dipoles[0,1,:]
    print dipoles[1,0,:]
    print dipoles[1,1,:]

def continuum_orbitals_h2_cubes():
    """
    compute the continuum orbitals of the hydrogen molecule and 
    save them to cube files
    """
    from DFTB.Analyse.Cube import function_to_cubefile
    
    # H2 with bond length r=1.4 bohr
    #atomlist = [(1,[0,0,-0.7]), (1,[0,0,+0.7])]
    rHH = 1.4
    theta = 45.0 * np.pi / 180.0
    atomlist = [(1,[0,0,0]), (1,[np.cos(theta)*rHH,np.sin(theta)*rHH,0])]    
    # energy PKE=2.7 eV
    E = 0.1

    # compute continuum orbitals
    continuum_orbitals, phase_shifts, lms = variational_kohn(atomlist, E, lmax=4)

    # and save them to cube files
    for wfn,(l,m) in zip(continuum_orbitals, lms):
        print "continuum orbital l=%d m=%d" % (l,m)
        
        def func(grid,dV):
            x,y,z = grid
            return wfn.amp(x,y,z)
        
        function_to_cubefile(atomlist, func,
                             filename="/tmp/h2_continuum_l%d_m%d.cube" % (l,m),
                             dbuff=15.0)

def test_continuum_normalization():
    """
    check that atomic continuum orbitals are correctly normalized
    """
    # single atom at origin
    atomlist = [(1,[0,0,0])]

    # energy E=1/2 k^2
    for energy in np.linspace(0.1,0.5,10):
        # atomic continuum orbitals
        basis = AtomicScatteringBasisSet(atomlist, energy, lmax=3)
        # 
        scaling_factors = continuum_normalization(basis.bfs, energy)
        print "energy= %s" % energy
        print "scaling factors"
        print scaling_factors
        # Since atomic continuum orbitals are correctly normalized by
        # default, all scaling factors should be 1
        assert la.norm(abs(scaling_factors - 1.0)) < 1.0e-2
    
    
if __name__ == "__main__":
    #plot_phase_shifts_h2()
    """
    # H2
    atomlist = [(1,[0,0,-0.7]), (1,[0,0,+0.7])]
    E = 0.1

    variational_kohn(atomlist, E)

    basis = AtomicScatteringBasisSet(atomlist, E)
    basis0 = AsymptoticSolutionsBasisSet(atomlist, E)
    pot0 = AtomicPotentialSuperposition(atomlist, confined=False)

    bfs = basis.bfs + basis0.bfsS + basis0.bfsC
    L = scattering_integrals(atomlist, bfs, pot0)

    import matplotlib.pyplot as plt
    plt.imshow(L)
    plt.show()
    """
    #test_transition_dipole_integrals("../MolecularIntegrals/test/h2o_hf_sv.fchk")
    continuum_orbitals_h2_cubes()
    #test_continuum_normalization()
