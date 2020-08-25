#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
implementation of Dill and Dehmer's continuum multiple scattering method

References
----------
[1] D.Dill, J.Dehmer, 
    "Electron-molecule scattering and molecular photoionization using the multiple-scattering method",
    J. Chem. Phys. 61, 692 (1974)
[2] M.Danos, L.Maximon
    "Multipole Matrix Elements of the Translation Operator"
    Journal of Mathematical Physics 6, 766 (1965)

"""
from DFTB import AtomicData
from DFTB.AtomicData import atom_names, slater_radii

from DFTB.MolecularIntegrals.LebedevQuadrature import outerN
from DFTB.MolecularIntegrals.Ints1e import integral
from DFTB.MolecularIntegrals import settings
from DFTB.MolecularIntegrals.MulticenterIntegration import select_angular_grid, multicenter_grids, join_grids
from DFTB.MolecularIntegrals.SphericalCoords import cartesian2spherical
from DFTB.MolecularIntegrals.BasissetFreeDFT import BasissetFreeDFT, effective_potential_func, density_func


from DFTB.SlaterKoster import XCFunctionals
from DFTB.SlaterKoster.free_pseudo_atoms import pseudo_atoms_list
from DFTB.SlaterKoster.SKIntegrals import spline_wavefunction
from DFTB.SlaterKoster.PseudoAtomDFT import PseudoAtomDFT
from DFTB.SlaterKoster.RadialPotential import CoulombExpGrid

from DFTB.Scattering.SlakoScattering import CoulombWave

from DFTB.Modeling import MolecularCoords as MolCo

# Most parts of the CMS method are implemented in Fortran in the module `cms.so`.
# The python code just acts as a wrapper. The old inefficient implementation in python
# and lots of tests can be found in `ContinuumMultipleScattering_tests.py`.
from DFTB.MolecularIntegrals import cms
# Photoionization cross sections are computed in `photo.so`
from DFTB.MolecularIntegrals import photo

import numpy as np
import numpy.linalg as la
from scipy import interpolate
from scipy import special
from scipy import optimize

import mpmath # for gammainc
from DFTB.MolecularIntegrals.sphharm import sphharm

def close_packed_spheres(atomlist, debug=0):
    """
    In the muffin tin method space is partitioned into three spherical regions:

      region I    -  sphere of radius ri around each atom
      region III  -  sphere of radius r0 around the origin, which
                     encloses all other spheres
      region II   -  interstitial region between I and III

    This function chooses the radii r0 and ri (i=1,...,Nat) based
    on atomic Slater radii, so as to obtain a relatively closely
    packed assembly. 

    Parameters
    ----------
    atomlist     :  list of tuples (Zat,[xi,yi,zi]) with atomic number
                    Zat and nuclear positions [xi,yi,zi] for each atom
    
    Optional
    --------
    debug        :  if set to 1, the spheres around regions I and III
                    are plotted in the yz-plane

    Returns
    -------
    rhoIII       :  float, radius of sphere around origin
    radii        :  numpy array of shape (Nat,) with radii of region I 
                    spheres around each atom
    """
    # Find the radii for regions I
    Nat = len(atomlist)
    assert Nat > 1
    
    # None of the atomic centers should lie exactly at the origin,
    # since in this case in N_coeffs(...) one has to evaluate the
    # spherical Bessel functions of the second kind n_l(r) at r=0
    # where n_l(r=0)=inf.
    for i,(Zi,posi) in enumerate(atomlist):
        assert la.norm(posi) > 0.0, "No atom should lie exactly at the origin, \n but atom %d lies at %s !" % (i, str(posi))
            
    
    atomic_names = [atom_names[Z-1] for Z,pos in atomlist]
    # put a sphere around each atom which does not penetrate
    # any other sphere.
    # The distance between two atoms is divided according to the
    # Slater radii
    radii = np.zeros(Nat)
    for i,(Zi,posi) in enumerate(atomlist):
        radius_i = np.inf
        for j,(Zj,posj) in enumerate(atomlist):
            if i == j:
                continue
            # distance between atoms i and j
            Rij = la.norm(np.array(posi) - np.array(posj))
            # Slater radii
            rsi = slater_radii[atomic_names[i]]
            rsj = slater_radii[atomic_names[j]]
            # s*Rij belongs to atom i, (1-s)*Rij to atom j
            s =  rsi/(rsi+rsj)
            radius_i = min(radius_i, s*Rij)

        radii[i] = radius_i

    # Radius for region III
    rhoIII = 0.0
    for i,(Zi,posi) in enumerate(atomlist):
        rhoIII = max(rhoIII, la.norm(np.array(posi)) + radii[i])

    print "rhoIII = %s" % rhoIII
    print "radii = %s" % radii
        
    if debug:
        # show partitioning in yz-plane
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()
        ax.set_aspect(aspect='equal')
        ax.set_axis_off()
        
        ax.set_xlim((-5, 5))
        ax.set_ylim((-5, 5))
        
        # plot circle that separates region II from region III
        regionIII = plt.Circle((0.0, 0,0), rhoIII, color='black', fill=False)
        ax.add_artist(regionIII)

        # plot circles that separate regions I from regions II
        for i,(Zi, posi) in enumerate(atomlist):
            regionII = plt.Circle((posi[1], posi[2]), radii[i], color='blue', fill=False)
            ax.add_artist(regionII)

        plt.show()
    
        
    return rhoIII, radii

def averaged_regionII_potential(atomlist, rhoIII, radii, potential):
    """
    compute the volume average of the molecular potential V(x,y,z)
    over the region II. Inside region II the potential is replaced
    by a constant equal to the average value 

        (II)
       V    = < V >
                   II
    
    The volume integral is performed using Becke's numerical integration scheme.

    Parameters
    ----------
    atomlist     :  list of tuples (Zat,(x,y,z)) with the atomic
                    positions that define the multicenter grid
    rhoIII       :  float, radius of sphere around origin separating
                    region II from region III
    radii        :  list with radii around each atom that separate
                    region I from region II
    potential    :  callable, potential(x,y,z) evaluates the V on
                    on a grid
    
    Returns
    -------
    VII          :  float, average value of potential over region II
    """
    # This function gives the potential V(r)
    # if the position vector r is in region II
    # and 0 otherwise.
    def potential_in_II(x,y,z):
        pot = potential(x,y,z)
        # distance to origin
        r = np.sqrt(x*x+y*y+z*z)
        # exclude region III
        pot[rhoIII < r] = 0.0

        for i,(Zat,posi) in enumerate(atomlist):
            xi,yi,zi = posi
            # distance to atom i
            r = np.sqrt((x-xi)**2+(y-yi)**2+(z-zi)**2)
            # exclude region I = inside of sphere of radius ri around atom i
            pot[r < radii[i]] = 0.0

        return pot

    def volume_in_II(x,y,z):
        vol = np.ones(x.shape)
        # distance to origin
        r = np.sqrt(x*x+y*y+z*z)
        # exclude region III
        vol[rhoIII < r] = 0.0

        for i,(Zat,posi) in enumerate(atomlist):
            xi,yi,zi = posi
            # distance to atom i
            r = np.sqrt((x-xi)**2+(y-yi)**2+(z-zi)**2)
            # exclude region I = inside of sphere of radius ri around atom i
            vol[r < radii[i]] = 0.0

        return vol
        
    # integral of potential over region II
    pot_II_integ = integral(atomlist, potential_in_II)
    # volume of region II
    vol_II_integ = integral(atomlist, volume_in_II)
    
    # average potential <V>_II
    VII = pot_II_integ / vol_II_integ
    
    return VII


# solver for radial Schroedinger equation
class AtomicPotential(CoulombExpGrid):
    def setRadialPotential(self, Vr):
        self.Vr = Vr
    def getRadialPotential(self):
        return self.Vr

def schroedinger_regionI(atomlist, potential, energy,
                         lmax=10, charge=+1,
                         rmin=0.0001, rmax=30.0, Nr=10000):
    """
    solve the Schroedinger equation for given energy E > 0    
           __2          (sph)
      -1/2 \/   phi + (V   (r) - E) phi = 0
                        i
    in each atomic region. 

    In order to decouple different angular momenta the equation is not 
    solved for the full potential V but for a simplified potential

            (sph)               /
           V       =   1/(4 pi) | V(r-R ) dOmega
            i                   /      i        i

    which is the spherical average of the potential around each atom i
    with position R_i.

    Parameters
    ----------
    atomlist     : list of tuples (Zat,(x,y,z)) with the atomic
                   positions that define the multicenter grid
    potential    : callable, potential(x,y,z) should evaluate the potential V at the
                   grid points specified by x = [x0,x1,...,xn], y = [y0,y1,...yn]
                   and z = [z0,z1,...,zn]
    energy       : float, energy E

    Optional
    --------
    lmax         : highest angular momentum of radial wavefunctions,
                   l=0,1,...,lmax
    charge       : int > 0, the continuum orbital is phase matched to a Coulomb wave
                   in the electrostatic field -charge/r.
    rmin, rmax   : floats, lower and upper bound of radial grid 
    Nr           : int, number of radial grid points

    Returns
    -------
    radial_pots  : list of callables, radial_pots[i](r) evaluates the spherically
                   symmetric effective potential in region I around atom i,
                         (sph)
                        V   (r)
                         i 
    radial_wfns  : list of lists, radial_wfns[i][l] evaluates the 
                   radial wavefunction f^(i)_{l} of atom i in region I with 
                   angular momentum l. The total atomic wavefunction would be
                           (i)    (i)
                        psi    = f   (r) Y  (th,ph)
                           l,m    l       l,m 
    phase_shifts : list of lists, phase_shifts[i][l] is the phase-shift
                   of the partial wave of atom i with angular momentum l.
    """
    # wave vector
    k = np.sqrt(2*energy)
    # number of atoms
    Nat = len(atomlist)
    # angular (Lebedev) grid for averaging the potential spherically around
    # each atom
    Lmax, (th,ph,angular_weights) = select_angular_grid(settings.lebedev_order)
    Nang = len(th)
    # cartesian coordinates of points on unit sphere
    sc = np.sin(th)*np.cos(ph)
    ss = np.sin(th)*np.sin(ph)
    c  = np.cos(th)

    # radial potentials, list of callables
    radial_potentials = []
    # radial wavefunctions, list of lists of callables
    radial_wavefunctions = []
    # phase shifts delta_l, list of lists of floats
    phase_shifts = []
    print "solving radial Schroedinger equation for ..."
    for i in  range(0, Nat):
        Zi,posi = atomlist[i]
        print "  ... atom %s%d" % (atom_names[Zi-1].upper(), i+1)
        # radial grid, spacing between points increases exponentially
        rho = np.linspace(np.log(rmin), np.log(rmax), Nr)
        r = np.exp(rho)
        
        # cartesian coordinates of grid
        x = outerN(r, sc) + posi[0]
        y = outerN(r, ss) + posi[1]
        z = outerN(r, c ) + posi[2]

        # differential for solid angle 
        dOmega = 4*np.pi * outerN(np.ones(Nr), angular_weights)
        # 
        vI = potential(x,y,z)
        # After spherical averaging only the l=0,m=0 component survives
        vI_sph = 1.0/(4*np.pi) * np.sum(vI*dOmega, axis=-1)
        
        # function for radial potential
        pot = interpolate.interp1d(r, vI_sph.real,
                                   kind='cubic', fill_value="extrapolate")
        radial_potentials.append(pot)
             
        # solve radial Schroedinger equation on an exponential grid
        # using Numerov's method
        atom_solver = AtomicPotential(charge)
        atom_solver.setRadialGrid(rho)
        atom_solver.setRadialPotential(vI_sph)

        # solve Schroedinger equation for each l component
        radial_wavefunctions.append( [] )
        phase_shifts.append( [] )

        for l in range(0, lmax+1):
            delta_l, u_l = atom_solver.scattering_state(energy, l)

            # spline radial part of wavefunction f^(i)_l
            radial_wfn_l = interpolate.interp1d(r, u_l/(k*r),
                                                kind='cubic', fill_value='extrapolate')
            phase_shifts[-1].append(delta_l)
            radial_wavefunctions[-1].append(radial_wfn_l)

    return radial_potentials, radial_wavefunctions, phase_shifts
        
def precalculated_regionI(atomlist, energy, radii, lmax=10, chargeIII=+1.0):
    """
    load precalculated atomic effective potentials and compute scattering wavefunctions
    in region I.

    The DFTB pseudoatoms are stored together with their effective atomic DFT potentials.
    These radial potentials for each atom type are loaded from files in `free_pseudo_atoms/`.
    Then the atomic scattering wavefunctions at the given `energy` are obtained by numerical
    integration. 

    Parameters
    ----------
    atomlist     : list of tuples (Zat,(x,y,z)) with the atomic
                   positions that define the multicenter grid
    energy       : float, energy E (in Hartree)
    radii        : array with radii of atomic spheres
    
    Optional
    --------
    lmax         : int, highest angular momentum
    chargeIII    : float > 0, asymptotic monopole charge of effective potential,
                   Veff(r) ----> -chargeIII/r  for r --> oo, typically chargeIII=+1 for
                   a neutral molecule. This is the charge felt by an electron in region III. 

    Returns
    -------
    radial_pots  : list of callables, radial_pots[i](r) evaluates the spherically
                   symmetric effective potential in region I around atom i,
                         (sph)
                        V   (r)
                         i 
    radial_wfns  : list of lists, radial_wfns[i][l] evaluates the 
                   radial wavefunction f^(i)_{l} of atom i in region I with 
                   angular momentum l. The total atomic wavefunction would be
                           (i)    (i)
                        psi    = f   (r) Y  (th,ph)
                           l,m    l       l,m 
    phase_shifts : list of lists, phase_shifts[i][l] is the phase-shift
                   of the partial wave of atom i with angular momentum l.    
    """
    # maximal deviation at matching point for Numerov method
    numerov_conv = 1.0e-6
    # threshold for convergence of SCF calculation (not needed)
    en_conv = 1.0e-5

    # default grid for numerical integration of the radial Schroedinger equation.
    Npts = 25000 # number of radial grid points
    rmin = 0.0
    rmax = 500.0 # in bohr

    # data for each atom type (key = atomic number Z)
    # radial potential
    rad_pots_Z = {}
    # radial wavefunctions 
    rad_wfns_Z = {}
    # phase shifts
    deltas_Z = {}

    # unique atom types 
    atomtypes = list(set([Zi for (Zi,posi) in atomlist]))
    atomtypes.sort()
    
    for Zi in atomtypes:
        print "load potential and continuum orbitals for atom type %s" % Zi
        # load pseudo atom data
        at = pseudo_atoms_list[Zi-1]
        Nelec = at.Z
        # prepare solver for radial Schroedinger equation
        atomdft = PseudoAtomDFT(at.Z, Nelec, numerov_conv,en_conv)
        atomdft.setRadialGrid(rmin,rmax, Npts)
        atomdft.initialDensityGuess((at.r, at.radial_density * Nelec/at.Z))
        atomdft.KS_pot.getRadialPotential()

        # interpolate radial potential
        # The first point r=0 is excluded because there the potential
        # diverges.
        pot = interpolate.interp1d(atomdft.KS_pot.getRadialGrid()[1:],
                                   atomdft.KS_pot.getRadialPotential()[1:],
                                   kind='cubic', fill_value="extrapolate")
        rad_pots_Z[Zi] = pot

        # compute scattering orbitals for each l-value 
        rad_wfns_Z[Zi] = []
        deltas_Z[Zi] = []

        # The scattering orbitals at a given energy are labeled by their
        # angular momentum quantum number l. l is truncated at lmax.
        for l in range(0, lmax+1): 
            delta_l, u_l = atomdft.KS_pot.scattering_state(energy,l)
            # create a function for evaluating the continuum orbitals, at large
            # distances the radial wavefunction coincides with a shifted Coulomb wave
            R_spl = spline_wavefunction(atomdft.getRadialGrid(), u_l,
                                        ug_asymptotic=CoulombWave(energy,chargeIII, l, delta_l))
            rad_wfns_Z[Zi].append( R_spl )
            deltas_Z[Zi].append( delta_l )
            
    # copy atom type data to atoms, atoms belonging to the same type are assigned the
    # same potentials and wavefunctions.
    
    # radial potentials, callables
    rad_pots = []
    # radial wavefunctions, callables
    rad_wfns = []
    # phase shifts, not really needed
    phase_shifts = []
        
    for i,(Zi,posi) in enumerate(atomlist):
        rad_pots.append( rad_pots_Z[Zi] )
        rad_wfns.append( rad_wfns_Z[Zi] )
        phase_shifts.append( deltas_Z[Zi] )
        
    return rad_pots, rad_wfns, phase_shifts

             
class MuffinTinPotential(object):
    def __init__(self, atomlist, lmax):
        """
        set up a muffin tin potential for a continuum multiple scattering calculation

        Parameters
        ----------
        atomlist  :  list of tuples (Zat,[x,y,z]) with atomic number and cartesian
                     coordinates (in bohr) for each atom
        lmax      :  int, highest angular momentum at which the spherical wave
                     expansion is truncated
        """
        self.atomlist = atomlist
        self.lmax = lmax
        # nuclear charge felt by an electron in region III
        self.chargeIII = +1
        # partition volume into region I, II and III
        self.rhoIII, self.radii = close_packed_spheres(atomlist)

    def load_regionI(self, energy):
        """
        radial potentials and wavefunctions for region III are obtained
        for DFTB pseudo atoms.

        Parameters
        ----------
        energy     : float, energy of continuum states (in Hartree)
        """
        assert energy > 0.0
        self.energy = energy
        
        self.rad_pots, self.rad_wfns, self.phase_shifts = precalculated_regionI(self.atomlist, self.energy, self.radii, lmax=self.lmax, chargeIII=self.chargeIII)

        # The constant potential in region II is obtained by averaging
        # the potentials of region I and III over the molecular and atomic spheres
        nat = len(self.atomlist)
        
        pot_avg = 0.0
        area_tot = 0.0
        for i in range(0, nat):
            rho_i = self.radii[i]
            # area of atomic sphere i
            area_i = 4.0 * np.pi * rho_i**2
            # potential I on the sphere
            potI_i = self.rad_pots[i](rho_i)

            pot_avg += area_i * potI_i
            area_tot += area_i

        # molecular sphere
        potIII = -self.chargeIII / self.rhoIII
        areaIII = 4.0 * np.pi * self.rhoIII**2

        pot_avg += potIII
        area_tot += areaIII

        # average over area
        pot_avg /= area_tot

        self.Vconst = pot_avg
        
        print "constant potential in region II  Vconst= %e" % self.Vconst    
            
    def solve_regionI(self, energy, potential):
        """
        The molecular potential is averaged spherically around each atom. 

        Parameters
        ----------
        energy     : float, energy of continuum states (in Hartree)
        potential  : callable, potential(x,y,z) evaluates the molecular potential
                     on a grid
        """
        self.potential = potential
        
        self.Vconst = averaged_regionII_potential(self.atomlist, self.rhoIII, self.radii, self.potential)
        print "constant potential in region II  Vconst= %e" % self.Vconst
        
        print "solve atomic Schroedinger equations ..."
        assert energy > 0.0
        self.energy = energy
        self.rad_pots, self.rad_wfns, self.phase_shifts = schroedinger_regionI(self.atomlist, self.potential, energy, lmax=self.lmax)

        ### DEBUG
        debug = 1
        if debug:
            # plot radial potentials and wavefunctions for each atom
            import matplotlib.pyplot as plt
            r = np.linspace(0.0, 3.0, 1000)
            for i in range(0, len(self.atomlist)):
                plt.plot(r, self.rad_pots[i](r), lw=2, label="$V_{%d}(r)$" % i)
                for l in range(0, self.lmax+1):
                    plt.plot(r, self.rad_wfns[i][l](r), ls="--", label="i=%d l=%d" % (i,l))
            plt.show()
        
        ###

    def prepare_data(self):
        """
        bring data for geometry, radial potentials and wavefunctions 
        into a format that can be passed to the Fortran extensions.
        """
        # number of angular momentum channels (l,m)
        self.nlm = (self.lmax+1)**2
        # The angular momentum quantum numbers (l,m) are enumerated in the order
        # (0,0),
        # (1,0), (1,+1), (1,-1),
        # (2,0), (2,+1), (2,-1), (2,+2), (2,-2),
        # ...., (lmax,-lmax)
        self.angmoms = []
        for l in range(0, self.lmax+1):
            for m in range(0, l+1):
                self.angmoms.append( (l,m) )
                if m > 0:
                    self.angmoms.append( (l,-m) )

        assert len(self.angmoms) == self.nlm

        # atomic positions
        self.nat = len(self.atomlist)
        self.atpos = np.zeros((3,self.nat))
        for i,(Zi,posi) in enumerate(self.atomlist):
            self.atpos[:,i] = posi

        # The radial wavefunction is computed on an exponential grid,
        # but the Fortran code can only deal with B-splines with equidistant
        # knots, so we interpolate the potentials and wavefunctions on an equidistant grid.
        # Only the values inside region I (r < rho_i) are needed.
        r = np.linspace(0.001, 1.5*self.radii.max(), 5000)

        # Find out the number of knots and the degree used by `splrep`
        t,c,k = interpolate.splrep(r,r)
        
        # number of knots
        self.nspl = len(t)
        # order of spline
        self.kspl = k

        # knots and coefficients of B-splines for atomic radial potentials
        self.pot_knots  = np.zeros((self.nspl, self.nat))
        self.pot_coeffs = np.zeros((self.nspl, self.nat))
        for i in range(0, self.nat):
            # create B-spline
            t,c,k = interpolate.splrep(r, self.rad_pots[i](r))
            assert k == self.kspl
            self.pot_knots[:,i] = t
            self.pot_coeffs[:,i] = c

        # knots and coefficients of B-splines for atomic radial wavefunctions
        # in region I
        self.fIknots = np.zeros((self.nspl, self.lmax+1, self.nat))
        self.fIcoeffs = np.zeros((self.nspl, self.lmax+1, self.nat))

        for i in range(0, self.nat):
            for l in range(0, self.lmax+1):
                # create B-spline
                t,c,k = interpolate.splrep(r, self.rad_wfns[i][l](r))
                self.fIknots[:,l,i] = t
                self.fIcoeffs[:,l,i] = c

        
    def potential(self, x,y,z):
        """
        evaluate the muffin tin potential on a grid
                         
          region I     -   atomic potentials, which are spherically symmetric
          region II    -   constant potential
          region III   -   Coulomb potential -1/r
                 
        Parameters
        ----------
        x,y,z      :  arrays with cartesian coordinates of grid points

        Returns
        -------
        pot        :  array with muffin tin potential at the grid points, V(x,y,z)
        """
        # call Fortran extension
        pot = cms.muffin_tin_potential(self.Vconst, self.chargeIII, self.rhoIII, self.radii,
                                       self.atpos, self.kspl, self.pot_knots, self.pot_coeffs,
                                       x,y,z)

        return pot
    
    def match_regions(self):
        """
        construct the matrix for the inhomogeneous system of linear equations
        (16) and (17) in Ref. [1] and solve 

            M.x = rhs

        for all right-hand sides.
        """
        # call Fortran extension
        # The (S) wavefunctions fulfill the boundary conditions
        #                    -1/2  -1                                   
        #   Psi      ~  (pi k)     r    sum   [ sin(th  ) delta     +  K     cos(th  ) ] Y
        #      III,L                0      L'         l'       L,L'     L,L'       l'     L'
        self.matchM, self.rhs = cms.matching_matrix(self.energy, self.Vconst, self.chargeIII, self.rhoIII, self.radii,
                                                    self.atpos, self.kspl, self.fIknots, self.fIcoeffs)

        # solve matching equation M.x = b
        self.sol = la.solve(self.matchM, self.rhs)

    def get_wavefunction(self, boundcond='standing', wavetype='spherical',
                         l=0,m=0, kth=0.0, kph=0.0):
        """
        find the wavefunction which has the asymptotic form in eqn. (20)

        Solutions are either labeled by the quantum numbers (l,m) of the partial
        wave expansion in region III (if wavetype is set to 'spherical') 
        or by the angles kth and kph of the propagation direction of the plane wave
        (if wavetype is set to 'plane').

        Parameters
        ----------

        Optional
        --------
        boundcond  :  impose boundary conditions on wavefunctions 
                      'standing' - see eqn. (20), wavefunction is real everywhere
                      'incoming' - incoming wave (-) normalization according to eqn. (26),
                                   wavefunction is complex
                      'outgoing' - outgoiing wave (+) normalization according to eqn. (25),
                                   wavefunction is complex
        wavetype   :  type of wavefunction
                      'spherical'  - The wavefuntions are labeled by the quantum numbers (`l`,`m`).
                      'plane'      - The partial waves are combined to form a plane wave with the 
                                     propagation direction determined by the angles `kth` and `kph`.
                      
        l,m        :  Asymptotic angular momentum in region III of continuum orbital for
                      'spherical' waves
        kth,kph    :  propagation direction of 'plane' wave, the spherical coordinates
                      of the wave vector are (k,kth,kph)


        Returns
        -------
        wfn        :  callable, wfn(x,y,z) evaluates the (l,m) wavefunction
                      on a grid
        """        
        def wave(x,y,z):
            # evaluate standing waves
            kmat, wfn = cms.wavefunctions(self.energy, self.Vconst, self.chargeIII, self.rhoIII, self.radii,
                                              self.atpos, self.kspl, self.fIknots, self.fIcoeffs, self.sol, 0,
                                              x,y,z)
            K = kmat.real
            Id = np.eye(K.shape[0])
            
            # check that K-matrix is real and symmetric
            err_im = la.norm(kmat.real)
            print "|Im(K^(S))|= %e" % err_im
            err_sym = la.norm(kmat - kmat.transpose())
            print "|K - K^T|= %e" % err_sym
            err_harm = la.norm(kmat - kmat.conjugate().transpose())
            print "|K - K^dagger|= %e" % err_harm
            # check that S-matrix is unitary
            # S (1 - i*K) = (1 + i*K)
            S = np.dot(Id+1j*K , la.inv(Id-1j*K))
            err_unitary = la.norm( Id - np.dot(S, S.conjugate().transpose()) )
            print "|S.S^dagger-Id|= %e" % err_unitary
            
            debug = 0
            if debug == 1:
                # plot K-matrix
                import matplotlib.pyplot as plt
                fig,axes = plt.subplots(1,2)
                axes[0].set_title("K-matrix (real part)")
                axes[0].imshow(kmat.real)
                axes[1].set_title("K-matrix (imag. part)")
                axes[1].imshow(kmat.imag)
                plt.show()
            
            # construct complex wavefunction from real wavefunctions according to
            # eqn. (28) and (29)
            if boundcond == "incoming":
                print "incoming wave (-) normalization"
                # exp(-i*k*r) = cos(kr) - i * sin(kr)
                # eqn. (29-)
                C = la.inv(Id + 1j*K)
                wfn = np.dot(C, wfn)
            elif boundcond == "outgoing":
                print "outgoing wave (+) normalization"
                # exp(+i*k*r) = cos(kr) + i * sin(kr)
                # eqn. (29+)
                C = -la.inv(Id - 1j*K)
                wfn = np.dot(C, wfn)
            elif boundcond == "standing":
                # Wavefunction
                print "standing wave"
            else:
                raise ValueError("Illegal value '%s' for option 'boundcond'" % boundcond)

            if wavetype == "spherical":
                # Which partial wave solution (l,m) is required?
                # lm is an index into the array of angular momentum channels (l,m)
                #  [(0,0), (1,0),(1,-1),(1,+1), ...]
                for lm1,(l1,m1) in enumerate(self.angmoms):
                    if l1 == l and m1 == m:
                        # (l,m) corresponds to multiindex lm
                        lm = lm1
                        break

                return wfn[lm,:]

            elif wavetype == "plane":
                nlm = len(self.angmoms)
                if boundcond == "outgoing":
                    # Partial waves are combined to have the asymptotic form of eqn. (31)
                    # a^(+)_L coefficients of eqn. (37) for rotatione matrix R=Id
                    A = np.zeros(nlm, dtype=complex)
                    for lm1,(l1,m1) in enumerate(self.angmoms):
                        if m1 == 0:
                            A[lm1] = -1j**l1 * np.sqrt((2*l1+1)/(4.0*np.pi))
                elif boundcond == "incoming":
                    # Partial waves are combined to have the asymptotic form of eqn. (32)
                    # a^(-)_L coefficients of eqn. (41) for rotatione matrix R=Id
                    k = np.sqrt(2*self.energy)
                    eta = -self.chargeIII/k

                    A = np.zeros(nlm, dtype=complex)
                    # spherical harmonics Y_{l,m}(kth,kph)
                    ysph = sphharm(kth,kph, self.lmax)
                    for lm1,(l1,m1) in enumerate(self.angmoms):
                        # Coulomb phase shift, sigma_l = arg Gamma(l+1+i*eta)
                        g = mpmath.gammainc(l1+1.0+1.0j*eta)
                        sigma_l1 = np.angle( complex(g) )
                        # 
                        A[lm1] = 1j**l1 * np.exp(-1j*sigma_l1) * ysph[lm1].conjugate()
                else:
                    raise ValueError("Options boundcond='%s' and wavetype='%s' are not compatible." % (boundcond, wavetype))
                
                # linear combination of partial waves
                psi = np.dot(A, wfn)

                return psi
            
            else:
                raise ValueError("Illegal value '%s' for option 'wavetype'" % wavetype)        
                    
        def wavefunction(x,y,z):
            # The coordinate arrays may have any shape, but the
            # fortran code can only deal with rank 1 arrays
            shape = x.shape
            x,y,z = x.flatten(), y.flatten(), z.flatten()
            
            wfn = wave(x,y,z)
            
            # Convert wavefunction array to the same shape as
            # the input arrays
            wfn = np.reshape(wfn, shape)
            return wfn
        
        return wavefunction

    def set_bound_orbitals(self, orbitals):
        """
        The bound orbitals, which are the initial states in the photoionization
        process are evaluated on a multicenter Becke grid. The resolution
        of the radial and angular grid is set in `MolecularIntegrals.settings`.

        Parameters
        ----------
        orbitals   :  list of `norb` callables, orbitals[b](x,y,z) should evaluate the b-th bound
                      (or Dyson) orbital on a grid

        Returns
        -------
        grid       :  tuple of numpy arrays (x,y,z, w) with cartesian coordinates and weights
                      for numerical quadrature
        orbs       :  numpy array of shape (norb,len(x)) with values of bound orbitals on the grid
        """
        # spherical Becke grids around each atom
        points, weights, volumes = multicenter_grids(self.atomlist,
                                                     radial_grid_factor=settings.radial_grid_factor,
                                                     lebedev_order=settings.lebedev_order)
        # combine multicenter grids into a single grid
        # The quadrature rule for integration on this grid reads:
        #   /
        #   | f(x,y,z) dV = sum  w  f(x ,y ,z ) 
        #   /                  i  i    i  i  i
        x,y,z, w = join_grids(points, weights, volumes)
        # number of grid points
        npts = len(x)
        
        # evaluate all bound orbitals on the grid
        # number of orbitals
        self.norb = len(orbitals)
        orbs = np.zeros((self.norb, npts))
        print "evaluating bound orbitals..."
        for b,orbital in enumerate(orbitals):
            orbs[b,:] = orbital(x,y,z)

        grid = (x,y,z, w)
        
        return grid, orbs
            
    def transition_dipoles(self, grid, orbs):
        """
        compute transition dipoles matrix elements between bound orbitals (index by `b`)
        and the CMS continuum wavefunctions (labeled by the asymptotic angular momentum `(l,m)`)

            ( TDx )                 ( x )
            ( TDy )   = <Psi (k)  | ( y ) | Psi >
            ( TDz )         l,m     ( z )      b

        which are needed to evaluate eqn. (59).

        The integration is done numerically using the multicenter Becke grid generated in
        `set_bound_orbitals()`.

        Parameters
        ----------
        grid       :  tuple of numpy arrays (x,y,z, w) with cartesian coordinates and weights
                      for numerical quadrature
        orbs       :  numpy array of shape (norb,len(x)) with values of bound orbitals on the grid

        Returns
        -------
        tdip       :  transition dipoles between bound and free orbitals,
                      tdip[lm,b,:] is the cartesian transition dipole vector between the bound orbital with index `b`
                      and the K-matrix normalized continuum orbital with asymptotic momentum `lm`.
        norms2     :  numpy array with norms**2 of bound orbitals, for checking if the resolution
                      of the multicenter grid is sufficient
        """
        print "computing transition amplitudes by numerical integration..."
        x,y,z, w = grid
        # call Fortran extension for computing wavefunctions and
        # summing over grid points
        kmat, tdip, norms2, projs2 = cms.transition_dipoles(self.energy, self.Vconst, self.chargeIII, self.rhoIII, self.radii,
                                                            self.atpos, self.kspl, self.fIknots, self.fIcoeffs, self.sol, 0,
                                                            x,y,z, w,
                                                            orbs)

        print ""
        print "The bound orbitals should be normalized and approximately orthogonal"
        print "to all continuum orbitals, otherwise the transition dipoles depend"
        print "on the choice the origin."
        print ""
        print " Orbital            Norm              Proj. onto Continuum"
        print "    b         |<Psi_b|Psi_b>|^2     sum_lm |<Psi_lm|Psi_b>|^2"
        print " ------------------------------------------------------------"
        for b,(nrm,prj) in enumerate(zip(norms2,projs2)):
            print "  %3.1d           %e           %e" % (b+1,nrm,prj)
        print ""

        # save K-matrix and cartesian transition dipoles
        self.kmat = kmat
        self.tdip = tdip

        return tdip, norms2
        
    def photoelectron_distribution(self, pol=0):
        """
        photoelectron angular distribution for isotropically oriented
        emsemble of molecules characterized by the parameters

          sigma (total photoionization cross section),
          beta1 (only non-zero for chiral molecules) and
          beta2 (anisotropy parameter)

        Optional
        --------
        pol      : polarization of light, 0 (linear), -1 (left), +1 (right)

        Returns
        -------
        pad      : numpy array of shape (norb,3) with PAD parameters
                   sigma,beta1,beta2 = pad[b,:] for each bound orbital b                
        """
        # convert cartesian transition amplitudes to 
        tdip_in = cms.transform_tdip_in(self.energy, self.chargeIII, self.kmat.real, self.tdip.real, self.lmax)
        # The array returned by `transform_tdip_in` has the shape (nlm,norb,3),
        # but `pad_iso` expects an array of shape (norb,3,nlm).
        tdip_in = np.moveaxis(tdip_in, 0, -1)
        print "photoelectron angular distribution ..."
        pad = photo.pad_iso(self.energy, pol, tdip_in, self.lmax)
        
        print ""
        print "     Photoionization cross section (isotropic averaging)      "
        print "                                                              "
        print "  dsigma   sigma                                              "
        print "  ------ = ----- [1 + beta  P (cos(th)) + beta  P (cos(th)) ] "
        print "  d Omega  4 pi           1  1                2  2            "
        print "                                                              "
        pol2str = {0 : "0 (linear)", -1 : "-1 (left)", +1: "+1 (right)"}
        print "  polarization = %s" % pol2str[pol]
        print "  photokinetic energy = %e Hartree ( %e eV )" % (self.energy, self.energy * AtomicData.hartree_to_eV)
        print " "
        print "  Orbital      sigma             beta1             beta2       "
        print "  -------------------------------------------------------------"
        for b in range(0, self.norb):
            sigma, beta1, beta2 = pad[b,:]
            print "    %3.1d       %e      %e      %e" % (b+1, sigma, beta1, beta2)
        print ""
        return pad
        
##################################################
#
# Testing
#
##################################################

def test_transform_waves_in():
    """
    compare Psi^(-) computed with Fortran and python
    """
    # experimental geometry of water
    #  r(OH) = 0.958 Ang, angle(H-O-H) = 104.4776 degrees
    atomlist = [
        (8, (0.000000000000000,  0.000000000000000, -0.222540557483415)),
        (1, (0.000000000000000, +1.431214118579765,  0.886071388908105)),
        (1, (0.000000000000000, -1.431214118579765,  0.886071388908105))]

    charge = 0
    # highest angular momentum l of partial waves
    lmax = 10
    # energy of continuum orbital
    energy = 1.7 #5.1 #1.7 #0.77

    muffin = MuffinTinPotential(atomlist, lmax)

    muffin.load_regionI(energy)
    muffin.prepare_data()
    muffin.match_regions()

    # plot wavefunction in the xy-plane
    import matplotlib.pyplot as plt

    xx,yy = np.mgrid[-5:5:100j,-5:5:100j]
    zz = 0*xx
    
    x = xx.ravel()
    y = yy.ravel()
    z = zz.ravel()

    wfn_in = muffin.get_wavefunction(boundcond='incoming', wavetype='plane')
    wfn_in_xyz = wfn_in(x,y,z)

    # evaluate standing waves
    kmat, wfn_xyz = cms.wavefunctions(muffin.energy, muffin.Vconst, muffin.chargeIII, muffin.rhoIII, muffin.radii,
                                      muffin.atpos, muffin.kspl, muffin.fIknots, muffin.fIcoeffs, muffin.sol, 0,
                                      x,y,z)            
    # compute psi^(-) with Fortran
    psi_fortran = cms.transform_waves_in(muffin.energy, muffin.chargeIII, kmat.real, wfn_xyz.real, muffin.lmax)
    # spherical harmonics Y_{l,m}(kth,kph)
    ysph = sphharm(0.0, 0.0, muffin.lmax)
    wfn_in_xyz_fortran = np.dot(ysph.conjugate(), psi_fortran)

    err = la.norm(wfn_in_xyz - wfn_in_xyz_fortran)
    print wfn_in_xyz_fortran / wfn_in_xyz
    print "|Psi^(-) (python) - Psi^(-) (Fortran)|= %e" % err
    
    assert err < 1.0e-9

    
def cms_pads(atomlist,
             pol=0,
             potential_type='atomic superposition',
             lmax=10,
             energy_range_eV=(0.1,40.0,40),
             charge=0):
    """
    compute photoelectron angular distributions (PADs) for isotropic ensemble 
    as a function of kinetic energy of the photoelectron using the CMS method

    Optional
    --------
    pol             :  polarization of light, 0 (linear), -1 (left), +1 (right)
    potential_type  :  How should the spherically symmetric potential in region I 
                       be determined? "atomic superposition" uses atomic DFT potentials
                       while "molecular" spherically averages the molecular DFT potential
                       around each atom
    lmax            :  int,  highest angular momentum l of partial waves
    energy_range_eV :  The range of photokinetic energies is specified by the tuple
                       (pke_min, pke_max, nr.points), pke_min and pke_max are in eV.
    charge          :  charge of the molecule, should be 0

    Returns
    -------
    pke             :  array of length `npke` with photoelectron kinetic energies in Hartree
    pad             :  numpy array of shape (npke,norb,3), pad[k,b,:] contains the three parameters
                       sigma,beta1 and beta2 for ionization from orbital `b` into the continuum
                       at energy pke[k].
    """
    # shift molecule to center of mass
    pos = XYZ.atomlist2vector(atomlist)
    masses = AtomicData.atomlist2masses(atomlist)
    pos_shifted = MolCo.shift_to_com(pos, masses)
    atomlist = XYZ.vector2atomlist(pos_shifted, atomlist)

    muffin = MuffinTinPotential(atomlist, lmax)

    # choose resolution of multicenter grids
    settings.radial_grid_factor = 10      # controls size of radial grid 
    settings.lebedev_order = 23          # controls size of angular grid
    # XC-functional
    xc = XCFunctionals.libXCFunctional('gga_x_pbe', 'gga_c_pbe')
    
    dft = BasissetFreeDFT(atomlist, xc, charge=0)
    # bound orbitals
    orbitals = dft.getOrbitalGuess()
    norb = len(orbitals)
    grid, orbs = muffin.set_bound_orbitals(orbitals)
    
    #
    if potential_type == "molecular":
        rho = density_func(orbitals)
        potential = effective_potential_func(atomlist, rho, xc)
    
    # photoelectron kinetic energy (in Hartree)
    pke = np.linspace(*energy_range_eV) / AtomicData.hartree_to_eV
    # number of energies
    npke = len(pke)
    # compute orientation-averaged PAD for each energy
    pad = np.zeros((npke,norb,3))
    for i,energy in enumerate(pke):
        print "%d of %d   PKE = %e Hartree ( %e eV )" % (i+1,npke, energy,energy * AtomicData.hartree_to_eV)
        if potential_type == "molecular":
            muffin.solve_regionI(energy, potential)
        else:
            muffin.load_regionI(energy)
        muffin.prepare_data()
        muffin.match_regions()
        muffin.transition_dipoles(grid, orbs)
        pad[i,:,:] = muffin.photoelectron_distribution(pol=pol)

    plot_pads(pke, pad, pol)
    save_pads(pke, pad, pol, "/tmp/pad.dat")

    return pke,pad

def save_pads(pke,pad, pol, tbl_file, units="eV-Mb"):
    """
    A table with the PAD is written to `tbl_file`. 
    It contains the 4 columns   PKE   SIGMA  BETA1   BETA_2
    which define the PAD(th) at each energy according to
                                   
      PAD(th) = SIMGA/(4pi) [ 1 + BETA  P (cos(th)) + BETA  P (cos(th)) ]
                                      1  1                2  2

    For each orbital a block separated by a newline is written.
    """
    npke,norb,dummy = pad.shape
    sigma = pad[:,:,0]
    beta1 = pad[:,:,1]
    beta2 = pad[:,:,2]

    fh = open(tbl_file, "w")
    
    pol2str = {0 : "0 (linear)", -1 : "-1 (left)", +1: "+1 (right)"}

    print>>fh,"""
#
# photoelectron angular distributions (PAD) for an isotropic ensemble
#
#  PAD(th) = SIMGA/(4pi) [ 1 + BETA  P (cos(th)) + BETA  P (cos(th)) ]
#                                  1  1                2  2
#
# light polarization = %s
#
    """ % pol2str[pol]
    
    if units == "eV-Mb":
        pke = pke * AtomicData.hartree_to_eV
        # convert cross section sigma from bohr^2 to Mb
        sigma = sigma * AtomicData.bohr2_to_megabarn

        header = "# PKE/eV        SIGMA/Mb        BETA1        BETA2"
    else:
        header = "# PKE/Hartree   SIGMA/bohr^2    BETA1        BETA2"
    
    for b in range(0, norb):
        print>>fh, "# photoionization from orbital %d" % (b+1)
        print>>fh, header
        block = np.vstack((pke, sigma[:,b], beta1[:,b], beta2[:,b])).transpose()
        np.savetxt(fh, block, fmt=["%e","%e","%+10.7f", "%+10.7f"])
        print>>fh, ""

    print "PAD written to '%s'" % tbl_file
    fh.close()

def plot_pads(pke,pad, pol, units='eV-Mb'):
    """
    plot PAD parameters sigma, beta1 and beta2 as functions of PKE
    for different orbitals
    """
    npke,norb,dummy = pad.shape
    sigma = pad[:,:,0]
    beta1 = pad[:,:,1]
    beta2 = pad[:,:,2]
    
    import matplotlib
    matplotlib.rc('xtick', labelsize=17)
    matplotlib.rc('ytick', labelsize=17)
    matplotlib.rc('legend', fontsize=17)
    matplotlib.rc('axes', labelsize=17)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1,3)
    plt.title("polarization = %s" % pol)
    
    if units == "eV-Mb":
        pke = pke * AtomicData.hartree_to_eV
        # convert cross section sigma from bohr^2 to Mb
        sigma = sigma * AtomicData.bohr2_to_megabarn

        for ax in [0,1,2]:
            axes[ax].set_xlabel("PKE / eV")
        axes[0].set_ylabel(r"$\sigma$ / Mb")
    else:
        for ax in [0,1,2]:
            axes[ax].set_xlabel("PKE / Hartree")
        axes[0].set_ylabel(r"$\sigma$ / bohr$^2$")
        
    axes[1].set_ylabel(r"$\beta_1$")
    axes[2].set_ylabel(r"$\beta_2$")
    
    axes[2].set_ylim((-1.1,2.1))
    
    for b in range(0, norb):
        l, = axes[0].plot(pke, sigma[:,b], lw=2, label=r"Orb. %d" % (b+1))
        axes[1].plot(pke, beta1[:,b], lw=2, color=l.get_color())
        axes[2].plot(pke, beta2[:,b], lw=2, color=l.get_color())

    axes[0].legend()

    plt.subplots_adjust(wspace=0.5)
    
    plt.show()
    
    
if __name__ == "__main__":
    #test_transform_waves_in()
    
    from DFTB.MolecularIntegrals.BasissetFreeDFT import residual_func
    from DFTB import XYZ

    
    import sys
    import warnings
    warnings.filterwarnings("error")

    atomlist = XYZ.read_xyz(sys.argv[1])[0]

    ###
    cms_pads(atomlist, energy_range_eV=(0.1, 80.0, 240),
             potential_type="atomic superposition",
             lmax=20)
    #cms_pads(atomlist)
    ###
    
    # shift molecule to center of mass
    pos = XYZ.atomlist2vector(atomlist)
    masses = AtomicData.atomlist2masses(atomlist)
    pos_shifted = MolCo.shift_to_com(pos, masses)
    atomlist = XYZ.vector2atomlist(pos_shifted, atomlist)

    charge = 0
    # highest angular momentum l of partial waves
    lmax = 10
    # energy of continuum orbital
    energy = 0.07 #1.7 #5.1 #1.7 #0.77

    muffin = MuffinTinPotential(atomlist, lmax)

    #
    # 
    # 
    potential_type = "molecular" #"atomic superposition" # "molecular"
    if potential_type == "molecular":
        # choose resolution of multicenter grids
        settings.radial_grid_factor = 10      # controls size of radial grid 
        settings.lebedev_order = 23          # controls size of angular grid
        # XC-functional
        xc = XCFunctionals.libXCFunctional('gga_x_pbe', 'gga_c_pbe')
        
        dft = BasissetFreeDFT(atomlist, xc, charge=0)
        orbitals = dft.getOrbitalGuess()

        rho = density_func(orbitals)
        potential = effective_potential_func(atomlist, rho, xc)

        muffin.solve_regionI(energy, potential)

    else:
        muffin.load_regionI(energy)

    muffin.prepare_data()
    muffin.match_regions()
    
    # solution with asymptotic angular momentum (l,m)
    l,m = (0,0)
    wfn = muffin.get_wavefunction(boundcond='standing', l=l,m=m)

    """
    # ------- Residual ---------------------------------
    # The residual, R(x,y,z) = (H-E) wfn(x,y,z), should
    # be zero everywhere if the solution is correct.

    # choose resolution of multicenter grids
    settings.radial_grid_factor = 20     # controls size of radial grid 
    settings.lebedev_order = 23          # controls size of angular grid

    print "computing residual (H-E)wfn ..."
    print type(wfn)
    residual = residual_func(atomlist, wfn, muffin.potential, energy)
    #-----------------------------------------------------
    """
    
    # bound orbitals
    # XC-functional
    xc = XCFunctionals.libXCFunctional('gga_x_pbe', 'gga_c_pbe')
    dft = BasissetFreeDFT(atomlist, xc, charge=0)
    orbitals = dft.getOrbitalGuess()

    grid, orbs = muffin.set_bound_orbitals(orbitals)
    muffin.transition_dipoles(grid, orbs)
    muffin.photoelectron_distribution(pol=0)

    # plot wavefunction in the xy-plane
    import matplotlib.pyplot as plt

    xx,yy = np.mgrid[-5:5:100j,-5:5:100j]
    zz = 0*xx
    
    x = xx.ravel()
    y = yy.ravel()
    z = zz.ravel()
    
    wfn_xyz = np.reshape(wfn(x,y,z).real, xx.shape)

    extent = [x.min(), x.max(), y.min(), y.max()]
    plt.imshow(wfn_xyz, extent=extent)

    # show atom labels
    for Zi,posi in atomlist:
        plt.text(posi[0], posi[1], AtomicData.atom_names[Zi-1],
                 horizontalalignment='center',
                 verticalalignment='center')

    plt.show()
    
    # plot of CMS wavefunction and potential along one axis
    r = np.linspace(-5.0, 5.0, 1000)
    x = 0*r
    y = 0*r
    z = r

    plt.plot(r, muffin.potential(x,y,z), label=r"$V^{muffin}$")
    plt.plot(r, wfn(x,y,z).real, label=r"$\Psi_{%d,%d}$" % (l,m))
    plt.plot(r, residual(x,y,z).real, label=r"residual $(T+V^{muffin}-E)\Psi_{%d,%d}$" % (l,m))
    plt.legend()
    plt.show()

    
    
