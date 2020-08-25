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
from DFTB.AtomicData import atom_names, slater_radii

from DFTB.MolecularIntegrals.LebedevQuadrature import outerN, spherical_harmonics_it

from DFTB.MolecularIntegrals.Ints1e import integral
from DFTB.MolecularIntegrals import settings
from DFTB.MolecularIntegrals.MulticenterIntegration import select_angular_grid
from DFTB.MolecularIntegrals.SphericalCoords import cartesian2spherical
from DFTB.SlaterKoster import XCFunctionals
from DFTB.Scattering import Wigner

from DFTB.SlaterKoster.RadialPotential import CoulombExpGrid


import numpy as np
import numpy.linalg as la
from scipy import interpolate
from scipy import special
from scipy import optimize
import mpmath             # needed for coulombf() and coulombg()

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
    radial_pots_tck 
                 : list of tuples (t,c,k) with uniformly spaced knots t, coefficients c
                   and degree k of spline for interpolating radial potentials,
                   (ti,ci,ki) = radial_pots_tck[i] for atom i
    radial_pots  : list of callables, radial_pots[i](r) evaluates the spherically
                   symmetric effective potential in region I around atom i,
                         (sph)
                        V   (r)
                         i 
    radial_wfns_tck 
                 : list of tuples (t,c,k) with uniformly spaced knots t, coefficients c
                   and degree k of spline for interpolating radial wavefunctions,
                   (til,cil,kil) = radial_pots_tck[i][l] for partial wave of atom i with
                   angular momentum l.
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
    # angular grid for averaging the potential spherically around
    # each atom
    Lmax, (th,ph,angular_weights) = select_angular_grid(settings.lebedev_order)
    Nang = len(th)
    sc = np.sin(th)*np.cos(ph)
    ss = np.sin(th)*np.sin(ph)
    c  = np.cos(th)

    # knots, coefficients and degrees of B-splines for radial potentials
    radial_potentials_tck = []
    # radial potentials, callables
    radial_potentials = []
    # knots, coefficients and degrees of B-splines for radial wavefunctions
    radial_wavefunctions_tck = []
    # radial wavefunctions, callables
    radial_wavefunctions = []
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

        # spline radial potential
        spline_00 = interpolate.splrep(r, vI_sph.real, s=None)
        radial_potentials_tck.append(spline_00)
        
        # function for evaluating spline
        pot_00 = interpolate.interp1d(r, vI_sph.real,
                                      kind='cubic', fill_value="extrapolate")
        radial_potentials.append(pot_00)
             
        # solve radial Schroedinger equation on an exponential grid
        # using Numerov's method
        atom_solver = AtomicPotential(charge)
        atom_solver.setRadialGrid(rho)
        atom_solver.setRadialPotential(vI_sph)

        # solve Schroedinger equation for each l component
        radial_wavefunctions_tck.append( [] )
        radial_wavefunctions.append( [] )
        phase_shifts.append( [] )

        for l in range(0, lmax+1):
            delta_l, u_l = atom_solver.scattering_state(energy, l)

            # spline wavefunction
            tck = interpolate.splrep(r, u_l/(k*r), s=None)
            # spline radial part of wavefunction f^(i)_l
            radial_wfn_l = interpolate.interp1d(r, u_l/(k*r),
                                                kind='cubic', fill_value='extrapolate')
            phase_shifts[-1].append(delta_l)
            radial_wavefunctions_tck[-1].append( tck )
            radial_wavefunctions[-1].append(radial_wfn_l)

    return radial_potentials_tck, radial_potentials, radial_wavefunctions_tck, radial_wavefunctions, phase_shifts


from DFTB.SlaterKoster.free_pseudo_atoms import pseudo_atoms_list
from DFTB.SlaterKoster.SKIntegrals import spline_wavefunction
from DFTB.SlaterKoster.PseudoAtomDFT import PseudoAtomDFT
from DFTB.Scattering.SlakoScattering import CoulombWave
        
def precalculated_regionI(atomlist, energy, radii, lmax=10, chargeIII=+1):
    """
    load precalculated atomic effective potentials and compute scattering wavefunctions
    """
    # maximal deviation at matching point for Numerov method
    numerov_conv = 1.0e-6
    # threshold for convergence of SCF calculation
    en_conv = 1.0e-5

    # default grid
    Npts = 25000 # number of radial grid points
    rmin = 0.0
    rmax = 500.0

    # data for each atom type (key = atomic number)
    rad_pots_Z = {}
    rad_pots_tck_Z = {}
    rad_wfns_tck_Z = {}
    rad_wfns_Z = {}
    deltas_Z = {}
    
    atomtypes = list(set([Zi for (Zi,posi) in atomlist]))
    atomtypes.sort()
    for Zi in atomtypes:
        print "load potential and continuum orbitals for atom type %s" % Zi
        # load pseudo atoms
        at = pseudo_atoms_list[Zi-1]
        Nelec = at.Z
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
        # spline potential on equidistant grid
        r = np.linspace(0.001, 1.5*radii.max(), 5000)

        """
        ### DEBUG
        import matplotlib.pyplot as plt
        plt.plot(r, pot(r))
        plt.plot(atomdft.KS_pot.getRadialGrid(),
                                   atomdft.KS_pot.getRadialPotential())
        plt.show()
        ###
        """
        
        pot_tck = interpolate.splrep(r, pot(r))
        
        rad_pots_Z[Zi] = pot
        rad_pots_tck_Z[Zi] = pot_tck

        rad_wfns_tck_Z[Zi] = []
        rad_wfns_Z[Zi] = []
        deltas_Z[Zi] = []
        
        # compute scattering orbitals
        for l in range(0, lmax+1): # each atom has an s-, 3 p-, 5-d,  ... scattering orbitals
            delta_l, u_l = atomdft.KS_pot.scattering_state(energy,l)
            R_spl = spline_wavefunction(atomdft.getRadialGrid(), u_l,
                                        ug_asymptotic=CoulombWave(energy,chargeIII, l, delta_l))
            tck = interpolate.splrep(r, R_spl(r))
            rad_wfns_tck_Z[Zi].append( tck )
            
            rad_wfns_Z[Zi].append( R_spl )
            deltas_Z[Zi].append( delta_l )
            
    # copy atom type date to atoms
    
    # knots, coefficients and degrees of B-splines for radial potentials
    radial_potentials_tck = []
    # radial potentials, callables
    radial_potentials = []
    # knots, coefficients and degrees of B-splines for radial wavefunctions
    radial_wavefunctions_tck = []
    # radial wavefunctions, callables
    radial_wavefunctions = []
    phase_shifts = []
        
    for i,(Zi,posi) in enumerate(atomlist):
        radial_potentials_tck.append( rad_pots_tck_Z[Zi] )
        radial_potentials.append( rad_pots_Z[Zi] )
        radial_wavefunctions_tck.append( rad_wfns_tck_Z[Zi] )
        radial_wavefunctions.append( rad_wfns_Z[Zi] )
        phase_shifts.append( deltas_Z[Zi] )
        
    return radial_potentials_tck, radial_potentials, radial_wavefunctions_tck, radial_wavefunctions, phase_shifts

             
class MuffinTinPotential(object):
    def __init__(self, atomlist, lmax):
        """
        """
        self.atomlist = atomlist
        self.lmax = lmax
        # nuclear charge felt by an electron in region III
        self.chargeIII = +1
        # partition volume into region I, II and III
        self.rhoIII, self.radii = close_packed_spheres(atomlist)

    def load_regionI(self, energy):
        assert energy > 0.0
        self.energy = energy
        
        self.radial_potentials_tck, self.radial_potentials, self.radial_wavefunctions_tck, self.radial_wavefunctions, self.phase_shifts = precalculated_regionI(self.atomlist, self.energy, self.radii, lmax=self.lmax, chargeIII=self.chargeIII)

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
            potI_i = self.radial_potentials[i](rho_i)

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
        
        """
        self.potential = potential
        
        self.Vconst = averaged_regionII_potential(self.atomlist, self.rhoIII, self.radii, self.potential)
        print "constant potential in region II  Vconst= %e" % self.Vconst
        
        print "solve atomic Schroedinger equations ..."
        assert energy > 0.0
        self.energy = energy
        self.radial_potentials_tck, self.radial_potentials, self.radial_wavefunctions_tck, self.radial_wavefunctions, self.phase_shifts = schroedinger_regionI(self.atomlist, self.potential, energy, lmax=self.lmax)

        ### DEBUG
        debug = 1
        if debug:
            # plot radial potentials and wavefunctions for each atom
            import matplotlib.pyplot as plt
            r = np.linspace(0.0, 3.0, 1000)
            for i in range(0, len(self.atomlist)):
                plt.plot(r, self.radial_potentials[i](r), lw=2, label="$V_{%d}(r)$" % i)
                for l in range(0, self.lmax+1):
                    plt.plot(r, self.radial_wavefunctions[i][l](r), ls="--", label="i=%d l=%d" % (i,l))
            plt.show()
        
        ###

    def potential(self, x,y,z):
        """
        evaluate the muffin tin potential on a grid
                         
          region I     -   atomic potentials, which are spherically symmetric
          region II    -   constant potential
          region III   -   Coulomb potential -1/r
                            
        """
        # region II is everwhere outside regions I and III
        pot = self.Vconst * np.ones(x.shape)

        # region I
        for i,(Zat,posi) in enumerate(self.atomlist):
            # distance to center i
            xi,yi,zi = posi
            r = np.sqrt((x-xi)**2+(y-yi)**2+(z-zi)**2)
            pot[r < self.radii[i]] = self.radial_potentials[i](r[r < self.radii[i]])
 
        # region III
        r = np.sqrt(x**2+y**2+z**2)
        pot[self.rhoIII < r] = -1.0/r[self.rhoIII < r]

        return pot

    def potential_fortran(self, x,y,z):
        from DFTB.MolecularIntegrals import cms
        # atomic positions
        nat = len(self.atomlist)
        atpos = np.zeros((3,nat))
        for i,(Zi,posi) in enumerate(self.atomlist):
            atpos[:,i] = posi

        t,c,k = self.radial_potentials_tck[0]
        # number of nodes
        nspl = len(t)
        # order of spline
        kspl = k

        # knots and coefficients of B-splines for atomic radial potentials
        knots = np.zeros((nspl, nat))
        coeffs = np.zeros((nspl, nat))
        for i in range(0, nat):
            t,c,k = self.radial_potentials_tck[i]
            knots[:,i] = t
            coeffs[:,i] = c
            
        pot = cms.muffin_tin_potential(self.Vconst, self.chargeIII, self.rhoIII, self.radii, atpos, kspl, knots, coeffs, x,y,z)

        return pot
    
    def matching_matrix(self):
        """
        construct the matrix for the inhomogeneous system of linear equations
        (16) and (17) in Ref. [1]
        """
        import warnings
        warnings.filterwarnings("error")
        
        print "build matching matrix..."
        # E = 1/2 k^2
        k = np.sqrt(2.0*self.energy)
        # wavevector in region II
        kappa = np.sqrt(k**2 - 2*self.Vconst)
        print "k     = %e" % k
        print "kappa = %e" % kappa
        
        # radial wavefunctions in region I
        fI = self.radial_wavefunctions
        
        # radial wavefunctions in region II
        jII = lambda l,r : special.spherical_jn(l, kappa * r)
        nII = lambda l,r : special.spherical_yn(l, kappa * r)
        
        # radial wavefunctions in region III (around origin)
        # The functions evaluate fIII(k*r) and gIII(k*r)
        fIII, gIII = coulomb_func_factory(self.chargeIII, k)

        # number of angular momentum channels (l,m)
        Nlm = (self.lmax+1)**2
        # number of atomic centers
        Nat = len(self.atomlist)

        Ndim = Nlm*(Nat+1)

        # matrix defining inhomogeneous system of linear equations
        #  M.x = b
        matchM = np.zeros((Ndim,Ndim), dtype=complex)

        # intermediate variables needed for constructing M
        print "Wronskians..."
        # ratios of Wronskians
        wrQ = np.zeros((Nat,self.lmax+1))
        #          [n_l(kappa*rho_i), f_l^i(rho_j)]
        # wrQ    = --------------------------------
        #    i,l   [j_l(kappa*rho_i), f_l^i(rho_j)]
        for i in range(0, Nat):
            # radius around region I of atom i
            rho_i = self.radii[i]
            for l in range(0, self.lmax+1):
                wrQ[i,l] =   wronskian(lambda r: nII(l,r), lambda r: fI[i][l](r), rho_i) \
                           / wronskian(lambda r: jII(l,r), lambda r: fI[i][l](r), rho_i)

        #        [j_l(kappa*rhoIII), g_l^III(k*rhoIII)]
        # wrR  = --------------------------------------
        #    l   [n_l(kappa*rhoIII), g_l^III(k*rhoIII)]
        #
        #        [f_l^III(k*rhoIII), g_l^III(k*rhoIII)]
        # wrS  = --------------------------------------
        #    l   [n_l(kappa*rhoIII), g_l^III(k*rhoIII)]
        wrR = np.zeros(self.lmax+1)
        wrS = np.zeros(self.lmax+1)
        for l in range(0, self.lmax+1):
            wrR[l] =   wronskian(lambda r: jII(l,r) , lambda r: gIII(l,r), self.rhoIII) \
                     / wronskian(lambda r: nII(l,r) , lambda r: gIII(l,r), self.rhoIII)
            wrS[l] =   wronskian(lambda r: fIII(l,r), lambda r: gIII(l,r), self.rhoIII) \
                     / wronskian(lambda r: nII(l,r) , lambda r: gIII(l,r), self.rhoIII)
        # wrS is needed for constructing RHS of matching equation, so keep a copy
        self.wrS = wrS
            
        # To find the coefficients A^{I_i}_{l,m} and B^{III}_{l,m} from
        # the solution of the matching conditions according to eqns.
        # (18) and (19) we need to store some additional Wronskians
        #
        #                                 i
        # wronskJF    = [j (kappa*rho ), f (rho )]
        #         i,l     l          i    l    i
        #
        self.wronskJF = np.zeros((Nat,self.lmax+1))
        for i in range(0, Nat):
            # radius around region I of atom i
            rho_i = self.radii[i]
            for l in range(0, self.lmax+1):
                self.wronskJF[i,l] = wronskian(lambda r: jII(l,r), lambda r: fI[i][l](r), rho_i)
        #
        #                                 III
        # wronskNG  = [n (kappa*rhoIII), g (kappa*rhoIII)]
        #         l     l                 l
        #        
        #                                 III
        # wronskNF  = [n (kappa*rhoIII), f (kappa*rhoIII)]
        #         l     l                 l
        #
        self.wronskNG = np.zeros(self.lmax+1)
        self.wronskNF = np.zeros(self.lmax+1)
        for l in range(0, self.lmax+1):
            self.wronskNG[l] = wronskian(lambda r: nII(l,r) , lambda r: gIII(l,r), self.rhoIII)
            self.wronskNF[l] = wronskian(lambda r: nII(l,r) , lambda r: fIII(l,r), self.rhoIII)
            
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

        assert len(self.angmoms) == Nlm

        print "matching matrix..."
        # Fill in elements of matching matrix
        # Rows and columns are enumerated by multiindices ii and jj,
        # which stand for (i,li,mi) and (j,lj,mj). 
        
        # rows
        ii = 0
        # enumerate atomic centers (i=0 means the origin)
        for i in range(0, Nat+1):
            # enumerate angular momentum channels (li,mi) on center i
            for li,mi in self.angmoms:

                print "filling row %d (of %d)" % (ii+1, Ndim)
                # columns
                jj = 0
                # enumerate atomic centers (j=0 means the origin)
                for j in range(0, Nat+1):
                    # enumerate angular momentum channels (lj,mj) on center j
                    for lj,mj in self.angmoms:


                        if ii == jj:
                            # diagonal elements of matrix
                            if i == 0:
                                # (i=0,j=0) diagonal block, origin
                                matchM[ii,jj] = wrR[li]
                            else:
                                # (i==j,i>0) diagonal block, atomic centers
                                # Atomic indices start at zero, but in the matrix
                                # i=0 is the origin and i=1 refers to the first atomic center.
                                matchM[ii,jj] = wrQ[i-1,li]
                        else:
                            # off-diagonal elemenets
                            if i == 0 and j > 0:
                                # vector from origin to atom j (j is a 1-based index)
                                vec_0j = np.array(self.atomlist[j-1][1])
                                matchM[ii,jj] = J_coeffs(lj,mj, li,mi, -kappa*vec_0j)
                            elif j == 0 and i > 0:
                                # vector from origin to atom i (i is a 1-based index)
                                vec_0i = np.array(self.atomlist[i-1][1])
                                matchM[ii,jj] = J_coeffs(lj,mj, li,mi, kappa*vec_0i)
                            elif i > 0 and j > 0 and i != j:
                                # positions of atom i and j (i and j are 1-based indices)
                                posi = np.array(self.atomlist[i-1][1])
                                posj = np.array(self.atomlist[j-1][1])
                                # vector from atom j to atom i
                                vec_ji = posi - posj
                                matchM[ii,jj] = N_coeffs(lj,mj, li,mi, kappa*vec_ji)
                                
                        # increase column counter
                        jj += 1

                # increase row counter
                ii += 1

        print "matching matrix"
        print matchM

        # save matching matrix
        self.k = k
        self.kappa = kappa
        
        self.Nat = Nat
        self.Nlm = Nlm
        self.Ndim = Ndim
        self.matchM = matchM
        
        return matchM

    def matching_matrix_fortran(self):
        """
        compute matching matrix using Fortran implementation
        """
        from DFTB.MolecularIntegrals import cms

        nat = len(self.atomlist)
        # 
        t,c,k = self.radial_wavefunctions_tck[0][0]
        # number of nodes
        nspl = len(t)
        # order of spline
        kspl = k

        # knots and coefficients of B-splines for atomic radial wavefunctions
        fIknots = np.zeros((nspl, self.lmax+1, nat))
        fIcoeffs = np.zeros((nspl, self.lmax+1, nat))
        for i in range(0, nat):
            for l1 in range(0, self.lmax+1):
                t,c,k = self.radial_wavefunctions_tck[i][l1]
                fIknots[:,l1,i] = t
                fIcoeffs[:,l1,i] = c
                
        # atomic positions
        nat = len(self.atomlist)
        atpos = np.zeros((3,nat))
        for i,(Zi,posi) in enumerate(self.atomlist):
            atpos[:,i] = posi

            
        matchM,rhs = cms.matching_matrix(self.energy, self.Vconst, self.chargeIII, self.rhoIII, self.radii, atpos, kspl, fIknots, fIcoeffs)

        print "matching matrix (Fortran)"
        print matchM

        print "rhs l,m = (0,0) (fortran)"
        print rhs[:,0]
        
        return matchM, rhs
        
    def solve_matching_conditions(self, l,m):
        """
        find the wavefunction which has the asymptotic form in eqn. (20)
        """
        print "solve matching conditions for l=%d m=%d" % (l,m)
        # construct the right-hand side of the matching equation
        AIII = np.zeros(self.Nlm)
        rhs = np.zeros(self.Ndim)
        for lm1,(l1,m1) in enumerate(self.angmoms):
            if l == l1 and m == m1:
                # eqn. (23)
                AIII[lm1] = np.sqrt(self.k/np.pi)
                rhs[lm1] = AIII[lm1] * self.wrS[l]

        print "rhs l,m = (%d,%d)" % (l,m)
        print rhs
        
        # solve matching equation M.x = b
        x = la.solve(self.matchM, rhs)

        # extract coefficients A^{II_0}_{l,m} and B^{II_i}_{l,m}
        AII0 = x[:self.Nlm]
        BII = np.reshape(x[self.Nlm:], (self.Nat, self.Nlm))
        # find coefficients A^{I_i}_{l,m} according to eqn. (18)
        AI = np.zeros((self.Nat, self.Nlm), dtype=complex)
        for i in range(0, self.Nat):
            rho_i = self.radii[i]
            for lm1,(l1,m1) in enumerate(self.angmoms):
                AI[i,lm1] = BII[i,lm1] / (self.kappa * rho_i**2 * self.wronskJF[i,l1])
        # find coefficients B^{III}_{l,m} according to eqn. (19)
        BIII = np.zeros(self.Nlm, dtype=complex)
        for lm1,(l1,m1) in enumerate(self.angmoms):
            BIII[lm1] = - ( AII0[lm1] / (self.kappa * self.rhoIII**2) + self.wronskNF[l1] * AIII[lm1] ) \
               / self.wronskNG[l1]

        # L=(l,m) row of the K-matrix
        K_row_lm = np.sqrt(np.pi/self.k) * BIII

        # wavefunction
        wfn_lm = self.build_wavefunction(AI, AII0, BII, AIII, BIII)

        return K_row_lm, wfn_lm

    def solve_matching_conditions_fortran(self, l,m):
        from DFTB.MolecularIntegrals import cms
        
        self.matchM, rhs = self.matching_matrix_fortran()
        # solve matching equation M.x = b
        sol = la.solve(self.matchM, rhs)

        # atomic positions
        nat = len(self.atomlist)
        atpos = np.zeros((3,nat))
        for i,(Zi,posi) in enumerate(self.atomlist):
            atpos[:,i] = posi

        # 
        t,c,k = self.radial_wavefunctions_tck[0][0]
        # number of nodes
        nspl = len(t)
        # order of spline
        kspl = k

        # knots and coefficients of B-splines for atomic radial wavefunctions
        fIknots = np.zeros((nspl, self.lmax+1, nat))
        fIcoeffs = np.zeros((nspl, self.lmax+1, nat))
        for i in range(0, nat):
            for l1 in range(0, self.lmax+1):
                t,c,k = self.radial_wavefunctions_tck[i][l1]
                fIknots[:,l1,i] = t
                fIcoeffs[:,l1,i] = c

        # Which solution is required?
        for lm1,(l1,m1) in enumerate(self.angmoms):
            if l1 == l and m1 == m:
                lm = lm1
                break
        
        def wavefunction(x,y,z):
            kmat, wfn = cms.wavefunctions(self.energy, self.Vconst, self.chargeIII, self.rhoIII, self.radii,
                                          atpos, kspl, fIknots, fIcoeffs, sol, 0,
                                          x,y,z)
            #
            assert lm == 0
            return wfn[lm,:]

        return wavefunction
    
    def build_wavefunction(self, AI, AII0, BII, AIII, BIII):
        """
        construct a function for evaluating the wavefunction
        """
        # radial wavefunctions in region II
        jII = lambda l,r : special.spherical_jn(l, self.kappa * r)
        nII = lambda l,r : special.spherical_yn(l, self.kappa * r)
        # radial wavefunctions in region III (around origin)
        # The functions evaluate fIII(k*r) and gIII(k*r)
        fIII, gIII = coulomb_func_factory(self.chargeIII, self.k)

        def wavefunction(x,y,z):

            # region II    
            wfnII = 0j*x
            r0,th0,ph0 = cartesian2spherical((x,y,z))
            sph_it = spherical_harmonics_it(th0,ph0)
            for lm,(Ylm,l,m) in enumerate(sph_it):
                # first half of eqn. (7)
                wfnII += AII0[lm] * jII(l,r0) * Ylm
                
                if m == -self.lmax:
                    break

                
            for i,(Zat,posi) in enumerate(self.atomlist):
                # coordinates relative to center i 
                xi = x - posi[0]
                yi = y - posi[1]
                zi = z - posi[2]
                # convert them to spherical coordinates
                ri,thi,phi = cartesian2spherical((xi,yi,zi))

                sph_it = spherical_harmonics_it(thi,phi)
                for lm,(Ylm,l,m) in enumerate(sph_it):
                    # second half of eqn. (7)
                    wfnII += BII[i,lm] * nII(l,ri) * Ylm

                    if m == -self.lmax:
                        break
                    
            wfn = wfnII

            # region III
            if len(r0[r0 > self.rhoIII]) > 0:
                rIII  = r0[ r0 > self.rhoIII]
                thIII = th0[r0 > self.rhoIII]
                phIII = ph0[r0 > self.rhoIII]
                wfnIII = 0j*rIII
                sph_it = spherical_harmonics_it(thIII,phIII)
                for lm,(Ylm,l,m) in enumerate(sph_it):
                    # eqn. (8)
                    wfnIII = wfnIII + (AIII[lm] * fIII(l,rIII) + BIII[lm] * gIII(l,rIII)) * Ylm
                    if m == -self.lmax:
                        break

                wfn[r0 > self.rhoIII] = wfnIII
            else:
                print "no points inside region III"
                
            # region I
            for i,(Zat,posi) in enumerate(self.atomlist):
                # distance to atom i
                ri = np.sqrt((x-posi[0])**2+(y-posi[1])**2+(z-posi[2])**2)
                # radius around region I of atom i
                rho_i = self.radii[i]

                if len(ri[ri < rho_i]) == 0:
                    print "no points inside region I of atom %d" % i
                    continue
                
                # coordinates relative to center i of points
                # which lie inside region I_i
                xIi = x[ri < rho_i] - posi[0]
                yIi = y[ri < rho_i] - posi[1]
                zIi = z[ri < rho_i] - posi[2]
                # convert them to spherical coordinates
                rIi,thIi,phIi = cartesian2spherical((xIi,yIi,zIi))
                # wavefunction in region I_i
                wfnIi = 0j*rIi
                sph_it = spherical_harmonics_it(thIi,phIi)
                for lm,(Ylm,l,m) in enumerate(sph_it):

                    wfnIi += AI[i,lm] * self.radial_wavefunctions[i][l](rIi) * Ylm
                    if m == -self.lmax:
                        break
                    
                wfn[ri < rho_i] = wfnIi

            return wfn
        
        return wavefunction
            
def wronskian(func1, func2, x0, h=1.0e-5):
    """
    compute the Wronskian of two functions f1(x) and f2(x)
    numerically at the x=x0

                        df2         df1
       [f1,f2] = f1(x0) ---(x0)  -  ---(x0) f2(x0)
                        dx          dx

               = (f1 * f2' - f1' * f2)(x0)

    The derivatives df1/dx and df2/dx are approximated by finite
    differences [1] with a step size of dx=h.

    Parameters
    ----------
    func1, func2  :  callables, func1(x) and func2(x) should 
                     by able to operate on a numpy array `x`
    x0            :  float

    Optional
    --------
    h             :  small float, step size for numerical differentiation

    References
    ----------
    [1] W. Bickley, "Formulae for Numerical Differentiation",
        The Mathematical Gazette, vol. 25, no. 263, pp. 19-27 (1941)
    """
    # displacements
    x = x0 + np.array([-2.0, -1.0, 0.0, 1.0, 2.0])*h
    # Bickley's coefficients for n=4, m=1, p=2
    # taken from Ref. [1]
    # This is a 5-term central difference approximation for
    # the 1st derivative.
    D1 = np.array([2.0, -16.0, 0.0, 16.0, -2.0])/(24.0*h)

    # evaluate function f1 at sample points
    y1 = func1(x)
    # finite difference approximation for f1'(x0)
    df1dx = np.dot(D1,y1)
    # evaluate function f2 at sample points
    y2 = func2(x)
    # finite difference approximation for f2'(x0)
    df2dx = np.dot(D1,y2)

    # f1(x0)
    f1 = y1[2]
    # f2(x0)
    f2 = y2[2]
    
    # Wronskian
    w = f1 * df2dx - df1dx * f2

    return w

def cart2sph_scalar(x,y,z):
    """
    convert cartesian to spherical coordinates by inverting 
    the equations

         x = r*sin(th)*cos(ph)
         y = r*sin(th)*sin(ph)
         z = r*cos(th)

    Parameters
    ----------
    x,y,z      : 3 floats, cartesian coordinates

    Returns
    -------
    r,th,phi   : 3 floats, spherical coordinates
    """
    r = np.sqrt(x*x+y*y+z*z)
    
    # theta
    if r > 0.0:
        th = np.arccos(z/r)
    else:
        # for r=0 the angle theta is not well defined, choose th = 0
        th = 0.0

    # phi
    if x != 0.0:
        if y != 0.0:
            phi = np.arctan2(y,x)
            if phi < 0.0:
                #translate angles from the range [-pi, pi] to the range [0, 2.0*pi]
                phi = 2.0*np.pi+phi
        else:
            # if y == 0, phi=0 for x positive and phi=pi for x negative
            phi = (1.0-np.sign(x))/2.0*np.pi
    else:
        # if x == 0, phi = pi/2 if y is positive and phi=3/2*pi if y is negative
        phi = np.pi/2.0 + (1.0-np.sign(y))/2.0*np.pi

    return r,th,phi
    

def J_coeffs(l1,m1, l2,m2, vec,
             lmax=10):
    """
    coefficients for expanding spherical Bessel functions of
    first and second kind around different center. The new center
    differs j from the old center i by the shift vector R_ij.

    J-part of eqn. 14 in Dill & Dehmer (1974) (Ref. [1])

    Parameters
    ----------
    l1,m1      : angular momentum quantum numbers, 0 <= l1, -l1 <= m1 <= +l1
    l2,m2      : angular momentum quantum numbers, 0 <= l2, -l2 <= m1 <= +l2
    vec        : cartesian shift vector R_ij, vec = [x,y,z]
    
    Optional
    --------
    lmax       : highest value of l included in the summation

    Returns
    -------
    J          : complex, coefficient J^{ij}_{L1,L2} in the notation
                 of Ref. [1]
    """
    # spherical coordinates (r,theta,phi)
    # of cartesian vector vec = (x,y,z)
    x,y,z = vec
    r,th,ph = cart2sph_scalar(x,y,z)
    
    J = 0.0j
    sph_it = spherical_harmonics_it(th,ph)
    for Ylm,l,m in sph_it:
        J_lm = 1.0j**(l+l2-l1) * (-1)**m1 \
               * np.sqrt(4.0*np.pi*(2*l+1)*(2*l1+1)*(2*l2+1)) \
               * Wigner.Wigner3J(l,0, l2,0,  l1,0) \
               * Wigner.Wigner3J(l,m, l2,m2, l1,-m1) \
               * Ylm * special.spherical_jn(l,r)
        J += J_lm
        
        if m == -lmax:
            break

    return J

def N_coeffs(l1,m1, l2,m2, vec,
             lmax=10):
    """
    same as `J_coeffs(...)` with the difference that the spherical
    Bessel function of the first kind j_l is replaced by that of
    the second kind n_l
    """
    # spherical coordinates (r,theta,phi)
    # of cartesian vector vec = (x,y,z)
    x,y,z = vec
    r,th,ph = cart2sph_scalar(x,y,z)

    N = 0.0j    
    sph_it = spherical_harmonics_it(th,ph)
    for Ylm,l,m in sph_it:
        N_lm = 1.0j**(l+l2-l1) * (-1)**m1 \
               * np.sqrt(4.0*np.pi*(2*l+1)*(2*l1+1)*(2*l2+1)) \
               * Wigner.Wigner3J(l,0, l2,0,  l1,0) \
               * Wigner.Wigner3J(l,m, l2,m2, l1,-m1) \
               * Ylm * special.spherical_yn(l,r)
        N += N_lm
        
        if m == -lmax:
            break

    return N

def coulomb_func_factory(Z, k):
    """
    create regular and irregular functions f^(III)_l(r0) 
    and g^(III)_l(r0) for region III, where the wavefunction
    has the following form (see eqn. (8) in Ref. [1]):

                       III   III           III  III
      Psi    = sum  [ A     f  (k*r0)  +  B    g  (k*r0) ] Y (th0,ph0)
         III      L    L     l             L    l           L

    Parameters
    ----------
    Z          : int > 0, asymptotic charge of molecule,
                 Even if the molecule is neutral, the asymptotic charge should 
                 be set to Z=1, since the effective Kohn-Sham potential has the 
                 limit 

                       V  (r)  -------->   - 1/r
                        eff      r->oo

                 due to the exchange part of the functional.
    k          : float, length of wave vector

    Returns
    -------
    f          : callable, the universal function f(l,r) evaluates f^(III)_l(k*r)
    g          : callable, the universal function g(l,r) evaluates g^(III)_l(k*r)
    """
    assert Z > 0
    assert k > 0
    
    def f(l, r):
        # regular Coulomb function
        #  III
        # f  (k*r) = 1/(k*r) F (eta=-Z/k; k*r)
        #  l                  l
        fl = 1.0/(k*r) * mpmath.coulombf(l, -Z/k, k*r)
        # convert from mpc to float
        fl = float(complex(fl).real)
        return fl
        
    def g(l, r):
        # irregular Coulomb function
        #  III
        # g  (k*r) = 1/(k*r) G (eta=-Z/k; k*r)
        #  l                  l
        gl = 1.0/(k*r) * mpmath.coulombg(l, -Z/k, k*r)
        # convert from mpc to float
        gl = float(complex(gl).real)
        return gl

    # universal functions that can operator on numpy arrays
    # element-wise
    f_ufunc = np.frompyfunc(f, 2, 1)
    g_ufunc = np.frompyfunc(g, 2, 1)
    
    return f_ufunc, g_ufunc

def radial_wfn_derivs(radial_wavefunctions, radii, h=1.0e-5):
    """
    evaluate radial wavefunctions and their radial derivatives
    on the atomic sphere at r=rho_i
    """
    nat = len(radial_wavefunctions)
    lmax = len(radial_wavefunctions[0])-1

    #  (i)
    # f  (r=rho )
    #  l       i
    fI = np.zeros((lmax+1,nat))
    #  '(i)
    # f   (r=rho )
    #   l       i
    fIp = np.zeros((lmax+1,nat))

    for i in range(0, nat):
        # radius of atomic sphere i
        rho_i = radii[i]
        for l in range(0, lmax+1):
            # radial wavefunction 
            func = radial_wavefunctions[i][l]
            # evaluate f^(i)_l(r) at r=rho_i
            fI[l,i] = func(rho_i)
            # evaluate radial derivative d(f^(i)_l(r))/dr at r=rho_io
            # displacements
            r = rho_i + np.array([-2.0, -1.0, 0.0, 1.0, 2.0])*h
            # Bickley's coefficients for n=4, m=1, p=2
            # taken from Ref. [1]
            # This is a 5-term central difference approximation for
            # the 1st derivative.
            D1 = np.array([2.0, -16.0, 0.0, 16.0, -2.0])/(24.0*h)

            # evaluate radial function at sample points
            fr = func(r)
            # finite difference approximation for f1'(x0)
            fIp[l,i] = np.dot(D1,fr)

    return fI, fIp

##################################################
#
# Testing
#
##################################################

def test_partitioning():
    """
    water molecule is partitioned into region I,II and III
    """
    # experimental geometry of water
    #  r(OH) = 0.958 Ang, angle(H-O-H) = 104.4776 degrees
    atomlist = [
        (8, (0.000000000000000,  0.000000000000000, -0.222540557483415)),
        (1, (0.000000000000000, +1.431214118579765,  0.886071388908105)),
        (1, (0.000000000000000, -1.431214118579765,  0.886071388908105))]

    rhoIII, radii = close_packed_spheres(atomlist, debug=1)


def test_muffin_tin_potential_h2():
    import time
    
    # hydrogen molecule
    atomlist = [(1, (0.0, 0.0, -0.7)),
                (1, (0.0, 0.0, +0.7))]
    charge = 0    
    # highest angular momentum l of partial waves
    lmax = 5
    # energy of continuum orbital
    energy = 0.77

    muffin = MuffinTinPotential(atomlist, lmax)

    from DFTB.MolecularIntegrals.BasissetFreeDFT import BasissetFreeDFT, effective_potential_func, density_func

    # XC-functional
    xc = XCFunctionals.libXCFunctional('gga_x_pbe', 'gga_c_pbe')
    
    # choose resolution of multicenter grids
    settings.radial_grid_factor = 10      # controls size of radial grid 
    settings.lebedev_order = 23          # controls size of angular grid
    
    dft = BasissetFreeDFT(atomlist, xc, charge=0)
    orbitals = dft.getOrbitalGuess()

    rho = density_func(orbitals)
    potential = effective_potential_func(atomlist, rho, xc)

    """
    # compute atomic wavefunctions in region I
    muffin.solve_regionI(energy, potential)
    """

    # load precalculated atomic potentials (the same as used in DFTB)
    muffin.load_regionI(energy)

    # compare muffin tin potential with the true molecular potential
    print "compare muffin tin potential with true molecular potential"
    # compare Fortran and python implementations if muffin tin potential
    r = np.linspace(-5.0, +5.0, 10000)
    x = 0*r
    y = 0*r+0.2
    z = r
    Vpy   = muffin.potential_fortran(x,y,z)
    Vfort = muffin.potential(x,y,z)
    err = la.norm(Vpy-Vfort)

    import matplotlib.pyplot as plt
    plt.plot(r, Vpy, label="V(python)")
    plt.plot(r, Vfort, ls="-.", label="V(fortran)")
    plt.legend()
    plt.show()
    print "|V(python)-V(fortran)|= %e" % err
    
    
    # save tables with cuts along the z-axis of V, V^(rec) and U
    r = np.linspace(-5.0, +5.0, 10000)
    dat_file = "/tmp/h2_muffin_tin_potential.dat"
    fh = open(dat_file, "w")
    # y-positions of cuts along z-axis, x=0 in all cuts
    for y in [-0.222540557483415, 0.0, 0.886071388908105]:
        data = np.vstack(( r,
                           potential(0*r,0*r+y,r),
                           muffin.potential(   0*r,0*r+y,r),
                         )).transpose()
        print>>fh, "# cut through potential along the z-axis at x=0, y=%e" % y
        print>>fh, "# "
        print>>fh, "# R/bohr    V/Hartree       V^(muffin)/Hartree"
        np.savetxt(fh, data, fmt="%+10.8e")
        print>>fh, " "
    print "tables with V and V^(muffin) written to '%s'" % dat_file
    fh.close()

    # matching matrix

    # with Fortran
    ta = time.time()
    matchM_fortran, rhs = muffin.matching_matrix_fortran()
    tb = time.time()
    print "computation of matching matrix (Fortran) took %s seconds" % (tb-ta)

    # with python
    ta = time.time()
    matchM_python = muffin.matching_matrix()
    tb = time.time()
    print "computation of matching matrix (python) took %s seconds" % (tb-ta)

    err = la.norm(matchM_python - matchM_fortran)
    print "|M(python) - M(fortran)|= %e" % err

    import matplotlib.pyplot as plt
    # relative error
    rel_error = abs(matchM_python - matchM_fortran)
    rel_error[rel_error > 1.0e-4] = rel_error[rel_error > 1.0e-4] / abs(matchM_python[rel_error > 1.0e-4])
    plt.imshow(rel_error.transpose())
    plt.colorbar()
    plt.show()
    
    # s-wave
    K_row_00, wfn_00 = muffin.solve_matching_conditions(0,0)
    # compare with fortran code
    wfn_00_fortran = muffin.solve_matching_conditions_fortran(0,0)

    r = np.linspace(-5.0, +5.0, 500)
    x = 0*r
    y = 0*r+0.4
    z = r

    ta = time.time()
    plt.plot(r, wfn_00_fortran(x,y,z).real, ls="-", label="Fortran")
    tb = time.time()
    print "computation of wavefunction (Fortran) took %s seconds" % (tb-ta)
    ta = time.time()
    plt.plot(r, wfn_00(x,y,z).real, ls="-.", label="python")
    tb = time.time()
    print "computation of wavefunction (python) took %s seconds" % (tb-ta)
    plt.legend()
    plt.show()
    
    
    print "cuts through wavefunction..."
    # save tables with cuts through the s-wave continuum orbital
    r = np.linspace(-5.0, +5.0, 10000)
    dat_file = "/tmp/h2_continuum_orbital_s.dat"
    fh = open(dat_file, "w")
    # y-positions of cuts along z-axis, x=0 in all cuts
    for y in [-0.222540557483415, 0.0, 0.886071388908105]:
        wfn_r = wfn_00(0*r,0*r+y,r)
        data = np.vstack(( r,
                           muffin.potential(0*r,0*r+y,r),
                           wfn_r.real, wfn_r.imag
                         )).transpose()
        print>>fh, "# cut through potential along the z-axis at x=0, y=%e" % y
        print>>fh, "# "
        print>>fh, "# R/bohr       V^(muffin)/Hartree       Re(Psi_00)       Im(Psi_00)"
        np.savetxt(fh, data, fmt="%+10.8e")
        print>>fh, " "
    print "tables with V and wavefunction written to '%s'" % dat_file
    fh.close()


def test_muffin_tin_potential_h2_interactive():
    # hydrogen molecule
    atomlist = [(1, (0.0, 0.0, -0.7)),
                (1, (0.0, 0.0, +0.7))]
    charge = 0    
    # XC-functional
    xc = XCFunctionals.libXCFunctional('gga_x_pbe', 'gga_c_pbe')
    
    # choose resolution of multicenter grids
    settings.radial_grid_factor = 10      # controls size of radial grid 
    settings.lebedev_order = 23          # controls size of angular grid
    # highest angular momentum l of partial waves
    lmax = 10
    
    from DFTB.MolecularIntegrals.BasissetFreeDFT import BasissetFreeDFT, effective_potential_func, density_func
    
    dft = BasissetFreeDFT(atomlist, xc, charge=0)
    orbitals = dft.getOrbitalGuess()

    rho = density_func(orbitals)
    potential = effective_potential_func(atomlist, rho, xc)

    energy = 0.5

    muffin = MuffinTinPotential(atomlist, lmax)
    # compute atomic wavefunctions in region I
    muffin.solve_regionI(energy, potential)

    return muffin

    
def test_muffin_tin_potential_water():
    # experimental geometry of water
    #  r(OH) = 0.958 Ang, angle(H-O-H) = 104.4776 degrees
    atomlist = [
        (8, (0.000000000000000,  0.000000000000000, -0.222540557483415)),
        (1, (0.000000000000000, +1.431214118579765,  0.886071388908105)),
        (1, (0.000000000000000, -1.431214118579765,  0.886071388908105))]
    charge = 0    
    # XC-functional
    xc = XCFunctionals.libXCFunctional('gga_x_pbe', 'gga_c_pbe')
    
    # choose resolution of multicenter grids
    settings.radial_grid_factor = 10      # controls size of radial grid 
    settings.lebedev_order = 23          # controls size of angular grid
    # highest angular momentum l of partial waves
    lmax = 5
    
    from DFTB.MolecularIntegrals.BasissetFreeDFT import BasissetFreeDFT, effective_potential_func, density_func
    
    dft = BasissetFreeDFT(atomlist, xc, charge=0)
    orbitals = dft.getOrbitalGuess()

    rho = density_func(orbitals)
    potential = effective_potential_func(atomlist, rho, xc)

    energy = 0.5

    muffin = MuffinTinPotential(atomlist, lmax)
    # compute atomic wavefunctions in region I
    muffin.solve_regionI(energy, potential)
    
    # compare muffin tin potential with the true molecular potential
    
    # save tables with cuts along the y-axis of V, V^(rec) and U
    r = np.linspace(-5.0, +5.0, 10000)
    dat_file = "/tmp/water_muffin_tin_potential.dat"
    fh = open(dat_file, "w")
    # z-positions of cuts along y-axis, x=0 in all cuts
    for z in [-0.222540557483415, 0.0, 0.886071388908105]:
        data = np.vstack(( r,
                           potential(0*r,r,0*r+z),
                           muffin.potential(0*r,r,0*r+z),
                         )).transpose()
        print>>fh, "# cut through potential along the y-axis at x=0, z=%e" % z
        print>>fh, "# "
        print>>fh, "# R/bohr    V/Hartree       V^(muffin)/Hartree"
        np.savetxt(fh, data, fmt="%+10.8e")
        print>>fh, " "
    print "tables with V and V^(muffin) written to '%s'" % dat_file
    fh.close()
    
    muffin.matching_matrix()
    # s-wave
    K_row_00, wfn_00 = muffin.solve_matching_conditions(0,0)

    print "cuts through wavefunction..."
    # save tables with cuts through the s-wave continuum orbital
    r = np.linspace(-5.0, +5.0, 10000)
    dat_file = "/tmp/water_continuum_orbital_s.dat"
    fh = open(dat_file, "w")
    # z-positions of cuts along y-axis, x=0 in all cuts
    for z in [-0.222540557483415, 0.0, 0.886071388908105]:
        wfn_r = wfn_00(0*r,r,0*r+z)
        data = np.vstack(( r,
                           muffin.potential(0*r,r,0*r+z),
                           wfn_r.real, wfn_r.imag
                         )).transpose()
        print>>fh, "# cut through potential along the y-axis at x=0, z=%e" % z
        print>>fh, "# "
        print>>fh, "# R/bohr       V^(muffin)/Hartree       Re(Psi_00)       Im(Psi_00)"
        np.savetxt(fh, data, fmt="%+10.8e")
        print>>fh, " "
    print "tables with V and wavefunction written to '%s'" % dat_file
    fh.close()
    
def spherical_hankel(l, x, kind=1):
    """
    compute spherical Hankel functions h^(1) and h^(2) by inverting
    the relation

                   (1)    (2)
         j = 1/2 (h    + h   )
          l        l      l
                      (1)     (2)
         n = -1/2 i (h    -  h   )
          l           l       l

    """
    assert kind in [1,2]
    j = special.spherical_jn(l, x)
    n = special.spherical_yn(l, x)
    if kind == 1:
        h1 = j + 1.0j*n
        return h1
    else:
        h2 = j - 1.0j*n
        return h2
        
        
def test_hankel_functions():
    """
    check relations between spherical Bessel functions of 1st and 2nd kind
    and Hankel functions of 1st and 2nd kind
    """
    x = 2.0 * (np.random.rand(10) -0.5)

    for l in range(0, 1):
        j = special.spherical_jn(l, x)
        n = special.spherical_yn(l, x)
        h1 = special.hankel1(l, x)
        h2 = special.hankel2(l, x)

        print j
        print 0.5*(h1+h2)
        
        err_j = la.norm( j -  0.5*(h1+h2) )
        err_n = la.norm( n +  0.5j*(h1-h2) )

        assert err_j < 1.0e-10
        assert err_n < 1.0e-10
        

def test_wronskian_trigonometric():
    """
    check the relation

       [sin(x), cos(x)] = -1

    numerically
    """
    xs = 10.0 * 2.0*(np.random.rand(10) - 0.5)

    for x0 in xs:
        w0 = wronskian(np.sin, np.cos, x0)
        assert abs(w0 + 1.0) < 1.0e-10

        
def test_wronskian_spherical_bessel():
    """
    spherical Bessel functions of the first and second kind,
    j_l and n_l, are supposed to have the Wronskian

                          -2
        [j (z), n (z)] = z
          l      l

    This relation is checked numerically by computing the Wronskian
    at a number of randomly selected points by finite differences.
    """
    # positive random values
    zs = 10.0 * np.random.rand(10)

    # check the relation for all angular momenta up to lmax
    lmax = 10
    for l in range(0, lmax+1):
        def jl(z):
            return special.spherical_jn(l,z)
        def nl(z):
            return special.spherical_yn(l,z)
        
        for z0 in zs:
            # left hand side of equation, [jl,nl]
            lhs = wronskian(jl, nl, z0)
            # right hand side
            rhs = z0**(-2)

            #print "lhs = %s" % lhs
            #print "rhs = %s" % rhs
            err = abs(lhs-rhs)
            assert err < 1.0e-10, "error = %e" % err

    # check analytical derivatives of Bessel functions
    from DFTB.MolecularIntegrals import coul90
    for z0 in zs:
        jl,nl,jlp,nlp = coul90.coul90_scalar(z0, 0.0, lmax, 1)
        # left hand side, [jl,nl] = jl*nl' - jl'*nl
        lhs = jl*nlp - jlp*nl
        # right hand side
        rhs = z0**(-2)

        #print "lhs = %s" % lhs
        #print "rhs = %s" % rhs
        err = la.norm(lhs-rhs)
        #print err
        assert err < 1.0e-10, "error = %e" % err
            
def test_spherical_bessel_shift():
    """
    r_i are the cartesian coordinates of a point P relative
    to the atomic center i, r_j are the coordinates relative to
    different center j, the vector R_ij points from center i to
    center j.

    We want to convert a spherical wave expansion around center i
    into an expansion around center j. 

    The spherical Bessel function of the first kind j_l transforms
    as

      j (|r_i|) Y (th_i,ph_i) = sum      J(l,m,l',m',R_ij) Y  (th_j,ph_j) j (|r_j|)       (1)
       l         l,m               l',m'                    l',m'          l' 


    For the spherical Bessel function of the second kind, n_l, t
    wo cases have to be distinguished, depending on the ratio |r_j|/|R_ij|

    Case 1:     |r_j| < |R_ij|

      n (|r_i|) Y (th_i,ph_i) = sum      N(l,m,l',m',R_ij) Y  (th_j,ph_j) j (|r_j|)       (2)
       l         l,m               l',m'                    l',m'          l'

    Case 2:     |r_j| > |R_ij|

      n (|r_i|) Y (th_i,ph_i) = sum      J(l,m,l',m',R_ij) Y  (th_j,ph_j) n (|r_j|)       (3)
       l         l,m               l',m'                    l',m'          l'
    """
    # reproduciple sequence of random numbers
    #np.random.seed(seed=10000)
    np.random.seed(seed=30000)
    
    # position vector of point P
    r = 10.0 * 2.0*(np.random.rand(3)-0.5)
    # center i
    posi = 5.0 * 2.0*(np.random.rand(3)-0.5)
    # center j
    posj = 5.0 * 2.0*(np.random.rand(3)-0.5)
    # shift vector R_ij
    shift_ij = posj-posi
    
    # coordinates of point P relative to center i
    veci = r-posi
    # and relative to center j
    vecj = r-posj

    # convert cartesian to spherical coordinates
    r_i,th_i,ph_i = cart2sph_scalar(*veci)
    r_j,th_j,ph_j = cart2sph_scalar(*vecj)

    print "|r_j| < |R_ij|  ?"
    print r_j < la.norm(shift_ij)
    
    # Now we verify eqns. (1),(2) and (3) by computing the left
    # and right hand sides for angular momenta l=0,...,lmax
    lmax = 40

    # spherical harmonics around center i
    sph_it_i = spherical_harmonics_it(th_i,ph_i)
    for Ylm_i,l_i,m_i in sph_it_i:
        print "l_i = %d  m_i = %d" % (l_i, m_i)

        print "checking eqn. (1)  -  transformation of spherical Bessel of first kind"
        # left hand side of eqn. (1)
        lhs = special.spherical_jn(l_i,r_i) * Ylm_i

        # right hand side of eqn. (1)
        rhs = 0.0j
        # spherical harmonics around center j
        sph_it_j = spherical_harmonics_it(th_j,ph_j)
        for Ylm_j,l_j,m_j in sph_it_j:
            rhs += J_coeffs(l_i,m_i, l_j,m_j, shift_ij, lmax=lmax) * special.spherical_jn(l_j,r_j) * Ylm_j

            if m_j == -lmax:
                break
            
        # compare LHS and RHS
        print "LHS = %s" % lhs
        print "RHS = %s" % rhs
        err = abs(lhs-rhs)
        assert err < 1.0e-10

        if r_j < la.norm(shift_ij):
            # case 1
            print "case |r_j| < |R_ij|"
            print "checking eqn. (2)  -  transformation of spherical Bessel of second kind"
            # left hand side of eqn. (2)
            lhs = special.spherical_yn(l_i,r_i) * Ylm_i

            # right hand side of eqn. (1)
            rhs = 0.0j
            # spherical harmonics around center j
            sph_it_j = spherical_harmonics_it(th_j,ph_j)
            for Ylm_j,l_j,m_j in sph_it_j:
                rhs += N_coeffs(l_i,m_i, l_j,m_j, shift_ij, lmax=lmax) * special.spherical_jn(l_j,r_j) * Ylm_j

                if m_j == -lmax:
                    break
            
            # compare LHS and RHS
            print "LHS = %s" % lhs
            print "RHS = %s" % rhs
            err = abs(lhs-rhs)
            assert err < 1.0e-10

        else:
            # case 2
            print "case |r_j| > |R_ij|"
            print "checking eqn. (3)  -  transformation of spherical Bessel of second kind"
            # left hand side of eqn. (2)
            lhs = special.spherical_yn(l_i,r_i) * Ylm_i

            # right hand side of eqn. (1)
            rhs = 0.0j
            # spherical harmonics around center j
            sph_it_j = spherical_harmonics_it(th_j,ph_j)
            for Ylm_j,l_j,m_j in sph_it_j:
                rhs += J_coeffs(l_i,m_i, l_j,m_j, shift_ij, lmax=lmax) * special.spherical_yn(l_j,r_j) * Ylm_j

                if m_j == -lmax:
                    break
            
            # compare LHS and RHS
            print "LHS = %s" % lhs
            print "RHS = %s" % rhs
            err = abs(lhs-rhs)
            assert err < 1.0e-10

        
        if m_i == -lmax:
            break

    Wigner.Wigner3J.statistics()

    
def test_coulomb_functions():
    """
    compute Wronskian of region III wavefunctions and compare
    with expected asymptotic forms
    """
    
    # molecular charge
    Z = 1.0
    # wave vector
    k = 0.5
    # angular momentum
    l = 2
    
    # radial wavefunctions of region III
    #  f(l,r) = f^(III)_l(k*r)
    #  g(l,r) = g^(III)_l(k*r)
    f, g = coulomb_func_factory(Z, k)

    # Wronskian
    # Since regular and irregular Coulomb functions F and G
    # have Wronskian [F(z),G(z)] = -1, we should have for
    # f(z) = 1/z F(z) and g(z) = 1/z G(z) the Wronskian
    #   [f(z),g(z)] = -1/z^2
    # and
     #  [f(k*r),g(k*r)] = -k/(k*r)^2 = -1/(k*r^2)
    r0 = 100.0
    l = 0
    lhs = wronskian(lambda r: f(l,r), lambda r: g(l,r), r0)
    rhs = -1.0/(k*r0**2)
    print "lhs = %s" % lhs
    print "rhs = %s" % rhs
    err = abs(lhs-rhs)
    assert err < 1.0e-10

    
    # for large distances asymptotically the functions
    # should be have like
    #          r->oo
    #  f(l,r) -------->  1/(k*r) sin(Theta_l(-Z/k; k*r))
    #
    #          r->oo
    #  g(l,r) -------->  1/(k*r) cos(Theta_l(-Z/k; k*r))
    #
    # with
    #
    #  Theta_l(eta,rho) = rho - eta*ln(2*rho) - 1/2*l*pi + sigma_l(eta)
    # and the Coulomb phase shift
    #
    #  sigma_l = arg{ Gamma(l+1 + i*eta) }


    # asymptotic forms
    def f_large_r(l, r):
        eta = -Z/k
        rho = k*r
        # Coulomb phase shift
        sigma_l = np.angle(special.gamma(l+1+1.0j*eta))
        sigma_l = sigma_l % (2.0*np.pi)
        # argument
        Theta_l = rho - eta*np.log(2*rho) - 0.5*l*np.pi + sigma_l
        
        f = 1.0/rho * np.sin(Theta_l)
        return f

    def g_large_r(l, r):
        eta = -Z/k
        rho = k*r
        # Coulomb phase shift
        sigma_l = np.angle(special.gamma(l+1+1.0j*eta))
        sigma_l = sigma_l % (2.0*np.pi)
        # argument
        Theta_l = rho - eta*np.log(2*rho) - 0.5*l*np.pi + sigma_l
        
        return 1.0/rho * np.cos(Theta_l)


    import matplotlib.pyplot as plt
    rs = np.linspace(40.0, 500.0, 1000)
    plt.xlabel("r")
    # exact
    plt.plot(rs, f(l,rs), label=r"$f_l^{(III)}(r)$")
    plt.plot(rs, g(l,rs), label=r"$g_l^{(III)}(r)$")
    # limit form
    plt.plot(rs, f_large_r(l,rs), ls="--", label=r"$f_l^{(III)}(r)$ (asymptotic)")
    plt.plot(rs, g_large_r(l,rs), ls="--", label=r"$g_l^{(III)}(r)$ (asymptotic)")

    plt.legend()
    plt.show()


def test_schroedinger_regionI():
    """
    verify that for the hydrogen atom the continuum orbitals are Coulomb waves
    """
    import matplotlib.pyplot as plt
    
    # hydrogen atom
    atomlist = [(1, (0.0,0.0,0.0))]
    def potential(x,y,z):
        r = np.sqrt(x*x+y*y+z*z)
        return -1.0/r
    # energy of continuum orbital
    energy = 1.0
    k = np.sqrt(2*energy)
    # asymptotic charge
    Z = +1.0
    
    radial_pots, radial_wfns, deltas = schroedinger_regionI(atomlist, potential, energy, charge=Z)

    r = np.linspace(0.0001, 20.0, 10000)
    plt.xlabel("r / bohr")
    plt.ylabel("radial wavefunction")

    # compare with exact Coulomb functions
    f, g = coulomb_func_factory(Z, k)
    for l,R_l in enumerate(radial_wfns[0]):
        plt.plot(r, R_l(r), label="$R_{%d}(r)$" % l)
        plt.plot(r, f(l, r), ls="-.", label="$F_l(k r)/(k r)$")
    plt.legend()
    plt.show()
                 

def test_wigner3j():
    """
    compare Wigner3j symbols from SHTOOLS' implementation with
    my own slow implementation
    """
    from DFTB.MolecularIntegrals.Wigner3j import wigner3j
    nsamples = 10
    # comparison
    for n in range(0, nsamples):
        jmax = 6
        l2 = np.random.randint(0,jmax+1)
        m2 = np.random.randint(-jmax,jmax+1)
        l3 = np.random.randint(0,jmax+1)
        m3 = np.random.randint(-jmax,jmax+1)
        
        m = -m2-m3
        
        w3js = []
        lmin = max(abs(l2-l3), abs(m))
        lmax = l2+l3
        for l in range(lmin,lmax+1):
            w3j = Wigner.Wigner3J(l,m, l2,m2, l3,m3)
            w3js.append(w3j)
        w3js = np.array(w3js)
            
        w3j_shtools, jmin_, jmax_, exitstatus = wigner3j(l2,l3, m, m2, m3)

        print "slow implementation"
        print w3js
        print "fast implementation of SHTOOLS"
        print w3j_shtools

        err = la.norm(w3js - w3j_shtools[:len(w3js)])
        assert err < 1.0e-5

    # timing
    print "Timing..."
    import time
    ta = time.time()
    nsamples = 1000
    for n in range(0, nsamples):
        jmax = 21
        l2 = np.random.randint(0,jmax+1)
        m2 = np.random.randint(-jmax,jmax+1)
        l3 = np.random.randint(0,jmax+1)
        m3 = np.random.randint(-jmax,jmax+1)
        
        m = -m2-m3

        
        for l in range(lmin,lmax+1):
            w3j = Wigner.Wigner3J(l,m, l2,m2, l3,m3)

    tb = time.time()
    print "old implementation, timing: %s seconds" % (tb-ta)

    import time
    ta = time.time()
    nsamples = 100
    for n in range(0, nsamples):
        jmax = 21
        l2 = np.random.randint(0,jmax+1)
        m2 = np.random.randint(-jmax,jmax+1)
        l3 = np.random.randint(0,jmax+1)
        m3 = np.random.randint(-jmax,jmax+1)
        
        m = -m2-m3
        
        w3j_shtools, jmin_, jmax_, exitstatus = wigner3j(l2,l3, m, m2, m3)

    tb = time.time()
    print "SHTOOLS implementation, timing: %s seconds" % (tb-ta)
    
def test_coul90():
    """
    compare value of Coulomb functions computed with Barnett's COUL90
    and mpmath
    """
    from DFTB.MolecularIntegrals import coul90
    import mpmath
    Z = 1.0
    k = 0.5
    eta = -Z/k

    lmax = 5
    
    x = 0.1

    # Coulomb functions
    mp_coulombf = np.zeros(lmax+1)
    mp_coulombg = np.zeros(lmax+1)
    for l in range(0, lmax+1):
        mp_coulombf[l] = mpmath.coulombf(l, eta, x)
        mp_coulombg[l] = mpmath.coulombg(l, eta, x)

    # 
    coul90_coulombf, coul90_coulombg, coul90_deriv_coulombf, coul90_deriv_coulombg = coul90.coul90_scalar(x,eta,lmax,0)

    print mp_coulombf
    print coul90_coulombf
    err_f = la.norm(mp_coulombf - coul90_coulombf)
    print "error of Coulomb F function = %e" % err_f

    print mp_coulombg
    print coul90_coulombg
    err_g = la.norm(mp_coulombg - coul90_coulombg)
    print "error of Coulomb G function = %e" % err_g

    # spherical Bessel functions
    scipy_besselj = np.zeros(lmax+1)
    scipy_bessely = np.zeros(lmax+1)
    for l in range(0, lmax+1):
        scipy_besselj[l] = special.spherical_jn(l, x)
        scipy_bessely[l] = special.spherical_yn(l, x)

    # 
    coul90_besselj, coul90_bessely, coul90_deriv_besselj, coul90_deriv_bessely = coul90.coul90_scalar(x,eta,lmax,1)

    print scipy_besselj
    print coul90_besselj
    err_j = la.norm(scipy_besselj - coul90_besselj)
    print "error of spherical Bessel j function = %e" % err_j

    print scipy_bessely
    print coul90_bessely
    err_y = la.norm(scipy_bessely - coul90_bessely)
    print "error of spherical Bessel y function = %e" % err_y

    
def test_sphharm():
    from DFTB.MolecularIntegrals.sphharm import sphharm, sphharm_arr, sphharm_const_m
    from DFTB.MolecularIntegrals.associatedLegendrePolynomials import sphvec
    import time
    
    th = 0.5
    ph = -0.45

    lmax = 23
    nchannels = (lmax+1)**2
    ysph_fortran = sphharm(th,ph, lmax)
    ysph_python = sphvec(th,ph, nchannels)
    print ysph_fortran
    print ysph_python
    dev = ysph_fortran - ysph_python
    print "deviation"
    print dev
    err = la.norm(ysph_fortran - ysph_python)
    print "error |Y_lm (fortran) - Y_lm (python)|= %e" % err
    assert err < 1.0e-10

    # timing
    nangle = 10000
    th = np.random.rand(nangle) * np.pi
    ph = np.random.rand(nangle) * 2*np.pi

    ta = time.time()
    ysph_python = sphvec(th,ph, nchannels)
    tb = time.time()
    print "Python code took  %s seconds" % (tb-ta)

    
    ta = time.time()
    ysph_fortran = sphharm_arr(th,ph, lmax)
    tb = time.time()
    print "Fortran code took  %s seconds" % (tb-ta)

    err = la.norm(ysph_fortran - ysph_python)
    print "error |Y_lm (fortran) - Y_lm (python)|= %e" % err

    # test computation of spherical harmonics for fixed m
    # and different l
    mconst = -4
    ysph_mconst = sphharm_const_m(th[0],ph[0], mconst, lmax)

    def lm_tuples_it(lmax):
        for l in range(0, lmax+1):
            for m in range(0,l+1):
                yield (l,m)
                if m > 0:
                    yield (l,-m)

    for lm,(l,m) in enumerate(lm_tuples_it(lmax)):
        if m == mconst:
            # compare
            err = abs(ysph_mconst[l] - ysph_python[lm,0])
            print "l= %d m= %d  Y_{l,m}= %s  ?=  %s" % (l,m, ysph_mconst[l], ysph_python[lm,0])
            assert err < 1.0e-10

def test_jn_coeffs():
    from DFTB.MolecularIntegrals import cms

    vec = np.random.rand(3)
    print "vec= ", vec
    lmax = 5
    for l1 in range(0, lmax):
        for m1 in range(-l1,l1+1):
            for l2 in range(0, lmax):
                for m2 in range(-l2,l2+1):
                    j_python = J_coeffs(l1,m1, l2,m2, vec, lmax=2*lmax+1)
                    n_python = N_coeffs(l1,m1, l2,m2, vec, lmax=2*lmax+1)
                    j_fortran, n_fortran = cms.jn_coeffs(l1,m1, l2,m2, vec)
                    # relative errors
                    err_j = abs(j_python - j_fortran)/abs(j_python)
                    err_n = abs(n_python - n_fortran)/abs(n_python)

                    print "J(l1=%d,m1=%d, l2=%d,m2=%d)= %s  ?=  %s" % (l1,m1,l2,m2, j_python, j_fortran)
                    print "N(l1=%d,m1=%d, l2=%d,m2=%d)= %s  ?=  %s" % (l1,m1,l2,m2, n_python, n_fortran)
                    
                    assert err_j < 1.0e-10
                    assert err_n < 1.0e-10
                    

def test_phi_iso():
    """
    compare Fortran and python implementations of 
    eqn. (11b) of 
      B.Ritchie "Theory of angular distribution of photoelectrons..." 
      Phys. Rev. A, vol. 13, num. 4, 1976
    """
    from DFTB.MolecularIntegrals import photo
    W3j = Wigner.Wigner3J
    lmax = 10
    for p in [-1,0,1]:
        for l1 in range(0, lmax+1):
            for m1 in range(-l1,l1+1):
                for l2 in range(0, lmax+1):
                    for m2 in range(-l2,l2+1):
                        for mu1 in [-1,0,1]:
                            for mu2 in [-1,0,1]:
                                phi_iso_fortran = photo.phi_iso(p,l1,m1,mu1, l2,m2,mu2)
                                phi_iso_python = np.zeros(2+1)
                                for l in [0,1,2]:
                                    fac = (-1)**(p+mu2+m2) * (2*l+1) * np.sqrt((2*l1+1)*(2*l2+1))
                                    wprod =   W3j(l2,0  , l1,0   , l,0) \
                                            * W3j(1 ,p  , 1 ,-p  , l,0)  \
                                            * W3j(l2,m2 , l1,-m1 , l, -(m2-m1)) \
                                            * W3j(1 ,mu2, 1 ,-mu1, l, -(m2-m1))
                                    phi_iso_python[l] = fac * wprod

                                err = la.norm(phi_iso_fortran - phi_iso_python)
                                assert err < 1.0e-10

                                
if __name__ == "__main__":
    #test_partitioning()
    #test_muffin_tin_potential_h2()
    #test_muffin_tin_potential_water()
                
    #test_hankel_functions()

    #test_wronskian_trigonometric()
    #test_wronskian_spherical_bessel()

    #test_spherical_bessel_shift()

    #test_coulomb_functions()

    #test_schroedinger_regionI()

    #test_wigner3j()
    #test_coul90()
    #test_sphharm()
    #test_jn_coeffs()

    test_phi_iso()
