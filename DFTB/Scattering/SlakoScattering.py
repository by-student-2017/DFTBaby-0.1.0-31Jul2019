"""
Slater-Koster tables for matrix elements between bound valence orbitals and unbound
scattering orbitals.
"""
from DFTB import AtomicData
from DFTB.SlaterKoster.PseudoAtomDFT import PseudoAtomDFT
from DFTB.SlaterKoster import slako_transformations as T
from DFTB.SlaterKoster import slako_transformations_dipole as Tdip
from DFTB.SlaterKoster.free_pseudo_atoms import pseudo_atoms_list
from DFTB.SlaterKoster.SKIntegrals import Atom, AtomPair, integrands_tau, integrands_tau_dipole, spline_wavefunction
from DFTB.SlaterKoster.SKIntegrals import SlakoTransformations
from DFTB.BasisSets import load_pseudo_atoms
from DFTB.SKMatrixElements import count_orbitals

import numpy as np
import numpy.linalg as la
import scipy.linalg as sla

from matplotlib import pyplot as plt

# default grid
Npts = 25000 # number of radial grid points
rmin = 0.0
rmax = 500.0

# maximal deviation at matching point for Numerov method
numerov_conv = 1.0e-6
# threshold for convergence of SCF calculation
en_conv = 1.0e-5

class AtomPairBoundFree(AtomPair):
    def __init__(self, atom_data1, atom_data2):
        self.A1 = Atom(atom_data1)
        self.A2 = Atom(atom_data2)
        self.A1.splineValenceOrbitals()
        self.A1.splineEffectivePotential()
        self.A2.splineValenceOrbitals()
        self.A2.splineEffectivePotential()
    def getSKIntegrals(self, d, grid, E):
        k = np.sqrt(2*E)
        wavelength = 2.0 * np.pi/k

        Npts = 25000 # number of radial grid points
        rmin = 0.0
        # make sure the grid is large enough so that the amplitude of the radial wavefunction
        # is close to that of the asymptotic solution
        rmax = 500.0 + 2.0*wavelength
        print "rmax = %s bohr" % rmax
        
        # E is the energy of the scattering state
        R_valence1 = self.A1.getValenceOrbitals()
        R_valence2 = self.A2.getValenceOrbitals()
        # for the second atom the unbound orbitals are calculated
        # in the effective potential of the atom
        Z2 = self.A2.atom.Z
        at = pseudo_atoms_list[Z2-1]
        # cation, Nelec = Z - 1
        Nelec = at.Z-0.99999
        atomdft = PseudoAtomDFT(at.Z, Nelec, numerov_conv,en_conv)
        atomdft.setRadialGrid(rmin,rmax, Npts)
        atomdft.initialDensityGuess((at.r, at.radial_density * Nelec/at.Z))
        atomdft.KS_pot.getRadialPotential()
        en2 = E
        # effective potentials
        Veff1_spl = self.A1.getEffectivePotential()
        Veff2_spl = self.A2.getEffectivePotential()

        self.S = {}
        self.H = {}
        self.PhaseShifts = {}
        self.d = d
        
        # overlaps and hamiltonian matrix elements
        print "OVERLAPS AND HAMILTONIANS"
        for i in T.index2tau.keys():
            l1,m1,l2,m2 = T.index2tau[i]
            print "l1=%s m1=%s l2=%s m2=%s" % (l1,m1,l2,m2)
            if (not R_valence1.has_key(l1)):
                continue
            """
            if (not R_valence2.has_key(l2)):
                # only add scattering orbitals if there is also a bound
                # orbital with the same l
                #  H: 1s bound      => only s scattering state
                #  C: 2s,2p bound   => s and p scattering states
                continue
            """
            en1, R1_spl = R_valence1[l1]
            # compute scattering orbital
            delta_l2, u_l2 = atomdft.KS_pot.scattering_state(en2,l2)
            R2_spl = spline_wavefunction(atomdft.getRadialGrid(), u_l2)
            
            # integration on two center polar grid
            olap = []
            Hpart = []
            for k,dk in enumerate(self.d):
                rhos,zs,areas = grid[0][k], grid[1][k], grid[2][k]
                #
                s,h,p1,p2 = integrands_tau((zs,rhos),dk/2.0, \
                        en1,l1,m1,R1_spl,Veff1_spl,self.A1.atom.r0, \
                        en2,l2,m2,R2_spl,Veff2_spl,self.A2.atom.r0)
                norm1 = sum(p1*areas)
                norm2 = sum(p2*areas)
#                print "norm1 = %s" % norm1
#                print "norm2 = %s" % norm2
                olap.append( sum(s*areas) )
                Hpart.append( sum(h*areas) )
            olap = np.array(olap)
            Hpart = np.array(Hpart)
            self.S[(l1,l2,i)] = olap
#            self.H[(l1,l2,i)] = en2*olap + Hpart
            self.H[(l1,l2,i)] = en1*olap + Hpart
            self.PhaseShifts[l2] = delta_l2
        # dipoles
        print "DIPOLES"
        self.Dipole = {}
        for i in Tdip.index2tau.keys():
            l1,m1,lM,mM,l2,m2 = Tdip.index2tau[i]
            print "l1=%s m1=%s l2=%s m2=%s" % (l1,m1,l2,m2)
            if not R_valence1.has_key(l1):
                continue
            """
            if (not R_valence2.has_key(l2)):
                # only add scattering orbitals if there is also a bound
                # orbital with the same l
                #  H: 1s bound      => only s scattering state
                #  C: 2s,2p bound   => s and p scattering states
                continue
            """
            en1, R1_spl = R_valence1[l1]
            # compute scattering orbital
            delta_l2, u_l2 = atomdft.KS_pot.scattering_state(en2,l2)
            R2_spl = spline_wavefunction(atomdft.getRadialGrid(), u_l2)
       
            # integration on two center polar grid
            Dippart = []
            for k,dk in enumerate(self.d):
                rhos,zs,areas = grid[0][k], grid[1][k], grid[2][k]
                #
                """
                if dk > 3.0:
                    from matplotlib.pyplot import plot,show
                    plot(rhos, zs, "o")
                    show()
                """
                #
                dip = integrands_tau_dipole((zs,rhos),dk/2.0, \
                        l1,m1,R1_spl, lM,mM, l2,m2,R2_spl)
                Dippart.append( sum(dip*areas) )
            Dippart = np.array(Dippart)
            self.Dipole[(l1,l2,i)] = Dippart

        return self.S,self.H,(self.Dipole)

class AtomPairFreeFree(AtomPair):
    def __init__(self, atom_data1, atom_data2):
        self.A1 = Atom(atom_data1)
        self.A2 = Atom(atom_data2)
        self.A1.splineValenceOrbitals()
        self.A1.splineEffectivePotential()
        self.A2.splineValenceOrbitals()
        self.A2.splineEffectivePotential()
    # both atoms have free scattering orbitals
    def getSKIntegrals(self, d, grid, E):
        k = np.sqrt(2*E)
        wavelength = 2.0 * np.pi/k

        Npts = 25000 # number of radial grid points
        rmin = 0.0
        rmax = 500.0 + 2.0*wavelength

        # E is the energy of the scattering state
        Z1 = self.A1.atom.Z
        at1 = pseudo_atoms_list[Z1-1]
        atomdft1 = PseudoAtomDFT(at1.Z, at1.Z, numerov_conv,en_conv)
        atomdft1.setRadialGrid(rmin,rmax, Npts)    
        atomdft1.initialDensityGuess((at1.r, at1.radial_density))
        atomdft1.KS_pot.getRadialPotential()
        en1 = E
        # for the second atom the unbound orbitals are calculated
        # in the effective potential of the atom
        Z2 = self.A2.atom.Z
        at2 = pseudo_atoms_list[Z2-1]
        atomdft2 = PseudoAtomDFT(at2.Z, at2.Z, numerov_conv,en_conv)
        atomdft2.setRadialGrid(rmin,rmax, Npts)    
        atomdft2.initialDensityGuess((at2.r, at2.radial_density))
        atomdft2.KS_pot.getRadialPotential()
        en2 = E
        # effective potentials
        Veff1_spl = self.A1.getEffectivePotential()
        Veff2_spl = self.A2.getEffectivePotential()

        self.S = {}
        self.H = {}
        self.d = d
        
        # overlaps and hamiltonian matrix elements
        for i in T.index2tau.keys():
            l1,m1,l2,m2 = T.index2tau[i]
            print "l1 = %s  m1 = %s  l2 = %s  m2 = %s" % (l1,m1,l2,m2)
            # compute scattering orbital on atom 1
            delta_l1, u_l1 = atomdft1.KS_pot.scattering_state(en1,l1)
            R1_spl = spline_wavefunction(atomdft1.getRadialGrid(), u_l1)
            # compute scattering orbital on atom 2
            delta_l2, u_l2 = atomdft2.KS_pot.scattering_state(en2,l2)
            R2_spl = spline_wavefunction(atomdft2.getRadialGrid(), u_l2)
            
            # integration on two center polar grid
            olap = []
            Hpart = []
            for k,dk in enumerate(self.d):
                rhos,zs,areas = grid[0][k], grid[1][k], grid[2][k]
                #
                s,h,p1,p2 = integrands_tau((zs,rhos),dk/2.0, \
                        en1,l1,m1,R1_spl,Veff1_spl,self.A1.atom.r0, \
                        en2,l2,m2,R2_spl,Veff2_spl,self.A2.atom.r0)
                norm1 = sum(p1*areas)
                norm2 = sum(p2*areas)
#                print "norm1 = %s" % norm1
#                print "norm2 = %s" % norm2
                olap.append( sum(s*areas) )
                Hpart.append( sum(h*areas) )
            olap = np.array(olap)
            Hpart = np.array(Hpart)
            self.S[(l1,l2,i)] = olap
#            self.H[(l1,l2,i)] = en2*olap + Hpart
            self.H[(l1,l2,i)] = en1*olap + Hpart
            self.PhaseShifts[(l1,l2)] = np.exp(1.0j*(-delta_l1 + delta_l2))
        # dipoles
        self.Dipole = {}

        for i in Tdip.index2tau.keys():
            l1,m1,lM,mM,l2,m2 = Tdip.index2tau[i]
            # compute scattering orbital on atom 1
            delta_l1, u_l1 = atomdft1.KS_pot.scattering_state(en1,l1)
            R1_spl = spline_wavefunction(atomdft1.getRadialGrid(), u_l1)
            # compute scattering orbital on atom 2
            delta_l2, u_l2 = atomdft2.KS_pot.scattering_state(en2,l2)
            R2_spl = spline_wavefunction(atomdft2.getRadialGrid(), u_l2)
       
            # integration on two center polar grid
            Dippart = []
            for k,dk in enumerate(self.d):
                rhos,zs,areas = grid[0][k], grid[1][k], grid[2][k]
                #
                """
                if dk > 3.0:
                    from matplotlib.pyplot import plot,show
                    plot(rhos, zs, "o")
                    show()
                """
                #
                dip = integrands_tau_dipole((zs,rhos),dk/2.0, \
                        l1,m1,R1_spl, lM,mM, l2,m2,R2_spl)
                Dippart.append( sum(dip*areas) )
            Dippart = np.array(Dippart)
            self.Dipole[(l1,l2,i)] = Dippart

        return self.S,self.H,(self.Dipole)

    
def create_directory_structure(energies):
    import os
    import pprint
    import sys
    slako_dirs_bf = []
    slako_dirs_ff = []
    slako_base = "slako_tables_scattering"
    # BOUND-FREE
    for i,en in enumerate(energies):
        slako_dir=os.path.join(slako_base, "BF_en_%.4d" % i)
        slako_dirs_bf.append(slako_dir)
        if not os.path.exists(slako_dir):
            os.makedirs(slako_dir)
            # create __init__.py file so that this module can be imported
            open(os.path.join(slako_dir, "__init__.py"), "a").close()
    # FREE-FREE
    for i,en in enumerate(energies):
        slako_dir=os.path.join(slako_base, "FF_en_%.4d" % i)
        slako_dirs_ff.append(slako_dir)
        if not os.path.exists(slako_dir):
            os.makedirs(slako_dir)
            # create __init__.py file so that this module can be imported
            open(os.path.join(slako_dir, "__init__.py"), "a").close()
            
    pp = pprint.PrettyPrinter(depth=10)
    fh = open( os.path.join(slako_base, "__init__.py"), "w")
    print>>fh, "# This file has been generated automatically by %s" % sys.argv[0]
    print>>fh, "from numpy import array"
    print>>fh, "# energies of scattering states in Hartree"
    print>>fh, "energies = \\\n%s" % pp.pformat(energies)
    print>>fh, "# directories where Slater-Koster tables are stored. Matrix elements are between"
    print>>fh, "# the valence orbitals of the 1st atom and the scattering orbitals of the 2nd atom (bound-free)"
    print>>fh, "# or between scattering orbitals on both atoms (free-free)"
    slako_modules_bf = [d.replace("/", ".") for d in slako_dirs_bf]
    print>>fh, "# Slater-Koster tables for BOUND-FREE integrals"
    print>>fh, "slako_modules_bf = \\\n%s" % pp.pformat(slako_modules_bf)
    slako_modules_ff = [d.replace("/", ".") for d in slako_dirs_ff]
    print>>fh, "# Slater-Koster tables for FREE-FREE integrals"
    print>>fh, "slako_modules_ff = \\\n%s" % pp.pformat(slako_modules_ff)
    fh.close()
    return slako_dirs_bf, slako_dirs_ff

class SlakoTransformationsSpherical(SlakoTransformations):
    """
    In DFTB the atomic orbitals are real spherical harmonics Yreal_lm(th,ph), while in Burke's formula
    for the orientation-averaged photoangular distribution the angular parts of the orbitals
    are spherical harmonics Ylm(th,ph). The Wigner-D matrices rotate the Ylm's, therefore we need
    to construct the matrix elements <Yreal_l1m1*Rl1|r|Yl2m2*Rl2> from <Yreal_lm*Rl|r|Yreal_l2m2*Rl2>
    """
    # ket orbital is a spherical harmonics and not a REAL spherical harmonics
    def getDipole(self, l1,m1,pos1, l2,m2,pos2):
        r,x,y,z = self._directional_cosines(pos1,pos2)
        Dx_p = Tdip.slako_transformations[(l1,m1,1, 1,l2,abs(m2) )](r,x,y,z, self.Dipole_spl)
        Dy_p = Tdip.slako_transformations[(l1,m1,1,-1,l2,abs(m2) )](r,x,y,z, self.Dipole_spl)
        Dz_p = Tdip.slako_transformations[(l1,m1,1, 0,l2,abs(m2) )](r,x,y,z, self.Dipole_spl)

        Dx_m = Tdip.slako_transformations[(l1,m1,1, 1,l2,-abs(m2) )](r,x,y,z, self.Dipole_spl)
        Dy_m = Tdip.slako_transformations[(l1,m1,1,-1,l2,-abs(m2) )](r,x,y,z, self.Dipole_spl)
        Dz_m = Tdip.slako_transformations[(l1,m1,1, 0,l2,-abs(m2) )](r,x,y,z, self.Dipole_spl)

        Sp = T.slako_transformations[(l1,m1,l2, abs(m2) )](r,x,y,z, self.S_spl)
        Sm = T.slako_transformations[(l1,m1,l2,-abs(m2) )](r,x,y,z, self.S_spl)

        if m2 < 0:
            fac = 1.0/np.sqrt(2.0)
            S  = fac * (Sp - 1.0j*Sm)
            Dx = fac * (Dx_p - 1.0j*Dx_m)
            Dy = fac * (Dy_p - 1.0j*Dy_m)
            Dz = fac * (Dz_p - 1.0j*Dz_m)
        elif m2 > 0:
            fac = pow(-1,m2)/np.sqrt(2.0)
            S = fac * (Sp + 1.0j*Sm)
            Dx = fac * (Dx_p + 1.0j*Dx_m)
            Dy = fac * (Dy_p + 1.0j*Dy_m)
            Dz = fac * (Dz_p + 1.0j*Dz_m)
        else:
            assert m2 == 0
            S = Sp
            Dx = Dx_p
            Dy = Dy_p
            Dz = Dz_p

        Dx += pos2[0]*S
        Dy += pos2[1]*S
        Dz += pos2[2]*S

        return [Dx,Dy,Dz]

def load_slako_scattering(atomlist, E):
    """
    load the Slater-Koster tables for scattering states at energy E
    """
    # find unique atom types
    atomtypes = list(set([Zi for (Zi,posi) in atomlist]))
    atomtypes.sort()
    # select the tables at the right energy
    from DFTB.Scattering import slako_tables_scattering
    imin = np.argmin(abs(slako_tables_scattering.energies-E))
    
    SKT_bf = {}
    SKT_ff = {}
    for Zi in atomtypes:
        for Zj in atomtypes:
            # bound-free
            module_name_bf = "DFTB.Scattering.%s.%s_%s" % (slako_tables_scattering.slako_modules_bf[imin],
                                                                    AtomicData.atom_names[Zi-1], AtomicData.atom_names[Zj-1])
            slako_module_bf = __import__(module_name_bf, fromlist=['Z1','Z2'])

            for (l1,l2,i),olap in slako_module_bf.S.iteritems():
                # The overlap between a bound and a continuum orbital for d=0 should be exactly zero
                # since the two are different eigenfunctions of the same hamiltonian. The value
                # obtained from the numerical integration will be close to 0 but not exactly 0.
                olap[0] = 0.0
                
            #SKT_bf[(Zi,Zj)] = SlakoTransformationsSpherical(slako_module_bf)
            SKT_bf[(Zi,Zj)] = SlakoTransformations(slako_module_bf)
            # free-free
            try:
                module_name_ff = "DFTB.Scattering.%s.%s_%s" % (slako_tables_scattering.slako_modules_ff[imin],
                                                                    AtomicData.atom_names[Zi-1], AtomicData.atom_names[Zj-1])
                slako_module_ff = __import__(module_name_ff, fromlist=['Z1','Z2'])

                SKT_ff[(Zi,Zj)] = SlakoTransformations(slako_module_ff)
            except ImportError:
                SKT_ff[(Zi,Zj)] = None

    return SKT_bf, SKT_ff

def ScatteringDipoleMatrix(atomlist, valorbs, SKT, inter_atomic=False):
    """
    compute matrix elements of dipole operator between valence orbitals
    using Slater-Koster Rules

    Parameters:
    ===========
    atomlist: list of tuples (Zi,[xi,yi,zi]) of atom types and positions
    valorbs: list of bound valence orbitals with quantum numbers (ni,li,mi)
    SKT: Slater Koster table
    inter_atomic: enables inter-atomic photoionization, otherwise transitions between
       a bound orbital on one atom and a continuum orbital on a different atom
       are neglected

    Returns:
    ========
    Dipoles: matrix with shape (Norb,Norb,3)
       Dipoles[i,j,0] for instance would be <i|x|j>

    """
    # count bound valence orbitals
    Norb1 = count_orbitals(atomlist, valorbs)
    Nat = len(atomlist)
    scattering_valorbs = [(0,0),  # s
                          (1,-1), (1,0), (1,1), # py, pz ,px
                          (2,-2), (2,-1), (2,0), (2,1), (2,2) # dxy, dyz, dz2, dzx, dx2y2
                      ]
    Norb2 = Nat * len(scattering_valorbs)
    Dipole = np.zeros((Norb1,Norb2,3), dtype=complex)

    # iterate over atoms
    mu = 0
    for i,(Zi,posi) in enumerate(atomlist):
        # iterate over bound orbitals on center i
        for (ni,li,mi) in valorbs[Zi]:
            # iterate over atoms
            nu = 0
            for j,(Zj,posj) in enumerate(atomlist):
                # iterate over scattering orbitals on center j
                for (lj,mj) in scattering_valorbs:

                    if not inter_atomic:
                        # no 'inter-atomic' photoionization. Transitions between bound and continuum
                        # orbitals on different atomic centers are excluded
                        if i != j:
                            nu +=1
                            continue
                    #
                    if Zj == 1:
                        # hydrogen atom should only have an s- and a p-continuum orbital
                        if lj > 1:
                            nu += 1
                            continue
                    #
                    delta_lj = SKT[(Zi,Zj)].getPhaseShift(lj)
                    phase = 1.0 #np.exp(-1.0j*delta_lj)
                    # angular part of orbitals are real spherical harmonics
                    # but we need the dipole matrix elements between 
                    Dx,Dy,Dz  = SKT[(Zi,Zj)].getDipole(li,mi,posi, lj,mj,posj)
                    
                    Dipole[mu,nu,0] = Dx
                    Dipole[mu,nu,1] = Dy
                    Dipole[mu,nu,2] = Dz
                    Dipole[mu,nu,:] *= phase

                    nu += 1
            mu += 1
    return Dipole

def ScatteringHamiltonianMatrix(atomlist, valorbs, SKT):
    """
    compute matrix elements of Hamiltonian between bound and scattering orbitals
    using Slater-Koster Rules

    Parameters:
    ===========
    atomlist: list of tuples (Zi,[xi,yi,zi]) of atom types and positions
    valorbs: list of bound valence orbitals with quantum numbers (ni,li,mi)
    SKT: Slater Koster table
    
    Returns:
    ========
    H,S: matrices with shape (Norb,Norb) with overlap and hamiltonian at reference density
    """
    # count bound valence orbitals and unbound scattering orbitals
    Norb1 = count_orbitals(atomlist, valorbs)
    Nat = len(atomlist)
    scattering_valorbs = [(0,0), (1,-1), (1,0), (1,1)]
    Norb2 = Nat * len(scattering_valorbs)
    H0 = np.zeros((Norb1,Norb2),dtype=complex)
    S = np.zeros((Norb1,Norb2),dtype=complex)
    
    # iterate over atoms
    mu = 0
    for i,(Zi,posi) in enumerate(atomlist):
        # iterate over bound orbitals on center i
        for (ni,li,mi) in valorbs[Zi]:
            # iterate over atoms
            nu = 0
            for j,(Zj,posj) in enumerate(atomlist):
                # iterate over scattering orbitals on center j
                """
                for (nj,lj,mj) in valorbs[Zj]:
                """
                for (lj,mj) in scattering_valorbs:
                    delta_lj = SKT[(Zi,Zj)].getPhaseShift(lj)
                    phase = np.exp(-1.0j*delta_lj)
                    S[mu,nu]  = SKT[(Zi,Zj)].getOverlap(li,mi,posi, lj,mj,posj) * phase
                    H0[mu,nu] = SKT[(Zi,Zj)].getHamiltonian0(li,mi,posi, lj,mj,posj) * phase

                    nu += 1
            mu += 1
    return S, H0

def continuum_flux(atomlist, SKT):
    Nat = len(atomlist)
    scattering_valorbs = [(0,0), (1,-1), (1,0), (1,1)]
    Norb = Nat * len(scattering_valorbs)

    Scont = np.zeros((Norb,Norb), dtype=complex)
    
    # iterate over atoms
    mu = 0
    for i,(Zi,posi) in enumerate(atomlist):
        # iterate over scattering orbitals on center i
        for (li,mi) in scattering_valorbs:
            delta_li = SKT[(Zi,Zi)].getPhaseShift(li)  #
            # iterate over atoms
            nu = 0
            for j,(Zj,posj) in enumerate(atomlist):
                # iterate over scattering orbitals on center j
                for (lj,mj) in scattering_valorbs:
                    delta_lj = SKT[(Zj,Zj)].getPhaseShift(lj)
                    
                    Scont[mu,nu] = np.exp(-1.0j*np.pi/2.0 * (li-lj) - 1.0j*(delta_li - delta_lj))
                    
                    nu += 1
            mu += 1
    print Scont
    print la.det(Scont)
    return Scont

############ PLOTTING SCATTERING ORBITALS #############################
from DFTB.BasisSets import AtomicBasisFunction
from DFTB.Analyse import Cube

import mpmath
from scipy.special import factorial, gamma, hyp1f1

class CoulombWave:
    def __init__(self, E, Z, l, delta_l):
        """
        evaluate phase-shifted Coulomb wave

        CoulombF(k*r-delta_l)
        """
        self.k = np.sqrt(2.0*E)
        self.l = l
        self.Z = Z
        self.delta_l = delta_l
    def __call__(self, r):
        fl_asymptotic = self.asymptotic_expansion(r)
        """
        # compare with exact values
        fl_exact = self.mpmath_implementation(r)
        err = np.sum(abs(fl_asymptotic-fl_exact))
        print "|Coulomb(asymptotic) - Coulomb(exact)|= %e" % err
        """
        return fl_asymptotic
    
    def asymptotic_expansion(self, r):
        """
        asymptotic expansion for large rho=k*r-delta_l of the regular Coulomb function F_l according
        to Abramowitz & Stegun, section 14.5.
        """
        # fast implementation
        rho = self.k*r-self.delta_l
        eta = -self.Z/self.k
        l = self.l
        # iteration according to 14.5.8
        eps = 1.0e-10
        f = 1.0
        g = 0.0
        fk = 1.0
        gk = 0.0
        for k in range(0, 1000):
            ak = (2*k+1)*eta / ( (2*k+2)*rho )
            bk = (l*(l+1) - k*(k+1) + eta**2)/( (2*k+2)*rho )
            fk = ak*fk - bk*gk
            gk = ak*gk + bk*fk
            if np.all(abs(fk) < eps) and np.all(abs(gk) < eps):
                break
            f += fk
            g += gk
        # 14.5.6
        gm = gamma(l+1+1.0j*eta)
        sigma_l = np.arctan2(gm.imag, gm.real)
        # 14.5.5
        theta_l = 0*rho
        theta_l[rho > 0] = rho[rho > 0] - eta*np.log(2*rho[rho > 0]) - l*np.pi/2.0 + sigma_l

        Fl = g*np.cos(theta_l) + f*np.sin(theta_l)
        return Fl
    
    def mpmath_implementation(self, r):
        # slow mpmath implementation
        kr = self.k*r
        f = []
        for rho in kr:
            fi = mpmath.coulombf(self.l, -self.Z/self.k, rho-self.delta_l)
            f.append(float(fi))
        return np.array(f)
    
def load_pseudo_atoms_scattering(atomlist, E, rmin=rmin, rmax=rmax, Npts=Npts,
                                 unscreened_charge=0.0, lmax=2):
    """
    find the continuum orbitals at the photokinetic energy E for each atom type
    present in the molecule. The density of the pseudoatoms are read from file and
    the continuum orbitals are calculated by solving the radial Schroedinger equation
    for a single electron in the effective atomic potential of the cation (electrostatic + xc).
    The radial grid on which the continuum orbital is calculated can be specified using
    the keywords rmin, rmax and Npts.

    lmax is the angular momentum of the highest shell
    """
    atomtypes = list(set([Zi for (Zi,posi) in atomlist]))
    atomtypes.sort()
    valorbs = {}
    radial_val = {}
    phase_shifts = {}
    for Zi in atomtypes:
        # load pseudo atoms
        at = pseudo_atoms_list[Zi-1]
        # cation
        Nelec = at.Z - unscreened_charge
        atomdft = PseudoAtomDFT(at.Z, Nelec, numerov_conv,en_conv)
        atomdft.setRadialGrid(rmin,rmax, Npts)
        atomdft.initialDensityGuess((at.r, at.radial_density * Nelec/at.Z))
        atomdft.KS_pot.getRadialPotential()

        # definition of valence shell
        valorbs[Zi] = []
        radial_val[Zi] = []
        phase_shifts[Zi] = []
        # compute scattering orbitals
        for l in range(0, lmax+1): # each atom has an s-, 3 p-, 5-d,  ... scattering orbitals
            delta_l, u_l = atomdft.KS_pot.scattering_state(E,l)
            R_spl = spline_wavefunction(atomdft.getRadialGrid(), u_l,
                                        ug_asymptotic=CoulombWave(E, atomdft.KS_pot.unscreened_charge, l, delta_l))
            n = np.inf
            for m in range(-l,l+1):
                valorbs[Zi].append((n,l,m))
                radial_val[Zi].append(R_spl)
                phase_shifts[Zi].append(delta_l)
    return valorbs, radial_val, phase_shifts


class AtomicScatteringBasisSet:
    def __init__(self, atomlist, E, rmin=rmin, rmax=rmax, Npts=Npts, lmax=2):
        valorbs, radial_val, phase_shifts = load_pseudo_atoms_scattering(atomlist, E, rmin=rmin, rmax=rmax, Npts=Npts, lmax=lmax)
        self.bfs = []
        for i,(Zi,posi) in enumerate(atomlist):
            for indx,(ni,li,mi) in enumerate(valorbs[Zi]):
                basis_function = AtomicBasisFunction(Zi, posi, ni,li,mi, radial_val[Zi][indx], i)
                self.bfs.append(basis_function)

#################### READ and WRITE Dyson orbitals ######################
            
def load_dyson_orbitals(dyson_file):
    """
    The file dyson_file should contain the molecular orbital coefficients of the Dyson orbital
    in the basis of atomic orbitals as used by DFTB. Each Dyson orbital occupied two lines:
    The first line contains its name and ionization energy (IE). The second line contains 
    a list of coefficients:
        Example:
           DYSON-ORBITALS
           S0->D0     -10.0
           0.0 0.0 0.0 0.0.
           S0->D1     -5.4
           0.0 0.0 0.0 0.0.
           ...

    Returns:
    ========
    names: list of names of the Dyson orbitals
    ionization_energies: list of IEs
    orbs: orbs[:,i] are the MO coefficients of the i-th Dyson orbital
    """
    fh = open(dyson_file, "r")
    lines = fh.readlines()
    # header marks file type
    assert lines[0].strip() == "DYSON-ORBITALS", "Strange header '%s' expected 'DYSON-ORBITALS'?!" % lines[0]
    lines = lines[1:]
    ndyson = len(lines)/2
    #
    names = []
    ionization_energies = []
    orbs = []
    for i in range(0, ndyson):
        name, IE = lines[2*i].split()
        IE = float(IE)
        # load coefficients of Dyson orbital
        orb = np.array( map(float, lines[2*i+1].split()) )
        names.append(name)
        ionization_energies.append(IE)
        orbs.append(orb)
    orbs = np.array(orbs).transpose()

    return names, ionization_energies, orbs

def save_dyson_orbitals(dyson_file, names, ionization_energies, orbs, mode="w"):
    fh = open(dyson_file, mode)
    if mode == "w":
        print>>fh, "DYSON-ORBITALS"
    for i,(name,IE) in enumerate(zip(names, ionization_energies)):
        print>>fh, "%s         %8.6f" % (name, IE)
        print>>fh, "  ".join(map(str, orbs[:,i]))
    fh.close()

###############################################################################

def test_scattering_orbitals():
    from DFTB.LR_TDDFTB import LR_TDDFTB
    from DFTB import XYZ
    
    atomlist = XYZ.read_xyz("h2.xyz")[0]

    tddftb = LR_TDDFTB(atomlist)
    tddftb.setGeometry(atomlist, charge=0)
    options={"nstates": 1}
    tddftb.getEnergies(**options)
    
    valorbs, radial_val = load_pseudo_atoms(atomlist)

    E = 5.0 / 27.211
    bs = AtomicScatteringBasisSet(atomlist, E)
    print bs.bfs
    
    SKT_bf, SKT_ff = load_slako_scattering(atomlist, E)
    S_bb, H0_bb = tddftb.dftb2._constructH0andS()
    S_bf, H0_bf = ScatteringHamiltonianMatrix(atomlist, valorbs, SKT_bf)
    #
    invS_bb = la.inv(S_bb)
    # (H-E*S)^t . Id . (H-E*S)
    HmE2 = np.dot(H0_bf.conjugate().transpose(), np.dot(invS_bb, H0_bf)) \
           - E * np.dot( S_bf.conjugate().transpose(), np.dot(invS_bb, H0_bf)) \
           - E * np.dot(H0_bf.conjugate().transpose(), np.dot(invS_bb,  S_bf)) \
           + E**2 * np.dot(S_bf.conjugate().transpose(), np.dot(invS_bb, S_bf))
    Scont = continuum_flux(atomlist, SKT_bf)
    S2 = np.dot( S_bf.conjugate().transpose(), np.dot(la.inv(S_bb), S_bf))
    """
    #
    H2 = np.dot(H0_bf.transpose(), np.dot(la.inv(S_bb), H0_bf))
    S2 = np.dot( S_bf.transpose(), np.dot(la.inv(S_bb), S_bf))
    print "H2"
    print H2
    print "S2"
    print S2
    scat_orbe2, scat_orbs = sla.eig(H2) #, S2)
    print "PKE = %s" % E
    print "Energies^2 = %s" % scat_orbe2
    scat_orbe = np.sqrt(scat_orbe2)
    sort_indx = np.argsort(scat_orbe)
    scat_orbe = scat_orbe[sort_indx]
    scat_orbs = scat_orbs[:,sort_indx]
    print "Energies of scattering orbitals: %s" % scat_orbe
    orbE = np.argmin(abs(scat_orbe-E))
    """
    assert np.sum(abs(HmE2.conjugate().transpose() - HmE2)) < 1.0e-10
    assert np.sum(abs(S2.conjugate().transpose() - S2)) < 1.0e-10
    lambdas, scat_orbs = sla.eigh(HmE2)
    print "lambdas = %s" % lambdas

    from DFTB.Scattering import PAD
    
    for i in range(0, len(lambdas)):
        if abs(lambdas[i]) > 1.0e-8:
            print "%d  lambda = %s" % (i, lambdas[i])
            ### 
            def wavefunction(grid, dV):
                # evaluate orbital
                amp = Cube.orbital_amplitude(grid, bs.bfs, scat_orbs[:,i], cache=False)
                return amp
            PAD.asymptotic_density(wavefunction, 20, E)
            ###
            
            for (flm,l,m) in classify_lm(bs, scat_orbs[:,i]):
                if abs(flm).max() > 1.0e-4:
                    print " %s %s     %s" % (l,m,abs(flm))
            Cube.orbital2grid(atomlist, bs.bfs, scat_orbs[:,i], \
                          filename="/tmp/scattering_orbital_%d.cube" % i, dbuff=25.0)
            delattr(Cube.orbital_amplitude, "cached_grid")

def classify_lm(bs, mo):
    """
    asymptotically the continuum orbitals should have spherical symmetry. 
    The continuum orbitals are evaluated at a radius R and are projected onto spherical
    harmonics to determine their (l,m) components.
    """
    from PI import LebedevQuadrature
    def f(x,y,z):
        # evaluate orbital
        grid = (x,y,z)
        amp = Cube.orbital_amplitude(grid, bs.bfs, mo, cache=False)
        return amp
    r = np.array([10.0,20.0])
    flm = list( LebedevQuadrature.spherical_wave_expansion_it(f,r, 20) )

    return flm
            
def test_photoangular_distribution():
    from DFTB.LR_TDDFTB import LR_TDDFTB
    from DFTB import XYZ
    from DFTB.Scattering import slako_tables_scattering

    # BOUND ORBITAL = HOMO
    atomlist = XYZ.read_xyz("./h2.xyz")[0]
    tddftb = LR_TDDFTB(atomlist)
    tddftb.setGeometry(atomlist, charge=0)
    options={"nstates": 1}
    tddftb.getEnergies(**options)
    
    valorbs, radial_val = load_pseudo_atoms(atomlist)

    HOMO, LUMO = tddftb.dftb2.getFrontierOrbitals()
    bound_orbs = tddftb.dftb2.getKSCoefficients()

    # according to Koopman's theorem the electron is ionized from the HOMO
    mo_indx = range(0, len(bound_orbs))
    nmo = len(mo_indx)
    energies = [[] for i in range(0, nmo)]
    sigmas = [[] for i in range(0, nmo)]
    betas = [[] for i in range(0, nmo)]
    tdip2s = [[] for i in range(0, nmo)]
    for imo in range(0, nmo):
        mo_bound = bound_orbs[:,mo_indx[imo]]

        # CONTINUUM ORBITALS at energy E
        for E in slako_tables_scattering.energies:
            
            SKT_bf, SKT_ff = load_slako_scattering(atomlist, E)
            S_bb, H0_bb = tddftb.dftb2._constructH0andS()
            S_bf, H0_bf = ScatteringHamiltonianMatrix(atomlist, valorbs, SKT_bf)
            #
            invS_bb = la.inv(S_bb)
            # (H-E*S)^t . Id . (H-E*S)
            HmE2 = np.dot(H0_bf.conjugate().transpose(), np.dot(invS_bb, H0_bf)) \
                   - E * np.dot( S_bf.conjugate().transpose(), np.dot(invS_bb, H0_bf)) \
                   - E * np.dot(H0_bf.conjugate().transpose(), np.dot(invS_bb,  S_bf)) \
                   + E**2 * np.dot(S_bf.conjugate().transpose(), np.dot(invS_bb, S_bf))
            #
            
            scat_lambdas, scat_orbs = sla.eigh(HmE2)
            print "PKE = %s" % E
            print "lambdas = %s" % scat_lambdas
        
            # photoangular distribution averaged over isotropically oriented ensemble of molecules
            Dipole = ScatteringDipoleMatrix(atomlist, valorbs, SKT_bf)
            sigma = 0.0
            beta = 0.0
            tdip2 = 0.0
            olap = 0.0
            for i in range(0, len(scat_lambdas)):
                if abs(scat_lambdas[i]) > 1.0e-10:
                    print "%d  lambda = %s" % (i, scat_lambdas[i])
                    mo_scatt = scat_orbs[:,i]
                    sigma_i, beta_i = angular_distribution(atomlist, valorbs, mo_bound, mo_scatt, Dipole)
                    sigma += sigma_i
                    beta += sigma_i * beta_i

                    tdip_i = np.zeros(3,dtype=complex)
                    for xyz in range(0, 3):
                        tdip_i[xyz] += np.dot(mo_bound, np.dot(Dipole[:,:,xyz], mo_scatt))
                    tdip2 += np.sum(abs(tdip_i)**2)
                    
            print "tdip2 = %s" % tdip2        
            beta /= sigma
                                        
            ####
            #sigma_test, beta_test = angular_distribution_old2(atomlist, valorbs, mo_bound, mo_scatt, Dipole)
            #assert abs(sigma-sigma_test) < 1.0e-10
            #assert abs(beta-beta_test) < 1.0e-10
            ####
            print "E= %s   sigma=%s   beta= %s" % (E,sigma, beta)
            energies[imo].append(E)
            sigmas[imo].append(sigma.real)
            betas[imo].append(beta.real)
            tdip2s[imo].append(tdip2)
            
    energies = np.array(energies)
    sigmas = np.array(sigmas)
    betas = np.array(betas)
    
    from matplotlib import pyplot as plt
    # total cross section
    plt.xlabel("PKE / eV")
    plt.ylabel("total photoionization cross section $\sigma$")
    for imo in range(0, nmo):
        plt.plot(energies[imo]*27.211, sigmas[imo], lw=2)
        #plt.plot(energies[imo]*27.211, tdip2s[imo], lw=2, ls="-.")
    plt.show()
    # anisotropy
    plt.cla()
    
    plt.ylim((-1.0,2.0))
    plt.xlabel("PKE / eV")
    plt.ylabel("anisotropy $\\beta_2$")
    for imo in range(0, nmo):
        plt.plot(energies[imo]*27.211, betas[imo], lw=2, label="%d" % imo)
    plt.legend()
    plt.show()
        
    return energies, sigmas, betas

def test_dipole_prepared_continuum():
    """
    see 
    G. Fronzoni, M. Stener, S. Furlan, P. Decleva
    Chemical Physics 273 (2001) 117-133
    """
    from DFTB.LR_TDDFTB import LR_TDDFTB
    from DFTB import XYZ
    from DFTB.Scattering import slako_tables_scattering

    # BOUND ORBITAL = HOMO
    atomlist = XYZ.read_xyz("./water.xyz")[0]
    tddftb = LR_TDDFTB(atomlist)
    tddftb.setGeometry(atomlist, charge=0)
    options={"nstates": 1}
    tddftb.getEnergies(**options)
    
    valorbs, radial_val = load_pseudo_atoms(atomlist)

    HOMO, LUMO = tddftb.dftb2.getFrontierOrbitals()
    bound_orbs = tddftb.dftb2.getKSCoefficients()

    # polarization direction of E-field
    epol = np.array([0.0, 1.0, 0.0])
    
    # according to Koopman's theorem the electron is ionized from the HOMO
    mo_indx = range(0, len(bound_orbs))
    nmo = len(mo_indx)
    for imo in range(0, nmo):
        mo_bound = bound_orbs[:,mo_indx[imo]]

        # CONTINUUM ORBITALS at energy E
        for E in slako_tables_scattering.energies:
            bs = AtomicScatteringBasisSet(atomlist, E)
            SKT_bf, SKT_ff = load_slako_scattering(atomlist, E)
            Dipole = ScatteringDipoleMatrix(atomlist, valorbs, SKT_bf)
            # projection of dipoles onto polarization direction
            Dipole_projected = np.zeros((Dipole.shape[0], Dipole.shape[1]))
            for xyz in [0,1,2]:
                Dipole_projected += Dipole[:,:,xyz] * epol[xyz]
            # unnormalized coefficients of dipole-prepared continuum orbitals
            mo_scatt = np.dot(mo_bound, Dipole_projected)
            nrm2 = np.dot(mo_scatt, mo_scatt)
            # normalized coefficients
            mo_scatt /= np.sqrt(nrm2)

            Cube.orbital2grid(atomlist, bs.bfs, mo_scatt, \
                          filename="/tmp/scattering_orbital_%d_to_%s.cube" % (imo, str(E).replace(".", "p")), dbuff=25.0)
            delattr(Cube.orbital_amplitude, "cached_grid")
    
def test_averaged_asymptotic_density():
    from DFTB.LR_TDDFTB import LR_TDDFTB
    from DFTB import XYZ
    from DFTB.Scattering import slako_tables_scattering
    from DFTB.Scattering import PAD

    # BOUND ORBITAL = HOMO
    atomlist = XYZ.read_xyz("./water.xyz")[0]
    tddftb = LR_TDDFTB(atomlist)
    tddftb.setGeometry(atomlist, charge=0)
    options={"nstates": 1}
    tddftb.getEnergies(**options)
    
    valorbs, radial_val = load_pseudo_atoms(atomlist)

    HOMO, LUMO = tddftb.dftb2.getFrontierOrbitals()
    bound_orbs = tddftb.dftb2.getKSCoefficients()

    # polarization direction of E-field
    epol = np.array([0.0, 0.0, 1.0])
    
    # according to Koopman's theorem the electron is ionized from the HOMO
    mo_indx = range(0, len(bound_orbs))
    nmo = len(mo_indx)
    for imo in range(0, nmo):
        print "IMO = %s" % imo
        import time
        time.sleep(1)
        mo_bound = bound_orbs[:,mo_indx[imo]]
        # CONTINUUM ORBITALS at energy E
        for E in slako_tables_scattering.energies[-2:-1]:
            bs = AtomicScatteringBasisSet(atomlist, E)
            SKT_bf, SKT_ff = load_slako_scattering(atomlist, E)
            Dipole = ScatteringDipoleMatrix(atomlist, valorbs, SKT_bf)
        
            PAD.averaged_asymptotic_density(mo_bound, Dipole, bs, 20.0, E)
            
if __name__ == "__main__":
    from DFTB.Scattering.PAD import angular_distribution, angular_distribution_old, angular_distribution_old2
    from matplotlib import pyplot as plt

    #test_scattering_orbitals()
    #test_dipole_prepared_continuum()
    test_averaged_asymptotic_density()
    #test_photoangular_distribution()
    exit(-1)
    
    #atomlist = [(8,(0,0,0))]
    atomlist = [(1,(0,0,0)), (1,(0,0,3.0))]

    
    energies = []
    sigmas = []
    betas = []
    for E in np.linspace(0.001, 5.0/27.211, 20):
        
        SKT_bf, SKT_ff = load_slako_scattering(atomlist, E)
        #print SKT
        valorbs, radial_val = load_pseudo_atoms(atomlist)
        
        Dipole = ScatteringDipoleMatrix(atomlist, valorbs, SKT_bf)
        #print Dipole
        mos_bound = np.array([1.0, 1.0])
        #mos_bound = np.array([1.0,-1.0])
        
        mos_bound /= la.norm(mos_bound)

        mos_scatt = np.array([0.0,  1.0,1.0,1.0] + [0.0,  1.0,1.0,1.0])
        #mos_scatt = np.array([0.0,  1.0,1.0,1.0] + [0.0,  1.0,1.0,1.0])
        mos_scatt /= la.norm(mos_scatt)
        #sigma, beta = angular_distribution_old(atomlist, valorbs, mos_bound, mos_scatt, Dipole)
        #print "OLD E= %s   sigma=%s   beta= %s" % (E,sigma, beta)
        sigma, beta = angular_distribution(atomlist, valorbs, mos_bound, mos_scatt, Dipole)
        print "NEW E= %s   sigma=%s   beta= %s" % (E,sigma, beta)
        energies.append(E)
        sigmas.append(sigma.real)
        betas.append(beta.real)

        
    plt.plot(energies, sigmas)
    plt.show()
    plt.cla()
    plt.ylim((-1.0,2.0))
    plt.plot(energies, betas)
    plt.show()
