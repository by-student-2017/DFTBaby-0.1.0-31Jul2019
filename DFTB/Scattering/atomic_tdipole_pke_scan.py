"""
This script plots the magnitude of the transition dipoles between a bound p-orbital and a continuum s- or d-orbital
as a function of the photokinetic energy. This should answer the question which channel (s- or d-) dominates at 
low or high PKE, respectively.
"""
import numpy as np

import DFTB
from DFTB.LR_TDDFTB import LR_TDDFTB
from DFTB import XYZ, AtomicData
from DFTB.Modeling import MolecularCoords as MolCo
from DFTB.BasisSets import load_pseudo_atoms
from DFTB.Analyse import Cube
from DFTB.Scattering import slako_tables_scattering
from DFTB.Scattering.SlakoScattering import load_pseudo_atoms_scattering, load_slako_scattering, ScatteringDipoleMatrix
from DFTB.Scattering import PAD
import mpmath

def atomic_pz_orbital(Z, data_file):
    atomlist = [(Z, (0.0,0.0,0.0))]

    # compute molecular orbitals with DFTB
    print "compute molecular orbitals with tight-binding DFT"
    tddftb = LR_TDDFTB(atomlist)
    tddftb.setGeometry(atomlist, charge=0)
    options={"nstates": 1}
    try:
        tddftb.getEnergies(**options)
    except DFTB.Solver.ExcitedStatesNotConverged:
        pass

    valorbs, radial_val = load_pseudo_atoms(atomlist)
    bound_orbs = tddftb.dftb2.getKSCoefficients()
    # order of orbitals s, py,pz,px, dxy,dyz,dz2,dzx,dx2y2, so the pz-orbital has index 2
    mo_bound = bound_orbs[:,2]    
    tdipole_data = []
    for iE,E in enumerate(slako_tables_scattering.energies):
        print "PKE = %s" % E
        try:
            SKT_bf, SKT_ff = load_slako_scattering(atomlist, E)
        except ImportError:
            break
        Dipole = ScatteringDipoleMatrix(atomlist, valorbs, SKT_bf).real

        # dipole between bound orbital and the continuum AO basis orbitals
        dipole_bf = np.tensordot(mo_bound, Dipole, axes=(0,0))
        tdip_pz_to_s = dipole_bf[0,2]   # points along z-axis
        tdip_pz_to_dyz = dipole_bf[5,1] # points along y-axis
        tdip_pz_to_dz2 = dipole_bf[6,2] # points along z-axis
        tdip_pz_to_dzx = dipole_bf[7,0] # points along x-axis
        
        tdipole_data.append( [E*AtomicData.hartree_to_eV] + [tdip_pz_to_s, tdip_pz_to_dyz, tdip_pz_to_dz2, tdip_pz_to_dzx] )

        ####
        # determine the radius of the sphere where the angular distribution is calculated. It should be
        # much larger than the extent of the molecule
        (xmin,xmax),(ymin,ymax),(zmin,zmax) = Cube.get_bbox(atomlist, dbuff=0.0)
        dx,dy,dz = xmax-xmin,ymax-ymin,zmax-zmin
        Rmax = max([dx,dy,dz]) + 500.0
        print "Radius of sphere around molecule, Rmax = %s bohr" % Rmax

        k = np.sqrt(2*E)
        wavelength = 2.0 * np.pi/k
        print "wavelength = %s" % wavelength
        valorbsE, radial_valE = load_pseudo_atoms_scattering(atomlist, E, rmin=0.0, rmax=Rmax+2*wavelength, Npts=90000)


        # Plot radial wavefunctions
        import matplotlib.pyplot as plt
        plt.ion()
        r = np.linspace(0.0, Rmax+2*wavelength, 5000)
        for i,(Zi,posi) in enumerate(atomlist):
            for indx,(ni,li,mi) in enumerate(valorbsE[Zi]):
                # only plot the dz2 continuum orbital
                if li == 2 and mi == 0 and (iE % 10 == 0):
                    R_spl = radial_valE[Zi][indx]
                    radial_orbital_wfn = R_spl(r)
                    plt.plot(r, radial_orbital_wfn, label="Z=%s n=%s l=%s m=%s E=%s" % (Zi,ni,li,mi,E))
                    plt.plot(r, np.sin(k*r)/r, ls="-.")
                    #plt.plot(r, np.sin(k*r + 1.0/k * np.log(2*k*r))/r, ls="--", lw=2)
                    plt.plot(r, np.array([float(mpmath.coulombf(li,-1.0/k, k*rx))/rx for rx in r]), ls="--", lw=2, label="CoulombF l=%s E=%s" % (li,E))
        plt.draw()
        ####

    # save table
    fh = open(data_file, "w")
    print>>fh, "# PKE/eV                   pz->s                    pz->dyz                  pz->dz2                  pz->dzx"
    np.savetxt(fh, tdipole_data)
    fh.close()
    print "Wrote table with transition dipoles to %s" % data_file


    # show radial wavefunctions
    plt.ioff()
    plt.show()

    
if __name__ == "__main__":
    atomic_pz_orbital(6, "carbon_pz_tdipole_scan.dat")
    atomic_pz_orbital(8, "oxygen_pz_tdipole_scan.dat")
    
