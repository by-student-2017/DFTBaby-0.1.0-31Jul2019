#!/usr/bin/env python
"""
This little program computes orientation-averaged angular distributions of photoelectrons
for ionization from Kohn-Sham or Hartree-Fock orbitals. It produces a table of total cross
sections (in arbitrary units) and anisotropies for each photokinetic energy
for which Slater-Koster tables are available. 

The orbitals can be taken either from a tight-binding calculation or a previous DFT calculation using Turbomole. 
The tight-binding calculation is done automatically, so that in this case only the geometry has to be provided in an xyz-file, e.g.:
 
  averaged_pad.py water.xyz

To use different orbitals, you can extract molecular orbitals from a Turbomole calculation using the utility script
'transform_turbomole_orbitals.py' which projects the Gaussian basis set onto the minimal basis used in tight-binding DFT.
The MO coefficients will be taken from this file instead of the tight-binding calculation if the MO file is passed as a second argument, e.g.:

  transform_turbomole_orbitals.py <Turbomole directory with mos-file> mos.dyson

  averaged_pad.py water.xyz mos.dyson

"""
import numpy as np

import DFTB
from DFTB.LR_TDDFTB import LR_TDDFTB
from DFTB import XYZ, AtomicData
from DFTB.Modeling import MolecularCoords as MolCo
from DFTB.BasisSets import load_pseudo_atoms
from DFTB.Analyse import Cube
from DFTB.Scattering import slako_tables_scattering
from DFTB.Scattering.SlakoScattering import AtomicScatteringBasisSet, load_slako_scattering, ScatteringDipoleMatrix, load_dyson_orbitals
from DFTB.Scattering import PAD

import os.path

def averaged_pad_scan(xyz_file, dyson_file,
                      selected_orbitals, npts_euler, npts_theta, nskip, inter_atomic, sphere_radius):
    molecule_name = os.path.basename(xyz_file).replace(".xyz", "")
    atomlist = XYZ.read_xyz(xyz_file)[-1]
    # shift molecule to center of mass
    print "shift molecule to center of mass"
    pos = XYZ.atomlist2vector(atomlist)
    masses = AtomicData.atomlist2masses(atomlist)
    pos_com = MolCo.shift_to_com(pos, masses)
    atomlist = XYZ.vector2atomlist(pos_com, atomlist)
    # compute molecular orbitals with DFTB
    tddftb = LR_TDDFTB(atomlist)
    tddftb.setGeometry(atomlist, charge=0)
    options={"nstates": 1}
    try:
        tddftb.getEnergies(**options)
    except DFTB.Solver.ExcitedStatesNotConverged:
        pass

    valorbs, radial_val = load_pseudo_atoms(atomlist)

    if dyson_file == None:
        print "tight-binding Kohn-Sham orbitals are taken as Dyson orbitals"
        HOMO, LUMO = tddftb.dftb2.getFrontierOrbitals()
        bound_orbs = tddftb.dftb2.getKSCoefficients()
        if selected_orbitals == None:
            # all orbitals
            selected_orbitals = range(0,bound_orbs.shape[1])
        else:
            selected_orbitals = eval(selected_orbitals, {}, {"HOMO": HOMO+1, "LUMO": LUMO+1})
            print "Indeces of selected orbitals (counting from 1): %s" % selected_orbitals
        orbital_names = ["orb_%s" % o for o in selected_orbitals]
        selected_orbitals = np.array(selected_orbitals, dtype=int)-1 # counting from 0
        dyson_orbs = bound_orbs[:,selected_orbitals]
        ionization_energies = -tddftb.dftb2.getKSEnergies()[selected_orbitals]
    else:
        print "coeffients for Dyson orbitals are read from '%s'" % dyson_file
        orbital_names, ionization_energies, dyson_orbs = load_dyson_orbitals(dyson_file)
        ionization_energies = np.array(ionization_energies) / AtomicData.hartree_to_eV

    print ""
    print "*******************************************"
    print "*  PHOTOELECTRON ANGULAR DISTRIBUTIONS    *"
    print "*******************************************"
    print ""
    
    # determine the radius of the sphere where the angular distribution is calculated. It should be
    # much larger than the extent of the molecule
    (xmin,xmax),(ymin,ymax),(zmin,zmax) = Cube.get_bbox(atomlist, dbuff=0.0)
    dx,dy,dz = xmax-xmin,ymax-ymin,zmax-zmin
    Rmax = max([dx,dy,dz]) + sphere_radius
    Npts = max(int(Rmax),1) * 50
    print "Radius of sphere around molecule, Rmax = %s bohr" % Rmax
    print "Points on radial grid, Npts = %d" % Npts
    
    nr_dyson_orbs = len(orbital_names)
    # compute PADs for all selected orbitals
    for iorb in range(0, nr_dyson_orbs):
        print "computing photoangular distribution for orbital %s" % orbital_names[iorb]
        data_file = "betas_" + molecule_name + "_" + orbital_names[iorb] + ".dat"
        pad_data = []
        print "  SCAN"
        nskip = max(1, nskip)
        # save table
        fh = open(data_file, "w")
        print "  Writing table with betas to %s" % data_file
        print>>fh, "# ionization from orbital %s   IE = %6.3f eV" % (orbital_names[iorb], ionization_energies[iorb]*AtomicData.hartree_to_eV)
        print>>fh, "# inter_atomic: %s  npts_euler: %s  npts_theta: %s  rmax: %s" % (inter_atomic, npts_euler, npts_theta, Rmax)
        print>>fh, "# PKE/eV     sigma          beta1          beta2      beta3          beta4"
        for i,E in enumerate(slako_tables_scattering.energies):
            if i % nskip != 0:
                continue
            print "    PKE = %6.6f Hartree  (%4.4f eV)" % (E, E*AtomicData.hartree_to_eV)
            k = np.sqrt(2*E)
            wavelength = 2.0 * np.pi/k
            bs_free = AtomicScatteringBasisSet(atomlist, E, rmin=0.0, rmax=Rmax+2*wavelength, Npts=Npts)
            SKT_bf, SKT_ff = load_slako_scattering(atomlist, E)
            Dipole = ScatteringDipoleMatrix(atomlist, valorbs, SKT_bf, inter_atomic=inter_atomic).real

            orientation_averaging = PAD.OrientationAveraging_small_memory(Dipole, bs_free, Rmax, E, npts_euler=npts_euler, npts_theta=npts_theta)

            pad,betasE = orientation_averaging.averaged_pad(dyson_orbs[:,iorb])
            pad_data.append( [E*AtomicData.hartree_to_eV] + list(betasE) )
            # save PAD for this energy
            print>>fh, "%10.6f   %10.6e  %+10.6e  %+10.6f  %+10.6e  %+10.6e" % tuple(pad_data[-1])
            fh.flush()
        fh.close()

    
if __name__ == "__main__":
    import sys
    import optparse
    usage = "Usage: python %s <xyz-file> [<file with Dyson MO coefficients>]\n" % os.path.basename(sys.argv[0])
    usage += " computes angular distributions of photoelectrons for ionization from selected molecular orbitals\n"
    usage += " and produces tables with sigma(PKE) and beta(PKE) for a range of photokinetic energies.\n\n"
    usage += " If you specify the optional file with Dyson orbitals, those orbitals with replace \n"
    usage += " the tight-binding Kohn-Sham orbitals.\n\n"
    usage += " The output files are named according to the pattern 'betas_<xyz basename>_<orbital name>.dat'.\n"
    usage += " basename of the xyz-file.\n\n"

    usage += " type --help to see all options.\n"

    parser = optparse.OptionParser(usage)
    parser.add_option("--selected_orbitals", dest="selected_orbitals", help="list of indeces of molecular tight-binding orbitals from which the electron is ionized, starting from 1. 'HOMO' and 'LUMO' is also understood. If left empty, photoangular distributions are calculated for all orbitals. [default: %default]", default=None)
    parser.add_option("--npts_euler", dest="npts_euler", help="Number of grid points N for numerical integration over molecular orientations for each dimension, size of the grid is (2*N)^3. [default: %default]", default=5, type=int)
    parser.add_option("--npts_theta", dest="npts_theta", help="Number of grid points for the polar angle theta. Increase this number until beta4 converges to 0. [default: %default]", default=1000, type=int)
    parser.add_option("--inter_atomic", dest="inter_atomic", help="enables inter-atomic photoionization, otherwise transitions between a bound orbital on one atom and a continuum orbital on a different atom are neglected. [default: %default]", type=int, default=1)
    parser.add_option("--rmax", dest="rmax", help="The photoangular distribution is determined by the angular distribution of the continuum orbital on a sphere of radius rmax (in bohr). Ideally the radius should be infinity, but a larger radius requires propagating the continuum orbital to greater distances. [default: %default]", default=300.0, type=float)
    parser.add_option("--nskip", dest="nskip", help="Skip every nskip energy point in the scan [default: %default]", default=0, type=int)
    
    (opts,args) = parser.parse_args()
    if len(args) < 1:
        print usage
        exit(-1)
    xyz_file = args[0]
    # optionally read MO coefficients from file instead of using tight-binding orbitals
    if len(args) > 1:
        dyson_file = args[1]
    else:
        dyson_file = None
        
    averaged_pad_scan(xyz_file, dyson_file,
                      opts.selected_orbitals, opts.npts_euler, opts.npts_theta,
                      opts.nskip, opts.inter_atomic, opts.rmax)
