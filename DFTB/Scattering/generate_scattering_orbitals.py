#!/usr/bin/env python
"""
compute atomic scattering orbitals
"""
from DFTB import AtomicData
from DFTB.Scattering.SlakoScattering import create_directory_structure, AtomPairBoundFree, AtomPairFreeFree
import numpy as np


from DFTB.SlaterKoster.confined_pseudo_atoms import \
       h,  he, \
       li, be,                         b, c, n, o, f, ne, \
       na, mg,                         al,si,p, s, cl,ar,\
                     fe,  cu,  zn, \
                          ag

def slako_table_scattering(atom1,atom2,energies,slako_dirs):
    """
    For the atom pair 1-2 the Slater-Koster tables for matrix elements 
    between a bound valence orbital on atom 1 and an unbound orbital on atom 2 are calculated.
    The energy of the scattering orbital can vary. The tables are computed for a range
    of energies (in the array 'energies') and are stored for each energy in a separate directory.
    """
    print "-compute Slater-Koster integrals for atom pair %s-%s" \
        % (AtomicData.atom_names[atom1.Z-1], AtomicData.atom_names[atom2.Z-1])
    dimerBF = AtomPairBoundFree(atom1,atom2)
#    dimerFF = AtomPairFreeFree(atom1,atom2)

    slako_dirs_bf, slako_dirs_ff = slako_dirs
    for en,slako_dir_bf,slako_dir_ff in zip(energies, slako_dirs_bf, slako_dirs_ff):
        print "  - scattering energy: %4.6f Hartree (%4.6f eV)" % (en, en*AtomicData.hartree_to_eV)
        print "    BOUND-FREE"
        dimerBF.getSKIntegrals(d, grid, en)
        dimerBF.write_py_slako(slako_dir_bf)
        # disable plotting when working through ssh
#        dimerBF.plotSKIntegrals(slako_dir_bf)
#        print "    FREE-FREE"
#        dimerFF.getSKIntegrals(d, grid, en)
#        dimerFF.write_py_slako(slako_dir_ff)
#        # disable plotting when working through ssh
#        dimerFF.plotSKIntegrals(slako_dir_ff)



if __name__ == "__main__":
    import sys
    import optparse
    usage = "Usage: python %s <atom 1> <atom 2>\n" % sys.argv[0]
    usage += " computes Slater-Koster tables for matrix elements between the bound valence orbitals\n"
    usage += " of atom 1 and the unbound orbitals on atom 2 for a range of photokinetic energies\n"
    usage += " and stores them in the folder 'slako_tables_scattering'.\n"
    usage += " type --help to see all options"

    parser = optparse.OptionParser(usage)
    parser.add_option("--pke_range", dest="pke_range", help="Range of photokinetic energies in eV, (Emin,Emax,Npts). The range [Emin,Emax] is subdivided into Npts equidistant intervals. [default: %default]", default="(0.01, 5.0, 50)")

    (opts,args) = parser.parse_args()
    if len(args) < 2:
        print usage
        exit(-1)

    atom1 = args[0].lower()
    atom2 = args[1].lower()
    # energies of scattering state ~ photokinetic energy
    pke_range = eval(opts.pke_range)
    energies = np.linspace(*pke_range) / 27.211 # in Hartree
    #
    slako_dirs = create_directory_structure(energies)

    print "-load extended polar two-center grid"
    try:
        from DFTB.Scattering.slako_tables_scattering.double_polar_grid import d, grid
    except ImportError as e:
        print e
        raise Exception("Maybe you first have to generate a double polar grid for integration. Try: \n  python DFTB/Scattering/generate_extended_ptcgrid.py")
    
    if atom1 == "all" or atom2 == "all":
        print "=> compute tables for all combinations of H-C-N-O-S <="
        slako_table_scattering(h,h, energies, slako_dirs)
    
        slako_table_scattering(h,o, energies, slako_dirs)
        slako_table_scattering(o,h, energies, slako_dirs)
        slako_table_scattering(o,o, energies, slako_dirs)
    
        slako_table_scattering(c,c, energies, slako_dirs)
        slako_table_scattering(h,c, energies, slako_dirs)
        slako_table_scattering(c,h, energies, slako_dirs)
        slako_table_scattering(c,o, energies, slako_dirs)
        slako_table_scattering(o,c, energies, slako_dirs)
    
        slako_table_scattering(n,n, energies, slako_dirs)
        slako_table_scattering(h,n, energies, slako_dirs)
        slako_table_scattering(n,h, energies, slako_dirs)
        slako_table_scattering(c,n, energies, slako_dirs)
        slako_table_scattering(n,c, energies, slako_dirs)
        slako_table_scattering(o,n, energies, slako_dirs)
        slako_table_scattering(n,o, energies, slako_dirs)

        slako_table_scattering(s,s, energies, slako_dirs)
        slako_table_scattering(h,s, energies, slako_dirs)
        slako_table_scattering(s,h, energies, slako_dirs)
        slako_table_scattering(c,s, energies, slako_dirs)
        slako_table_scattering(s,c, energies, slako_dirs)
        slako_table_scattering(o,s, energies, slako_dirs)
        slako_table_scattering(s,o, energies, slako_dirs)
        slako_table_scattering(n,s, energies, slako_dirs)
        slako_table_scattering(s,n, energies, slako_dirs)

    else:
        # convert atom names to atom objects
        try:
            at1 = eval(atom1)
        except NameError as e:
            print e
            raise Exception("There is not pseudoatom with name %s in DFTB.SlaterKoster.confined_pseudo_atoms!" % atom1)
        try:
            at2 = eval(atom2)
        except NameError as e:
            print e
            raise Exception("There is not pseudoatom with name %s in DFTB.SlaterKoster.confined_pseudo_atoms!" % atom2)

            
        slako_table_scattering(at1,at2, energies, slako_dirs)

    print "FINISHED"
