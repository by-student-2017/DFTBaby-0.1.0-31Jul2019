#!/usr/bin/env python
"""
extracts molecular orbitals from a Turbomole calculation and transforms from the Gaussian
basis to the minimal basis of DFTB. 

The output file has the following format

 DYSON-ORBITALS
 <orbital name 1>      <ionization energy 1 in Hartree>
 <row with tight-binding MO coefficients for orbital 1>  
 <orbital name 2>      <ionization energy 1 in Hartree>
 <row with tight-binding MO coefficients for orbital 2>  
 ...

"""
import numpy as np
import numpy.linalg as la

from DFTB import XYZ
from DFTB.DFTB2 import DFTB2
from DFTB.Formats import Turbomole
from DFTB.BasisSets import GaussianBasisSet, AtomicBasisSet
from DFTB.Scattering.SlakoScattering import save_dyson_orbitals
from DFTB.Analyse import Cube
    
def transform_turbomole_orbitals(tm_dir, dyson_file, method="atomwise", selected_orbitals=None):
    """
    The molecular orbitals from a Turbomole calculation are transformed
    into the basis used by DFTB. Since in DFTB there are much less atomic orbitals
    this transformation is not bijective and does not conserve the normalization and
    orthogonality among the MOs. 

    Parameters:
    ===========
    tm_dir: directory with coord, basis and mos files
    dyson_file: transformed MO coefficients for DFTB are written to this file
    method: Method used to find the coefficients in the minimal basis
      'atomwise': overlaps are only calculated between orbitals on the same atom (very fast).
      'grid': the DFT orbital mo(r) and the minimal DFT atomic orbitals ao_i(r) are calculated 
         on a grid and the coefficients are obtained by minimizing the deviation
             error(c_i) = integral |mo(r) - sum_i c_i ao_i(r)|^2 dr
         in a least square sense (slower, but more accurate).
    selected_orbitals: list of indeces (starting at 1, e.g. '[1,2,3]') of orbitals that should be transformed.
      If not set all orbitals are transformed.
    """
    print "Reading geometry and basis from Turbomole directory %s" % tm_dir
    Data = Turbomole.parseTurbomole(tm_dir + "/coord")
    atomlist = Data["coord"]
    Data = Turbomole.parseTurbomole(tm_dir + "/basis")
    basis_data = Data["basis"]
    bs_gaussian = GaussianBasisSet(atomlist, basis_data)
    # load Turbomole orbitals
    try:
        Data = Turbomole.parseTurbomole(tm_dir + "/mos")
        orbe,orbs_turbomole = Data["scfmo"]
    except IOError:
        print "'mos' file not found! Maybe this is an open-shell calculation. Trying to read file 'alpha'"
        Data = Turbomole.parseTurbomole(tm_dir + "/alpha")
        orbe,orbs_turbomole = Data["uhfmo_alpha"]
    # Which orbitals should be transformed?
    if selected_orbitals == None:
        selected_mos = np.array(range(0, len(orbe)), dtype=int)
    else:
        selected_mos = np.array(eval(selected_orbitals), dtype=int)-1 
    nmo = len(selected_mos)
    # transform them to DFTB basis
    if method == "atomwise":
        T = bs_gaussian.getTransformation_GTO_to_DFTB()
        orbs = np.dot(T, orbs_turbomole[:,selected_mos])
    elif method == "grid":
        # define grid for integration
        (xmin,xmax),(ymin,ymax),(zmin,zmax) = Cube.get_bbox(atomlist, dbuff=7.0)
        dx,dy,dz = xmax-xmin,ymax-ymin,zmax-zmin
        ppb = 5.0 # Points per bohr
        nx,ny,nz = int(dx*ppb),int(dy*ppb),int(dz*ppb)
        x,y,z = np.mgrid[xmin:xmax:nx*1j, ymin:ymax:ny*1j, zmin:zmax:nz*1j]
        grid = (x,y,z)
        dV = dx/float(nx-1) * dy/float(ny-1) * dz/float(nz-1)
        # numerical atomic DFTB orbitals on the grid
        bs_atomic = AtomicBasisSet(atomlist)
        aos = [bf.amp(x,y,z) for bf in bs_atomic.bfs]
        nao = len(aos)
        # overlaps S2_mn = <m|n>
        S2 = np.zeros((nao,nao))
        for m in range(0, nao):
            S2[m,m] = np.sum(aos[m]*aos[m]*dV)
            for n in range(m+1,nao):
                S2[m,n] = np.sum(aos[m]*aos[n]*dV)
                S2[n,m] = S2[m,n]
        # DFT molecular orbitals on the grid
        mos = [Cube.orbital_amplitude(grid, bs_gaussian.bfs, orbs_turbomole[:,i]).real for i in selected_mos]
        # overlaps S1_mi = <m|i>, m is an atomic DFTB orbital, i a molecular DFT orbital
        S1 = np.zeros((nao,nmo))
        for m in range(0, nao):
            for i in range(0, nmo):
                S1[m,i] = np.sum(aos[m]*mos[i]*dV)
        # Linear regression leads to the matrix equation
        #   sum_n S2_mn c_ni = S1_mi
        # which has to be solved for the coefficients c_ni.
        orbs = la.solve(S2, S1)

    else:
        raise ValueError("Method should be 'atomwise' or 'grid' but not '%s'!" % method)
    # normalize orbitals, due to the incomplete transformation they are not necessarily
    # orthogonal
    dftb2 = DFTB2(atomlist)
    dftb2.setGeometry(atomlist)
    S = dftb2.getOverlapMatrix()
    for i in range(0, nmo):
        norm2 = np.dot(orbs[:,i], np.dot(S, orbs[:,i]))
        orbs[:,i] /= np.sqrt(norm2)
    # save orbitals 
    xyz_file = "geometry.xyz"
    XYZ.write_xyz(xyz_file, [atomlist])
    print "Molecular geometry was written to '%s'." % xyz_file
    ionization_energies = -orbe[selected_mos]
    save_dyson_orbitals(dyson_file, ["%d" % (i+1) for i in selected_mos], ionization_energies, orbs, mode="w")
    print "MO coefficients for DFTB were written to '%s'." % dyson_file



    
if __name__ == "__main__":
    import sys
    import optparse

    usage  = "Usage: python %s <Turbomole directory> <output file .dyson>\n" % sys.argv[0]
    usage += "  extracts MOs from a Turbomole calculation and transforms them into the minimal basis set used by DFTB.\n"
    usage += "  Not all orbitals can be represented in a minimal basis set.\n"
    usage += "  The geometry extracted from the 'coord' file is saved to 'geometry.xyz'\n"
    usage += "  Type --help to see all options.\n"

    parser = optparse.OptionParser(usage)
    parser.add_option("--method", dest="method", type=str,
                      help=\
"""Method used to find the coefficients in the minimal basis. 
   'atomwise': overlaps are only calculated between orbitals on the same atom (very fast).
   'grid': the DFT orbital mo(r) and the minimal DFT atomic orbitals ao_i(r) are calculated 
     on a grid and the coefficients are obtained by minimizing the deviation
           error(c_i) = integral |mo(r) - sum_i c_i ao_i(r)|^2 dr
     in a least square sense (slower, but more accurate). [default: %default]""", default="grid")
    parser.add_option("--selected_orbitals", dest="selected_orbitals", type=str, help=\
"""list of indeces (starting at 1, e.g. '[1,2,3]') of orbitals that should be transformed.
      If not set all orbitals are transformed. [default: %default]""", default=None)
    
    (opts, args) = parser.parse_args()
    if len(args) < 2:
        print usage
        exit(-1)
        
    tm_dir = args[0]
    dyson_file = args[1]
    transform_turbomole_orbitals(tm_dir, dyson_file, method=opts.method, selected_orbitals=opts.selected_orbitals)
    
