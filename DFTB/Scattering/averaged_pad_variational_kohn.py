#!/usr/bin/env python
"""
This little program computes orientation-averaged angular distributions of photoelectrons
for ionization from Kohn-Sham or Hartree-Fock orbitals. It produces a table of total cross
sections (in arbitrary units) and anisotropies for each photokinetic energy.
"""
import numpy as np

# The calculation of PAD for different energies can be trivially parallelized
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()
    
import DFTB
from DFTB import AtomicData
from DFTB.Analyse import Cube

from DFTB.MolecularIntegrals.fchkfile import G09ResultsDFT
from DFTB.Scattering.VariationalKohn import GaussianMolecularOrbital, variational_kohn, transition_dipole_integrals, analyze_dipoles_angmom, orientation_averaged_pad

import os.path

def choose_initial_orbital(res, orbstr):
    """
    extract the index of an orbital from a string such as '0', 'HOMO+1', 'LUMO', etc.

    Parameters
    ----------
    res    :  instance of G09ResultsTDDFT object
    orbstr :  string with orbital name

    Returns
    -------
    orb    :  integer, index of orbital (counting starts at 0)
    """
    # save name of orbital for printing
    orbstr_ = orbstr
    # indices of HOMO and LUMO orbitals
    LUMO = max(res.nelec_alpha, res.nelec_beta)
    HOMO = LUMO-1
    if 'HOMO' in orbstr:
        origin = HOMO
        orbstr = orbstr.replace("HOMO", "")
    elif 'LUMO' in orbstr:
        origin = LUMO
        orbstr = orbstr.replace("LUMO", "")
    else:
        origin = 0
        
    if len(orbstr) == 0:
        orb = origin
    else:
        orb = origin + int(orbstr)

    print "index of HOMO: %d" % HOMO
    print "index of LUMO: %d" % LUMO
    assert orb >= 0, "Orbital index has to be positive, not %d (%s)" % (orb, orbstr_)
    print "initial bound orbital: %d" % orb  
    return orb


def averaged_pad_scan(res, initial_orbital, energy_range, data_file,
                      npts_euler=5, npts_theta=50, npts_r=60, rmax=300.0, lmax=1,
                      lebedev_order=23, radial_grid_factor=3,
                      units="eV-Mb"):
    """
    compute the photoelectron angular distribution for an ensemble of istropically
    oriented molecules. 

    Parameters
    ----------
    res             : instance of G09ResultsDFT,
                      contains basis set, MO coefficients of bound orbitals
    initial_orbital : index (starting from 0) of occupied orbital from which 
                      the electron is ionized, only the alpha orbitals are used
    energy_range    : numpy array with photoelectron kinetic energies (PKE)
                      for which the PAD should be calculated
    data_file       : path to file, a table with PKE, SIGMA and BETAs is written

    Optional
    --------
    npts_euler      : number of grid points N for numerical integration over 
                      molecular orientations for each Euler angle a,b,c
    npts_theta      : number of grid points for theta angle. The molecular frame PAD
                      is computed on the rotated grid for each orientation and averaged.
    npts_r          : number of radial grid points for integration on interval [rmax,rmax+2pi/k]
    rmax            : large radius at which the continuum orbitals can be matched 
                      with the asymptotic solution
    lmax            : maximal angular momentum of atomic continuum orbitals and asymptotic solution.
    lebedev_order   : order of Lebedev grid for angular integrals
    radial_grid_factor 
                    : factor by which the number of grid points is increased
                      for integration on the interval [0,+inf]
    units           : units for energies and photoionization cross section in output, 'eV-Mb' (eV and Megabarn) or 'a.u.'
    """
    # convert geometry to atomlist format
    atomlist = []
    for i in range(0, res.nat):
        atomlist.append( (res.atomic_numbers[i], res.coordinates[:,i]) )

    print ""
    print "*******************************************"
    print "*  PHOTOELECTRON ANGULAR DISTRIBUTIONS    *"
    print "*******************************************"
    print ""
    
    # determine the radius of the sphere where the angular distribution is calculated. It should be
    # much larger than the extent of the molecule
    (xmin,xmax),(ymin,ymax),(zmin,zmax) = Cube.get_bbox(atomlist, dbuff=0.0)
    dx,dy,dz = xmax-xmin,ymax-ymin,zmax-zmin
    # increase rmax by the size of the molecule
    rmax += max([dx,dy,dz])
    Npts = max(int(rmax),1) * 50
    print "Radius of sphere around molecule, rmax = %s bohr" % rmax
    print "Points on radial grid, Npts = %d" % Npts

    #assert res.nelec_alpha == res.nelec_beta
    # create wavefunction of bound initial orbital
    # MO coefficients of initial orbital
    orbs_initial = res.orbs_alpha[:,initial_orbital]
    # 
    bound_orbital = GaussianMolecularOrbital(res.basis, orbs_initial)
    
    # compute PADs for all energies
    pad_data = []
    print "  SCAN"
    
    print "  Writing table with betas to %s" % data_file
    # table headers
    header  = ""
    header += "# formatted checkpoint file: %s" % fchk_file + '\n'
    header += "# initial orbital: %s" % initial_orbital + '\n'
    header += "# npts_euler: %s  npts_theta: %s  npts_r: %s  rmax: %s  lmax: %s" % (npts_euler, npts_theta, npts_r, rmax, lmax) + '\n'
    header += "# lebedev_order: %s  radial_grid_factor: %s" % (lebedev_order, radial_grid_factor) + '\n'
    if units == "eV-Mb":
        header += "# PKE/eV     SIGMA/Mb       BETA1          BETA2      BETA3          BETA4" + '\n'
    else:
        header += "# PKE/Eh     SIGMA/bohr^2   BETA1          BETA2      BETA3          BETA4" + '\n'

    # save table
    access_mode = MPI.MODE_WRONLY | MPI.MODE_CREATE
    fh = MPI.File.Open(comm, data_file, access_mode)
    if rank == 0:
        # only process 0 write the header
        fh.Write_ordered(header)
    else:
        fh.Write_ordered('')
    
    for i,energy in enumerate(energy_range):
        if i % size == rank:
            print "    PKE = %6.6f Hartree  (%4.4f eV)" % (energy, energy*AtomicData.hartree_to_eV)
        
            # molecular continuum orbitals at energy E
            continuum_orbitals, phase_shifts, lms = variational_kohn(atomlist, energy,
                                                                     lmax=lmax, rmax=rmax, npts_r=npts_r,
                                                                     radial_grid_factor=radial_grid_factor, lebedev_order=lebedev_order)
        
            # transition dipoles between bound and free orbitals
            dipoles = transition_dipole_integrals(atomlist, [bound_orbital], continuum_orbitals,
                                                  radial_grid_factor=radial_grid_factor,
                                                  lebedev_order=lebedev_order)
            ### DEBUG
            analyze_dipoles_angmom(dipoles, lms)
            ###
        
            # compute PAD for ionization from orbital 0 (the bound orbital) 
            betasE = orientation_averaged_pad(dipoles[0,:,:], continuum_orbitals, energy,
                                              rmax=rmax, npts_euler=npts_euler, npts_r=npts_r, npts_theta=npts_theta)

            if units == "eV-Mb":
                energy *= AtomicData.hartree_to_eV
                # convert cross section sigma from bohr^2 to Mb
                betasE[0] *= AtomicData.bohr2_to_megabarn
                                
            pad_data.append( [energy] + list(betasE) )
            # save row with PAD for this energy to table
            row = "%10.6f   %10.6e  %+10.6e  %+10.6f  %+10.6e  %+10.6e" % tuple(pad_data[-1]) + '\n'
            fh.Write_ordered(row)
                    
    fh.Close()
        
    print "FINISHED"

if __name__ == "__main__":
    import sys
    import optparse
    import os.path

    usage  = """

       %s  <Gaussian 09 fchk-file>  <initial orbital>

    compute the photoangular distribution (PAD) for ionization from a bound orbital 
    to the continuum, which is approximated as a linear combination of atomic continuum orbitals.

    The uncontracted basis set, M.O. coefficients and orbital energies are read
    from the formatted checkpoint file. The initial orbital may be specified either by
    its index (starting from 0) or by a string such as 'HOMO', 'LUMO', 'HOMO-1' etc.
    Only the alpha orbitals are used.

    A table with the PAD is written to betas_<name of fchk-file>_<orbital>.dat 
    It contains the 6 columns   PKE/eV  SIGMA/Mb  BETA_1  BETA_2  BETA_3  BETA_4
    which define the PAD(th) at each energy according to

                                    k=4
      PAD(th) = SIMGA/(4pi) [1 + sum   BETA_k Pk(cos(th))]
                                    k=1

    Type --help to see all options.

    The computation of the PAD for a range of energies can be parallelized trivially. 
    The script can be run in parallel using `mpirun`. 
    """ % os.path.basename(sys.argv[0])

    parser = optparse.OptionParser(usage)
    # options
    parser.add_option("-u", "--units", dest="units", default="eV-Mb",
                      help="Units for energies and photoionization cross section in output, 'eV-Mb' (eV and Megabarn) or 'a.u.' [default: %default]")
    parser.add_option("--npts_euler", dest="npts_euler", help="Number of grid points N for numerical integration over molecular orientations for each dimension, size of the grid is (2*N)^3. [default: %default]", default=10, type=int)
    parser.add_option("--npts_theta", dest="npts_theta", help="Number of grid points for the polar angle theta. Increase this number until beta4 converges to 0. [default: %default]", default=100, type=int)
    parser.add_option("--npts_r", dest="npts_r", help="Number of radial points for integration on interval [rmax,rmax+2*pi/k] (2*pi/k is the wavelength at energy E=1/2 k^2) [default: %default]", default=60, type=int)
    parser.add_option("--lebedev_order", dest="lebedev_order", help="Order Lebedev grid for angular integrals [default: %default]", default=65, type=int)
    parser.add_option("--radial_grid_factor", dest="radial_grid_factor", help="Factor by which the number of radial grid points is increased for integration on the interval [0,+inf], [default: %default]", default=5, type=int)
    parser.add_option("--rmax", dest="rmax", help="The photoangular distribution is determined by the angular distribution of the continuum orbital on a spherical shell with inner radius rmax (in bohr) and outer radius rmax+2pi/k. Ideally the radius should be infinity, but a larger radius requires propagating the continuum orbital to greater distances. [default: %default]", default=300.0, type=float)
    parser.add_option("--lmax", dest="lmax", help="Maximal angular momentum of atomic continuum orbitals and asymptotic solution. [default: %default]", default=6, type=int)
    parser.add_option("--energy_range", dest="energy_range", help="Range of photoelectron kinetic energy E=1/2 k^2 given as a tuple (Emin,Emax,Npts) for which the angular distribution should be determined (in eV) [default: %default]", default="(0.05,5.0,20)", type=str)
    
    (opts, args) = parser.parse_args()
    
    if len(args) < 2:
        print "Usage:",
        print usage
        exit(-1)
    # path to Gaussian formatted checkpoint file
    fchk_file = args[0]
    # name of initial bound orbital
    orbstr = args[1]

    # ... read fchk-file
    fchk_file = args[0]
    res = G09ResultsDFT(fchk_file)
    
    iorb = choose_initial_orbital(res, orbstr)
            
    print "occupied orbitals: %s" % max(res.nelec_alpha, res.nelec_beta)
    print "occupied+virtual orbitals: %s" % res.nmo

    # energy range in Hartree
    energy_range = np.linspace(*eval(opts.energy_range)) / AtomicData.hartree_to_eV

    molecule_name = os.path.basename(fchk_file).replace(".fchk", "")
    data_file = "betas_" + molecule_name + "_" + orbstr + ".dat"

    print "computing photoangular distribution for orbital %s" % orbstr
    averaged_pad_scan(res, iorb, energy_range, data_file, 
                      npts_euler=opts.npts_euler, npts_theta=opts.npts_theta, npts_r=opts.npts_r,
                      rmax=opts.rmax, lmax=opts.lmax,
                      lebedev_order=opts.lebedev_order, radial_grid_factor=opts.radial_grid_factor,
                      units=opts.units)
