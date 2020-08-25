#!/usr/bin/env python
"""
This program samples initial conditions from the Wigner distribution.
It requires and optimized geometry (as an xyz-file) and a Hessian for
the same geometries (as a numpy data-file)
"""
from DFTB.Dynamics import HarmonicApproximation
from DFTB import XYZ, AtomicData

import numpy as np

if __name__ == "__main__":
    import sys
    from os.path import join
    import optparse

    usage  = "Usage: %s <xyz-file> <hessian file>\n" % sys.argv[0]
    usage += "  samples initial conditions from Wigner distribution\n"
    usage += "  and creates dynamics_####.in files with initial geometries\n"
    usage += "  and velocities.\n"
    usage += "  Type --help to see all options.\n"

    parser = optparse.OptionParser(usage)
    parser.add_option("--Nsample", dest="Nsample", type=int, default=30, help="Number of samples from the Wigner distribution [default: %default]")
    parser.add_option("--outdir", dest="outdir", type=str, default=".", help="Initial conditions are written to this directory [default: %default]")
    parser.add_option("--zero_threshold", dest="zero_threshold", type=float, default=1.0e-8, help="Normal modes with very low frequencies (freq^2 < zero_threshold) are treated as zero modes. This allows to freeze modes that would lead to large unrealistic displacements. [default: %default]")
    
    (opts, args) = parser.parse_args()
    if len(args) < 2:
        print usage
        exit(-1)

    xyz_file = args[0]
    hess_file = args[1]
    # should be options

    # optimized geometry
    atomlist = XYZ.read_xyz(xyz_file)[-1]
    xopt = XYZ.atomlist2vector(atomlist)
    # load hessian
    hess = np.loadtxt(hess_file)
    
    masses = AtomicData.atomlist2masses(atomlist)
    vib_freq, vib_modes = HarmonicApproximation.vibrational_analysis(xopt, hess, masses, \
                                 zero_threshold=opts.zero_threshold, is_molecule=True)

    # SAMPLE INITIAL CONDITIONS FROM WIGNER DISTRIBUTION
    qs,ps = HarmonicApproximation.initial_conditions_wigner(xopt, hess, masses,
                                                            Nsample=opts.Nsample, zero_threshold=opts.zero_threshold)
    # make hydrogens slower
    for i in range(0, opts.Nsample):
        for A,(Z,pos) in enumerate(atomlist):
            if Z == 1:
                ps[3*A:3*(A+1),i] *= 0.001
    #
    # STATISTICS 
    # kinetic energy
    Ekin = np.zeros(opts.Nsample)
    for i in range(0, opts.Nsample):
        Ekin[i] = np.sum( ps[:,i]**2 / (2*np.array(masses)) )
    # temperature  (3*N-6)/2 k T = Ekin
    f = len(masses)-6 # degrees of freedom minus rotation and translation
    T = Ekin * 2.0/(f*AtomicData.kBoltzmann)
    print "Initial Condition Statistics"
    print "============================"
    print "averages are over %d trajectories" % opts.Nsample
    print "kinetic energy = (%2.7f +- %2.7f) Hartree" % (np.mean(Ekin), np.std(Ekin))
    print "temperature    = (%2.7f +- %2.7f) Kelvin" % (np.mean(T), np.std(T))
    print ""

    # geometries only
    wigner_xyz = join(opts.outdir, "wigner.xyz")
    geometries = [XYZ.vector2atomlist(qs[:,i], atomlist) for i in range(0, opts.Nsample)]
    XYZ.write_xyz(wigner_xyz, geometries)
    print "Wigner ensemble written to file %s" % wigner_xyz

    # initial conditions
    HarmonicApproximation.save_initial_conditions(atomlist, qs, ps, opts.outdir, "dynamics")

