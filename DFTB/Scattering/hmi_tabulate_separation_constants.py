#!/usr/bin/env python
"""
Tabulate separation constants for the hydrogen molecular ion
"""
import numpy as np
#from hydrogen_molecular_ion import SeparationConstants
from HMI import SeparationConstants

if __name__ == "__main__":
    # bond length in bohr
    R = 2.0
    # two protons
    Za = 1.0
    Zb = 1.0
    atomlist = [(1,(0,0,-R/2.0)), (1,(0,0,+R/2.0))]

    Lsep = SeparationConstants(R, Za, Zb)
    # maximum quantum numbers
    nmax = 10  # number of nodes in S(eta), for bound states
               # this index designates the energy level, e.g. 1\sigma
    mmax = 6   # number of nodes in P(phi), m=0 (sigma), m=1 (pi), etc.
    print "generating separation constants..."
    #nc2 = 10
    nc2 = 20
    energy_range = np.linspace(-35.0, 20.0, nc2) * 2.0/R**2
    
    Lsep.tabulate_separation_constants(energy_range, nmax=nmax, mmax=mmax)

