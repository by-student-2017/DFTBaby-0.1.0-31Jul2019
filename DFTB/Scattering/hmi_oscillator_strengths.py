#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
oscillator strength for the 

      1s\sigma_g - 1s\sigma_u 

transition
"""

import numpy as np
from hydrogen_molecular_ion import DimerWavefunctions
from hmi_photoelectron_angular_distribution import WrapperWavefunction
from DFTB.Scattering.VariationalKohn import transition_dipole_integrals

if __name__ == "__main__":
    R = 2.0
    Za = 1.0
    Zb = 1.0
    atomlist = [(1,(0,0,-R/2.0)), (1,(0,0,+R/2.0))]
    
    wfn = DimerWavefunctions(R,Za,Zb)

    ### 1s\sigma_g -> 1s\sigma_u transition ###

    trig = 'cos'
    # 1s\sigma_g
    m,n, q = 0,0, 0

    energy1,(R1,S1,P1),wavefunction1 = wfn.getBoundOrbital(m,n,trig,q)
    print "1\sigma_g  Energy: %s Hartree" % energy1

    # 1s\sigma_u (?? not sure about quantum numbers)
    m,n, q = 0,1, 0
    
    energy2,(R2,S2,P2),wavefunction2 = wfn.getBoundOrbital(m,n,trig,q)
    print "1s\sigma_u  Energy: %s Hartree" % energy2

    # transition dipoles from factorization R*S*P
    dipole = wfn.transition_dipoles(R1,S1,P1, R2,S2,P2)
    #
    orbital1 = WrapperWavefunction(wavefunction1)
    orbital2 = WrapperWavefunction(wavefunction2)

    dipole_test = transition_dipole_integrals(atomlist, [orbital1], [orbital2], radial_grid_factor=7, lebedev_order=65)
    
    # excitation energy
    en_exc = energy2-energy1
    # oscillator strength
    f = 2.0/3.0 * (energy2-energy1) * np.sum(dipole**2)
    
    print "transition  1s\sigma_g -> 1s\sigma_u"
    print "  dipole D from factorization R*S*P      = %s" % dipole
    print "  dipole D from num. integration (Becke) = %s" % dipole_test
    print "  |D|^2 = %s" % np.sum(dipole**2)
    print "  excitation energy E = %s (Hartree)      %s (Rydberg)" % (en_exc, 2*en_exc)
    print "  oscillator strength"
    print "  f = %s" % f


    
