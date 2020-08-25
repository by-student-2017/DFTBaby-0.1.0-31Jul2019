#!/usr/bin/env python
"""
compute energies of lowest bound states of hydrogen molecular ion (H2+)
"""
import numpy as np
from hydrogen_molecular_ion import DimerWavefunctions

if __name__ == "__main__":
    R = 2.0
    Za = 1.0
    Zb = 1.0
    
    wfn = DimerWavefunctions(R,Za,Zb)

    ### \sigma_g states ###
    m,n = 0,0
    trig = 'cos'
    
    # 1\sigma_g
    q = 0
    energy,(Rfunc,Sfunc,Pfunc),wavefunction = wfn.getBoundOrbital(m,n,trig,q)
    print "1\sigma_g  Energy: %s Hartree" % energy

    # 2\sigma_g
    q = 1
    energy,(Rfunc,Sfunc,Pfunc),wavefunction = wfn.getBoundOrbital(m,n,trig,q)
    print "2\sigma_g  Energy: %s Hartree" % energy

    # 3\sigma_g
    q = 2
    energy,(Rfunc,Sfunc,Pfunc),wavefunction = wfn.getBoundOrbital(m,n,trig,q)
    print "3\sigma_g  Energy: %s Hartree" % energy

    ### \sigma_u states ###
    m,n = 0,1

    # 1\sigma_u
    q = 0
    energy,(Rfunc,Sfunc,Pfunc),wavefunction = wfn.getBoundOrbital(m,n,trig,q)
    print "1\sigma_u  Energy: %s Hartree" % energy

    # 2\sigma_u
    q = 1
    energy,(Rfunc,Sfunc,Pfunc),wavefunction = wfn.getBoundOrbital(m,n,trig,q)
    print "2\sigma_u  Energy: %s Hartree" % energy

    # 3\sigma_u
    q = 2
    energy,(Rfunc,Sfunc,Pfunc),wavefunction = wfn.getBoundOrbital(m,n,trig,q)
    print "3\sigma_u  Energy: %s Hartree" % energy

    ### \pi_g states ###
    m,n = 1,1

    # 1\pi_g
    q = 0
    energy,(Rfunc,Sfunc,Pfunc),wavefunction = wfn.getBoundOrbital(m,n,trig,q)
    print "1\pi_g  Energy: %s Hartree" % energy

    # 2\pi_g
    q = 1
    energy,(Rfunc,Sfunc,Pfunc),wavefunction = wfn.getBoundOrbital(m,n,trig,q)
    print "2\pi_g  Energy: %s Hartree" % energy

    ### \delta_g states ###
    m,n = 2,0

    # 1\delta_g
    q = 0
    energy,(Rfunc,Sfunc,Pfunc),wavefunction = wfn.getBoundOrbital(m,n,trig,q)
    print "1\delta_g  Energy: %s Hartree" % energy

    # 2\pi_g
    q = 1
    energy,(Rfunc,Sfunc,Pfunc),wavefunction = wfn.getBoundOrbital(m,n,trig,q)
    print "2\delta_g  Energy: %s Hartree" % energy
