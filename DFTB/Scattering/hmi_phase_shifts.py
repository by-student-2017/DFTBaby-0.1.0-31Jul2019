#!/usr/bin/env python
"""
compute phase shifts of the continuum states of the hydrogen molecular ion (H2+)
"""
import numpy as np
from scipy.special import gamma
from hydrogen_molecular_ion import DimerWavefunctions

def coulomb_phase_shift(L,E,Z=1):
    """
    Coulomb phase shift at energy E

    \sigma_L = arg{ Gamma(L+1 - i Z/k) }
    """
    k = np.sqrt(2*E)
    sigma_L = np.angle(gamma(L+1-1.0j*Z/k))
    sigma_L = sigma_L % (2.0*np.pi)
    return sigma_L

def plot_coulomb_phase_shifts():
    import matplotlib.pyplot as plt

    """
    L = 0
    eta = np.linspace(0.0, 4.0, 100)
    plt.plot(eta, np.angle(gamma(L+1+1.0j*eta))/np.pi)
    plt.show()
    """
    
    E = np.linspace(0.0, 10.0, 200)
    for L in range(0, 5):
        plt.plot(E, coulomb_phase_shift(L,E,Z=1), label="L=%d" % L)

    plt.xlabel(r"energy / Hartree")
    plt.ylabel(r"Coulomb phase shift $\sigma_L$")
    plt.legend()
    
    plt.show()

def hmi_phase_shifts(m=1,n=0, trig='cos', plot=False):
    R = 2.0
    Za = 1.0
    Zb = 1.0
    
    wfn = DimerWavefunctions(R,Za,Zb, plot=plot)

    # same energies as in table 2 of Richards & Larkins (1985)
    energies = np.array([0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 3.0, 4.0, 5.0, 6.0]) #, 10.0, 20.0]
    #
    deltas = []

    L = m+n
    
    for E in energies:
        delta, (Rfunc,Sfunc,Pfunc),wavefunction = wfn.getContinuumOrbital(m,n,trig,E)
        deltas.append(delta)
    # The term (L+1) * pi/2 is missing in my definition of the phase shift
    deltas = np.array(deltas) + (L+1)*np.pi/2 
    
    # Coulomb phase shifts, for Z=2 since there are two protons
    sigmas = coulomb_phase_shift(L,energies,Z=2)

    # phase shifts are measured relative to the Coulomb phase shifts in units of pi
    phase_shifts = (deltas-sigmas)/np.pi
    
    print "  Phase Shifts  for m= %d , n= %d" % (m,n)
    print "energy / e.u.     delta_l         sigma_l    (delta_l-sigma_l)/pi"
    print "-----------------------------------------------------------------"
    for (E,delta,sigma,shift) in zip(energies, deltas, sigmas, phase_shifts):
        print "   %.5f      %+10.6f      %+10.6f      %+10.6f" % (E,delta,sigma, shift)
    print ""

    if plot == True:
        import matplotlib.pyplot as plt
    
        plt.clf()
        plt.plot(energies, deltas, label="$\delta(m=%d,n=%d)$" % (m,n))
        plt.plot(energies, sigmas, label=r"$\sigma_0$")
        plt.plot(energies, (deltas-sigmas)/np.pi, label=r"$(\delta_0 - \sigma_0)/\pi$")
        
    
        plt.xlabel(r"energy / Hartree")
        plt.ylabel(r"phase shifts")
        
        plt.legend()
        plt.show()

    # Since phase shifts are angles, adding or subtracting a multiple of 2*pi does
    # not make a difference. If the phase shifts are expressed in units of pi,
    # we can add or subtract a multiple of 2. To avoid ambiguity we choose the 
    # multiple 2*k such that the modulous of the phaseshift is as small as possible:
    #     min_k  |shift + 2*k|

    phase_shifts_unique = np.zeros(phase_shifts.shape)
    ks = np.array(range(-10,10+1))
    for i,shift in enumerate(phase_shifts):
        ik = np.argmin( abs(shift + 2*ks) )
        kmin = ks[ik]
        phase_shifts_unique[i] = shift + 2*kmin
        
    return energies, phase_shifts_unique

def table_of_hmi_phase_shifts():
    """
    create a table of phase shifts for continuum orbitals of H_2^+ with different symmetry.
    The table has the same form as table 2 in Richards & Larkins to facilitate comparison.
    """
    # Continuum orbitals are designated by
    #
    #     * their energy \epsilon
    #     * the angular momentum L=m+n, L=0 (s), L=1 (p), L=2 (d), L=3 (f), L=4 (g), L=5 (h)
    #     * the number of nodes in P(phi), m=0 (sigma), m=1 (pi), m=2 (delta)
    #     * and the parity p=(-1)^(m+n), p=0 (g for gerade), p=1 (u for ungerade)
    #
    # as
    #     \epsilon L m parity
    #
    # For example, the orbitals with energy \epsilon, m=1,n=0
    # would be denoted by
    #   \epsilon p \pi u

    # \epsilon p \pi_u
    energies, shifts_p_pi_u = hmi_phase_shifts(m=1,n=0)
    # \epsilon p \sigma_u
    energies, shifts_p_sigma_u = hmi_phase_shifts(m=0,n=1)
    # \epsilon f \pi_u
    energies, shifts_f_pi_u = hmi_phase_shifts(m=1,n=2)
    # \epsilon f \sigma_u
    energies, shifts_f_sigma_u = hmi_phase_shifts(m=0,n=3)
    # \epsilon h \pi_u
    energies, shifts_h_pi_u = hmi_phase_shifts(m=1,n=4)
    # \epsilon h \sigma_u
    energies, shifts_h_sigma_u = hmi_phase_shifts(m=0,n=5)

    # combine phase shifts into table
    data = np.vstack((energies, shifts_p_pi_u, shifts_p_sigma_u, shifts_f_pi_u,
                                shifts_f_sigma_u, shifts_h_pi_u, shifts_h_sigma_u)).transpose()
    # save phase shifts to file
    fh = open("hmi_phase_shifts.dat", "w")
    print>>fh, "# hydrogen molecular ion H2+"
    print>>fh, "# "
    print>>fh, "# photoelectron                          phase shifts"
    print>>fh, "# energy / a.u.                    (delta_L - sigma_L)/pi"
    print>>fh, "#                           continuum wavefunction symmetry  "
    print>>fh, "# \epsilon   p \pi_u   p \sigma_u  f \pi_u   f \sigma_u  h \pi_u    h \sigma_u"
    np.savetxt(fh, data, fmt="%+10.6f")

    fh.close()
    
    
if __name__ == "__main__":
    table_of_hmi_phase_shifts()
    #plot_coulomb_phase_shifts()
    exit(-1)
    
    R = 2.0
    Za = 1.0
    Zb = 1.0
    
    wfn = DimerWavefunctions(R,Za,Zb, plot=False)

    E = 0.2 # energy in a.u.
    
    # k\sigma_g
    m,n = 0,0
    trig = 'cos'
    delta, (Rfunc,Sfunc,Pfunc),wavefunction = wfn.getContinuumOrbital(m,n,trig,E)
    # phase shifts are given relative to the Coulomb phase shift
    sigma = coulomb_phase_shift(0,E)
    print "sigma= %s  delta= %s" % (sigma, delta)
    print "k\sigma_g  phase shift: %s \pi rad" % ((delta-sigma)/np.pi)

    # k\sigma_u
    m,n = 0,1
    delta, (Rfunc,Sfunc,Pfunc),wavefunction = wfn.getContinuumOrbital(m,n,trig,E)
    sigma = coulomb_phase_shift(0,E)
    print "k\sigma_u  phase shift: %s \pi rad" % ((delta-sigma)/np.pi)
