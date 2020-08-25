#!/usr/bin/env python
from DFTB.Scattering.SlakoScattering import load_pseudo_atoms_scattering

import numpy as np
import matplotlib.pyplot as plt
import mpmath

def mp_coulomb_wave_func(kr, Z, k, l):
    eta = -Z/k
    f = []
    for rho in kr:
        fi = mpmath.coulombf(l, eta, rho)
        f.append(float(fi))
    return(np.array(f))


if __name__ == "__main__":
    atomlist = [(6,[0.0, 0.0, 0.0])]
    E = 0.1
    k = np.sqrt(2*E)
    unscreened_charge = 0.0
    valorbs, radial_val, phase_shifts = load_pseudo_atoms_scattering(atomlist, E, unscreened_charge=unscreened_charge)
    #
    (n,l,m) = valorbs[6][0]
    print "n= %s l= %s m= %s" % (n,l,m)
    R_spl = radial_val[6][0]
    delta = phase_shifts[6][0]
    print "phase shift delta= %s" % delta
    
    r = np.linspace(300.0, 700.0, 5000)
    plt.plot(r, R_spl(r), label=r"$R_{l}(r)$")
    plt.plot(r, np.sin(k*r)/r, ls="-.", label=r"$\sin(kr)$")
    plt.plot(r, np.sin(k*r-delta)/r, ls="--", label=r"$\sin(k r - \delta_l)$")
    coul = mp_coulomb_wave_func(k*r-delta, unscreened_charge, k, 0)
    plt.plot(r, coul/r, ls="--", label=r"CoulombF$(kr-\delta)$ for $Z=%s$" % unscreened_charge)

    
    plt.xlim((440, 540))
    plt.ylim((-0.003, +0.003))
    plt.legend()
    plt.show()

    
