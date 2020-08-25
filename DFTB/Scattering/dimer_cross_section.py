#!/usr/bin/env python
"""
plots the total photoionization cross section from the sigma_g orbital of as dimer
into the p-continuum orbital as a function of the following parameters:
   - kinetic energy of the photoelectron
   - bond length of the dimer R
   - exponent a of the bound atomic s-orbitals b(r) = A exp(-a*r^2)
"""
import numpy as np

def total_cross_section(pke, a,R):
    """
    total photoionization cross section

    Parameters:
    ===========
    pke: photokinetic energy in Hartree
    a: exponent in 1/bohr^2
    R: bond length in bohr
    """
    # only the component of the transition dipole moment along the dimer axis (z-axis)
    # is non-zero
    # Dz = <continuum|z|bound>
    k = np.sqrt(2*pke)
    Dz = (np.pi/a)**(3.0/4.0) * np.exp(-k**2/(4*a)) * (
               (1.0/(a*R**2) - 1.0) * np.sin(k*R) - 1.0/(a*R**2) * (k*R) * np.cos(k*R))
    D2 = Dz**2

    fine_structure_const = 1.0/137.0
    
    sigma = 8.0 * fine_structure_const * pke * k * D2

    return sigma

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from matplotlib.widgets import Slider
    
    ax = plt.subplot(111)
    plt.subplots_adjust(left=0.15, bottom=0.3)
    ax.set_xlabel("PKE / eV")
    ax.set_ylabel("total cross section")
    
    # Slider for varying  the parameters a,R and IE
    a_ax = plt.axes([0.15, 0.1, 0.65, 0.03])
    a_slider = Slider(a_ax, "a / bohr^(-2)", 0.01, 4.0, valinit=0.23)
    R_ax = plt.axes([0.15, 0.15, 0.65, 0.03])
    R_slider = Slider(R_ax, "R / bohr", 0.1, 5.0, valinit=1.0)
    
    pke = np.linspace(0.0, 100.0, 1000)  # in eV
    l, = ax.plot(pke, total_cross_section(pke/27.2111, a_slider.val, R_slider.val))

    def update(dummy):
        sigma = total_cross_section(pke/27.211, a_slider.val, R_slider.val)
        l.set_ydata(sigma)
        ax.set_ylim((0.0, sigma.max()))
        
    a_slider.on_changed(update)
    R_slider.on_changed(update)

    plt.show()
    
