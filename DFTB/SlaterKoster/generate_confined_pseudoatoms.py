#!/usr/bin/env python
"""
generate the DFT orbitals for first and second row atoms in a confining potential
and save them to a file.

Note:
=====
If we start with an initial density of zero everywhere the eigen energies
after the first iteration can be considerably lower than the final values. Therefore
the energy range has to extend to much lower energies than the expected lowest
orbital energy.
"""
from PseudoAtomDFT import PseudoAtomDFT, occupation_numbers
from DFTB.Parameters import confinement_radii_byZ
  # if the previous line gives an error,it's because
  # you tried to run this script from within the package, instead call it from the 
  # parent directory of DFTB
from numpy import linspace, array, inf
import os.path

script_dir = os.path.dirname(os.path.realpath(__file__))
orbdir = os.path.join(script_dir, "confined_pseudo_atoms/")

Npts = 3000 # number of radial grid points
rmin = 0.0
rmax = 15.0

# maximal deviation at matching point for Numerov method
numerov_conv = 1.0e-7
# threshold for convergence of SCF calculation
en_conv = 1.0e-5

def hydrogen():
    energy_range = linspace(-4.0, 2.0, 200)

    Z = 1
    Nelec = 1

    atomdft = PseudoAtomDFT(Z,Nelec,numerov_conv,en_conv, grid_spacing="exponential", r0=confinement_radii_byZ[Z])
    atomdft.setRadialGrid(rmin, rmax, Npts)
    try:
        from confined_pseudo_atoms import h
        atomdft.initialDensityGuess((h.r, h.radial_density))
    except ImportError:
        atomdft.initialDensityGuess()
    atomdft.setEnergyRange(energy_range)
    atomdft.solveKohnSham()
    fh = open(orbdir + "h.py", "w")
    atomdft.saveSolution(fh)
    fh.close()

def helium():
    dE = 0.1
    energy_range = linspace(-4.0, 1.0, 200)

    Z = 2
    Nelec = 2
    atomdft = PseudoAtomDFT(Z,Nelec,numerov_conv,en_conv, grid_spacing="exponential", r0=confinement_radii_byZ[Z])
    atomdft.setRadialGrid(rmin, rmax, Npts)
    try:
        from confined_pseudo_atoms import he
        atomdft.initialDensityGuess((he.r, he.radial_density))
    except ImportError:
        atomdft.initialDensityGuess()
    atomdft.setEnergyRange(energy_range)
    atomdft.solveKohnSham()
    fh = open(orbdir + "he.py", "w")
    atomdft.saveSolution(fh)
    fh.close()

def lithium():
    energy_range = list(linspace(-3.0, -1.5, 100)) \
        + list(linspace(-1.5, 1.5, 100))

    Z = 3
    Nelec = 3
    atomdft = PseudoAtomDFT(Z,Nelec,numerov_conv,en_conv, grid_spacing="exponential", r0=confinement_radii_byZ[Z])
    atomdft.setRadialGrid(rmin, rmax, Npts)
    try:
        from confined_pseudo_atoms import li
        atomdft.initialDensityGuess((li.r, li.radial_density))
    except ImportError:
        atomdft.initialDensityGuess()
    atomdft.setEnergyRange(energy_range)
    atomdft.solveKohnSham()
    fh = open(orbdir + "li.py", "w")
    atomdft.saveSolution(fh)
    fh.close()

def beryllium():
    energy_range = list(linspace(-5.0, -2.0, 100)) \
        + list(linspace(-2.0, -0.001, 100)) \
        + list(linspace(-0.001, 3.0, 100))

    Z = 4
    Nelec = 4
    atomdft = PseudoAtomDFT(Z,Nelec,numerov_conv,en_conv, grid_spacing="exponential", r0=confinement_radii_byZ[Z])
    atomdft.setRadialGrid(rmin, rmax, Npts)
    try:
        from confined_pseudo_atoms import be
        atomdft.initialDensityGuess((be.r, be.radial_density))
    except ImportError:
        atomdft.initialDensityGuess()
    atomdft.setEnergyRange(energy_range)
    atomdft.solveKohnSham()
    fh = open(orbdir + "be.py", "w")
    atomdft.saveSolution(fh)
    fh.close()

def boron():
    energy_range = list(linspace(-7.0, 3.0, 300))

    Z = 5
    Nelec = 5
    atomdft = PseudoAtomDFT(Z,Nelec,numerov_conv,en_conv, grid_spacing="exponential", r0=confinement_radii_byZ[Z])
    atomdft.setRadialGrid(rmin, rmax, Npts)
    try:
        from confined_pseudo_atoms import b
        atomdft.initialDensityGuess((b.r, b.radial_density))
    except ImportError:
        atomdft.initialDensityGuess()
    atomdft.setEnergyRange(energy_range)
    atomdft.solveKohnSham()
    fh = open(orbdir + "b.py", "w")
    atomdft.saveSolution(fh)
    fh.close()

def carbon():
    energy_range = list(linspace(-11.0, 2.0, 1000))

    Z = 6
    Nelec = 6
    atomdft = PseudoAtomDFT(Z,Nelec,numerov_conv,en_conv, grid_spacing="exponential", r0=confinement_radii_byZ[Z])
#    # add polarization functions => 3d orbitals
#    atomdft.setValenceOrbitals(["2s", "2p", "3d"], format="spectroscopic")
#    #
    atomdft.setRadialGrid(rmin, rmax, Npts)
    try:
        from confined_pseudo_atoms import c
        atomdft.initialDensityGuess((c.r, c.radial_density))
    except ImportError:
        atomdft.initialDensityGuess()
    atomdft.setEnergyRange(energy_range)
    atomdft.solveKohnSham()
    fh = open(orbdir + "c.py", "w")
    atomdft.saveSolution(fh)
    fh.close()

def nitrogen():
    energy_range = list(linspace(-16.0, 2.0, 1000))

    Z = 7
    Nelec = 7
    atomdft = PseudoAtomDFT(Z,Nelec,numerov_conv*0.1,en_conv, grid_spacing="exponential", r0=confinement_radii_byZ[Z])
    atomdft.setRadialGrid(rmin, rmax, Npts)
#    # add polarization functions => 3d orbitals
#    atomdft.setValenceOrbitals(["2s", "2p", "3d"], format="spectroscopic")
#    #
    try:
        from confined_pseudo_atoms import n
        atomdft.initialDensityGuess((n.r, n.radial_density))
    except ImportError:
        atomdft.initialDensityGuess()
    atomdft.setEnergyRange(energy_range)
    atomdft.solveKohnSham()
    fh = open(orbdir + "n.py", "w")
    atomdft.saveSolution(fh)
    fh.close()

def oxygen():
    energy_range = list(linspace(-30.0, 2.0, 300)) \

    Z = 8
    Nelec = 8
    atomdft = PseudoAtomDFT(Z,Nelec,numerov_conv*0.1,en_conv*2.0, grid_spacing="exponential", r0=confinement_radii_byZ[Z])
    atomdft.setRadialGrid(rmin, rmax, Npts)
    try:
        from confined_pseudo_atoms import o
        atomdft.initialDensityGuess((o.r, o.radial_density))
    except ImportError:
        atomdft.initialDensityGuess()
    atomdft.setEnergyRange(energy_range)
    atomdft.solveKohnSham()
    fh = open(orbdir + "o.py", "w")
    atomdft.saveSolution(fh)
    fh.close()
    
def fluorine():
    energy_range = list(linspace(-25.0, 3.0, 300)) \

    Z = 9
    Nelec = 9

    atomdft = PseudoAtomDFT(Z,Nelec,numerov_conv,en_conv, grid_spacing="exponential", r0=confinement_radii_byZ[Z])
    atomdft.setRadialGrid(rmin, rmax, Npts)
    try:
        from confined_pseudo_atoms import f
        atomdft.initialDensityGuess((f.r, f.radial_density))
    except ImportError:
        atomdft.initialDensityGuess()
    atomdft.setEnergyRange(energy_range)
    atomdft.solveKohnSham()
    fh = open(orbdir + "f.py", "w")
    atomdft.saveSolution(fh)
    fh.close()
    
def neon():
    energy_range = list(linspace(-31.0, 3.0, 300))

    Z = 10
    Nelec = 10
    atomdft = PseudoAtomDFT(Z,Nelec,numerov_conv,en_conv, grid_spacing="exponential", r0=confinement_radii_byZ[Z])
    atomdft.setRadialGrid(rmin, rmax, Npts)
    try:
        from confined_pseudo_atoms import ne
        atomdft.initialDensityGuess((ne.r, ne.radial_density))
    except ImportError:
        atomdft.initialDensityGuess()
    atomdft.setEnergyRange(energy_range)
    atomdft.solveKohnSham()
    fh = open(orbdir + "ne.py", "w")
    atomdft.saveSolution(fh)
    fh.close()

# third row atoms
rmax_3rd_row = rmax + 10.0 # increase radial grid by 3 bohr for 3rd row atoms

def natrium():
    energy_range = list(linspace(-45.0, 10.00, 500))

    Z = 11
    Nelec = 11
    atomdft = PseudoAtomDFT(Z,Nelec,numerov_conv,en_conv, grid_spacing="exponential", r0=confinement_radii_byZ[Z])
    atomdft.setRadialGrid(rmin, rmax_3rd_row, Npts)
    try:
        from confined_pseudo_atoms import na
        atomdft.initialDensityGuess((na.r, na.radial_density))
    except ImportError:
        atomdft.initialDensityGuess()
    atomdft.setEnergyRange(energy_range)
    atomdft.solveKohnSham()
    fh = open(orbdir + "na.py", "w")
    atomdft.saveSolution(fh)
    fh.close()

# Add magnesium as it is contained in bacteriochlorophyll
def magnesium():
    energy_range = list(linspace(-50.0, -10.00, 250)) \
        + list(linspace(-10.0, -0.001, 300)) \
        + list(linspace(-0.001, 5.0, 100))

    Z = 12
    Nelec = 12
    atomdft = PseudoAtomDFT(Z,Nelec,numerov_conv,en_conv, grid_spacing="exponential", r0=confinement_radii_byZ[Z])
    atomdft.setRadialGrid(rmin, rmax_3rd_row - 5.0, Npts)
    try:
        from confined_pseudo_atoms import mg
        atomdft.initialDensityGuess((mg.r, mg.radial_density))
    except ImportError:
        atomdft.initialDensityGuess()
    atomdft.setEnergyRange(energy_range)
    atomdft.solveKohnSham()
    fh = open(orbdir + "mg.py", "w")
    atomdft.saveSolution(fh)
    fh.close()

def aluminum():
    energy_range = list(linspace(-60.0, 3.00, 600))

    Z = 13
    Nelec = 13
    atomdft = PseudoAtomDFT(Z,Nelec,numerov_conv,en_conv, grid_spacing="exponential", r0=confinement_radii_byZ[Z])
    atomdft.setRadialGrid(rmin, rmax_3rd_row, Npts)
    try:
        from confined_pseudo_atoms import al
        atomdft.initialDensityGuess((al.r, al.radial_density))
    except ImportError:
        atomdft.initialDensityGuess()
    atomdft.setEnergyRange(energy_range)
    atomdft.solveKohnSham()
    fh = open(orbdir + "al.py", "w")
    atomdft.saveSolution(fh)
    fh.close()

def silicon():
    energy_range = list(linspace(-70.0, -50.0, 50)) \
                + list(linspace(-20.0,-2.0, 100)) \
                + list(linspace(-2.0, 5.0, 100))

    Z = 14
    Nelec = 14
    atomdft = PseudoAtomDFT(Z,Nelec, numerov_conv,en_conv, grid_spacing="exponential", r0=confinement_radii_byZ[Z])
    atomdft.setRadialGrid(rmin, rmax_3rd_row, Npts)
    ## add unoccupied d orbitals to minimal basis
    atomdft.setValenceOrbitals(["3s","3p","3d"], format="spectroscopic")
    try:
        from confined_pseudo_atoms import si
        atomdft.initialDensityGuess((si.r, si.radial_density))
    except ImportError:
        atomdft.initialDensityGuess()
    atomdft.setEnergyRange(energy_range)
    atomdft.solveKohnSham()
    fh = open(orbdir + "si.py", "w")
    atomdft.saveSolution(fh)
    fh.close()

def phosphorus():
    energy_range = list(linspace(-80.0, 3.00, 600)) \

    Z = 15
    Nelec = 15
    atomdft = PseudoAtomDFT(Z,Nelec,numerov_conv,en_conv, grid_spacing="exponential", r0=confinement_radii_byZ[Z])
    atomdft.setRadialGrid(rmin, rmax_3rd_row, Npts+1000)
    try:
        from confined_pseudo_atoms import p
        atomdft.initialDensityGuess((p.r, p.radial_density))
    except ImportError:
        atomdft.initialDensityGuess()
    atomdft.setEnergyRange(energy_range)
    atomdft.solveKohnSham()
    fh = open(orbdir + "p.py", "w")
    atomdft.saveSolution(fh)
    fh.close()

def sulfur():
    energy_range = list(linspace(-90.0, 3.00, 600))
    Z = 16
    Nelec = 16
    atomdft = PseudoAtomDFT(Z,Nelec,numerov_conv,en_conv, grid_spacing="exponential", r0=confinement_radii_byZ[Z])
    # add unoccupied d orbitals to minimal basis
    atomdft.setValenceOrbitals(["3s","3p","3d"], format="spectroscopic")
    atomdft.setRadialGrid(rmin, rmax_3rd_row, Npts)
    try:
        from confined_pseudo_atoms import s
        atomdft.initialDensityGuess((s.r, s.radial_density))
    except ImportError:
        atomdft.initialDensityGuess()
    atomdft.setEnergyRange(energy_range)
    atomdft.solveKohnSham()
    fh = open(orbdir + "s.py", "w")
    atomdft.saveSolution(fh)
    fh.close()

def chlorine():
    energy_range = list(linspace(-110.0, -10.00, 200)) \
        + list(linspace(-10.0, -1.0, 300)) \
        + list(linspace(-1.0, 3.0, 300))

    Z = 17
    Nelec = 17
    atomdft = PseudoAtomDFT(Z,Nelec,numerov_conv,en_conv, grid_spacing="exponential", r0=confinement_radii_byZ[Z])
    atomdft.setRadialGrid(rmin, rmax_3rd_row, Npts)
    try:
        from confined_pseudo_atoms import cl
        atomdft.initialDensityGuess((cl.r, cl.radial_density))
    except ImportError:
        atomdft.initialDensityGuess()
    atomdft.setEnergyRange(energy_range)
    atomdft.solveKohnSham()
    fh = open(orbdir + "cl.py", "w")
    atomdft.saveSolution(fh)
    fh.close()


def argon():
    energy_range = list(linspace(-120.0, 3.00, 600)) 

    Z = 18
    Nelec = 18
    atomdft = PseudoAtomDFT(Z,Nelec,numerov_conv,en_conv, grid_spacing="exponential", r0=confinement_radii_byZ[Z])
    atomdft.setRadialGrid(rmin, rmax_3rd_row, Npts)
    try:
        from confined_pseudo_atoms import ar
        atomdft.initialDensityGuess((ar.r, ar.radial_density))
    except ImportError:
        atomdft.initialDensityGuess()
    atomdft.setEnergyRange(energy_range)
    atomdft.solveKohnSham()
    fh = open(orbdir + "ar.py", "w")
    atomdft.saveSolution(fh)
    fh.close()

# fourth row atoms
rmax_4th_row = rmax + 10.0 # increase radial grid by 10 bohr for 4th row atoms
Npts_4th_row = Npts + 1000 

def potassium():
    energy_range = list(linspace(-135.0, -40.00, 100)) \
                  +list(linspace(-40, -13.0, 50)) \
                  +list(linspace(-13.0, -1.0, 200)) \
                  +list(linspace(-1.0, -0.0001, 200))

    Z = 19
    Nelec = 19
    atomdft = PseudoAtomDFT(Z,Nelec,numerov_conv,en_conv, grid_spacing="exponential", r0=confinement_radii_byZ[Z])
    # in potassium 4s is filled before 3d: [Ar] 4s^1
    occupation = [occ for occ in occupation_numbers(Nelec)]
    assert occupation[-1] == (1, 3-1, 2) # 1e in 3d
    occupation[-1] = (1,4-1,0) # 1e in 4s
    atomdft.setOccupation(occupation)
    atomdft.setRadialGrid(rmin, rmax_4th_row, Npts_4th_row)
    try:
        from confined_pseudo_atoms import k
        atomdft.initialDensityGuess((k.r, k.radial_density))
    except ImportError:
        atomdft.initialDensityGuess()
    atomdft.setEnergyRange(energy_range)
    atomdft.solveKohnSham()
    fh = open(orbdir + "k.py", "w")
    atomdft.saveSolution(fh)
    fh.close()

# test version !!!
def calcium():
    energy_range = list(linspace(-160.0, -40.00, 200)) \
                  +list(linspace(-40, -13.0, 200)) \
                  +list(linspace(-13.0, -2.5, 200)) \
                  +list(linspace(-2.5, 1, 200))

    Z = 20
    Nelec = 20
    atomdft = PseudoAtomDFT(Z,Nelec,numerov_conv,en_conv, grid_spacing="exponential", r0=confinement_radii_byZ[Z])
    # in calcium 4s is filled before 3d: [Ar] 4s^2
    occupation = [occ for occ in occupation_numbers(Nelec)]
    assert occupation[-1] == (2, 3-1, 2) # 2e in 3d
    occupation[-1] = (0,3-1,2)     # put 0e in 3d
    occupation.append( (2,4-1,0) ) # and 2e in 4s
    atomdft.setOccupation(occupation)
    atomdft.setRadialGrid(rmin, rmax_4th_row, Npts_4th_row)
    try:
        from confined_pseudo_atoms import k
        atomdft.initialDensityGuess((k.r, float(Nelec)/float(k.Nelec) * k.radial_density))
        #from confined_pseudo_atoms import ca
        #atomdft.initialDensityGuess((ca.r, ca.radial_density))
    except ImportError:
        atomdft.initialDensityGuess()
    atomdft.setEnergyRange(energy_range)
    atomdft.solveKohnSham()
    fh = open(orbdir + "ca.py", "w")
    atomdft.saveSolution(fh)
    fh.close()

# test version !!!
def scandium():
    energy_range = list(linspace(-200.0, -100.0, 200)) \
        + list(linspace(-100.0, -15.5, 200)) \
        + list(linspace(-15.5, -4.0, 400)) \
        + list(linspace(-4.0, 5.0, 200)) \
    # equidistant grid, use larger grid for heavy atoms
              
    Z = 21
    Nelec = 21
    atomdft = PseudoAtomDFT(Z,Nelec,numerov_conv,100*en_conv, grid_spacing="exponential", r0=confinement_radii_byZ[Z])
    # scandium has the electron configuration: [Ar] 3d^1 4s^2
    occupation = [occ for occ in occupation_numbers(Nelec)]
    assert occupation[-1] == (3,3-1,2) # 3e in 3d
    occupation[-1] = (1,3-1,2)     # put 1e in 3d
    occupation.append( (2,4-1,0) ) # and 2e in 4s
    atomdft.setOccupation(occupation)
    atomdft.setRadialGrid(0.0000004, 16.0, 6000)

    try:
        from confined_pseudo_atoms import ca
        atomdft.initialDensityGuess((ca.r, float(Nelec)/float(ca.Nelec) * ca.radial_density))
        #from confined_pseudo_atoms import sc
        #satomdft.initialDensityGuess((sc.r, sc.radial_density))
    except ImportError:
        atomdft.initialDensityGuess()
    atomdft.setEnergyRange(energy_range)
    atomdft.solveKohnSham()
    fh = open(orbdir + "sc.py", "w")
    atomdft.saveSolution(fh)
    fh.close()

def titanium():
    energy_range = list(linspace(-400.0, -200.0, 200)) \
        + list(linspace(-200.0, -30.0, 200)) \
        + list(linspace(-30.0, -5.0, 400)) \
        + list(linspace(-5.0, 5.0, 200)) \
    # equidistant grid, use larger grid for heavy atoms
              
    Z = 22
    Nelec = 22
    atomdft = PseudoAtomDFT(Z,Nelec,numerov_conv,100*en_conv, grid_spacing="exponential", r0=confinement_radii_byZ[Z])
    # titanium has the electron configuration: [Ar] 3d^2 4s^2
    occupation = [occ for occ in occupation_numbers(Nelec)]
    assert occupation[-1] == (4,3-1,2) # 4e in 3d
    occupation[-1] = (2,3-1,2)     # put 2e in 3d
    occupation.append( (2,4-1,0) ) # and 2e in 4s
    atomdft.setOccupation(occupation)
    atomdft.setRadialGrid(0.0000004, 16.0, 6000)

    try:
        from confined_pseudo_atoms import ti
        atomdft.initialDensityGuess((ti.r, ti.radial_density))
    except ImportError:
        atomdft.initialDensityGuess()
    atomdft.setEnergyRange(energy_range)
    atomdft.solveKohnSham()
    fh = open(orbdir + "ti.py", "w")
    atomdft.saveSolution(fh)
    fh.close()
    
# test version !!!
def vanadium():
    energy_range = list(linspace(-250.0, -200.0, 200)) \
        + list(linspace(-200.0, -15.0, 200)) \
        + list(linspace(-15.0, -5.0, 400)) \
        + list(linspace(-5.0, 5.0, 200)) \
    # equidistant grid, use larger grid for heavy atoms
              
    Z = 23
    Nelec = 23
    atomdft = PseudoAtomDFT(Z,Nelec,numerov_conv,100*en_conv, grid_spacing="exponential", r0=confinement_radii_byZ[Z])
    # vanadium has the electron configuration: [Ar] 3d^3 4s^2
    occupation = [occ for occ in occupation_numbers(Nelec)]
    assert occupation[-1] == (5,3-1,2) # 5e in 3d
    occupation[-1] = (3,3-1,2)     # put 3e in 3d
    occupation.append( (2,4-1,0) ) # and 2e in 4s
    atomdft.setOccupation(occupation)
    atomdft.setRadialGrid(0.0000004, 16.0, 6500)

    try:
        from confined_pseudo_atoms import cr
        atomdft.initialDensityGuess((cr.r, float(Nelec)/float(cr.Nelec) * cr.radial_density))
        #from confined_pseudo_atoms import v
        #atomdft.initialDensityGuess((v.r, v.radial_density))
    except ImportError:
        atomdft.initialDensityGuess()
    atomdft.setEnergyRange(energy_range)
    atomdft.solveKohnSham()
    fh = open(orbdir + "v.py", "w")
    atomdft.saveSolution(fh)
    fh.close()
    
# test version !!!
def chromium():
    energy_range = list(linspace(-250.0, -200.0, 200)) \
        + list(linspace(-200.0, -20.0, 200)) \
        + list(linspace(-20.0, -5.0, 400)) \
        + list(linspace(-5.0, 5.0, 200)) \
    # equidistant grid, use larger grid for heavy atoms
              
    Z = 24
    Nelec = 24
    atomdft = PseudoAtomDFT(Z,Nelec,numerov_conv,100*en_conv, grid_spacing="exponential", r0=confinement_radii_byZ[Z])
    # chromium has the electron configuration: [Ar] 3d^4 4s^2
    occupation = [occ for occ in occupation_numbers(Nelec)]
    assert occupation[-1] == (6,3-1,2) # 6e in 3d
    occupation[-1] = (4,3-1,2)     # put 4e in 3d
    occupation.append( (2,4-1,0) ) # and 2e in 4s
    atomdft.setOccupation(occupation)
    atomdft.setRadialGrid(0.0000004, 16.0, 7000)

    try:
        from confined_pseudo_atoms import mn
        atomdft.initialDensityGuess((mn.r, float(Nelec)/float(mn.Nelec) * mn.radial_density))
        #from confined_pseudo_atoms import cr
        #atomdft.initialDensityGuess((cr.r, cr.radial_density))
    except ImportError:
        atomdft.initialDensityGuess()
    atomdft.setEnergyRange(energy_range)
    atomdft.solveKohnSham()
    fh = open(orbdir + "cr.py", "w")
    atomdft.saveSolution(fh)
    fh.close()
    
# test version !!!
def manganese():
    energy_range = list(linspace(-300.0, -200.0, 200)) \
        + list(linspace(-200.0, -26.0, 200)) \
        + list(linspace(-26.0, -5.0, 400)) \
        + list(linspace(-5.0, 5.0, 200)) \
    # equidistant grid, use larger grid for heavy atoms
              
    Z = 25
    Nelec = 25
    atomdft = PseudoAtomDFT(Z,Nelec,numerov_conv,100*en_conv, grid_spacing="exponential", r0=confinement_radii_byZ[Z], damping=0.6)    
    # manganese has the electron configuration: [Ar] 3d^(5) 4s^2
    occupation = [occ for occ in occupation_numbers(Nelec)]
    assert occupation[-1] == (7,3-1,2) # 7e in 3d
    occupation[-1] = (5,3-1,2)     # put 5e in 3d
    occupation.append( (2,4-1,0) ) # and 2e in 4s
    atomdft.setOccupation(occupation)

    atomdft.setRadialGrid(0.0000004, 16.0, 7500)
    try:
        #from confined_pseudo_atoms import fe
        #atomdft.initialDensityGuess((fe.r, float(Nelec)/float(fe.Nelec) * fe.radial_density))
        from confined_pseudo_atoms import mn
        atomdft.initialDensityGuess((mn.r, mn.radial_density))
    except ImportError:
        atomdft.initialDensityGuess()
    atomdft.setEnergyRange(energy_range)
    atomdft.solveKohnSham()
    fh = open(orbdir + "mn.py", "w")
    atomdft.saveSolution(fh)
    fh.close()
    
def iron():
    energy_range = list(linspace(-400.0, -200.0, 200)) \
        + list(linspace(-200.0, -30.0, 200)) \
        + list(linspace(-30.0, -5.0, 400)) \
        + list(linspace(-5.0, 5.0, 200)) \
    # equidistant grid, use larger grid for heavy atoms
              
    Z = 26
    Nelec = 26
    atomdft = PseudoAtomDFT(Z,Nelec,numerov_conv,100*en_conv, grid_spacing="exponential", r0=confinement_radii_byZ[Z], damping=0.6)    
    # iron has the electron configuration: [Ar] 3d^(6) 4s^2
    occupation = [occ for occ in occupation_numbers(Nelec)]
    assert occupation[-1] == (8,3-1,2) # 8e in 3d
    occupation[-1] = (6,3-1,2)     # put 6e in 3d
    occupation.append( (2,4-1,0) ) # and 2e in 4s
    atomdft.setOccupation(occupation)

    atomdft.setRadialGrid(0.0000004, 16.0, 8000)
    try:
        from confined_pseudo_atoms import fe
        atomdft.initialDensityGuess((fe.r, fe.radial_density))
    except ImportError:
        atomdft.initialDensityGuess()
    atomdft.setEnergyRange(energy_range)
    atomdft.solveKohnSham()
    fh = open(orbdir + "fe.py", "w")
    atomdft.saveSolution(fh)
    fh.close()

# test version !!!
def cobalt():
    energy_range = list(linspace(-300.0, -200.0, 200)) \
        + list(linspace(-200.0, -31.0, 200)) \
        + list(linspace(-31.0, -7.0, 400)) \
        + list(linspace(-7.0, 5.0, 200)) \
    # equidistant grid, use larger grid for heavy atoms
              
    Z = 27
    Nelec = 27
    atomdft = PseudoAtomDFT(Z,Nelec,numerov_conv,100*en_conv, grid_spacing="exponential", r0=confinement_radii_byZ[Z], damping=0.6)    
    # cobalt has the electron configuration: [Ar] 3d^(7) 4s^2
    occupation = [occ for occ in occupation_numbers(Nelec)]
    assert occupation[-1] == (9,3-1,2) # 9e in 3d
    occupation[-1] = (7,3-1,2)     # put 7e in 3d
    occupation.append( (2,4-1,0) ) # and 2e in 4s
    atomdft.setOccupation(occupation)

    atomdft.setRadialGrid(0.0000004, 16.0, 7400)
    try:
        from confined_pseudo_atoms import fe
        atomdft.initialDensityGuess((fe.r, float(Nelec)/float(fe.Nelec) * fe.radial_density))
        #from confined_pseudo_atoms import co
        #atomdft.initialDensityGuess((co.r, co.radial_density))
    except ImportError:
        atomdft.initialDensityGuess()
    atomdft.setEnergyRange(energy_range)
    atomdft.solveKohnSham()
    fh = open(orbdir + "co.py", "w")
    atomdft.saveSolution(fh)
    fh.close()

# test version !!!
def nickel():
    energy_range = list(linspace(-350.0, -200.0, 200)) \
        + list(linspace(-200.0, -31.6, 200)) \
        + list(linspace(-31.6, -6.0, 400)) \
        + list(linspace(-6.0, 5.0, 200)) \
    # equidistant grid, use larger grid for heavy atoms
              
    Z = 28
    Nelec = 28
    atomdft = PseudoAtomDFT(Z,Nelec,numerov_conv,100*en_conv, grid_spacing="exponential", r0=confinement_radii_byZ[Z], damping=0.6)    
    # nickel has the electron configuration: [Ar] 3d^(9) 4s^1
    occupation = [occ for occ in occupation_numbers(Nelec)]
    assert occupation[-1] == (10,3-1,2) # 10e in 3d
    occupation[-1] = (9,3-1,2)     # put 9e in 3d
    occupation.append( (1,4-1,0) ) # and 1e in 4s
    atomdft.setOccupation(occupation)
    
    atomdft.setRadialGrid(0.0000004, 20.0, 6700)
    try:
        from confined_pseudo_atoms import cu
        atomdft.initialDensityGuess((cu.r, float(Nelec)/float(cu.Nelec) * cu.radial_density))
        #from confined_pseudo_atoms import ni
        #atomdft.initialDensityGuess((ni.r, ni.radial_density))
    except ImportError:
        atomdft.initialDensityGuess()
    atomdft.setEnergyRange(energy_range)
    atomdft.solveKohnSham()
    fh = open(orbdir + "ni.py", "w")
    atomdft.saveSolution(fh)
    fh.close()

def copper():
    energy_range = list(linspace(-400.0, -200.0, 200)) \
        + list(linspace(-200.0, -30.0, 200)) \
        + list(linspace(-30.0, -5.0, 400)) \
        + list(linspace(-5.0, 5.0, 200)) \
    # equidistant grid, use larger grid for heavy atoms
              
    Z = 29
    Nelec = 29
    atomdft = PseudoAtomDFT(Z,Nelec,numerov_conv,en_conv, grid_spacing="exponential", r0=confinement_radii_byZ[Z], damping=0.6)    
    atomdft.setRadialGrid(0.0000004, 20.0, 6000)
    try:
        from confined_pseudo_atoms import cu
        atomdft.initialDensityGuess((cu.r, cu.radial_density))
    except ImportError:
        atomdft.initialDensityGuess()
    atomdft.setEnergyRange(energy_range)
    atomdft.solveKohnSham()
    fh = open(orbdir + "cu.py", "w")
    atomdft.saveSolution(fh)
    fh.close()

def zinc():
    energy_range = list(linspace(-400.0, -200.0, 200)) \
        + list(linspace(-200.0, -30.0, 200)) \
        + list(linspace(-30.0, -5.0, 400)) \
        + list(linspace(-5.0, 5.0, 200)) \
    # equidistant grid, use larger grid for heavy atoms
              
    Z = 30
    Nelec = 30
    atomdft = PseudoAtomDFT(Z,Nelec,numerov_conv,en_conv, grid_spacing="exponential", r0=confinement_radii_byZ[Z], damping=0.6)    
    atomdft.setRadialGrid(0.0000004, 20.0, 6000)
    try:
        from confined_pseudo_atoms import zn
        atomdft.initialDensityGuess((zn.r, zn.radial_density))
    except ImportError:
        atomdft.initialDensityGuess()
    atomdft.setEnergyRange(energy_range)
    atomdft.solveKohnSham()
    fh = open(orbdir + "zn.py", "w")
    atomdft.saveSolution(fh)
    fh.close()

# test version !!!
def gallium():
    energy_range = list(linspace(-400.0, -200.0, 200)) \
        + list(linspace(-200.0, -45.0, 200)) \
        + list(linspace(-45.0, -8.0, 400)) \
        + list(linspace(-8.0, -4.5, 200)) \
        + list(linspace(-4.5, 5.0, 200)) \
    # equidistant grid, use larger grid for heavy atoms
              
    Z = 31
    Nelec = 31
    atomdft = PseudoAtomDFT(Z,Nelec,numerov_conv,en_conv, grid_spacing="exponential", r0=confinement_radii_byZ[Z], damping=0.6)    
    atomdft.setRadialGrid(0.0000004, 20.0, 6000)
    try:
        from confined_pseudo_atoms import zn
        atomdft.initialDensityGuess((zn.r, float(Nelec)/float(zn.Nelec) * zn.radial_density))
        #from confined_pseudo_atoms import ga
        #atomdft.initialDensityGuess((ga.r, ga.radial_density))
    except ImportError:
        atomdft.initialDensityGuess()
    atomdft.setEnergyRange(energy_range)
    atomdft.solveKohnSham()
    fh = open(orbdir + "ga.py", "w")
    atomdft.saveSolution(fh)
    fh.close()

# test version !!!
def germanium():
    energy_range = list(linspace(-450.0, -200.0, 200)) \
        + list(linspace(-200.0, -49.0, 200)) \
        + list(linspace(-49.0, -9.0, 400)) \
        + list(linspace(-9.0, -3.0, 200)) \
        + list(linspace(-3.0, 5.0, 200)) \
    # equidistant grid, use larger grid for heavy atoms
              
    Z = 32
    Nelec = 32
    atomdft = PseudoAtomDFT(Z,Nelec,numerov_conv,en_conv, grid_spacing="exponential", r0=confinement_radii_byZ[Z], damping=0.6)    
    atomdft.setRadialGrid(0.0000004, 20.0, 6000)
    try:
        from confined_pseudo_atoms import ga
        atomdft.initialDensityGuess((ga.r, float(Nelec)/float(ga.Nelec) * ga.radial_density))
        #from confined_pseudo_atoms import ge
        #atomdft.initialDensityGuess((ge.r, ge.radial_density))
    except ImportError:
        atomdft.initialDensityGuess()
    atomdft.setEnergyRange(energy_range)
    atomdft.solveKohnSham()
    fh = open(orbdir + "ge.py", "w")
    atomdft.saveSolution(fh)
    fh.close()

# test version !!!
def arsenic():
    energy_range = list(linspace(-470.0, -200.0, 200)) \
        + list(linspace(-200.0, -53.0, 200)) \
        + list(linspace(-53.0, -8.2, 400)) \
        + list(linspace(-8.2, -4.5, 200)) \
        + list(linspace(-4.5, 5.0, 200)) \
    # equidistant grid, use larger grid for heavy atoms
              
    Z = 33
    Nelec = 33
    atomdft = PseudoAtomDFT(Z,Nelec,numerov_conv,en_conv, grid_spacing="exponential", r0=confinement_radii_byZ[Z], damping=0.6)    
    atomdft.setRadialGrid(0.0000004, 20.0, 6000)
    try:
        from confined_pseudo_atoms import ge
        atomdft.initialDensityGuess((ge.r, float(Nelec)/float(ge.Nelec) * ge.radial_density))
        #from confined_pseudo_atoms import arsenic
        #atomdft.initialDensityGuess((arsenic.r, arsenic.radial_density))
    except ImportError:
        atomdft.initialDensityGuess()
    atomdft.setEnergyRange(energy_range)
    atomdft.solveKohnSham()
    fh = open(orbdir + "arsenic.py", "w")
    atomdft.saveSolution(fh)
    fh.close()

# test version !!!
def selenium():
    energy_range = list(linspace(-500.0, -200.0, 200)) \
        + list(linspace(-200.0, -57.0, 200)) \
        + list(linspace(-57.0, -7.0, 400)) \
        + list(linspace(-7.0, -4.0, 200)) \
        + list(linspace(-4.0, 5.0, 200)) \
    # equidistant grid, use larger grid for heavy atoms
              
    Z = 34
    Nelec = 34
    atomdft = PseudoAtomDFT(Z,Nelec,numerov_conv,en_conv, grid_spacing="exponential", r0=confinement_radii_byZ[Z], damping=0.6)    
    atomdft.setRadialGrid(0.0000004, 20.0, 6000)
    try:
        from confined_pseudo_atoms import br
        atomdft.initialDensityGuess((br.r, float(Nelec)/float(br.Nelec) * br.radial_density))
        #from confined_pseudo_atoms import se
        #atomdft.initialDensityGuess((se.r, se.radial_density))
    except ImportError:
        atomdft.initialDensityGuess()
    atomdft.setEnergyRange(energy_range)
    atomdft.solveKohnSham()
    fh = open(orbdir + "se.py", "w")
    atomdft.saveSolution(fh)
    fh.close()

def bromine():
    energy_range = list(linspace(-1500.0, -400.0, 200)) \
        + list(linspace(-400.0, -200.0, 200)) \
        + list(linspace(-200.0, -30.0, 200)) \
        + list(linspace(-30.0, -5.0, 400)) \
        + list(linspace(-5.0, 5.0, 200)) \
    # equidistant grid, use larger grid for heavy atoms
              
    Z = 35
    Nelec = 35
    atomdft = PseudoAtomDFT(Z,Nelec,numerov_conv,en_conv, grid_spacing="exponential", r0=confinement_radii_byZ[Z], damping=0.6)
    # 3d-shell is closed (occupied by 10 electrons)
    atomdft.setValenceOrbitals(["4s", "4p"], format="spectroscopic")    
    #atomdft.setRadialGrid(0.0000004, 14.0, 10000)
    atomdft.setRadialGrid(0.0000004, 18.0, 10000)
    try:
        from confined_pseudo_atoms import br
        atomdft.initialDensityGuess((br.r, br.radial_density))
    except ImportError:
        atomdft.initialDensityGuess()
    atomdft.setEnergyRange(energy_range)
    atomdft.solveKohnSham()
    fh = open(orbdir + "br.py", "w")
    atomdft.saveSolution(fh)
    fh.close()

# test version !!!
def krypton():
    energy_range = list(linspace(-550.0, -200.0, 200)) \
        + list(linspace(-200.0, -66.0, 200)) \
        + list(linspace(-66.0, -8.2, 200)) \
        + list(linspace(-8.2, -6.5, 400)) \
        + list(linspace(-6.5, 5.0, 200)) \
    # equidistant grid, use larger grid for heavy atoms
              
    Z = 36
    Nelec = 36
    atomdft = PseudoAtomDFT(Z,Nelec,numerov_conv,en_conv, grid_spacing="exponential", r0=confinement_radii_byZ[Z], damping=0.6)
    # 3d-shell is closed (occupied by 10 electrons)
    atomdft.setValenceOrbitals(["4s", "4p"], format="spectroscopic")    
    #atomdft.setRadialGrid(0.0000004, 14.0, 10000)
    atomdft.setRadialGrid(0.0000004, 18.0, 10000)
    try:
        from confined_pseudo_atoms import br
        atomdft.initialDensityGuess((br.r, float(Nelec)/float(br.Nelec) * br.radial_density))
        #from confined_pseudo_atoms import ky
        #atomdft.initialDensityGuess((ky.r, ky.radial_density))
    except ImportError:
        atomdft.initialDensityGuess()
    atomdft.setEnergyRange(energy_range)
    atomdft.solveKohnSham()
    fh = open(orbdir + "ky.py", "w")
    atomdft.saveSolution(fh)
    fh.close()

# fifth row atoms

# test version !!!
def rubidium():
    energy_range = list(linspace(-650.0, -550.0, 100)) \
        + list(linspace(-550.0, -42.0, 200)) \
        + list(linspace(-42.0, -7.0, 200)) \
        + list(linspace(-7.0, -5.0, 400)) \
        + list(linspace(-5.0, 5.0, 400)) 
    # equidistant grid, use larger grid for heavy atoms

    Z = 37
    Nelec = 37
    atomdft = PseudoAtomDFT(Z,Nelec,numerov_conv,en_conv, grid_spacing="exponential", r0=confinement_radii_byZ[Z])
    # in rubdium 5s is filled before 4d: [Kr] 5s^1
    occupation = [occ for occ in occupation_numbers(Nelec)]
    assert occupation[-1] == (1, 4-1, 2) # 1e in 4d
    occupation[-1] = (1,5-1,0) # 1e in 5s
    atomdft.setOccupation(occupation)
    atomdft.setRadialGrid(0.000000001, 14.0, 20000)
    try:
        from confined_pseudo_atoms import sr
        atomdft.initialDensityGuess((sr.r, float(Nelec)/float(sr.Nelec) * sr.radial_density))
        #from confined_pseudo_atoms import rb
        #atomdft.initialDensityGuess((rb.r, rb.radial_density))
    except ImportError:
        atomdft.initialDensityGuess()
    atomdft.setEnergyRange(energy_range)
    atomdft.solveKohnSham()
    fh = open(orbdir + "rb.py", "w")
    atomdft.saveSolution(fh)
    fh.close()

# test version !!!
def strontium():
    energy_range = list(linspace(-650.0, -550.0, 100)) \
        + list(linspace(-550.0, -42.0, 200)) \
        + list(linspace(-42.0, -7.0, 200)) \
        + list(linspace(-7.0, -5.0, 400)) \
        + list(linspace(-5.0, 5.0, 400)) \
    # equidistant grid, use larger grid for heavy atoms

    Z = 38
    Nelec = 38
    atomdft = PseudoAtomDFT(Z,Nelec,numerov_conv,en_conv, grid_spacing="exponential", r0=confinement_radii_byZ[Z])
    # in strontium 5s is filled before 4d: [Kr] 5s^2
    occupation = [occ for occ in occupation_numbers(Nelec)]
    assert occupation[-1] == (2,4-1,2) # 2e in 4d
    occupation[-1] = (0,4-1,2)     # put 0e in 4d
    occupation.append( (2,5-1,0) ) # and 2e in 5s
    atomdft.setOccupation(occupation)
    atomdft.setRadialGrid(0.000000001, 14.0, 20000)
    try:
        from confined_pseudo_atoms import y
        atomdft.initialDensityGuess((y.r, float(Nelec)/float(y.Nelec) * y.radial_density))
        #from confined_pseudo_atoms import sr
        #atomdft.initialDensityGuess((sr.r, sr.radial_density))
    except ImportError:
        atomdft.initialDensityGuess()
    atomdft.setEnergyRange(energy_range)
    atomdft.solveKohnSham()
    fh = open(orbdir + "sr.py", "w")
    atomdft.saveSolution(fh)
    fh.close()

# test version !!!
def yttrium():
    energy_range = list(linspace(-700.0, -580.0, 100)) \
        + list(linspace(-580.0, -45.0, 200)) \
        + list(linspace(-45.0, -8.0, 200)) \
        + list(linspace(-8.0, -5.0, 400)) \
        + list(linspace(-5.0, 5.0, 400)) \
    # equidistant grid, use larger grid for heavy atoms
              
    Z = 39
    Nelec = 39
    atomdft = PseudoAtomDFT(Z,Nelec,numerov_conv,en_conv, grid_spacing="exponential", r0=confinement_radii_byZ[Z], damping=0.6)
    # The electron configuration of ruthenium is  [Kr] 4d^(1) 5s^(2)
    occupation = [occ for occ in occupation_numbers(Nelec)]
    # `occupation` is a list of tuples
    #   (nr.electrons, quantum number n + 1, quantum number l)
    # that describe the occupied shells.
    # In the default occupation, yttrium would have the electronic configuration [Kr] 4d^(1).
    assert occupation[-1] == (3,4-1,2) # 3e in 4d
    # The default occupation is not correct, so we need to adjust it by
    # removing one electron from 4d shell,
    occupation[-1] = (1,4-1,2) # 1e in 4d
    # and adding a 5s shell with 1 electron.
    occupation.append( (2,5-1,0) ) # 2e in 5s
    # The new occupation is now [Kr] 4d^(1) 5s^(2)
    
    atomdft.setOccupation(occupation)
    atomdft.setValenceOrbitals(["4d", "5s"], format="spectroscopic")
    atomdft.setRadialGrid(0.000000001, 14.0, 20000)
    try:
        from confined_pseudo_atoms import zr
        atomdft.initialDensityGuess((zr.r, float(Nelec)/float(zr.Nelec) * zr.radial_density))
        #from confined_pseudo_atoms import y
        #atomdft.initialDensityGuess((y.r, y.radial_density)
    except ImportError:
        atomdft.initialDensityGuess()
    atomdft.setEnergyRange(energy_range)
    atomdft.solveKohnSham()
    fh = open(orbdir + "y.py", "w")
    atomdft.saveSolution(fh)
    fh.close()    

# test version !!!
def zirconium():
    energy_range = list(linspace(-730.0, -600.0, 100)) \
        + list(linspace(-600.0, -50.0, 200)) \
        + list(linspace(-50.0, -9.0, 200)) \
        + list(linspace(-9.0, -5.0, 400)) \
        + list(linspace(-5.0, 5.0, 400)) \
    # equidistant grid, use larger grid for heavy atoms
              
    Z = 40
    Nelec = 40
    atomdft = PseudoAtomDFT(Z,Nelec,numerov_conv,en_conv, grid_spacing="exponential", r0=confinement_radii_byZ[Z], damping=0.6)
    # The electron configuration of ruthenium is  [Kr] 4d^(2) 5s^(2)
    occupation = [occ for occ in occupation_numbers(Nelec)]
    # `occupation` is a list of tuples
    #   (nr.electrons, quantum number n + 1, quantum number l)
    # that describe the occupied shells.
    # In the default occupation, zirconium would have the electronic configuration [Kr] 4d^(2).
    assert occupation[-1] == (4,4-1,2) # 4e in 4d
    # The default occupation is not correct, so we need to adjust it by
    # removing one electron from 4d shell,
    occupation[-1] = (2,4-1,2) # 2e in 4d
    # and adding a 5s shell with 2 electron.
    occupation.append( (2,5-1,0) ) # 2e in 5s
    # The new occupation is now [Kr] 4d^(2) 5s^(2)
    
    atomdft.setOccupation(occupation)
    atomdft.setValenceOrbitals(["4d", "5s"], format="spectroscopic")
    atomdft.setRadialGrid(0.000000001, 14.0, 20000)
    try:
        from confined_pseudo_atoms import nb
        atomdft.initialDensityGuess((nb.r, float(Nelec)/float(nb.Nelec) * nb.radial_density))
        #from confined_pseudo_atoms import zr
        #atomdft.initialDensityGuess((zr.r, zr.radial_density)
    except ImportError:
        atomdft.initialDensityGuess()
    atomdft.setEnergyRange(energy_range)
    atomdft.solveKohnSham()
    fh = open(orbdir + "zr.py", "w")
    atomdft.saveSolution(fh)
    fh.close()    

# test version !!!
def niobium():
    energy_range = list(linspace(-750.0, -650.0, 100)) \
        + list(linspace(-650.0, -55.0, 200)) \
        + list(linspace(-55.0, -11.0, 200)) \
        + list(linspace(-11.0, -5.0, 400)) \
        + list(linspace(-5.0, 5.0, 400)) \
    # equidistant grid, use larger grid for heavy atoms
              
    Z = 41
    Nelec = 41
    atomdft = PseudoAtomDFT(Z,Nelec,numerov_conv,en_conv, grid_spacing="exponential", r0=confinement_radii_byZ[Z], damping=0.6)
    # The electron configuration of ruthenium is  [Kr] 4d^(3) 5s^(2)
    occupation = [occ for occ in occupation_numbers(Nelec)]
    # `occupation` is a list of tuples
    #   (nr.electrons, quantum number n + 1, quantum number l)
    # that describe the occupied shells.
    # In the default occupation, ruthenium would have the electronic configuration [Kr] 4d^(3).
    assert occupation[-1] == (5,4-1,2) # 5e in 4d
    # The default occupation is not correct, so we need to adjust it by
    # removing one electron from 4d shell,
    occupation[-1] = (3,4-1,2) # 3e in 4d
    # and adding a 5s shell with 2 electron.
    occupation.append( (2,5-1,0) ) # qe in 5s
    # The new occupation is now [Kr] 4d^(3) 5s^(2)
    
    atomdft.setOccupation(occupation)
    atomdft.setValenceOrbitals(["4d", "5s"], format="spectroscopic")
    atomdft.setRadialGrid(0.000000001, 14.0, 20000)
    try:
        from confined_pseudo_atoms import mo
        atomdft.initialDensityGuess((mo.r, float(Nelec)/float(mo.Nelec) * mo.radial_density))
        #from confined_pseudo_atoms import nb
        #atomdft.initialDensityGuess((nb.r, nb.radial_density)
    except ImportError:
        atomdft.initialDensityGuess()
    atomdft.setEnergyRange(energy_range)
    atomdft.solveKohnSham()
    fh = open(orbdir + "nb.py", "w")
    atomdft.saveSolution(fh)
    fh.close()    

# test version !!!
def molybdenum():
    energy_range = list(linspace(-750.0, -700.0, 100)) \
        + list(linspace(-700.0, -55.0, 200)) \
        + list(linspace(-55.0, -11.0, 200)) \
        + list(linspace(-11.0, -5.0, 400)) \
        + list(linspace(-5.0, 5.0, 400)) \
    # equidistant grid, use larger grid for heavy atoms
              
    Z = 42
    Nelec = 42
    atomdft = PseudoAtomDFT(Z,Nelec,numerov_conv,en_conv, grid_spacing="exponential", r0=confinement_radii_byZ[Z], damping=0.6)
    # The electron configuration of ruthenium is  [Kr] 4d^(4) 5s^(2)
    occupation = [occ for occ in occupation_numbers(Nelec)]
    # `occupation` is a list of tuples
    #   (nr.electrons, quantum number n + 1, quantum number l)
    # that describe the occupied shells.
    # In the default occupation, molybdenum would have the electronic configuration [Kr] 4d^(4).
    assert occupation[-1] == (6,4-1,2) # 6e in 4d
    # The default occupation is not correct, so we need to adjust it by
    # removing one electron from 4d shell,
    occupation[-1] = (4,4-1,2)  # 4e in 4d
    # and adding a 5s shell with 2 electron.
    occupation.append( (2,5-1,0) ) # 2e in 5s
    # The new occupation is now [Kr] 4d^(4) 5s^(2)
    
    atomdft.setOccupation(occupation)
    atomdft.setValenceOrbitals(["4d", "5s"], format="spectroscopic")
    atomdft.setRadialGrid(0.000000001, 14.0, 20000)
    try:
        from confined_pseudo_atoms import tc
        atomdft.initialDensityGuess((tc.r, float(Nelec)/float(tc.Nelec) * tc.radial_density))
        #from confined_pseudo_atoms import mo
        #atomdft.initialDensityGuess((mo.r, mo.radial_density)
    except ImportError:
        atomdft.initialDensityGuess()
    atomdft.setEnergyRange(energy_range)
    atomdft.solveKohnSham()
    fh = open(orbdir + "mo.py", "w")
    atomdft.saveSolution(fh)
    fh.close()    

# test version !!!
def technetium():
    energy_range = list(linspace(-800.0, -700.0, 100)) \
        + list(linspace(-700.0, -65.0, 200)) \
        + list(linspace(-65.0, -13.0, 200)) \
        + list(linspace(-13.0, -5.0, 400)) \
        + list(linspace(-5.0, 5.0, 400)) \
    # equidistant grid, use larger grid for heavy atoms
              
    Z = 43
    Nelec = 43
    atomdft = PseudoAtomDFT(Z,Nelec,numerov_conv,en_conv, grid_spacing="exponential", r0=confinement_radii_byZ[Z], damping=0.6)
    # The electron configuration of ruthenium is  [Kr] 4d^(5) 5s^(2)
    occupation = [occ for occ in occupation_numbers(Nelec)]
    # `occupation` is a list of tuples
    #   (nr.electrons, quantum number n + 1, quantum number l)
    # that describe the occupied shells.
    # In the default occupation, technetium would have the electronic configuration [Kr] 4d^(5).
    assert occupation[-1] == (7,4-1,2) # 7e in 4d
    # The default occupation is not correct, so we need to adjust it by
    # removing one electron from 4d shell,
    occupation[-1] = (5,4-1,2) # 5e in 4d
    # and adding a 5s shell with 2 electron.
    occupation.append( (2,5-1,0) ) # 2e in 5s
    # The new occupation is now [Kr] 4d^(5) 5s^(2)
    
    atomdft.setOccupation(occupation)
    atomdft.setValenceOrbitals(["4d", "5s"], format="spectroscopic")
    atomdft.setRadialGrid(0.000000001, 14.0, 20000)
    try:
        from confined_pseudo_atoms import ru
        atomdft.initialDensityGuess((ru.r, float(Nelec)/float(ru.Nelec) * ru.radial_density))
        #from confined_pseudo_atoms import tc
        #atomdft.initialDensityGuess((tc.r, tc.radial_density)
    except ImportError:
        atomdft.initialDensityGuess()
    atomdft.setEnergyRange(energy_range)
    atomdft.solveKohnSham()
    fh = open(orbdir + "tc.py", "w")
    atomdft.saveSolution(fh)
    fh.close()    

def ruthenium():
    energy_range = list(linspace(-1500.0, -800.0, 100)) \
        + list(linspace(-800.0, -200.0, 200)) \
        + list(linspace(-200.0, -30.0, 200)) \
        + list(linspace(-30.0, -5.0, 400)) \
        + list(linspace(-5.0, 5.0, 400)) \
    # equidistant grid, use larger grid for heavy atoms
              
    Z = 44
    Nelec = 44
    atomdft = PseudoAtomDFT(Z,Nelec,numerov_conv,en_conv, grid_spacing="exponential", r0=confinement_radii_byZ[Z], damping=0.6)
    # The electron configuration of ruthenium is  [Kr] 4d^(7) 5s^(1)
    occupation = [occ for occ in occupation_numbers(Nelec)]
    # `occupation` is a list of tuples
    #   (nr.electrons, quantum number n + 1, quantum number l)
    # that describe the occupied shells.
    # In the default occupation, ruthenium would have the electronic configuration [Kr] 4d^(8).
    assert occupation[-1] == (8,4-1,2) # 8e in 4d
    # The default occupation is not correct, so we need to adjust it by
    # removing one electron from 4d shell,
    occupation[-1] = (7,4-1,2) # 7e in 4d
    # and adding a 5s shell with 1 electron.
    occupation.append( (1,5-1,0) ) # 1e in 5s
    # The new occupation is now [Kr] 4d^(7) 5s^(1)
    
    atomdft.setOccupation(occupation)
    atomdft.setValenceOrbitals(["4d", "5s"], format="spectroscopic")
    atomdft.setRadialGrid(0.000000001, 14.0, 20000)
    try:
        """
        # If no initial density is available for Ru, we can take the density from silver 
        # and scale it, so that it integrates to the number of electrons in ruthenium. 
        from confined_pseudo_atoms import ag
        atomdft.initialDensityGuess((ag.r, float(Nelec)/float(ag.Nelec) * ag.radial_density))
        """
        # If an initial density is available for Ru, we use that one
        from confined_pseudo_atoms import ru
        atomdft.initialDensityGuess((ru.r, float(Nelec)/float(ru.Nelec) * ru.radial_density))
    except ImportError:
        atomdft.initialDensityGuess()
    atomdft.setEnergyRange(energy_range)
    atomdft.solveKohnSham()
    fh = open(orbdir + "ru.py", "w")
    atomdft.saveSolution(fh)
    fh.close()    

def ruthenium_2plus():
    energy_range = list(linspace(-1500.0, -800.0, 100)) \
        + list(linspace(-800.0, -200.0, 200)) \
        + list(linspace(-200.0, -30.0, 200)) \
        + list(linspace(-30.0, -5.0, 400)) \
        + list(linspace(-5.0, 5.0, 400)) \
    # equidistant grid, use larger grid for heavy atoms
              
    Z = 44
    Nelec = 42     # Ru^(2+)
    atomdft = PseudoAtomDFT(Z,Nelec,numerov_conv,en_conv, grid_spacing="exponential", r0=confinement_radii_byZ[Z], damping=0.6)
    # The electron configuration of Ru^(2+) is  [Kr] 4d^(6)
    occupation = [occ for occ in occupation_numbers(Nelec)]
    # `occupation` is a list of tuples
    #   (nr.electrons, quantum number n + 1, quantum number l)
    # that describe the occupied shells.
    # In the default occupation, ruthenium would have the electronic configuration [Kr] 4d^(6),
    # which is correct.
    assert occupation[-1] == (6,4-1,2) # 6e in 4d
    
    atomdft.setOccupation(occupation)
    atomdft.setValenceOrbitals(["4d", "5s"], format="spectroscopic")
    atomdft.setRadialGrid(0.000000001, 14.0, 20000)
    try:
        # If no initial density is available for Ru^(2+), we can take the density from silver 
        # and scale it, so that it integrates to the number of electrons in ruthenium 2+. 
        from confined_pseudo_atoms import ag
        atomdft.initialDensityGuess((ag.r, float(Nelec)/float(ag.Nelec) * ag.radial_density))
        """
        # If an initial density is available for Ru^(2+), we use that one
        from confined_pseudo_atoms import ru
        atomdft.initialDensityGuess((ru.r, ru.radial_density))
        """
    except ImportError:
        atomdft.initialDensityGuess()
    atomdft.setEnergyRange(energy_range)
    atomdft.solveKohnSham()
    # Although we have calculated Ru^(2+), the pseudoorbitals still have to be saved under the name 'ru.py'.
    # Only one oxidation state can be used for any atom at the same time.
    fh = open(orbdir + "ru.py", "w")
    atomdft.saveSolution(fh)
    fh.close()    

# test version !!!
def rhodium():
    energy_range = list(linspace(-900.0, -800.0, 100)) \
        + list(linspace(-800.0, -65.0, 200)) \
        + list(linspace(-65.0, -13.5, 200)) \
        + list(linspace(-13.5, -5.0, 400)) \
        + list(linspace(-5.0, 5.0, 400)) \
    # equidistant grid, use larger grid for heavy atoms
              
    Z = 45
    Nelec = 45
    atomdft = PseudoAtomDFT(Z,Nelec,numerov_conv,en_conv, grid_spacing="exponential", r0=confinement_radii_byZ[Z], damping=0.6)
    # The electron configuration of ruthenium is  [Kr] 4d^(8) 5s^(1)
    occupation = [occ for occ in occupation_numbers(Nelec)]
    # `occupation` is a list of tuples
    #   (nr.electrons, quantum number n + 1, quantum number l)
    # that describe the occupied shells.
    # In the default occupation, rhodium would have the electronic configuration [Kr] 4d^(8).
    assert occupation[-1] == (9,4-1,2) # 9e in 4d
    # The default occupation is not correct, so we need to adjust it by
    # removing one electron from 4d shell,
    occupation[-1] = (8,4-1,2) # 8e in 4d
    # and adding a 5s shell with 1 electron.
    occupation.append( (1,5-1,0) ) # 1e in 5s
    # The new occupation is now [Kr] 4d^(7) 5s^(1)
    
    atomdft.setOccupation(occupation)
    atomdft.setValenceOrbitals(["4d", "5s"], format="spectroscopic")
    atomdft.setRadialGrid(0.000000001, 14.0, 20000)
    try:
        from confined_pseudo_atoms import ru
        atomdft.initialDensityGuess((ru.r, float(Nelec)/float(ru.Nelec) * ru.radial_density))
        #from confined_pseudo_atoms import rh
        #atomdft.initialDensityGuess((rh.r, rh.radial_density))
    except ImportError:
        atomdft.initialDensityGuess()
    atomdft.setEnergyRange(energy_range)
    atomdft.solveKohnSham()
    fh = open(orbdir + "rh.py", "w")
    atomdft.saveSolution(fh)
    fh.close()    

# test version !!!
def palladium():
    energy_range = list(linspace(-950.0, -800.0, 100)) \
        + list(linspace(-800.0, -70.0, 200)) \
        + list(linspace(-70.0, -15.0, 200)) \
        + list(linspace(-15.0, -5.0, 400)) \
        + list(linspace(-5.0, 5.0, 400)) \
    # equidistant grid, use larger grid for heavy atoms
              
    Z = 46
    Nelec = 46
    atomdft = PseudoAtomDFT(Z,Nelec,numerov_conv,en_conv, grid_spacing="exponential", r0=confinement_radii_byZ[Z], damping=0.6)
    # In silver the 5s is filled before 4f: [Kr] 4d^(9) 5s^1
    occupation = [occ for occ in occupation_numbers(Nelec)]
    #assert occupation[-1] == (1,4-1,3)
    assert occupation[-1] == (10,4-1,2) # 10e in 4d
    occupation[-1] = (9,4-1,2) # 9e in 4d
    occupation.append( (1,5-1,0) ) # 1e in 5s
    atomdft.setOccupation(occupation)
    atomdft.setValenceOrbitals(["4d","5s","5p"], format="spectroscopic")
    atomdft.setRadialGrid(0.000000001, 14.0, 20000)
    try:
        from confined_pseudo_atoms import rh
        atomdft.initialDensityGuess((rh.r, float(Nelec)/float(rh.Nelec) * rh.radial_density))
        #from confined_pseudo_atoms import pd
        #atomdft.initialDensityGuess((pd.r, pd.radial_density))
    except ImportError:
        atomdft.initialDensityGuess()
    atomdft.setEnergyRange(energy_range)
    atomdft.solveKohnSham()
    fh = open(orbdir + "pd.py", "w")
    atomdft.saveSolution(fh)
    fh.close()    

def silver():
    energy_range = list(linspace(-1050.0, -800.0, 100)) \
        + list(linspace(-800.0, -76.0, 200)) \
        + list(linspace(-76.0, -17.0, 200)) \
        + list(linspace(-17.0, -5.0, 400)) \
        + list(linspace(-5.0, 5.0, 400)) \
    # equidistant grid, use larger grid for heavy atoms
              
    Z = 47
    Nelec = 47
    atomdft = PseudoAtomDFT(Z,Nelec,numerov_conv,en_conv, grid_spacing="exponential", r0=confinement_radii_byZ[Z], damping=0.6)
    # In silver the 5s is filled before 4f: [Kr] 4d^(10) 5s^1
    occupation = [occ for occ in occupation_numbers(Nelec)]
    assert occupation[-1] == (1,4-1,3) # 1e in 4f
    occupation[-1] = (1,5-1,0) # 1e in 5s
    atomdft.setOccupation(occupation)
    # I would like to include unoccupied f orbitals in minimal basis but 
    # sofar no Slater rules exist for f orbitals. So include 5p
    atomdft.setValenceOrbitals(["4d","5s","5p"], format="spectroscopic")
    atomdft.setRadialGrid(0.000000001, 14.0, 20000)
    try:
        from confined_pseudo_atoms import cu
        atomdft.initialDensityGuess((cu.r, float(Nelec)/float(cu.Nelec) * cu.radial_density))
        #from confined_pseudo_atoms import ag
        #atomdft.initialDensityGuess((ag.r, ag.radial_density))
    except ImportError:
        atomdft.initialDensityGuess()
    atomdft.setEnergyRange(energy_range)
    atomdft.solveKohnSham()
    fh = open(orbdir + "ag.py", "w")
    atomdft.saveSolution(fh)
    fh.close()    

# test version !!!
def cadmium():
    energy_range = list(linspace(-1500.0, -800.0, 100)) \
        + list(linspace(-800.0, -200.0, 200)) \
        + list(linspace(-200.0, -30.0, 200)) \
        + list(linspace(-30.0, -5.0, 400)) \
        + list(linspace(-5.0, 5.0, 400)) \
    # equidistant grid, use larger grid for heavy atoms
              
    Z = 48
    Nelec = 48
    atomdft = PseudoAtomDFT(Z,Nelec,numerov_conv,en_conv, grid_spacing="exponential", r0=confinement_radii_byZ[Z], damping=0.6)
    # In silver the 5s is filled before 4f: [Kr] 4d^(10) 5s^2
    #occupation = [occ for occ in occupation_numbers(Nelec)]
    #assert occupation[-1] == (1,4-1,3) # 1e in 4f
    #occupation[-1] = (2,5-1,0) # 2e in 5s
    #atomdft.setOccupation(occupation)
    # I would like to include unoccupied f orbitals in minimal basis but 
    # sofar no Slater rules exist for f orbitals. So include 5p
    #atomdft.setValenceOrbitals(["4d", "5s","5p"], format="spectroscopic")
    atomdft.setRadialGrid(0.000000001, 14.0, 20000)
    try:
        from confined_pseudo_atoms import ag
        atomdft.initialDensityGuess((ag.r, float(Nelec)/float(ag.Nelec) * ag.radial_density))
        #from confined_pseudo_atoms import cd
        #atomdft.initialDensityGuess((cd.r, cd.radial_density))
    except ImportError:
        atomdft.initialDensityGuess()
    atomdft.setEnergyRange(energy_range)
    atomdft.solveKohnSham()
    fh = open(orbdir + "cd.py", "w")
    atomdft.saveSolution(fh)
    fh.close()    

# test version !!!
def indium():
    energy_range = list(linspace(-1500.0, -800.0, 100)) \
        + list(linspace(-800.0, -200.0, 200)) \
        + list(linspace(-200.0, -30.0, 200)) \
        + list(linspace(-30.0, -5.0, 400)) \
        + list(linspace(-5.0, 5.0, 400)) \
    # equidistant grid, use larger grid for heavy atoms
              
    Z = 49
    Nelec = 49
    atomdft = PseudoAtomDFT(Z,Nelec,numerov_conv,en_conv, grid_spacing="exponential", r0=confinement_radii_byZ[Z], damping=0.6)    
    atomdft.setRadialGrid(0.000000001, 14.0, 20000)
    try:
        from confined_pseudo_atoms import cd
        atomdft.initialDensityGuess((cd.r, float(Nelec)/float(cd.Nelec) * cd.radial_density))
        #from confined_pseudo_atoms import indium
        #atomdft.initialDensityGuess((indium.r, indium.radial_density))
    except ImportError:
        atomdft.initialDensityGuess()
    atomdft.setEnergyRange(energy_range)
    atomdft.solveKohnSham()
    fh = open(orbdir + "indium.py", "w")
    atomdft.saveSolution(fh)
    fh.close()    

# test version !!!
def tin():
    energy_range = list(linspace(-1500.0, -800.0, 100)) \
        + list(linspace(-800.0, -200.0, 200)) \
        + list(linspace(-200.0, -30.0, 200)) \
        + list(linspace(-30.0, -5.0, 400)) \
        + list(linspace(-5.0, 5.0, 400)) \
    # equidistant grid, use larger grid for heavy atoms
              
    Z = 50
    Nelec = 50
    atomdft = PseudoAtomDFT(Z,Nelec,numerov_conv,en_conv, grid_spacing="exponential", r0=confinement_radii_byZ[Z], damping=0.6)    
    atomdft.setRadialGrid(0.000000001, 14.0, 20000)
    try:
        from confined_pseudo_atoms import sb
        atomdft.initialDensityGuess((sb.r, float(Nelec)/float(sb.Nelec) * sb.radial_density))
        #from confined_pseudo_atoms import sn
        #atomdft.initialDensityGuess((sn.r, sn.radial_density))
    except ImportError:
        atomdft.initialDensityGuess()
    atomdft.setEnergyRange(energy_range)
    atomdft.solveKohnSham()
    fh = open(orbdir + "sn.py", "w")
    atomdft.saveSolution(fh)
    fh.close()    

# test version !!!
def antimony():
    energy_range = list(linspace(-1500.0, -400.0, 200)) \
        + list(linspace(-400.0, -200.0, 200)) \
        + list(linspace(-200.0, -30.0, 200)) \
        + list(linspace(-30.0, -5.0, 400)) \
        + list(linspace(-5.0, 5.0, 200)) \
    # equidistant grid, use larger grid for heavy atoms
              
    Z = 51
    Nelec = 51
    atomdft = PseudoAtomDFT(Z,Nelec,numerov_conv,en_conv, grid_spacing="exponential", r0=confinement_radii_byZ[Z], damping=0.6)    
    atomdft.setRadialGrid(0.0000004, 14.0, 10000)
    try:
        from confined_pseudo_atoms import te
        atomdft.initialDensityGuess((te.r, float(Nelec)/float(te.Nelec) * te.radial_density))
        #from confined_pseudo_atoms import sb
        #atomdft.initialDensityGuess((sb.r, sb.radial_density))
    except ImportError:
        atomdft.initialDensityGuess()
    atomdft.setEnergyRange(energy_range)
    atomdft.solveKohnSham()
    fh = open(orbdir + "sb.py", "w")
    atomdft.saveSolution(fh)
    fh.close()

# test version !!!
def tellurium():
    energy_range = list(linspace(-1500.0, -400.0, 200)) \
        + list(linspace(-400.0, -200.0, 200)) \
        + list(linspace(-200.0, -30.0, 200)) \
        + list(linspace(-30.0, -5.0, 400)) \
        + list(linspace(-5.0, 5.0, 200)) \
    # equidistant grid, use larger grid for heavy atoms
              
    Z = 52
    Nelec = 52
    atomdft = PseudoAtomDFT(Z,Nelec,numerov_conv,en_conv, grid_spacing="exponential", r0=confinement_radii_byZ[Z], damping=0.6)    
    atomdft.setRadialGrid(0.0000004, 14.0, 10000)
    try:
        from confined_pseudo_atoms import i
        atomdft.initialDensityGuess((i.r, float(Nelec)/float(i.Nelec) * i.radial_density))
        #from confined_pseudo_atoms import te
        #atomdft.initialDensityGuess((te.r, te.radial_density))
    except ImportError:
        atomdft.initialDensityGuess()
    atomdft.setEnergyRange(energy_range)
    atomdft.solveKohnSham()
    fh = open(orbdir + "te.py", "w")
    atomdft.saveSolution(fh)
    fh.close()

def iodine():
    energy_range = list(linspace(-1500.0, -400.0, 200)) \
        + list(linspace(-400.0, -200.0, 200)) \
        + list(linspace(-200.0, -30.0, 200)) \
        + list(linspace(-30.0, -5.0, 400)) \
        + list(linspace(-5.0, 5.0, 200)) \
    # equidistant grid, use larger grid for heavy atoms
              
    Z = 53
    Nelec = 53
    atomdft = PseudoAtomDFT(Z,Nelec,numerov_conv,en_conv, grid_spacing="exponential", r0=confinement_radii_byZ[Z], damping=0.6)    
    atomdft.setRadialGrid(0.0000004, 14.0, 10000)
    try:
        from confined_pseudo_atoms import i
        atomdft.initialDensityGuess((i.r, i.radial_density))
    except ImportError:
        atomdft.initialDensityGuess()
    atomdft.setEnergyRange(energy_range)
    atomdft.solveKohnSham()
    fh = open(orbdir + "i.py", "w")
    atomdft.saveSolution(fh)
    fh.close()

# test version !!!
def xenon():
    energy_range = list(linspace(-1500.0, -400.0, 200)) \
        + list(linspace(-400.0, -200.0, 200)) \
        + list(linspace(-200.0, -30.0, 200)) \
        + list(linspace(-30.0, -5.0, 400)) \
        + list(linspace(-5.0, 5.0, 200)) \
    # equidistant grid, use larger grid for heavy atoms
              
    Z = 54
    Nelec = 54
    atomdft = PseudoAtomDFT(Z,Nelec,numerov_conv,en_conv, grid_spacing="exponential", r0=confinement_radii_byZ[Z], damping=0.6)    
    atomdft.setRadialGrid(0.0000004, 14.0, 10000)
    try:
        from confined_pseudo_atoms import xe
        atomdft.initialDensityGuess((xe.r, xe.radial_density))
    except ImportError:
        atomdft.initialDensityGuess()
    atomdft.setEnergyRange(energy_range)
    atomdft.solveKohnSham()
    fh = open(orbdir + "xe.py", "w")
    atomdft.saveSolution(fh)
    fh.close()

# sixth row atoms

# test version !!!
def lanthanum():
    energy_range = list(linspace(-1500.0, -800.0, 100)) \
        + list(linspace(-800.0, -200.0, 200)) \
        + list(linspace(-200.0, -30.0, 200)) \
        + list(linspace(-30.0, -5.0, 400)) \
        + list(linspace(-5.0, 5.0, 400)) \
    # equidistant grid, use larger grid for heavy atoms
              
    Z = 57
    Nelec = 57
    atomdft = PseudoAtomDFT(Z,Nelec,numerov_conv,en_conv, grid_spacing="exponential", r0=confinement_radii_byZ[Z], damping=0.6)
    # The electron configuration of ruthenium is  [Xe] 5d^(1) 6s^(2)
    occupation = [occ for occ in occupation_numbers(Nelec)]
    # `occupation` is a list of tuples
    #   (nr.electrons, quantum number n + 1, quantum number l)
    # that describe the occupied shells.
    # In the default occupation, lanthanum would have the electronic configuration [Kr] 4d^(1).
    assert occupation[-1] == (3,5-1,2) # 3e in 5d
    # The default occupation is not correct, so we need to adjust it by
    # removing one electron from 5d shell,
    occupation[-1] = (1,5-1,2) # 1e in 5d
    # and adding a 6s shell with 1 electron.
    occupation.append( (2,5-1,0) ) # 2e in 6s
    # The new occupation is now [Kr] 5d^(1) 6s^(2)
    
    atomdft.setOccupation(occupation)
    atomdft.setValenceOrbitals(["5d","6s","6p"], format="spectroscopic")
    atomdft.setRadialGrid(0.000000001, 14.0, 20000)
    try:
        from confined_pseudo_atoms import y
        atomdft.initialDensityGuess((y.r, float(Nelec)/float(y.Nelec) * y.radial_density))
        #from confined_pseudo_atoms import la
        #atomdft.initialDensityGuess((la.r, la.radial_density))
    except ImportError:
        atomdft.initialDensityGuess()
    atomdft.setEnergyRange(energy_range)
    atomdft.solveKohnSham()
    fh = open(orbdir + "la.py", "w")
    atomdft.saveSolution(fh)
    fh.close()    

# test version !!!
def hafnium():
    energy_range = list(linspace(-1500.0, -400.0, 200)) \
        + list(linspace(-400.0, -200.0, 200)) \
        + list(linspace(-200.0, -30.0, 200)) \
        + list(linspace(-30.0, -5.0, 400)) \
        + list(linspace(-5.0, 5.0, 200)) \
    # equidistant grid, use larger grid for heavy atoms
              
    Z = 72
    Nelec = 72
    atomdft = PseudoAtomDFT(Z,Nelec,numerov_conv,en_conv, grid_spacing="exponential", r0=confinement_radii_byZ[Z], damping=0.6)
    # The electron configuration of ruthenium is  [Xe] 5d^(2) 6s^(2)
    occupation = [occ for occ in occupation_numbers(Nelec)]
    # `occupation` is a list of tuples
    #   (nr.electrons, quantum number n + 1, quantum number l)
    # that describe the occupied shells.
    # In the default occupation, hafnium would have the electronic configuration [Xe] 5d^(2).
    assert occupation[-1] == (4,5-1,2) # 4e in 5d
    # The default occupation is not correct, so we need to adjust it by
    # removing one electron from 5d shell,
    occupation[-1] = (2,5-1,2) # 2e in 5d
    # and adding a 6s shell with 1 electron.
    occupation.append( (2,6-1,0) ) # 2e in 6s
    # The new occupation is now [Xe] 5d^(2) 6s^(2)
    
    atomdft.setOccupation(occupation)
    atomdft.setValenceOrbitals(["5d", "6s"], format="spectroscopic")
    atomdft.setRadialGrid(0.0000004, 14.0, 10000)
    try:
        from confined_pseudo_atoms import la
        atomdft.initialDensityGuess((la.r, float(Nelec)/float(la.Nelec) * la.radial_density))
        #from confined_pseudo_atoms import hf
        #atomdft.initialDensityGuess((hf.r, hf.radial_density))
    except ImportError:
        atomdft.initialDensityGuess()
    atomdft.setEnergyRange(energy_range)
    atomdft.solveKohnSham()
    fh = open(orbdir + "hf.py", "w")
    atomdft.saveSolution(fh)
    fh.close()    
    
# test version !!!
def tantalum():
    energy_range = list(linspace(-1500.0, -400.0, 200)) \
        + list(linspace(-400.0, -200.0, 200)) \
        + list(linspace(-200.0, -30.0, 200)) \
        + list(linspace(-30.0, -5.0, 400)) \
        + list(linspace(-5.0, 5.0, 200)) \
    # equidistant grid, use larger grid for heavy atoms
              
    Z = 73
    Nelec = 73
    atomdft = PseudoAtomDFT(Z,Nelec,numerov_conv,en_conv, grid_spacing="exponential", r0=confinement_radii_byZ[Z], damping=0.6)
    # The electron configuration of ruthenium is  [Xe] 5d^(3) 6s^(2)
    occupation = [occ for occ in occupation_numbers(Nelec)]
    # `occupation` is a list of tuples
    #   (nr.electrons, quantum number n + 1, quantum number l)
    # that describe the occupied shells.
    # In the default occupation, tantalum would have the electronic configuration [Xe] 5d^(3).
    assert occupation[-1] == (5,5-1,2) # 5e in 5d
    # The default occupation is not correct, so we need to adjust it by
    # removing one electron from 5d shell,
    occupation[-1] = (3,5-1,2) # 3e in 5d
    # and adding a 6s shell with 1 electron.
    occupation.append( (2,6-1,0) ) # 2e in 6s
    # The new occupation is now [Xe] 5d^(3) 6s^(2)
    
    atomdft.setOccupation(occupation)
    atomdft.setValenceOrbitals(["5d", "6s"], format="spectroscopic")
    atomdft.setRadialGrid(0.0000004, 14.0, 10000)
    try:
        from confined_pseudo_atoms import hf
        atomdft.initialDensityGuess((hf.r, float(Nelec)/float(hf.Nelec) * hf.radial_density))
        #from confined_pseudo_atoms import ta
        #atomdft.initialDensityGuess((ta.r, float(Nelec)/float(ta.Nelec) * ta.radial_density))
    except ImportError:
        atomdft.initialDensityGuess()
    atomdft.setEnergyRange(energy_range)
    atomdft.solveKohnSham()
    fh = open(orbdir + "ta.py", "w")
    atomdft.saveSolution(fh)
    fh.close()    
    
# test version !!!
def tungsten():
    energy_range = list(linspace(-1500.0, -400.0, 200)) \
        + list(linspace(-400.0, -200.0, 200)) \
        + list(linspace(-200.0, -30.0, 200)) \
        + list(linspace(-30.0, -5.0, 400)) \
        + list(linspace(-5.0, 5.0, 200)) \
    # equidistant grid, use larger grid for heavy atoms
              
    Z = 74
    Nelec = 74
    atomdft = PseudoAtomDFT(Z,Nelec,numerov_conv,en_conv, grid_spacing="exponential", r0=confinement_radii_byZ[Z], damping=0.6)
    # The electron configuration of ruthenium is  [Xe] 5d^(4) 6s^(2)
    occupation = [occ for occ in occupation_numbers(Nelec)]
    # `occupation` is a list of tuples
    #   (nr.electrons, quantum number n + 1, quantum number l)
    # that describe the occupied shells.
    # In the default occupation, tantalum would have the electronic configuration [Xe] 5d^(4).
    assert occupation[-1] == (6,5-1,2) # 6e in 5d
    # The default occupation is not correct, so we need to adjust it by
    # removing one electron from 5d shell,
    occupation[-1] = (4,5-1,2) # 4e in 5d
    # and adding a 6s shell with 1 electron.
    occupation.append( (2,6-1,0) ) # 2e in 6s
    # The new occupation is now [Xe] 5d^(4) 6s^(2)
    
    atomdft.setOccupation(occupation)
    atomdft.setValenceOrbitals(["5d", "6s"], format="spectroscopic")
    atomdft.setRadialGrid(0.0000004, 14.0, 10000)
    try:
        from confined_pseudo_atoms import ta
        atomdft.initialDensityGuess((ta.r, float(Nelec)/float(ta.Nelec) * ta.radial_density))
        #from confined_pseudo_atoms import w
        #atomdft.initialDensityGuess((w.r, float(Nelec)/float(w.Nelec) * w.radial_density))
    except ImportError:
        atomdft.initialDensityGuess()
    atomdft.setEnergyRange(energy_range)
    atomdft.solveKohnSham()
    fh = open(orbdir + "w.py", "w")
    atomdft.saveSolution(fh)
    fh.close()    
    
# test version !!!
def rhenium():
    energy_range = list(linspace(-1500.0, -400.0, 200)) \
        + list(linspace(-400.0, -200.0, 200)) \
        + list(linspace(-200.0, -30.0, 200)) \
        + list(linspace(-30.0, -5.0, 400)) \
        + list(linspace(-5.0, 5.0, 200)) \
    # equidistant grid, use larger grid for heavy atoms
              
    Z = 75
    Nelec = 75
    atomdft = PseudoAtomDFT(Z,Nelec,numerov_conv,en_conv, grid_spacing="exponential", r0=confinement_radii_byZ[Z], damping=0.6)
    # The electron configuration of ruthenium is  [Xe] 5d^(5) 6s^(2)
    occupation = [occ for occ in occupation_numbers(Nelec)]
    # `occupation` is a list of tuples
    #   (nr.electrons, quantum number n + 1, quantum number l)
    # that describe the occupied shells.
    # In the default occupation, tantalum would have the electronic configuration [Xe] 5d^(5).
    assert occupation[-1] == (7,5-1,2) # 7e in 5d
    # The default occupation is not correct, so we need to adjust it by
    # removing one electron from 5d shell,
    occupation[-1] = (5,5-1,2) # 5e in 5d
    # and adding a 6s shell with 1 electron.
    occupation.append( (2,6-1,0) ) # 2e in 6s
    # The new occupation is now [Xe] 5d^(5) 6s^(2)
    
    atomdft.setOccupation(occupation)
    atomdft.setValenceOrbitals(["5d", "6s"], format="spectroscopic")
    atomdft.setRadialGrid(0.0000004, 14.0, 10000)
    try:
        from confined_pseudo_atoms import w
        atomdft.initialDensityGuess((w.r, float(Nelec)/float(w.Nelec) * w.radial_density))
        #from confined_pseudo_atoms import re
        #atomdft.initialDensityGuess((re.r, re.radial_density))
    except ImportError:
        atomdft.initialDensityGuess()
    atomdft.setEnergyRange(energy_range)
    atomdft.solveKohnSham()
    fh = open(orbdir + "re.py", "w")
    atomdft.saveSolution(fh)
    fh.close()    
    
# test version !!!
def osmium():
    energy_range = list(linspace(-1500.0, -400.0, 200)) \
        + list(linspace(-400.0, -200.0, 200)) \
        + list(linspace(-200.0, -30.0, 200)) \
        + list(linspace(-30.0, -5.0, 400)) \
        + list(linspace(-5.0, 5.0, 200)) \
    # equidistant grid, use larger grid for heavy atoms
              
    Z = 76
    Nelec = 76
    atomdft = PseudoAtomDFT(Z,Nelec,numerov_conv,en_conv, grid_spacing="exponential", r0=confinement_radii_byZ[Z], damping=0.6)
    # The electron configuration of ruthenium is  [Xe] 5d^(6) 6s^(2)
    occupation = [occ for occ in occupation_numbers(Nelec)]
    # `occupation` is a list of tuples
    #   (nr.electrons, quantum number n + 1, quantum number l)
    # that describe the occupied shells.
    # In the default occupation, tantalum would have the electronic configuration [Xe] 5d^(6).
    assert occupation[-1] == (8,5-1,2) # 8e in 5d
    # The default occupation is not correct, so we need to adjust it by
    # removing one electron from 5d shell,
    occupation[-1] = (6,5-1,2) # 6e in 5d
    # and adding a 5s shell with 1 electron.
    occupation.append( (2,6-1,0) ) # 2e in 6s
    # The new occupation is now [Xe] 5d^(6) 6s^(2)
    
    atomdft.setOccupation(occupation)
    atomdft.setValenceOrbitals(["5d", "6s"], format="spectroscopic")
    atomdft.setRadialGrid(0.0000004, 14.0, 10000)
    try:
        from confined_pseudo_atoms import re
        atomdft.initialDensityGuess((re.r, float(Nelec)/float(re.Nelec) * re.radial_density))
        #from confined_pseudo_atoms import os
        #atomdft.initialDensityGuess((os.r, os.radial_density))
    except ImportError:
        atomdft.initialDensityGuess()
    atomdft.setEnergyRange(energy_range)
    atomdft.solveKohnSham()
    fh = open(orbdir + "os.py", "w")
    atomdft.saveSolution(fh)
    fh.close()    
    
# test version !!!
def iridium():
    energy_range = list(linspace(-1500.0, -400.0, 200)) \
        + list(linspace(-400.0, -200.0, 200)) \
        + list(linspace(-200.0, -30.0, 200)) \
        + list(linspace(-30.0, -5.0, 400)) \
        + list(linspace(-5.0, 5.0, 200)) \
    # equidistant grid, use larger grid for heavy atoms
              
    Z = 77
    Nelec = 77
    atomdft = PseudoAtomDFT(Z,Nelec,numerov_conv,en_conv, grid_spacing="exponential", r0=confinement_radii_byZ[Z], damping=0.6)
    # The electron configuration of ruthenium is  [Xe] 5d^(7) 6s^(2)
    occupation = [occ for occ in occupation_numbers(Nelec)]
    # `occupation` is a list of tuples
    #   (nr.electrons, quantum number n + 1, quantum number l)
    # that describe the occupied shells.
    # In the default occupation, tantalum would have the electronic configuration [Xe] 5d^(7).
    assert occupation[-1] == (9,5-1,2) # 9e in 5d
    # The default occupation is not correct, so we need to adjust it by
    # removing one electron from 5d shell,
    occupation[-1] = (7,5-1,2) # 7e in 5d
    # and adding a 6s shell with 1 electron.
    occupation.append( (2,6-1,0) ) # 2e in 6s
    # The new occupation is now [Xe] 5d^(7) 6s^(2)
    
    atomdft.setOccupation(occupation)
    atomdft.setValenceOrbitals(["5d", "6s"], format="spectroscopic")
    atomdft.setRadialGrid(0.0000004, 14.0, 10000)
    try:
        from confined_pseudo_atoms import os
        atomdft.initialDensityGuess((os.r, float(Nelec)/float(os.Nelec) * os.radial_density))
        #from confined_pseudo_atoms import ir
        #atomdft.initialDensityGuess((ir.r, ir.radial_density))
    except ImportError:
        atomdft.initialDensityGuess()
    atomdft.setEnergyRange(energy_range)
    atomdft.solveKohnSham()
    fh = open(orbdir + "ir.py", "w")
    atomdft.saveSolution(fh)
    fh.close()    
    
# test version !!!
def platinum():
    energy_range = list(linspace(-1500.0, -400.0, 200)) \
        + list(linspace(-400.0, -200.0, 200)) \
        + list(linspace(-200.0, -30.0, 200)) \
        + list(linspace(-30.0, -5.0, 400)) \
        + list(linspace(-5.0, 5.0, 200)) \
    # equidistant grid, use larger grid for heavy atoms
              
    Z = 78
    Nelec = 78
    atomdft = PseudoAtomDFT(Z,Nelec,numerov_conv,en_conv, grid_spacing="exponential", r0=confinement_radii_byZ[Z], damping=0.6)
    # The electron configuration of ruthenium is  [Xe] 5d^(8) 6s^(2)
    occupation = [occ for occ in occupation_numbers(Nelec)]
    # `occupation` is a list of tuples
    #   (nr.electrons, quantum number n + 1, quantum number l)
    # that describe the occupied shells.
    # In the default occupation, tantalum would have the electronic configuration [Xe] 5d^(8).
    assert occupation[-1] == (10,5-1,2) # 10e in 5d
    # The default occupation is not correct, so we need to adjust it by
    # removing one electron from 5d shell,
    occupation[-1] = (8,5-1,2) # 8e in 5d
    # and adding a 6s shell with 1 electron.
    occupation.append( (2,6-1,0) ) # 2e in 6s
    # The new occupation is now [Xe] 5d^(7) 6s^(2)
    
    atomdft.setOccupation(occupation)
    atomdft.setValenceOrbitals(["5d","6s","6p"], format="spectroscopic")
    atomdft.setRadialGrid(0.0000004, 14.0, 10000)
    try:
        from confined_pseudo_atoms import ir
        atomdft.initialDensityGuess((ir.r, float(Nelec)/float(ir.Nelec) * ir.radial_density))
        #from confined_pseudo_atoms import pt
        #atomdft.initialDensityGuess((pt.r, float(Nelec)/float(pt.Nelec) * pt.radial_density))
    except ImportError:
        atomdft.initialDensityGuess()
    atomdft.setEnergyRange(energy_range)
    atomdft.solveKohnSham()
    fh = open(orbdir + "pt.py", "w")
    atomdft.saveSolution(fh)
    fh.close()    
    
# test version !!!
def gold():
    energy_range = list(linspace(-1500.0, -400.0, 200)) \
        + list(linspace(-400.0, -200.0, 200)) \
        + list(linspace(-200.0, -30.0, 200)) \
        + list(linspace(-30.0, -5.0, 400)) \
        + list(linspace(-5.0, 5.0, 200)) \
    # equidistant grid, use larger grid for heavy atoms
              
    Z = 79
    Nelec = 79
    atomdft = PseudoAtomDFT(Z,Nelec,numerov_conv,en_conv, grid_spacing="exponential", r0=confinement_radii_byZ[Z], damping=0.6)
    # The electron configuration of ruthenium is  [Xe] 5d^(10) 6s^(1)
    occupation = [occ for occ in occupation_numbers(Nelec)]
    # `occupation` is a list of tuples
    #   (nr.electrons, quantum number n + 1, quantum number l)
    # that describe the occupied shells.
    # In the default occupation, tantalum would have the electronic configuration [Xe] 5d^(10).
    #assert occupation[-1] == (10,5-1,2) # 1e in 5d
    occupation[-1] = (1,6-1,0) # 1e in 6s
    # and adding a 5s shell with 1 electron.
    # The new occupation is now [Xe] 5d^(10) 6s^(1)
    
    atomdft.setOccupation(occupation)
    atomdft.setValenceOrbitals(["5d","6s"], format="spectroscopic")
    atomdft.setRadialGrid(0.0000004, 14.0, 10000)
    try:
        from confined_pseudo_atoms import ag
        atomdft.initialDensityGuess((ag.r, float(Nelec)/float(ag.Nelec) * ag.radial_density))
        #from confined_pseudo_atoms import au
        #atomdft.initialDensityGuess((au.r, float(Nelec)/float(au.Nelec) * au.radial_density))
    except ImportError:
        atomdft.initialDensityGuess()
    atomdft.setEnergyRange(energy_range)
    atomdft.solveKohnSham()
    fh = open(orbdir + "au.py", "w")
    atomdft.saveSolution(fh)
    fh.close()    
    
# test version !!!
def lead():
    energy_range = list(linspace(-1500.0, -800.0, 100)) \
        + list(linspace(-800.0, -200.0, 200)) \
        + list(linspace(-200.0, -30.0, 200)) \
        + list(linspace(-30.0, -5.0, 400)) \
        + list(linspace(-5.0, 5.0, 400)) \
    # equidistant grid, use larger grid for heavy atoms
              
    Z = 82
    Nelec = 82
    atomdft = PseudoAtomDFT(Z,Nelec,numerov_conv,en_conv, grid_spacing="exponential", r0=confinement_radii_byZ[Z], damping=0.6)    
    atomdft.setRadialGrid(0.000000001, 14.0, 20000)
    try:
        from confined_pseudo_atoms import sn
        atomdft.initialDensityGuess((sn.r, float(Nelec)/float(sn.Nelec) * sn.radial_density))
        #from confined_pseudo_atoms import pb
        #atomdft.initialDensityGuess((pb.r, pb.radial_density))
    except ImportError:
        atomdft.initialDensityGuess()
    atomdft.setEnergyRange(energy_range)
    atomdft.solveKohnSham()
    fh = open(orbdir + "pb.py", "w")
    atomdft.saveSolution(fh)
    fh.close()    

def bismuth():
    energy_range = list(linspace(-1500.0, -400.0, 200)) \
        + list(linspace(-400.0, -200.0, 200)) \
        + list(linspace(-200.0, -30.0, 200)) \
        + list(linspace(-30.0, -5.0, 400)) \
        + list(linspace(-5.0, 5.0, 200)) \
    # equidistant grid, use larger grid for heavy atoms
              
    Z = 83
    Nelec = 83
    atomdft = PseudoAtomDFT(Z,Nelec,numerov_conv,en_conv, grid_spacing="exponential", r0=confinement_radii_byZ[Z], damping=0.6)    
    atomdft.setRadialGrid(0.0000004, 14.0, 10000)
    try:
        from confined_pseudo_atoms import sb
        atomdft.initialDensityGuess((sb.r, float(Nelec)/float(sb.Nelec) * sb.radial_density))
        #from confined_pseudo_atoms import bi
        #atomdft.initialDensityGuess((bi.r, bi.radial_density))
    except ImportError:
        atomdft.initialDensityGuess()
    atomdft.setEnergyRange(energy_range)
    atomdft.solveKohnSham()
    fh = open(orbdir + "bi.py", "w")
    atomdft.saveSolution(fh)
    fh.close()
    
if __name__ == "__main__":
    ## first row
    #hydrogen()
    #helium()

    ## second row
    lithium()
    #beryllium()
    #boron()
    #carbon()
    #nitrogen()
    #oxygen()
    #fluorine()
    #neon()
    
    ## third row
    #natrium()
    #magnesium()
    #aluminum()
    #silicon()
    #phosphorus()
    #sulfur()
    #chlorine()
    #argon()
    
    ## fourth row
    #potassium()
    #calcium()
    #scandium()
    #titanium()
    #vanadium()
    #chromium()
    #manganese()
    #iron()
    #cobalt()
    #nickel()
    #copper()
    #zinc()
    #gallium()
    #germanium()
    #arsenic()
    #selenium()
    #bromine()
    #krypton()

    ## fifth row
    #rubidium()
    #strontium()
    #yttrium()
    #zirconium()
    #niobium()
    #molybdenum()
    #technetium()
    #ruthenium()
    #rhodium()
    #palladium()
    #silver()
    #cadmium()
    #indium()
    #tin()
    #antimony()
    #tellurium()
    #iodine()
    #xenon()
    
    ## sixth row
    #cesium()
    #barium()
    #
    ##lanthanoide
    #lanthanum()
    #cerium()
    #prasedymium()
    #neodymium()
    #promethium()
    #samarium()
    #europium()
    #gadolinium()
    #terbium()
    #dysprosium()
    #holmium()
    #erbium()
    #thulium()
    #ytterbium()
    #lutetium()
    #
    #hafnium()
    #tantalium()
    #tungsten()
    #rhenium()
    #osmium()
    #iridium()
    #platinum()
    #gold()
    #mercury()
    #thallium()
    #lead()
    #bismuth()
    #polonium()
    #astatine()
    #radon()

    ## seventh row
    #francium()
    #radium()
    #
    #actinoide
    #actinium()
    #thorium()
    #protactinium()
    #uranium()
    #neptunium()
    #plutonium()
    #americium()
    #curium()

    ## heavy atoms, for copper relativistic effects are negligible, 
    ## for silver they are small, however gold and mercury would require a relativistic treatment
    #titanium()
    #iron()
    #copper()
    #zinc()
    #bromine()
    ## fifth row
    #ruthenium()
    #ruthenium_2plus()   # Ru^(2+)
    #silver()
    #iodine()
