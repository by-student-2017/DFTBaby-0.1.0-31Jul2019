#!/usr/bin/env python
"""
computes the photoelectron angular distribution (PAD) for ionization of an atom using the numerically
exact bound and continuum orbitals of this single-electron Hamiltonian.
"""

import numpy as np
from numpy.polynomial import legendre

from StieltjesImaging.LebedevQuadrature import get_lebedev_grid, outerN

from StieltjesImaging.associatedLegendrePolynomials import spherical_harmonics_it
from StieltjesImaging.SphericalCoords import cartesian2spherical
from StieltjesImaging.Wigner import Wigner3J

# The calculation of PAD for different energies can be trivially parallelized
from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

from DFTB import AtomicData
from DFTB.Analyse import Cube
from DFTB.utils import annotated_matrix

from DFTB.Scattering.VariationalKohn import transition_dipole_integrals, orientation_averaged_pad
from DFTB.BasisSets import AtomicBasisSet, AtomicBasisFunction
from DFTB.Scattering.SlakoScattering import AtomicScatteringBasisSet, load_pseudo_atoms_scattering

def spherical_harmonics_vec(th,ph, lmax):
    """
    compute the first spherical harmonics in the order 

      Y_{0,0},
      Y_{1,0}, Y_{1,+1}, Y_{1,-1},
      Y_{2,0}, Y_{2,+1}, Y_{2,-1}, Y_{2,+2}, Y_{2,-2},
      ...
      Y_{l,0}, Y_{l,+1}, Y_{l,-1}, ..., Y_{l,+m}, Y_{l,-m}, ..., Y_{l,-l}, Y_{l,+l}

    up to l=lmax.

    The spherical harmonics are all evaluated at the angles `th`,`ph`.

    Parameters
    ----------
    th, ph  : numpy arrays with angles theta and phi
    lmax    : integer, maximum angular moment

    Returns
    -------
    Y       : a list of grids with the same shape as `th` and `ph`,
              len(Y) = lmax*(lmax+2)+1
    lm      : a list of tuples (l,m) with the angular quantum numbers for
              each spherical harmonic in `Y`
    """
    Y  = []
    lm = []
    n = lmax*(lmax+2)+1
    sph_it = spherical_harmonics_it(th,ph)
    for i in range(0, n):
        Ylm,l,m = sph_it.next()
        Y.append(Ylm)
        lm.append((l,m))
    assert l ==  lmax
    assert m == -lmax

    return Y, lm

def asymptotic_Ylm(continuum_orbs, energy, rmax=2500.0, npts_r=300, lebedev_order=65):
    """
    decompose continuum orbitals asymptotically into spherical harmonics
                          a
        wfn    = sum     R  (r)  Y (th,ph)
           a        L,M   L,M     L,M

          a       /rmax+2pi/k  2   |  a   |2
        C      =  |           r dr | R    |
          L,M     /rmax            |  L,M |

    Parameters
    ----------
    continuum_orbitals : list of continuum orbitals
    energy             : photokinetic energy of continuum orbitals, E = 1/2 k^2

    Optional
    --------
    rmax               : radius at which the molecular potential is indistinguishable from the asymptotic potential
    npts_r             : number of points for integrating over radial interval [rmax,rmax+2pi/k]
    lebedev_order      : integer, order of Lebedev grid

    Returns
    -------
    Cs                 : Cs[i,a] is the contribution of spherical harmonic Y_(Li,Mi) to orbital a
                         at large distance
    LMs                : list of tuples (L,M) in the same order as in the 1st axis of `Cs`.
    """
    # wave number for kinetic energy E=1/2 k^2
    k = np.sqrt(2*energy)
    # wavelength
    wavelength = 2.0 * np.pi / k
    
    # radial grid
    # sample points and weights for Gauss-Legendre quadrature on the interval [-1,1]
    leggauss_pts, leggauss_weights = legendre.leggauss(npts_r)
    # For Gaussian quadrature the integral [-1,1] has to be changed into [rmax,rmax+2pi/k].
    # new endpoints of interval
    a = rmax
    b = rmax+wavelength
    # transformed sampling points and weights
    r = 0.5*(b-a)*leggauss_pts + 0.5*(a+b)
    dr = 0.5*(b-a)*leggauss_weights
    # Lebedev grid for spherical wave expansion
    th,ph, weights_angular = get_lebedev_grid(lebedev_order)
    # The lebedev quadrature rule is
    #         /
    #  I[f] = | dOmega f(Omega) = 4 pi sum  weights_angular[i] * f(th[i],ph[i])
    #         /                           i
    # For convencience we multiply the factor of 4*pi into the weights
    weights_angular *= 4*np.pi

    # evaluate spherical harmonics for L=0,1,2,3 M=-L,..,L on angular Lebedev grid
    Lmax = 3
    Ys,LMs = spherical_harmonics_vec(th,ph, Lmax)
    Ys = np.array(Ys)
    # cartesian coordinates of radial and angular grids
    x = outerN(r, np.sin(th)*np.cos(ph))
    y = outerN(r, np.sin(th)*np.sin(ph))
    z = outerN(r, np.cos(th))
    
    # differential for r-integration
    r2dr = r**2*dr
    # volume element   r^2 dr dOmega
    dV = outerN(r2dr, weights_angular)
    # add r-axis to angular weights
    dOmega = outerN(np.ones(npts_r), weights_angular)
    
    # nc: number of continuum wavefunctions
    nc = len(continuum_orbs)
    # number of angular components (L,M) for Lmax=2
    LMdim = Lmax*(Lmax+2)+1
    assert LMdim == len(LMs)
    # set up array for expansion coefficients
    Cs = np.zeros((LMdim,nc))
    
    for a in range(0, nc):  # loop over continuum orbitals
        wfn_a = continuum_orbs[a].amp(x,y,z)
        # normalization constant
        nrm2 = np.sum(dV * abs(wfn_a)**2)
        wfn_a /= np.sqrt(nrm2)

        for iLM, (l,m) in enumerate(LMs):
            # add r-axis to spherical harmonics Y_(l,m)(r,th,ph) = Y_(l,m)(th,ph)
            Ylm = outerN(np.ones(npts_r), Ys[iLM])
            # radial wavefunction of continuum orbital belonging to L,M channel
            #   a       /pi           /2pi      *                     
            # R (r)  =  | sin(th) dth | dph  Y  (th,ph)  wfn (r,th,ph)
            #   L,M     /0            /0       L,M           a
            #
            #                    a
            # wfn_a = sum_(L,M) R (r)  Y (th,ph)
            #                    L,M    L,M
            
            # integrate over angles
            Rlm = np.sum(dOmega * Ylm.conjugate() * wfn_a, axis=1)
            # integrate over r
            #  a     /rmax+k/2pi  2    |   a  |2
            # C    = |           r  dr | R    |  1 / (4 pi)
            #  L,M   /rmax             |  L,M | 
            Cs[iLM,a] = np.sum(r2dr * abs(Rlm)**2)

    print "  Asymptotic Decomposition Y_(l,m)"
    print "  ================================"
    row_labels = ["orb. %2.1d" % a for a in range(0, nc)]
    col_labels = ["Y(%d,%+d)" % (l,m) for (l,m) in LMs]
    txt = annotated_matrix(Cs.transpose(), row_labels, col_labels)
    print txt
            
    return Cs,LMs

    
def angular_product_distribution(continuum_orbs, energy, rmax=2500.0, npts_r=300, lebedev_order=65):
    """
    expand the angular distribution of the product of two molecular orbitals wfn_a(r) and
    wfn_b(r) into spherical harmonics

          a,b     /rmax+2pi/k  2   /pi            /2pi      *            *
        A      =  |           r dr |  sin(th) dth |  dph  Y  (th,ph)  wfn (r,th,ph) wfn (r,th,ph)
          L,M     /rmax            /0             /0       L,M           a             b


    Parameters
    ----------
    continuum_orbitals : list of instances of AtomicBasisFunction
    energy             : photokinetic energy of continuum orbitals, E = 1/2 k^2

    Optional
    --------
    rmax               : radius at which the molecular potential is indistinguishable from the asymptotic potential
    npts_r             : number of points for integrating over radial interval [rmax,rmax+2pi/k]
    lebedev_order      : integer, order of Lebedev grid

    Returns
    -------
    Amo            : 3d numpy array with angular components for each pair of molecular orbitals
                                    a,b
                     Amo[LM,a,b] = A
                                    L,M
    LMs            : list of tuples (L,M) in the same order as in the 1st axis of `Amo`.
    """
    # wave number for kinetic energy E=1/2 k^2
    k = np.sqrt(2*energy)
    # wavelength
    wavelength = 2.0 * np.pi / k
    
    # radial grid
    # sample points and weights for Gauss-Legendre quadrature on the interval [-1,1]
    leggauss_pts, leggauss_weights = legendre.leggauss(npts_r)
    # For Gaussian quadrature the integral [-1,1] has to be changed into [rmax,rmax+2pi/k].
    # new endpoints of interval
    a = rmax
    b = rmax+wavelength
    # transformed sampling points and weights
    r = 0.5*(b-a)*leggauss_pts + 0.5*(a+b)
    dr = 0.5*(b-a)*leggauss_weights
    
    # Lebedev grid for spherical wave expansion
    th,ph, weights_angular = get_lebedev_grid(lebedev_order)
    # The lebedev quadrature rule is
    #         /
    #  I[f] = | dOmega f(Omega) = 4 pi sum  weights_angular[i] * f(th[i],ph[i])
    #         /                           i
    # For convencience we multiply the factor of 4*pi into the weights
    weights_angular *= 4*np.pi
    
    # evaluate spherical harmonics for L=0,1,2, M=-L,..,L on angular Lebedev grid
    Lmax = 2
    Ys,LMs = spherical_harmonics_vec(th,ph, Lmax)
    Ys = np.array(Ys)
    # cartesian coordinates of radial and angular grids
    x = outerN(r, np.sin(th)*np.cos(ph))
    y = outerN(r, np.sin(th)*np.sin(ph))
    z = outerN(r, np.cos(th))
    # make sure r**2*dr has the same shape as x,y and z
    r2dr = np.repeat( np.reshape(r**2 * dr, (len(r),1)), len(th) , axis=1)
    # nc: number of continuum wavefunctions
    nc = len(continuum_orbs)
    # number of angular components (L,M) for Lmax=2
    LMdim = Lmax*(Lmax+2)+1
    assert LMdim == len(LMs)
    # set up array for expansion coefficients
    Amo = np.zeros((LMdim,nc,nc), dtype=complex)

    for a in range(0, nc):  # loop over continuum orbitals
        print " %d of %d done" % (a, nc)
        wfn_a = continuum_orbs[a].amp(x,y,z)
        for b in range(0, nc):  # loop over continuum orbitals
            wfn_b = continuum_orbs[b].amp(x,y,z)
            #                /rmax+2pi/k  2      *
            # PAD   (th,ph)= |           r dr wfn (r,th,ph) wfn (r,th,ph)
            #    a,b         /rmax               a             b
            PAD_ab = np.sum(r2dr * wfn_a.conjugate() * wfn_b, axis=0)
            """
            ### DEBUG
            # plot PAD for orbitals a and b
            import matplotlib.pyplot as plt
            from scipy.interpolate import griddata
            th_interp, ph_interp = np.mgrid[0.0:np.pi:100j, 0.0:2*np.pi:100j]
            PAD_ab_interp = griddata((th,ph), PAD_ab, (th_interp, ph_interp))
            plt.cla()
            plt.title(r"PAD$_{%d,%d}$" % (a,b))
            plt.xlabel(r"$\theta$ / $\pi$")
            plt.ylabel(r"$\phi$ / $\pi$")
            plt.imshow(PAD_ab_interp.T, extend=(0.0, 1.0, 0.0, 2.0))
            plt.colorbar()
            plt.show()
            ###
            """
            Amo[:,a,b] =  np.dot(Ys.conjugate(), weights_angular * PAD_ab)

    print "normalize Amo's"
    # Continuum wavefunctions are not normalized, we normalize them in such a way
    # the integrating the product of two continuum orbitals
    #
    #   /rmax+2pi/k  2   /            *
    #   |           r dr | dOmega  wfn (r,Omega) wfn (r,Omega)   =  delta
    #   /rmax            /            a             b                    a,b
    #
    # over one wavelength at a very large r gives exactly 1, if a=b, and 0 otherwise.
    # The A's are the expansion coefficients of
    #
    #                                M=+L   a,b  
    #  PAD_{a,b}(Omega) = sum     sum      A    Y (Omega)
    #                        L=0     M=-L   L,M  L,M
    #
    #  Therefore we need to normalize the A's such that
    #
    #  /                            a,b               a,b             !
    #  | dOmega PAD_{a,b}(Omega) = A    4*pi Y     = A    sqrt(4 pi)  = 1
    #  /                            0,0       0,0     0,0
    #                   a,b
    #  which implies   A    = 1/sqrt(4*pi) delta
    #                   0,0                     a,b
    nrm = np.zeros(nc, dtype=complex)
    for a in range(0, nc):
        nrm[a] = np.sqrt(Amo[0,a,a])
    # normalize Amo's
    for a in range(0, nc):
        for b in range(0, nc):
            Amo[:,a,b] /= nrm[a]*nrm[b]
    Amo /= np.sqrt(4.0*np.pi)
            
    print "       PAD_ab(th,ph) - real part          "
    print "=========================================="
    print "expansion into spherical harmonics        "
    print "                        a,b               "
    print "PAD   (th,ph) = sum    A     Y   (th,hp)  "
    print "   a,b             L,M  L,M   L,M         "
    print "                                          "
    for iLM,(L,M) in enumerate(LMs):
        print " L=%d M=%+d        " % (L,M)
        print "   a,b   "
        print "  A      "
        print "   %d,%+d" % (L,M)
        row_labels = ["a %2.1d" % a for a in range(0, nc)]
        col_labels = ["b %2.1d" % b for b in range(0, nc)]
        txt = annotated_matrix(Amo[iLM,:,:].real, row_labels, col_labels)
        print txt

            
    """
    ### DEBUG
    Amo_test = np.zeros((LMdim,nc,nc), dtype=complex)
    print "plot normalized wavefunctions"
    for a in range(0, nc):  # loop over continuum orbitals
        print " %d of %d done" % (a, nc)
        wfn_a = continuum_orbs[a].amp(x,y,z) / nrm[a]
        for b in range(0, nc):  # loop over continuum orbitals
            wfn_b = continuum_orbs[b].amp(x,y,z) / nrm[b]
            ### DEBUG
            import matplotlib.pyplot as plt
            plt.cla()
            plt.xlabel("r / bohr")
            plt.plot(r, wfn_a[:,0], label=r"$\phi_{%d}(r;\theta=\theta_0)$" % a)
            plt.plot(r, wfn_b[:,0], label=r"$\phi_{%d}(r;\theta=\theta_0)$" % b)
            plt.plot(r, wfn_a[:,len(th)/2], label=r"$\phi_{%d}(r;\theta=\theta_1)$" % a)
            plt.plot(r, wfn_b[:,len(th)/2], label=r"$\phi_{%d}(r;\theta=\theta_1)$" % b)
            plt.legend()
            plt.show()
            ###

            #                /rmax+2pi/k  2      *
            # PAD   (th,ph)= |           r dr wfn (r,th,ph) wfn (r,th,ph)
            #    a,b         /rmax               a             b
            PAD_ab = np.sum(r2dr * wfn_a.conjugate() * wfn_b, axis=0)
            Amo_test[:,a,b] =  np.dot(Ys.conjugate(), weights_angular * PAD_ab)
            print "a= %d b= %d" % (a,b)
            print "A2m (L=0,M=0)= %s" % Amo_test[0,a,b]
            print "A2m (numerical) = %s" % Amo_test[-5:,a,b]
    ### 
    """
     
    return Amo,LMs

def photoangular_distribution(Dmo, Amo, LMs, energy):
    """
    construct the photoelectron angular distribution (PAD) by averaging the angular
    electronc distribution of the dipole-prepared continuum orbital over all orientations.

    Parameters
    ----------
    Dmo        : transition dipoles between the initial bound orbital and the continuum orbitals `a`
                 Dmo[a,:] = <i|r|a>
    Amo        : 3d numpy array with angular components for each pair of continuum orbitals `a` and `b`
                                    a,b
                     Aao[LM,a,b] = A
                                    L,M
    LMs        : list of tuples (L,M) in the same order as in the 1st axis of `Amo`.
    energy     : photoelectron kinetic energy (in Hartree) 

    Returns
    -------
    sigma, beta  : photoelectron angular distribution for the initial bound orbital at PKE=energy
                   which consists of the total cross section `sigma` and the anisotropy parameter `beta`
    """
    # number of continuum orbitals
    nmo = Amo.shape[-1]
    # photon energy
    omega = energy # + IE ???????

    tdipvecs = Dmo.transpose()
    
    # convert transition dipoles from cartesian to spherical coordinates
    #  <i|x|n>             sin(tdip_th) cos(tdip_ph)
    #  <i|y|n>  = tdip_r * sin(tdip_th) cos(tdip_ph)
    #  <i|z|n>                 cos(tdip_th)
    #
    tdip_r, tdip_th, tdip_ph = cartesian2spherical(tdipvecs)
    
    # spherical harmonics Y_(1,m1)(tdip_th, tdip_ph)
    Y1 = np.zeros((3,nmo), dtype=complex)
    for Y1m1,l1,m1 in spherical_harmonics_it(tdip_th, tdip_ph):
        if l1 > 1:
            break
        if l1 == 1:
            Y1[m1,:] = Y1m1

    print "angular distributions for pairs of orbitals ..."
    # coefficients of Legendre polynomials of degrees L=0,1,2
    # for pairs of molecular orbitals with indices a,b
    # coefsPmo[L,a,b]
    coefsPmo = np.zeros((3,nmo,nmo), dtype=complex)
    for LM,(L,M) in enumerate(LMs):
        cL = np.sqrt((2*L+1.)/(4.0*np.pi)) * Wigner3J(1,0, 1,0, L,0) * 4.0*np.pi/3.0 \
             * np.outer(tdip_r[:], tdip_r[:]) \
             * Amo[LM,:,:]
        yyLM = np.zeros((nmo,nmo), dtype=complex)
        for m1 in [-1,0,1]:
            for m2 in [-1,0,1]:
                if not (-m1 + m2 + M == 0):
                    continue
                
                yyLM += (-1)**m1 * Wigner3J(1,-m1, 1,m2, L,M) \
                        * np.outer( Y1[m1,:], Y1[m2,:].conjugate() )
        cL *= yyLM
        cL *= 2*omega
        
        coefsPmo[L,:,:] += cL
                
    coefsP = np.sum(coefsPmo, axis=(1,2))

    # anisotropy parameter in
    #   sigma(th) = sigma / (4 pi) [ 1  +  beta P2(cos(th)) ]
    #             =        c0               c2  P2(cos(th))
    sigma = coefsP[0] * 4.0*np.pi
    beta = coefsP[2]/coefsP[0]
    assert -1.0-0.01 <= beta <= 2.0+0.01

    # sigma should be equal to sum of oscillator strengths from bound to continuum
    sigma_oscsum = 2.0/3.0 * omega * np.sum(Dmo**2)
    assert abs(sigma - sigma_oscsum) < 1.0e-8
    
    return sigma.real, beta.real

def atomic_ion_averaged_pad_scan(energy_range, data_file,
                             npts_r=60, rmax=300.0,
                             lebedev_order=23, radial_grid_factor=3,
                             units="eV-Mb",
                             tdip_threshold=1.0e-5):
    """
    compute the photoelectron angular distribution for an ensemble of istropically
    oriented atomic ions.

    Parameters
    ----------
    energy_range    : numpy array with photoelectron kinetic energies (PKE)
                      for which the PAD should be calculated
    data_file       : path to file, a table with PKE, SIGMA and BETA is written

    Optional
    --------
    npts_r          : number of radial grid points for integration on interval [rmax,rmax+2pi/k]
    rmax            : large radius at which the continuum orbitals can be matched 
                      with the asymptotic solution
    lebedev_order   : order of Lebedev grid for angular integrals
    radial_grid_factor 
                    : factor by which the number of grid points is increased
                      for integration on the interval [0,+inf]
    units           : units for energies and photoionization cross section in output, 'eV-Mb' (eV and Megabarn) or 'a.u.'
    tdip_threshold  : continuum orbitals |f> are neglected if their transition dipole moments mu = |<i|r|f>| to the
                      initial orbital |i> are below this threshold.
    """
    print ""
    print "*******************************************"
    print "*  PHOTOELECTRON ANGULAR DISTRIBUTIONS    *"
    print "*******************************************"
    print ""
    Z = 1
    atomlist = [(Z,(0.0,0.0,0.0))]

    # determine the radius of the sphere where the angular distribution is calculated. It should be
    # much larger than the extent of the molecule
    (xmin,xmax),(ymin,ymax),(zmin,zmax) = Cube.get_bbox(atomlist, dbuff=0.0)
    dx,dy,dz = xmax-xmin,ymax-ymin,zmax-zmin
    # increase rmax by the size of the molecule
    rmax += max([dx,dy,dz])
    Npts = max(int(rmax),1) * 50
    print "Radius of sphere around molecule, rmax = %s bohr" % rmax
    print "Points on radial grid, Npts = %d" % Npts
    
    # load bound pseudoatoms
    basis = AtomicBasisSet(atomlist, confined=False)
    bound_orbital = basis.bfs[0]
        
    # compute PADs for all energies
    pad_data = []
    print "  SCAN"
    
    print "  Writing table with PAD to %s" % data_file
    # table headers
    header  = ""
    header += "# npts_r: %s  rmax: %s" % (npts_r, rmax) + '\n'
    header += "# lebedev_order: %s  radial_grid_factor: %s" % (lebedev_order, radial_grid_factor) + '\n'
    if units == "eV-Mb":
        header += "# PKE/eV     SIGMA/Mb       BETA2" + '\n'
    else:
        header += "# PKE/Eh     SIGMA/bohr^2   BETA2" + '\n'

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

            # continuum orbitals at a given energy
            continuum_orbitals = []
            # quantum numbers of continuum orbitals (n,l,m)
            quantum_numbers = []
            phase_shifts = []

            print "compute atomic continuum orbitals ..."
            valorbs, radial_val, phase_shifts_val = load_pseudo_atoms_scattering(atomlist, energy, rmin=0.0, rmax=2*rmax, Npts=100000, lmax=3)

            pos = np.array([0.0, 0.0, 0.0])
            for indx,(n,l,m) in enumerate(valorbs[Z]):
                continuum_orbital = AtomicBasisFunction(Z, pos, n,l, m, radial_val[Z][indx], 0)
                continuum_orbitals.append( continuum_orbital )
                quantum_numbers.append( (n,l,m) )
                phase_shifts.append( phase_shifts_val[Z][indx] )
                
            # transition dipoles between bound and free orbitals
            dipoles = transition_dipole_integrals(atomlist, [bound_orbital], continuum_orbitals,
                                                  radial_grid_factor=radial_grid_factor,
                                                  lebedev_order=lebedev_order)
            # Continuum orbitals with vanishing transition dipole moments to the initial orbital
            # do not contribute to the PAD. Filter out those continuum orbitals |f> for which
            # mu^2 = |<i|r|f>|^2  <  threshold
            mu = np.sqrt(np.sum(dipoles[0,:,:]**2, axis=-1))
            dipoles_important = dipoles[0,mu > tdip_threshold,:]
            continuum_orbitals_important = [continuum_orbitals[i] for i in range(0, len(continuum_orbitals)) if mu[i] > tdip_threshold]
            quantum_numbers_important = [quantum_numbers[i] for i in range(0, len(continuum_orbitals)) if mu[i] > tdip_threshold]
            phase_shifts_important = [[phase_shifts[i]] for i in range(0, len(continuum_orbitals)) if mu[i] > tdip_threshold]
            
            print " %d of %d continuum orbitals have non-vanishing transition dipoles " \
                % (len(continuum_orbitals_important), len(continuum_orbitals))
            print " to initial orbital (|<i|r|f>| > %e)" % tdip_threshold

            print "  Quantum Numbers"
            print "  ==============="
            row_labels = ["orb. %2.1d" % a for a in range(0, len(quantum_numbers_important))]
            col_labels = ["N", "L", "M"]
            txt = annotated_matrix(np.array(quantum_numbers_important), row_labels, col_labels, format="%s")
            print txt

            print "  Phase Shifts (in units of pi)"
            print "  ============================="
            row_labels = ["orb. %2.1d" % a for a in range(0, len(phase_shifts_important))]
            col_labels = ["Delta"]
            txt = annotated_matrix(np.array(phase_shifts_important)/np.pi, row_labels, col_labels, format="  %8.6f  ")
            print txt

            
            print "  Transition Dipoles"
            print "  =================="
            print "  threshold = %e" % tdip_threshold
            row_labels = ["orb. %2.1d" % a for a in range(0, len(quantum_numbers_important))]
            col_labels = ["X", "Y", "Z"]
            txt = annotated_matrix(dipoles_important, row_labels, col_labels)
            print txt

            # 
            Cs, LMs = asymptotic_Ylm(continuum_orbitals_important, energy, 
                                     rmax=rmax, npts_r=npts_r, lebedev_order=lebedev_order)
            
            # expand product of continuum orbitals into spherical harmonics of order L=0,2
            Amo, LMs = angular_product_distribution(continuum_orbitals_important, energy,
                                                    rmax=rmax, npts_r=npts_r, lebedev_order=lebedev_order)
            
            # compute PAD for ionization from orbital 0 (the bound orbital) 
            sigma, beta = photoangular_distribution(dipoles_important, Amo, LMs, energy)

            if units == "eV-Mb":
                energy *= AtomicData.hartree_to_eV
                # convert cross section sigma from bohr^2 to Mb
                sigma *= AtomicData.bohr2_to_megabarn
                                
            pad_data.append( [energy, sigma, beta] )
            # save row with PAD for this energy to table
            row = "%10.6f   %10.6e  %+10.6e" % tuple(pad_data[-1]) + '\n'
            fh.Write_ordered(row)
                    
    fh.Close()

    print "  Photoelectron Angular Distribution"
    print "  =================================="
    print "  units: %s" % units
    row_labels = [" " for en in energy_range]
    col_labels = ["energy", "sigma", "beta2"]
    txt = annotated_matrix(np.array(pad_data).real, row_labels, col_labels)
    print txt

    
    print "FINISHED"


if __name__ == "__main__":
    import sys
    import optparse
    import os.path

    usage  = """

       %s <output file>

    compute the photoangular distribution (PAD) for ionization from the 
    lowest atomic valence orbital to the continuum. 

    A table with the PAD is written to <output file>. 
    It contains the 3 columns   PKE/eV  SIGMA/Mb  BETA_2
    which define the PAD(th) at each energy according to

                                   
      PAD(th) = SIMGA/(4pi) [1 + BETA_2 P2(cos(th))]

    Type --help to see all options.

    The computation of the PAD for a range of energies can be parallelized trivially. 
    The script can be run in parallel using `mpirun`. 
    """ % os.path.basename(sys.argv[0])

    parser = optparse.OptionParser(usage)
    # options
    parser.add_option("-u", "--units", dest="units", default="eV-Mb",
                      help="Units for energies and photoionization cross section in output, 'eV-Mb' (eV and Megabarn) or 'a.u.' [default: %default]")
    parser.add_option("--npts_r", dest="npts_r", help="Number of radial points for integration on interval [rmax,rmax+2*pi/k] (2*pi/k is the wavelength at energy E=1/2 k^2) [default: %default]", default=60, type=int)
    parser.add_option("--lebedev_order", dest="lebedev_order", help="Order Lebedev grid for angular integrals [default: %default]", default=65, type=int)
    parser.add_option("--radial_grid_factor", dest="radial_grid_factor", help="Factor by which the number of radial grid points is increased for integration on the interval [0,+inf], [default: %default]", default=6, type=int)
    parser.add_option("--rmax", dest="rmax", help="The photoangular distribution is determined by the angular distribution of the continuum orbital on a spherical shell with inner radius rmax (in bohr) and outer radius rmax+2pi/k. Ideally the radius should be infinity, but a larger radius requires propagating the continuum orbital to greater distances. [default: %default]", default=2500.0, type=float)
    parser.add_option("--energy_range", dest="energy_range", help="Range of photoelectron kinetic energy E=1/2 k^2 given as a tuple (Emin,Emax,Npts) for which the angular distribution should be determined (in Hartree). The conversion factor 'eV' is also recognized. [default: %default]", default="(0.2*eV,6.0*eV,20)", type=str)
    
    (opts, args) = parser.parse_args()
    
    if len(args) < 1:
        print "Usage:",
        print usage
        exit(-1)

    data_file = args[0]

    # energy range in Hartree
    eV = AtomicData.hartree_to_eV
    energy_range = np.linspace(*eval(opts.energy_range))

    print "computing photoangular distribution for ionization from lower valence orbital"
    atomic_ion_averaged_pad_scan(energy_range, data_file, 
                                 npts_r=opts.npts_r, rmax=opts.rmax,
                                 lebedev_order=opts.lebedev_order, radial_grid_factor=opts.radial_grid_factor,
                                 units=opts.units)

    
    
