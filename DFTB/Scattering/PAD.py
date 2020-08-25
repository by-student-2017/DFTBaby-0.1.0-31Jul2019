"""
PAD = Photoangular distribution

angular distribution of angle-resolved photoemission spectroscopy for isotropically
oriented ensembles of molecules
"""
from DFTB.SKMatrixElements import count_orbitals
from DFTB.Scattering.Wigner import Wigner3J
from DFTB.Analyse import Cube
from DFTB.Modeling import MolecularCoords as MolCo

import numpy as np
import numpy.linalg as la

# spherical harmonics
def Y(l,m,rvec):
    x,y,z = rvec
    r = np.sqrt(x*x+y*y+z*z)
    if l == 1:
        if m == -1:
            res = 0.5 * np.sqrt(3.0/(2.0*np.pi)) * (x-1.0j*y)/r
        elif m == 0:
            res = 0.5 * np.sqrt(3.0/np.pi) * z/r
        elif m == 1:
            res = -0.5 * np.sqrt(3.0/(2.0*np.pi)) * (x+1.0j*y)/r
    else:
        raise NotImplemented("spherical harmonic Y_(%d,%d) not implemented" % (l,m))
    return res

def Yreal(l,m,rvec):
    x,y,z = rvec
    r = np.sqrt(x*x+y*y+z*z)
    if l == 1:
        if m == -1:
            res = np.sqrt(3.0/(4.0*np.pi)) * y/r
        elif m == 0:
            res = np.sqrt(3.0/(4.0*np.pi)) * z/r
        elif m == 1:
            res = np.sqrt(3.0/(4.0*np.pi)) * x/r
    else:
        raise NotImplemented("real spherical harmonic Yreal_(%d,%d) not implemented" % (l,m))
    return res
    

def angular_distribution_old(atomlist, valorbs, mos_bound, mos_scatt, Dipole, polarization="linear"):
    """
    compute angular distribution of photoelectrons for an isotropically oriented ensemble
    of molecules. 
    """
    assert polarization in ["linear", "left", "right"]
    if polarization == "linear":
        mp = 0
    elif polarization == "left":
        mp = -1
    elif polarization == "right":
        mp = +1
    # count bound valence orbitals and unbound scattering orbitals
    Norb1 = count_orbitals(atomlist, valorbs)
    valorbs_scattering = [(0,0), (1,-1), (1,0), (1,1)]
    Nat = len(atomlist)
    Norb2 = Nat * len(valorbs_scattering)
    # angular momentum quantum numbers of the scattering orbitals
    angmom_scatt = []
    for i,(Zi,posi) in enumerate(atomlist):
        # all atoms have the same number of scattering orbitals
        angmom_scatt += valorbs_scattering

    print Dipole
    xsec = np.zeros(3, dtype=complex)
    for i in range(0, Norb1):
        for j in range(0, Norb2):
            lj,mj = angmom_scatt[j]
            for k in range(0, Norb1):
                for l in range(0, Norb2):
                    ll,ml = angmom_scatt[l]
                    #
                    f1 = mos_bound[i].conjugate() * mos_bound[k] * mos_scatt[l].conjugate() * mos_scatt[j]
                    f2 = np.sqrt((2*lj+1)*(2*ll+1))
                    f12 = f1*f2
                    for m1 in [-1,0,1]:
                        for m2 in [-1,0,1]:
                            f3 = 1.0/3.0 * la.norm(Dipole[i,j,:]) * Y(1,m1,Dipole[i,j,:]) \
                                         * la.norm(Dipole[k,l,:]) * Y(1,m2,Dipole[k,l,:]).conjugate()
                            f123 = f12*f3
                            for J in [0,1,2]:
                                W1 = Wigner3J(1 ,-mp,  1 ,mp,  J,0)
                                W2 = Wigner3J(1 ,-m1,  1 ,m2,  J,m1-m2)
                                W3 = Wigner3J(ll,-ml,  lj,mj,  J,m1-m2)
                                W4 = Wigner3J(ll,0,    lj,0,   J,0)
                                Wprod = W1*W2*W3*W4
                                ####
                                dxsec = f123 * pow(-1,mp+ml+m2) * (2*J+1) * Wprod
                                if (abs(Wprod) > 1.0e-10) and (abs(dxsec.imag) < 1.0e-10):
                                    print "dxsec[%s] = %s   f3 = %s" % (J, dxsec, f3)
                                ####
                                xsec[J] += f123 * pow(-1,mp+ml+m2) * (2*J+1) * Wprod
    # energy dependent factors are still missing
    print "xsec = %s" % xsec
    sigma = xsec[0]/(4.0*np.pi)
    beta = xsec[2]/xsec[0]
    return sigma, beta

def Dmat(dij, dkl):
    D = np.zeros((3,3), dtype=complex)
    xij,yij,zij = dij
    xkl,ykl,zkl = dkl
    D[0,0] = (xij-1.0j*yij)*(xkl+1.0j*ykl)
    D[0,1] = np.sqrt(2.0) * (xij-1.0j*yij)*zkl
    D[0,2] = -(xij-1.0j*yij)*(xkl-1.0j*ykl)

    D[1,0] = np.sqrt(2.0)*zij*(xkl+1.0j*ykl)
    D[1,1] = 2.0*zij*zkl
    D[1,2] = -np.sqrt(2.0)*zij*(xkl-1.0j*ykl)

    D[2,0] = -(xij+1.0j*yij)*(xkl+1.0j*ykl)
    D[2,1] = -np.sqrt(2.0)*(xij+1.0j*yij)*zkl
    D[2,2] = (xij+1.0j*yij)*(xkl-1.0j*ykl)
    D /= 8.0*np.pi
    return D

def Dmat_real(dij, dkl):
    D = np.zeros((3,3), dtype=complex)
    xij,yij,zij = dij
    xkl,ykl,zkl = dkl
    
    D[0,0] = yij*ykl
    D[0,1] = yij*zkl
    D[0,2] = yij*xkl

    D[1,0] = zij*ykl
    D[1,1] = zij*zkl
    D[1,2] = zij*xkl

    D[2,0] = xij*ykl
    D[2,1] = xij*zkl
    D[2,2] = xij*xkl

    D /= 4.0*np.pi
    return D

def angular_distribution_old2(atomlist, valorbs, mos_bound, mos_scatt, Dipole, polarization="linear"):
    """
    compute angular distribution of photoelectrons for an isotropically oriented ensemble
    of molecules. 
    """
    assert polarization in ["linear", "left", "right"]
    if polarization == "linear":
        mp = 0
    elif polarization == "left":
        mp = -1
    elif polarization == "right":
        mp = +1
    # count bound valence orbitals and unbound scattering orbitals
    Norb1 = count_orbitals(atomlist, valorbs)
    valorbs_scattering = [(0,0), (1,-1), (1,0), (1,1)]
    Nat = len(atomlist)
    Norb2 = Nat * len(valorbs_scattering)
    # angular momentum quantum numbers of the scattering orbitals
    angmom_scatt = []
    for i,(Zi,posi) in enumerate(atomlist):
        # all atoms have the same number of scattering orbitals
        angmom_scatt += valorbs_scattering
    xsec = np.zeros(3, dtype=complex)
    for i in range(0, Norb1):
        for j in range(0, Norb2):
            lj,mj = angmom_scatt[j]
            for k in range(0, Norb1):
                for l in range(0, Norb2):
                    ll,ml = angmom_scatt[l]
                    #
                    f1 = mos_bound[i].conjugate() * mos_bound[k] * mos_scatt[l].conjugate() * mos_scatt[j]
                    f2 = np.sqrt((2*lj+1)*(2*ll+1))
                    f12 = f1*f2
                    D = Dmat(Dipole[i,j,:], Dipole[k,l,:].conjugate())
                    for m1 in [-1,0,1]:
                        for m2 in [-1,0,1]:
                            f3 = D[m1+1,m2+1]
                            f123 = f12*f3
                            for J in [0,1,2]:
                                W1 = Wigner3J(1 , mp,  1 ,-mp,  J,0)
                                W2 = Wigner3J(1 , m1,  1 ,-m2,  J,-(ml-mj))
                                W3 = Wigner3J(ll, ml,  lj,-mj,  J,-(ml-mj))
                                W4 = Wigner3J(ll,0,    lj,0,   J,0)
                                Wprod = W1*W2*W3*W4
                                dxsec = f123 * pow(-1,mp+m1+ml) * (2.0*J+1.0) * Wprod
                                xsec[J] += dxsec
    # energy dependent factors are still missing
    print "xsec = %s" % xsec
    sigma = xsec[0]/(4.0*np.pi)
    beta = xsec[2]/xsec[0]
    return sigma, beta

def angular_distribution(atomlist, valorbs, mos_bound, mos_scatt, Dipole, polarization="linear"):
    """
    compute angular distribution of photoelectrons for an isotropically oriented ensemble
    of molecules. 
    """
    assert polarization in ["linear", "left", "right"]
    if polarization == "linear":
        mp = 0
    elif polarization == "left":
        mp = -1
    elif polarization == "right":
        mp = +1
    # count bound valence orbitals and unbound scattering orbitals
    Norb1 = count_orbitals(atomlist, valorbs)
    valorbs_scattering = [(0,0), (1,-1), (1,0), (1,1)]
    Nat = len(atomlist)
    Norb2 = Nat * len(valorbs_scattering)
    # angular momentum quantum numbers of the scattering orbitals
    angmom_scatt = []
    for i,(Zi,posi) in enumerate(atomlist):
        # all atoms have the same number of scattering orbitals
        angmom_scatt += valorbs_scattering
    xsec = np.zeros(3, dtype=complex)
    for m1 in [-1,0,1]:
        for m2 in [-1,0,1]:
            for J in [0,1,2]:
                W1 = Wigner3J(1 , mp,  1 ,-mp,  J,0)
                W2 = Wigner3J(1 , m1,  1 ,-m2,  J,-(m1-m2))

                for j in range(0, Norb2):
                    lj,mj = angmom_scatt[j]
                    for l in range(0, Norb2):
                        ll,ml = angmom_scatt[l]
                        f1 = np.sqrt((2*lj+1)*(2*ll+1))

                        W3 = Wigner3J(ll, ml,  lj,-mj,  J,-(m1-m2))
                        W4 = Wigner3J(ll,0,    lj,0,   J,0)
                        Wprod = W1*W2*W3*W4
                        for i in range(0, Norb1):
                            for k in range(0, Norb1):
                                #
                                f2 = mos_bound[i].conjugate() * mos_bound[k] * mos_scatt[l].conjugate() * mos_scatt[j]
                                D = Dmat(Dipole[i,j,:], Dipole[k,l,:].conjugate())
                                f3 = D[m1+1,m2+1]
                                dxsec = f1*f2*f3 * pow(-1,mp+m1+ml) * (2.0*J+1.0) * Wprod
                                xsec[J] += dxsec
    # energy dependent factors are still missing
    print "xsec = %s" % xsec
    sigma = xsec[0]/(4.0*np.pi)
    beta = xsec[2]/xsec[0]
    return sigma, beta


def sigma2pi_ionization():
    mp = 0
    ll = 1
    xsec = np.zeros(3,dtype=complex)
    for ml in [-1,0,1]:
        m1 = ml
        lj = 1
        for mj in [-1,0,1]:
            m2 = mj
            for J in [0,1,2]:
                W1 = Wigner3J(1 , mp,  1 ,-mp,  J,0)
                W2 = Wigner3J(1 , m1,  1 ,-m2,  J,-(ml-mj))
                W3 = Wigner3J(ll, ml,  lj,-mj,  J,-(ml-mj))
                W4 = Wigner3J(ll,0,    lj,0,   J,0)
                Wprod = W1*W2*W3*W4

                xsec[J] += pow(-1,m1) * (2*J+1) * Wprod
    beta2 = xsec[2]/xsec[0]
    print beta2


def asymptotic_density(wavefunction, Rmax, E):
    from PI import LebedevQuadrature
    
    k = np.sqrt(2*E)
    wavelength = 2.0 * np.pi/k
    r = np.linspace(Rmax, Rmax+wavelength, 30)
    n = 5810
    th,phi,w = np.array(LebedevQuadrature.LebedevGridPoints[n]).transpose()
    
    x = LebedevQuadrature.outerN(r, np.sin(th)*np.cos(phi))
    y = LebedevQuadrature.outerN(r, np.sin(th)*np.sin(phi))
    z = LebedevQuadrature.outerN(r, np.cos(th))
    wfn2 = abs(wavefunction((x,y,z), 1.0))**2
    wfn2_angular = np.sum(w*wfn2, axis=0)

    from matplotlib.mlab import griddata
    import matplotlib.pyplot as plt

    th_resampled,phi_resampled = np.mgrid[0.0: np.pi: 30j, 0.0: 2*np.pi: 30j]
    resampled = griddata(th, phi, wfn2_angular, th_resampled, phi_resampled, interp="linear")

    # 2D PLOT
    plt.imshow(resampled.T, extent=(0.0, np.pi, 0.0, 2*np.pi))
    plt.colorbar()

    from DFTB.Modeling import MolecularCoords as MolCo
    R = MolCo.EulerAngles2Rotation(1.0, 0.5, -0.3)
    th, phi = rotate_sphere(R, th, phi)
    
    plt.plot(th, phi, "r.")
    plt.plot(th_resampled, phi_resampled, "b.")
    plt.show()
    # SPHERICAL PLOT
    from mpl_toolkits.mplot3d import Axes3D
    x_resampled = resampled * np.sin(th_resampled) * np.cos(phi_resampled)
    y_resampled = resampled * np.sin(th_resampled) * np.sin(phi_resampled)
    z_resampled = resampled * np.cos(th_resampled)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x_resampled, y_resampled, z_resampled, rstride=1, cstride=1)
    ax.scatter(x_resampled, y_resampled, z_resampled, color="k", s=20)
    plt.show()

def averaged_asymptotic_density(mo_bound, Dipole, bs, Rmax, E):
    from PI import LebedevQuadrature

    k = np.sqrt(2*E)
    wavelength = 2.0 * np.pi/k
    r = np.linspace(Rmax, Rmax+wavelength, 30)
    n = 5810
    #ths,phis,ws = np.array(LebedevQuadrature.LebedevGridPoints[n]).transpose()

    ths = []
    phis = []
    ws = []
    for theta in np.linspace(0.0, np.pi, 30):
        for phi in np.linspace(0.0, 2.0*np.pi, 30):
            ths.append(theta)
            phis.append(phi)
            ws.append(1)
    ths = np.array(ths)
    phis = np.array(phis)
    ws = np.array(ws)
        
            
    # dipole between bound orbital and the continuum AO basis orbitals
    dipole_bf = np.tensordot(mo_bound, Dipole, axes=(0,0))
    # average MF-PAD over all orientations of the laser polarization direction
    wfn2_angular_avg = np.zeros(ws.shape)

    #
    #weights, Rots = euler_average(10)
    weights, Rots = so3_quadrature(10)
    ez = np.array([0,0,1.0])
    for i,(w,R) in enumerate(zip(weights, Rots)):
        print "i = %d of %d" % (i, len(weights))
        # rotate laser polarization from lab frame into molecular frame, this is equivalent to
        # rotating the molecule from the molecular into the lab frame
        epol = np.dot(R, ez)
        # projection of dipoles onto polarization direction
        mo_scatt = np.zeros(Dipole.shape[1])
        for xyz in [0,1,2]:
            mo_scatt += dipole_bf[:,xyz] * epol[xyz]
        # normalized coefficients
        nrm2 = np.dot(mo_scatt, mo_scatt)
        mo_scatt /= np.sqrt(nrm2)
        print "sigma = %s" % nrm2
        print "mo_scatt = %s" % mo_scatt
        
        def wavefunction(grid, dV):
            # evaluate continuum orbital
            amp = Cube.orbital_amplitude(grid, bs.bfs, mo_scatt, cache=False)
            return amp

        # rotate continuum wavefunction from molecular frame into lab frame
        ths_rot, phis_rot = rotate_sphere(R, ths, phis)
        xs = LebedevQuadrature.outerN(r, np.sin(ths_rot)*np.cos(phis_rot))
        ys = LebedevQuadrature.outerN(r, np.sin(ths_rot)*np.sin(phis_rot))
        zs = LebedevQuadrature.outerN(r, np.cos(ths_rot))

        wfn2 = abs(wavefunction((xs,ys,zs), 1.0))**2
        wfn2_angular = np.sum(ws * wfn2, axis=0)

        """
        ##########
        from matplotlib.mlab import griddata
        import matplotlib.pyplot as plt

        th_resampled,phi_resampled = np.mgrid[0.0: np.pi: 30j, 0.0: 2*np.pi: 30j]
        resampled = griddata(ths, phis, wfn2_angular, th_resampled, phi_resampled, interp="linear")

        # 2D PLOT
        plt.imshow(resampled.T, extent=(0.0, np.pi, 0.0, 2*np.pi))
        plt.colorbar()
        #plt.plot(ths, phis, "r.")
        #plt.plot(xs, ys, "b.")
        plt.show()
        # SPHERICAL PLOT
        from mpl_toolkits.mplot3d import Axes3D
        x_resampled = resampled * np.sin(th_resampled) * np.cos(phi_resampled)
        y_resampled = resampled * np.sin(th_resampled) * np.sin(phi_resampled)
        z_resampled = resampled * np.cos(th_resampled)
        
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(x_resampled, y_resampled, z_resampled, rstride=1, cstride=1)
        ax.scatter(x_resampled, y_resampled, z_resampled, color="k", s=20)
        plt.show()
        ######
        print "W = %s" % w
        """
        
        #
        wfn2_angular_avg += w * wfn2_angular
    ####
    from matplotlib.mlab import griddata
    import matplotlib.pyplot as plt

    th_resampled,phi_resampled = np.mgrid[0.0: np.pi: 30j, 0.0: 2*np.pi: 30j]
    resampled = griddata(ths, phis, wfn2_angular_avg, th_resampled, phi_resampled, interp="linear")

    # 2D PLOT
    plt.imshow(resampled.T, extent=(0.0, np.pi, 0.0, 2*np.pi))
    plt.colorbar()
    #plt.plot(ths, phis, "r.")
    #plt.plot(xs, ys, "b.")
    plt.show()
    # SPHERICAL PLOT
    from mpl_toolkits.mplot3d import Axes3D
    x_resampled = resampled * np.sin(th_resampled) * np.cos(phi_resampled)
    y_resampled = resampled * np.sin(th_resampled) * np.sin(phi_resampled)
    z_resampled = resampled * np.cos(th_resampled)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x_resampled, y_resampled, z_resampled, rstride=1, cstride=1)
    ax.scatter(x_resampled, y_resampled, z_resampled, color="k", s=20)
    plt.show()

class OrientationAveraging:
    def __init__(self, Dipole, bs_free, Rmax, E, npts_euler=10):
        k = np.sqrt(2*E)
        wavelength = 2.0 * np.pi/k
        r = np.linspace(Rmax, Rmax+wavelength, 30)
        self.E = E
        
        self.Dipole = Dipole
        #self.weights, self.Rots = euler_average(npts_euler)
        self.weights, self.Rots = so3_quadrature(npts_euler)
        
        # spherical grid
        self.npts = 30
        self.rs, self.thetas, self.phis = np.mgrid[Rmax:(Rmax+wavelength):self.npts*1j, 0.0:np.pi:self.npts*1j, 0.0:2*np.pi:self.npts*1j]
        self.shape = self.rs.shape
        
        rs_rot, thetas_rot, phis_rot = [], [], []
        # unit vector along z-axis
        ez = np.array([0,0,1.0])
        # rotated polarization vector
        self.epol_rot = []
        # create all rotated versions of the spherical grid
        for i,(w,R) in enumerate(zip(self.weights, self.Rots)):
            # rotate grid
            thr,phr = rotate_sphere(R, self.thetas, self.phis)
            rs_rot += [self.rs]
            thetas_rot += [thr]
            phis_rot += [phr]
            # rotate E-field polarization
            self.epol_rot.append( np.dot(R, ez) )
            # 
        rs_rot = np.array(rs_rot)
        thetas_rot = np.array(thetas_rot)
        phis_rot = np.array(phis_rot)
        
        xs = rs_rot * np.sin(thetas_rot) * np.cos(phis_rot)
        ys = rs_rot * np.sin(thetas_rot) * np.sin(phis_rot)
        zs = rs_rot * np.cos(thetas_rot)

        grid = (xs,ys,zs)
        self.bf_grid = []
        # evaluate all continuum basis orbitals on the grid
        for bf in bs_free.bfs:
            self.bf_grid.append( bf.amp(*grid) )
            
    def averaged_pad(self, mo_bound):
        # dipole between bound orbital and the continuum AO basis orbitals
        dipole_bf = np.tensordot(mo_bound, self.Dipole, axes=(0,0))
        nmo = self.Dipole.shape[1] # number of scattering orbitals in continuum basis

        wfn_angular_avg = np.zeros((self.npts,self.npts))
        for i in range(0, len(self.Rots)):
            mo_scatt = np.zeros(nmo)
            for xyz in [0,1,2]:
                mo_scatt += dipole_bf[:,xyz] * self.epol_rot[i][xyz]
            mo_scatt *= np.sqrt(2.0*self.E)
            #
            nrm2 = np.dot(mo_scatt, mo_scatt)
            #mo_scatt /= np.sqrt(nrm2)

            wfn = np.zeros(self.shape)
            for mo in range(0, nmo):
                wfn += mo_scatt[mo] * self.bf_grid[mo][i,:,:,:]
            wfn_angular = np.sum(wfn**2 * self.rs**2, axis=0)

            wfn_angular_avg += self.weights[i] * wfn_angular
        pad = wfn_angular_avg
        
        # compute betas for each cut along the phi axis
        ntheta,nphi = pad.shape
        thetas = np.linspace(0.0, np.pi, ntheta)
        dtheta = thetas[1]-thetas[0]
        x = np.cos(thetas)
        dx = np.sin(thetas) * dtheta
        # Legendre polynomials
        P0 = np.ones(x.shape)
        P1 = x
        P2 = (3*x**2 - 1.0)/2.0
        P3 = (5*x**3 - 3*x)/2.0
        P4 = (35*x**4 - 30*x**2 + 3)/8.0
        P5 = (63*x**5 -70*x**3 + 15*x)/8.0
        P6 = (231*x**6 - 315*x**4 + 105*x**2 - 5)/16.0
        P = [P0,P1,P2,P3,P4,P5,P6]
        # check orthonormality
        """
        for n in range(0, len(P)):
            for m in range(0, len(P)):
                olap = np.sum(P[n]*P[m] * dx)
                print "%d %d  %s" % (n,m,olap)
                if n == m:
                    nrm = 2.0/(2*n+1.0)
                    assert abs(olap-nrm) < 1.0e-6
                else:
                    assert abs(olap) < 1.0e-6
        """
        #
        betas = np.zeros((5,nphi))
        for i in range(0, nphi):
            c = np.zeros(len(P))
            for n in range(0, len(P)):
                c[n] = (2.0*n+1.0)/2.0 * np.sum(pad[:,i] * P[n] * dx)
            sigma = c[0] * (4.0*np.pi)
            beta1 = c[1] / c[0]
            beta2 = c[2] / c[0]
            beta3 = c[3] / c[0]
            beta4 = c[4] / c[0]
            print "sigma = %s  beta1 = %s   beta2 = %s  beta3 = %s   beta4 = %s" % (sigma, beta1, beta2, beta3, beta4)
            betas[:,i] = np.array([sigma,beta1,beta2,beta3,beta4])
        # average
        betas = np.sum(betas, axis=1)/float(nphi)
        return pad, betas


class OrientationAveraging_small_memory:
    def __init__(self, Dipole, bs_free, Rmax, E, npts_euler=10, npts_theta=1000):
        k = np.sqrt(2*E)
        self.wavelength = 2.0 * np.pi/k
        self.E = E
        self.Dipole = Dipole
        
        # spherical grid
        self.npts_r = 30
        self.npts_theta = npts_theta
        self.npts_phi = 1
        self.rs, self.thetas, self.phis = np.mgrid[Rmax:(Rmax+self.wavelength):self.npts_r*1j, 0.0:np.pi:self.npts_theta*1j, 0.0:2*np.pi:self.npts_phi*1j]
        self.shape = self.rs.shape
        
        #self.weights, self.Rots = euler_average(npts_euler)
        self.weights, self.Rots = so3_quadrature(npts_euler)
        self.bs_free = bs_free
        
    def averaged_pad(self, mo_bound):
        # dipole between bound orbital and the continuum AO basis orbitals
        dipole_bf = np.tensordot(mo_bound, self.Dipole, axes=(0,0))
        #print "BOUND"
        #print mo_bound
        #print "DIPOLE"
        #print dipole_bf
        nmo = self.Dipole.shape[1] # number of scattering orbitals in continuum basis

        wfn2_angular_avg = np.zeros((self.npts_theta,self.npts_phi))
        # unit vector along z-axis
        ez = np.array([0,0,1.0])
        ez /= la.norm(ez)
        #### DEBUG
        epol_avg = np.zeros(3)
        ####
        for i,(w,R) in enumerate(zip(self.weights, self.Rots)):
            if i % min(100, len(self.weights)/2) == 0:
                print "%d of %d" % (i,len(self.weights))
            epol_rot = np.dot(R, ez)
            mo_scatt = np.zeros(nmo)
            for xyz in [0,1,2]:
                mo_scatt += dipole_bf[:,xyz] * epol_rot[xyz]
            # 
            mo_scatt *= np.sqrt(2.0*self.E)
            nrm2 = np.dot(mo_scatt.conjugate(), mo_scatt)
            #mo_scatt /= np.sqrt(nrm2)
            ### DEBUG
            #print "epol_rot = %s" % epol_rot
            #print "weight = %s" % w
            #print "MO_SCATT"
            #print mo_scatt
            epol_avg += epol_rot
            ###
            thetas_rot, phis_rot = rotate_sphere(R, self.thetas, self.phis)
            xs = self.rs * np.sin(thetas_rot) * np.cos(phis_rot)
            ys = self.rs * np.sin(thetas_rot) * np.sin(phis_rot)
            zs = self.rs * np.cos(thetas_rot)

            grid = (xs,ys,zs)

            amplitude_continuum = np.zeros(self.shape)
            for mo in range(0, nmo):
                amplitude_continuum += mo_scatt[mo] * self.bs_free.bfs[mo].amp(*grid)

            #amplitude_continuum = Cube.orbital_amplitude(grid, self.bs_free.bfs, mo_scatt, cache=False)

            wfn2 = abs(amplitude_continuum)**2
            dr = self.wavelength / float(self.npts_r)
            wfn2_angular = np.sum(wfn2 * dr, axis=0)

            """
            ###
            print "*********************"
            from matplotlib import pyplot as plt
            plt.ioff()
            plt.cla()
            mf_pad = wfn2_angular
            plt.imshow(np.fliplr(mf_pad.transpose()),
                       extent=[0.0, np.pi, 0.0, 2*np.pi], aspect=0.5, origin='lower')
            plt.xlim((0.0, np.pi))
            plt.ylim((0.0, 2*np.pi))
            plt.xlabel("$\\theta$")
            plt.ylabel("$\phi$")

            #plt.plot(thetas_rot[0,:,:].ravel(), phis_rot[0,:,:].ravel(), "x")
            plt.plot(self.thetas[0,:,:].ravel(), self.phis[0,:,:].ravel(), "x")

            # show piercing points of E-field vector
            # ... coming out of the plane
            r = la.norm(epol_rot)
            th_efield = np.arccos(epol_rot[2]/r) 
            phi_efield = np.arctan2(epol_rot[1],epol_rot[0]) + np.pi
            print "Rot"
            print R
            print "rotated epol = %s" % epol_rot
            print "rotated th, phi = %s %s" % (th_efield, phi_efield)
            plt.plot([th_efield],[phi_efield], "o", markersize=10, color=(0.0,1.0, 0.0))
            plt.text(th_efield, phi_efield, "E-field", color=(0.0,1.0,0.0), ha="center", va="top")
            # ... going into the plane
            th_efield = np.arccos(-epol_rot[2]/r) 
            phi_efield = np.arctan2(-epol_rot[1],-epol_rot[0]) + np.pi
            #print "- th, phi = %s %s" % (th_efield, phi_efield)
            plt.plot([th_efield],[phi_efield], "x", markersize=10, color=(0.0,1.0, 0.0))

            plt.show()
            if i % 10 == 0:
                plt.savefig("/tmp/rot_%d.png" % i)
            plt.ion()
            ###
            """
            wfn2_angular_avg += self.weights[i] * wfn2_angular
        ### DEBUG
        print "EPOL ROT AVG"
        print (epol_avg / float(len(self.weights)))
        ###
        pad = wfn2_angular_avg
        # compute betas for each cut along the phi axis
        ntheta,nphi = pad.shape
        thetas = np.linspace(0.0, np.pi, ntheta)
        dtheta = thetas[1]-thetas[0]
        x = np.cos(thetas)
        dx = np.sin(thetas) * dtheta
        # Legendre polynomials
        P0 = np.ones(x.shape)
        P1 = x
        P2 = (3*x**2 - 1.0)/2.0
        P3 = (5*x**3 - 3*x)/2.0
        P4 = (35*x**4 - 30*x**2 + 3)/8.0
        P5 = (63*x**5 -70*x**3 + 15*x)/8.0
        P6 = (231*x**6 - 315*x**4 + 105*x**2 - 5)/16.0
        P = [P0,P1,P2,P3,P4,P5,P6]
        # check orthonormality
        """
        for n in range(0, len(P)):
            for m in range(0, len(P)):
                olap = np.sum(P[n]*P[m] * dx)
                print "%d %d  %s" % (n,m,olap)
                if n == m:
                    nrm = 2.0/(2*n+1.0)
                    assert abs(olap-nrm) < 1.0e-6
                else:
                    assert abs(olap) < 1.0e-6
        """
        #
        betas = np.zeros((5,nphi))
        for i in range(0, nphi):
            c = np.zeros(len(P))
            for n in range(0, len(P)):
                c[n] = (2.0*n+1.0)/2.0 * np.sum(pad[:,i] * P[n] * dx)
            sigma = c[0] * (4.0*np.pi)
            beta1 = c[1] / c[0]
            beta2 = c[2] / c[0]
            beta3 = c[3] / c[0]
            beta4 = c[4] / c[0]
            print "sigma = %s  beta1 = %s   beta2 = %s  beta3 = %s   beta4 = %s" % (sigma, beta1, beta2, beta3, beta4)
            betas[:,i] = np.array([sigma,beta1,beta2,beta3,beta4])
        # average
        betas = np.sum(betas, axis=1)/float(nphi)
        return pad, betas
    
def euler_average(n):
    """
    grid for averaging over Euler angles a,b,g

    Parameters:
    ===========
    n: number of grid points for each of the 3 dimensions
    
    Returns:
    ========
    w: list of n^3 weights
    R: list of n^3 rotation matrices

    a function of a vector v, f(v), can be averaged over all roations as 

     <f> = sum_i w[i]*f(R[i].v)

    """
    alpha = np.linspace(0, 2.0*np.pi, n+1)[:-1]  # don't count alpha=0 and alpha=2pi twice
    beta  = np.linspace(0, np.pi, n)
    gamma = np.linspace(0, 2.0*np.pi, n+1)[:-1]  # don't count gamma=0 and gamma=2pi twice
    # list of weights
    weights = []
    # list of rotation matrices
    Rots = []
    da = alpha[1]-alpha[0]
    db = beta[1]-beta[0]
    dg = gamma[1]-gamma[0]
    dV = da*db*dg / (8.0*np.pi**2)
    for a in alpha:
        for b in beta:
            for g in gamma:
                w = np.sin(b) * dV
                if (abs(w) > 0.0):
                    weights += [w]
                    R = MolCo.EulerAngles2Rotation(a,b,g)
                    Rots += [ R ]
    return weights, Rots

def so3_quadrature(n):
    """
    grid for averaging over Euler angles a,b,g according to
    Kostoloc,P. et.al. FFTs on the Rotation Group, section 2.3
    """
    alphas = []
    betas = []
    gammas = []
    weights_beta = []
    for i in range(0, 2*n):
        ai = (2.0*np.pi*i)/(2.0*n)
        bi = (np.pi*(2.0*i+1))/(4.0*n)
        gi = (2.0*np.pi*i)/(2.0*n)
        # weights
        wi = 0.0
        for l in range(0, n):
            wi += 1.0/(2.0*l+1.0) * np.sin((2.0*l+1.0) * bi)
        wi *= 1.0/(1.0*n) * np.sin(bi)
        
        alphas.append(ai)
        betas.append(bi)
        gammas.append(gi)
        weights_beta.append(wi)
    # consistency check: The weights should be the solutions of the system of linear equations
    # sum_(k=0)^(2*n-1) wB(k) * Pm(cos(bk)) = delta_(0,m)   for 0 <= m < n
    from scipy.special import legendre
    for m in range(0, n):
        sm = 0.0
        Pm = legendre(m)
        for k in range(0, 2*n):
            sm += weights_beta[k] * Pm(np.cos(betas[k]))
        if m == 0:
            assert abs(sm-1.0) < 1.0e-10
        else:
            assert abs(sm) < 1.0e-10
    # 
    # list of weights
    weights = []
    # list of rotation matrices
    Rots = []
    
    dV = 1.0/(2.0*n)**2
    V = 0.0
    for i in range(0, 2*n):
        ai = alphas[i]
        for j in range(0, 2*n):
            bj = betas[j]
            wj = weights_beta[j]
            for k in range(0, 2*n):
                gk = gammas[k]
                #
                R = MolCo.EulerAngles2Rotation(ai,bj,gk)
                weights.append( wj * dV )
                Rots.append( R )
                #
                V += wj * dV
    assert abs(V-1.0) < 1.0e-10
    return weights, Rots


def rotate_sphere(R, th, phi):
    """
    rotate the points (th,phi) on a sphere by a rotation matrix R
    """
    # convert spherical to cartesian vectors
    er = np.array([
        np.sin(th)*np.cos(phi),
        np.sin(th)*np.sin(phi),
        np.cos(th)])
    # rotate this vector
    er_rot = np.tensordot(R, er, axes=(1,0))
    # convert back from cartesian to spherical coordinates
    x,y,z = er_rot[0,...], er_rot[1,...], er_rot[2,...]
    th_rot = np.arccos(z) 
    phi_rot = np.arctan2(y,x)
    # arctan2 returns the signed angle with the x-axis but we need an angle between 0 and 2 pi
    phi_rot[phi_rot < 0.0] = 2*np.pi - abs(phi_rot[phi_rot < 0.0])
    
    return th_rot, phi_rot
    
if __name__ == "__main__":
    # test
    sigma2pi_ionization()
    
    s=0.0
    l=2
    for m1 in [-1,0,1]:
        for m2 in [-1,0,1]:
            s += Wigner3J(1,m1, 1,-m2, l,m2-m1)**2
    assert abs(s - 1.0) < 1.0e-10
    assert abs(2.0 -  5 * Wigner3J(1,0, 1,0, 2,0) * Wigner3J(1,0,1,0,2,0) / ( Wigner3J(1,0, 1,0, 0,0) * Wigner3J(1,0, 1,0, 0,0) )) < 1.0e-10
