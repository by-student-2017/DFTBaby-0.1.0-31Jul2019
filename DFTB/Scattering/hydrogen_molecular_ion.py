#!/usr/bin/env python
"""
For the hydrogen molecular ion (H2+) the Schroedinger equation
may be solved exactly to arbitrary numerical precision.

Reference [3] contains a table of eigen energies (in Rydberg!) 
at different bond lengths for various values of m,n,q

References
----------
[1] J-P. Grivet, "The Hydrogen Molecular Ion Revisited",
    J. Chem. Educ., 2002, 79 (1), p 127
[2] L. Ponomarev, L. Somov, 
    "The Wave Functions of Continuum for the 
     Two-Center Problem in Quantum Mechanics",
    J. Comput. Phys. 1976, 20, 183-195.
[3] D. Bates, R. Reid in 
    "Electronic eigenenergies of the hydrogen molecular ion"
    Advances in atomic and molecular physics (1968), vol. 4, p. 13

"""
try:
    from DFTB.Analyse import Cube
except ImportError:
    print "DFTB module probably not available"

import os.path
# Tables with separation constants and discrete energies
# are loaded from the directory of the current file .
hmi_directory = os.path.dirname(os.path.abspath(__file__))
    
import numpy as np
import numpy.linalg as la
from scipy.integrate import ode
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt

def bisect(f, x1, x2):
    """between x1 and x2 f(x) changes sign."""
    f1 = f(x1)
    f2 = f(x2)
    if f1*f2 > 0.0:
        raise ValueError("f does not change sign between x1 = %s and x2 = %s, f(x1) = %s and f(x2) = %s" % (x1, x2, f1,f2))
    if f1 < 0.0:
        a = x1
        b = x2
    else:
        a = x2
        b = x1
    while True:
        midpoint = (a+b)/2.0
        fm = f(midpoint)
        if fm < 0.0:
            a = midpoint
        else:
            b = midpoint
        yield fm,(a,b)

def angular_equation_S(m,L,a,b,c2,parity, count_nodes=False, plot=True, Nx=None):
    assert b == 0.0, "Code is not correct for b != 0"
    
    def f(x, y):
        # y1 = f(x),  y2 = f'(x)
        y1,y2 = y
        y1deriv = y2
        y2deriv = (m*(m+1) - L + b*x + c2*x**2)/(1.0-x**2) * y1 + 2*(m+1)*x/(1.0-x**2) * y2
        return [y1deriv, y2deriv]

    def jac(x, y):
        # df/dy
        return [[1.0, 0.0],
                [(m*(m+1) - L + b*x + c2*x**2)/(1.0-x**2), 2*(m+1)*x/(1.0-x**2)]]

    I = ode(f, jac)
    I.set_integrator('lsoda', with_jacobian=True, atol=[1.0e-16, 1.0e-16])
    # integrate from x=-1.0 to x=0.0
    xl = -1.0 + 1.0e-8 #1.0e-12
    yl = [parity*1.0, parity*(m*(m+1) - L - b + c2)/(2.0*(m+1.0))]

    I.set_initial_value(yl, xl)
    I.integrate(0.0)
    val_l = I.y[0]
    deriv_l = I.y[1]
    # integrate from x=+1.0 to x=0.0
    xr = 1.0 - 1.0e-12
    yr = [1.0, (m*(m+1) - L - b + c2)/(2.0*(m+1.0))]    
    I.set_initial_value(yr, xr)
    I.integrate(0.0)
    val_r = I.y[0]
    deriv_r = I.y[1]
    if parity == 1:
        mismatch = deriv_l - deriv_r
    else:
        mismatch = val_l - val_r

    if count_nodes == True:
        # count nodes in the interval [-1,0]
        nodes_l = 0
        I.set_initial_value(yl,xl)
        y_last = yl[0]
        dx = 0.01
        while I.successful() and I.t < -1.0e-2:
            I.integrate(I.t+dx)
            if (y_last * I.y[0]) < 0.0:
                nodes_l += 1
            y_last = I.y[0]
        #
        if parity == +1:
            nodes = 2 * nodes_l
        else:
            nodes = 2 * nodes_l + 1

    if Nx != None:
        """
        eta_range = [-1.0]
        solution = [parity*1.0]
        # compute solution on a grid with Nx points between -1.0 and 1.0
        I.set_initial_value(yl,xl)
        dx = 2.0/float(Nx)
        while I.successful() and I.t < 2.0:
            eta_range.append(I.t+dx)
            I.integrate(I.t+dx)
            solution.append(I.y[0])
        eta_range = np.array(eta_range)
        solution = np.array(solution)
        """
        dx = 0.0005
        xs_left = [xl]
        fs_left = [yl[0]]
        I.set_initial_value(yl, xl)
        while I.successful() and I.t < 0.0:
            I.integrate(I.t+dx)
            xs_left.append( I.t )
            fs_left.append( I.y[0] )

        I.set_initial_value(yr, xr)
        xs_right = [xr]
        fs_right = [yr[0]]
        while I.successful() and I.t > 0.0:
            I.integrate(I.t-dx)
            xs_right.append( I.t )
            fs_right.append( I.y[0] )
        xs= xs_left + list(reversed(xs_right))
        fs = fs_left + list(reversed(fs_right))
        eta_range = np.linspace(-1.0, 1.0, Nx) 
        solution = np.interp(eta_range, xs, fs)
        
    if plot == True:
        # For plotting

        dx = 0.01
        xs_left = [xl]
        fs_left = [yl[0]]
        I.set_initial_value(yl, xl)
        while I.successful() and I.t < 0.0:
            I.integrate(I.t+dx)
            xs_left.append( I.t )
            fs_left.append( I.y[0] )

        I.set_initial_value(yr, xr)
        xs_right = [xr]
        fs_right = [yr[0]]
        while I.successful() and I.t > 0.0:
            I.integrate(I.t-dx)
            xs_right.append( I.t )
            fs_right.append( I.y[0] )

        plt.plot(xs_left, fs_left, label="left", color="black")
        plt.plot(xs_right,fs_right,label="right", color="red")

        """
        ### DEBUG
        # compare with series expansions around eta=-1 (left) and eta=+1 (right)
        from HMI import angular_equation_f_series_expansion
        xs_left = np.array(xs_left)
        xs_right = np.array(xs_right)
        fs_left, fs_der_left   = angular_equation_f_series_expansion(m,L,a,b,c2,-1, +1, xs_left, 20)
        fs_right, fs_der_right = angular_equation_f_series_expansion(m,L,a,b,c2,+1, parity,xs_right, 20)
        plt.plot(xs_left, fs_left, label="left (series)", color="blue", ls="-.", lw=3)
        plt.plot(xs_right, fs_right, label="right (series)", color="orange", ls="-.", lw=3)
        plt.draw()
        plt.show()
        ###
        """
        
    if Nx != None:
        return eta_range, solution    
        
    if count_nodes == True:
        return mismatch, nodes
    else:
        return mismatch

def find_roots(f, x_search_range, eps_conv=1.0e-12):
    """
    finds the roots of f(x) by bisection. 

    Parameters:
    ===========
    f: scalar function whose roots should be determined
    x_search_range: array with interval boundaries. Each interval may contain at most one root
    eps_conv: Iteration stops if  |f(x0)| < esp_conv

    Returns:
    ========
    roots: list of roots [x0_1,x0_2,...]
    """
    x_last = x_search_range[0]
    f_last = f(x_last)
    roots = []
    for i in range(1,len(x_search_range)):
        # look for sign change in the interval [x[i-1],x[i]]
        x1 = x_last
        f1 = f_last
        x2 = x_search_range[i]
        f2 = f(x2)
        if f1*f2 < 0.0:
            #print "sign change in interval (%s,%s) from %s to %s" % (x1,x2, f1,f2)
            for fm,(a,b) in bisect(f,x1,x2):
                #print "(%s,%s)   f = %s" % (a,b,fm)
                if abs(fm) < eps_conv:
                    x0 = (a+b)/2.0
                    #print "root found at x = %s" % x0
                    roots.append( x0 )
                    ####
                    plt.cla()
                    ####
                    break
                if abs(a-b) < eps_conv:
                    #print "No root despite sign change???"
                    x0 = (a+b)/2.0
                    if abs(fm) < 1.0e-3:
                        roots.append( x0 )
                    ####
                    plt.cla()
                    ####
                    break
        x_last = x2
        f_last = f2

    return roots

class SeparationConstants:
    def __init__(self, R, Za, Zb, plot=False):
        self.R = R
        self.Za = Za
        self.Zb = Zb
        assert Za == Zb
        self.table_file = os.path.join(hmi_directory, "separation_constants.dat")
        self.plot = plot
    def tabulate_separation_constants(self, energy_range, mmax=0, nmax=10):
        a = self.R*(self.Zb+self.Za)
        b = self.R*(self.Zb-self.Za)
        c2s = self.R**2 * energy_range / 2.0
        self.c2_range = (c2s.min(), c2s.max())
        self.mmax = mmax
        self.nmax = nmax
        
        fh = open(self.table_file, "w")
        print>>fh, "# Za Zb"
        print>>fh, "%d %d" % (self.Za, self.Zb)
        print>>fh, "# c2 range"
        print>>fh, "%10.20f %10.20f" % self.c2_range
        print>>fh, "# M   N   coefficients of polynomial fit in descending order"
        ###
        plt.ion()
        ###
        nc2 = len(c2s)
        self.deg = nc2 # degree of fitting polynomial

        Ls = np.zeros((nc2,nmax+1,mmax+1))
        for m in range(0, mmax+1):
            print "*** M = %d ***" % m
            for i,c2 in enumerate(c2s):
                print " c2 = 1/2 R^2 * energy = %s" % c2
                plt.cla()
                
                Lroots = []
                Lnodes = []
                for parity in [+1,-1]:
                    print " parity= %d" % parity
                    def f(L):
                        mismatch = angular_equation_S(m,L,a,b,c2,parity,count_nodes=False)
                        return mismatch
                    L_search_range = np.linspace(-30.0, 200.0, 200)
                    LrootsP = find_roots(f, L_search_range)
                    print "Found %s roots" % len(Lroots)
                    # count nodes
                    LnodesP = []
                    for L0 in LrootsP:
                        mismatch, nodes = angular_equation_S(m,L0,a,b,c2,parity, count_nodes=True)
                        LnodesP.append( nodes )
                    # added roots and node counts for this parity
                    Lnodes += LnodesP
                    Lroots += LrootsP

                print "roots      : %s" % Lroots
                print "node counts: %s" % Lnodes
                assert len(Lnodes) == len(Lroots)
                for n in range(0, nmax+1):
                    Ls[i,n,m] = Lroots[Lnodes.index(n)]

            # fit 4th degree polynomial
            L_poly_fit = np.polyfit(c2s, Ls[:,:,m], self.deg)
            #
            for n in range(0, nmax):
                print>>fh, "%d %d   " % (m,n),
                for d in range(0, self.deg+1):
                    print>>fh, "  %20.10f" % L_poly_fit[d,n],
                print>>fh, ""
            ###
            if self.plot == True:
                plt.cla()
                for n in range(0, nmax):
                    plt.plot(c2s, Ls[:,n])
                    plt.plot(c2s, np.polyval(L_poly_fit[:,n], c2s), ls="-.")
                plt.draw()
                plt.ioff()
                plt.show()
            ###
        fh.close()
        ###
        if self.plot == True:
            plt.ioff()
            plt.show()
        ###

    def load_separation_constants(self):
        fh = open(self.table_file)
        comment = fh.readline()
        self.Za,self.Zb = map(int, fh.readline().split())
        comment = fh.readline()
        self.c2_range = map(float, fh.readline().split())
        
        data = np.loadtxt(fh)
        ms,ns = data[:,0], data[:,1]
        self.mmax = int(ms.max())
        self.nmax = int(ns.max())
        
        self.polys = {}
        for i,(m,n) in enumerate(zip(ms,ns)):
           self.polys[(m,n)] = data[i,2:]
        """
        ###
        plt.cla()
        c2s = np.linspace(-35.0, 16.0, 100)
        for i,(m,n) in enumerate(zip(ms,ns)):
            plt.plot(c2s, np.polyval(self.polys[(m,n)], c2s), ls="-.", label="m=%d n=%d" % (m,n))
        plt.legend()
        plt.ioff()
        plt.show()
        plt.ion()
        ###
        """
    def L_interpolated(self, m, n):
        p = self.polys[(m,n)]
        def L(c2):
            assert self.c2_range[0] <= c2 <= self.c2_range[1], "c2 outside of range for interpolation"
            return (m,n, np.polyval(p, c2))
        return L
        
################### RADIAL EQUATION R(xi) ##################

def radial_equation_R_series_expansion(m,L,a,b,c2, xs, n):
    """
    solve radial equation for g(x) around x=1 by a series expansion
    of length n.
    
    Parameters
    ----------
    m      :  number of nodes in angular wavefunction S(eta)
    L      :  value of separation constant
    a      :  R*(Za+Zb) = 2*D, where D is the distance between the two protons
    b      :  R*(Za-Zb) = 0, has to be zero
    c2     :  related to energy E of solution, c^2 = 1/2 E D^2
    xs     :  1d numpy array with grid on which g(x) should be evaluated, 
              the points should lie close to x=1 for the solution to be valid
    n      :  degree of polynomial in expansion

    Returns
    -------
    gs     :  1d numpy array with g(xs)
    gs_der :  1d numpy array with derivative g'(xs)

    The differential equation to be solved is:

       (x^2-1) g''(x) + 2*(m+1) x g'(x) + [a*x + c^2*x^2 + m*(m+1) - L] g(x) = 0

    The radial wavefunction R(x) is related to g(x) by:

       R(x) = g(x) * (x^2-1)^(m/2)

    The ansatz for g(x) around x=1 is given by

       g(x) = sum_i^n d_i (1 - x)^i

    Substituting this ansatz into the differential equation and equating coefficients
    leads to a four term recurrence relation (see eqn. 14 in Ref. [2]) for the d's:

       p_i d_(i+1)  +  q_i d_i  +  r_i d_(i-1)  +  s_i d_(i-2)

    with the coefficients

       p_i = -2*(i+1)*(i+m+1)
       q_i = i*(i-1) + 2*i*(m+1) + m*(m+1) - L + c^2 + a
       r_i = -a - 2*c^2
       s_i = c^2

    and the initial conditions
       
       d_(-2) = 0 ,  d_(-1) = 0,  d_0 = 1      
    """
    assert b == 0.0, "Code does not work for Za != Zb"
    # compute coefficients of 4 term recurrence
    p = np.zeros(n+1)
    q = np.zeros(n+1)
    r = np.zeros(n+1)
    s = np.zeros(n+1)
    for i in range(0, n+1):
        p[i] = -2*(i+1)*(i+m+1)
        q[i] = i*(i-1)+2*(m+1)*i+m*(m+1)-L+c2 + a
        r[i] = -a - 2*c2
        s[i] = c2
    # initialize recurrence relation
    d = np.zeros(n+1+2)
    d[-2] = 0.0
    d[-1] = 0.0
    d[0]  = 1.0
    # iterate recurrence relation
    for i in range(0,n):
        d[i+1] = (-q[i]*d[i]-r[i]*d[i-1]-s[i]*d[i-2])/p[i]
        
    # remove entries for d_(-2) and d_(-1)
    d = d[0:-2]
        
    # evaluate the polynomial
    # g(x) = sum_i=0^n d_i (1-x)^i
    omxs = 1.0-xs
    # poly1d needs the coefficients in reverse order
    poly = np.poly1d(d[::-1])
    # derivative
    poly_der = np.polyder(poly, m=1)
    # evaluate polynomial
    gs = np.polyval(poly, omxs)
    # evaluate derivative of polynomial
    # The minus sign comes from the chain rule
    #  d/d(x) = - d/d(1-x)
    gs_der = -np.polyval(poly_der, omxs)
    
    return gs, gs_der
    
def radial_equation_R(m,L,a,b,c2,count_nodes=False, plot=True, Xmax=10.0, Nx=None):
    assert b == 0.0, "Code does not work for Za != Zb"
    
    def g(x, y):
        # y1 = g(x),  y2 = g'(x)
        y1,y2 = y
        y1deriv = y2
        y2deriv = -(a*x + c2*x**2 + m*(m+1) - L)/(x**2-1.0) * y1 - (2*(m+1)*x)/(x**2-1.0) * y2
        return [y1deriv, y2deriv]

    def jac(x, y):
        # df/dy
        return [[1.0, 0.0],
                [-(a*x + c2*x**2 + m*(m+1) - L)/(x**2-1.0), - (2*(m+1)*x)/(x**2-1.0)]]

    I = ode(g, jac)
    I.set_integrator('lsoda', with_jacobian=True, atol=[1.0e-16, 1.0e-16])
    # integrate from x=1.0 to Xmax
    x1 = 1.0 + 1.0e-12
    y1 = [1.0, - (a + c2 + m*(m+1) - L)/(2.0*(m+1.0))]
    I.set_initial_value(y1, x1)
    I.integrate(Xmax)

    yl_vec = I.y
    # integrate from x=2*Xmax to Xmax
    xend = 25.0
    c = np.sqrt(abs(c2))
    yend = [np.exp(-c*xend), -c*np.exp(-c*xend)]
    I.set_initial_value(yend,xend)
    I.integrate(Xmax)
    yr_vec = I.y

    # scale right solution
    scale = yl_vec[0] / yr_vec[0]
    yr_vec *= scale
    # match derivatives
    mismatch = yr_vec[1] - yl_vec[1]

    """
    if abs(I.y[0]) < abs(I.y[1]):
        mismatch = I.y[1]
    else:
        mismatch = I.y[0]
    """
    
    if count_nodes == True:
        # count nodes in the interval [0, Xmax]
        nodes = 0
        I.set_initial_value(y1,x1)
        dx = 0.01
        y_last = y1
        while I.successful() and I.t < Xmax:
            I.integrate(I.t+dx)
            if (y_last * I.y[0]) < 0.0:
                nodes += 1
            y_last = I.y[0]

    if Nx != None:
        dx = 0.0005
        # outward integration
        xs_left = [x1]
        gs_left = [y1[0]]
        I.set_initial_value(y1,x1)
        while I.successful() and I.t < Xmax:
            I.integrate(I.t+dx)
            xs_left.append( I.t )
            gs_left.append( I.y[0] )

        # inward integration
        xs_right = [xend]
        gs_right = [yend[0]]
        I.set_initial_value(yend,xend)
        while I.successful() and I.t > Xmax:
            I.integrate(I.t-dx)
            xs_right.append( I.t )
            gs_right.append( scale * I.y[0] )

        xs = xs_left + list(reversed(xs_right))
        gs = gs_left + list(reversed(gs_right))
        xi_range = np.linspace(1.0, xend, Nx) 
        solution = np.interp(xi_range, xs, gs)
            
    if plot == True:
        dx = 0.01
        # outward integration
        xs_left = [x1]
        gs_left = [y1[0]]
        I.set_initial_value(y1,x1)
        while I.successful() and I.t < Xmax:
            I.integrate(I.t+dx)
            xs_left.append( I.t )
            gs_left.append( I.y[0] )
        xs_left = np.array(xs_left)

        plt.ylim((-1.0, 1.0))
        plt.plot(xs_left,gs_left, color="black")

        # inward integration
        xs_right = [xend]
        gs_right = [yend[0]]
        I.set_initial_value(yend,xend)
        while I.successful() and I.t > Xmax:
            I.integrate(I.t-dx)
            xs_right.append( I.t )
            gs_right.append( scale * I.y[0] )

        plt.plot(xs_right,gs_right, color="red")        

        """
        # asymptotic oscillating functions
        c = np.sqrt(abs(c2))
        plt.plot(xs, np.cos(c*xs) * pow(xs, -(m+1)), ls="-.")
        """
        ##
        plt.draw()

    if Nx != None:
        return xi_range, solution
        
    if count_nodes == True:
        return mismatch, nodes
    else:
        return mismatch

def radial_equation_R_oscillating(m,L,a,b,c2,count_nodes=False, plot=True, Xmax=2500.0, Nx=None):
    assert b == 0.0, "Code does not work for Za != Zb"
    assert c2 >= 0.0
    
    def g(x, y):
        # y1 = g(x),  y2 = g'(x)
        y1,y2 = y
        y1deriv = y2
        y2deriv = -(a*x + c2*x**2 + m*(m+1) - L)/(x**2-1.0) * y1 - (2*(m+1)*x)/(x**2-1.0) * y2
        return [y1deriv, y2deriv]

    def jac(x, y):
        # df/dy
        return [[1.0, 0.0],
                [-(a*x + c2*x**2 + m*(m+1) - L)/(x**2-1.0), - (2*(m+1)*x)/(x**2-1.0)]]

    I = ode(g, jac)
    I.set_integrator('lsoda', with_jacobian=True, atol=[1.0e-16, 1.0e-16])
    # integrate from x=1.0 to Xmax
    """
    # Initial values at x=1.0, starting to integrate directly
    # from x=1.0+eps leads to low accuracy.
    x1 = 1.0 + 1.0e-12
    y1 = [1.0, - (a + c2 + m*(m+1) - L)/(2.0*(m+1.0))]
    I.set_initial_value(y1, x1)
    """
    # At x=1.0 the direct integration fails and we have to use
    # a series expansion in (1-x) to get the initial values
    # at x=1+dx. 
    # integration method again.
    x1 = 1.0 + 0.0001 #1.0e-12
    y1 = radial_equation_R_series_expansion(m,L,a,b,c2, x1, 20)
    I.set_initial_value(y1, x1)
    
    dx = 0.001
    while I.successful() and I.t < Xmax:
        I.integrate(I.t + dx)
    assert I.successful()
    
    # oscillating solution
    c = np.sqrt(c2)
    def ginf(x, Delta):
        return np.cos(c * x + a/(2*c) * np.log(2*c*x) + Delta) * pow(x,-(m+1))
    # compute phase shift and normalization factor by matching solution at Xmax to the
    # oscillating solution
    print "Xmax = %s    I.t = %s" % (Xmax, I.t)
    #Delta = np.arctan2(-I.y[1], c*I.y[0]) - c*I.t
    #Delta = np.arctan2(-I.y[1], c*I.y[0]) - c*I.t - a/(2*c)*np.log(2*c*I.t)
    Delta = np.arctan2(-I.y[1], (c+a/(2*c*I.t))*I.y[0]) - c*I.t - a/(2*c)*np.log(2*c*I.t)
    Delta = Delta % (2.0*np.pi)
    A = ginf(I.t, Delta) / I.y[0]
    print "phase shift Delta = %s" % Delta
    print "normalization constant A = %s" % A
    
    xend = Xmax
    if count_nodes == True:
        # count nodes in the interval [0, Xmax]
        nodes = 0
        I.set_initial_value(y1,x1)
        dx = 0.001
        y_last = y1[0]
        while I.successful() and I.t < Xmax:
            I.integrate(I.t+dx)
            if (y_last * I.y[0]) < 0.0:
                nodes += 1
            y_last = I.y[0]

    if Nx != None:
        dx = 0.001
        # outward integration only
        xs = []
        gs = []
        I.set_initial_value(y1,x1)
        while I.successful() and I.t < xend:
            I.integrate(I.t+dx)
            xs.append( I.t )
            gs.append( A * I.y[0] )

        xi_range = np.linspace(1.0, xend, Nx) 
        solution = np.interp(xi_range, xs, gs)
            
    if plot == True:
        dx = 0.01
        # outward integration
        xs = []
        gs = []
        I.set_initial_value(y1,x1)
        while I.successful() and I.t < xend:
            I.integrate(I.t+dx)
            xs.append( I.t )
            gs.append( A * I.y[0] )
        xs = np.array(xs)

        plt.cla()
        plt.ylim((-1.0, 1.0))
        plt.plot(xs,gs, lw=2, color="black", label=r"$g(\xi)$")

        # series expansion around x=1
        gs_series, gs_der_series = radial_equation_R_series_expansion(m,L,a,b,c2, xs, 20)
        plt.plot(xs, A * gs_series, lw=1, color="green", label=r"$g(\xi)$ series around $\xi=1$")
        
        # asymptotic oscillating functions
        c = np.sqrt(abs(c2))
        #for Delta_test in np.linspace(1.11-0.01, 1.11+0.01, 8):
        #    #plt.plot(xs, np.cos(c*xs + Delta) * pow(xs, -(m+1)), ls="-.")
        #    plt.plot(xs, np.cos(c*xs + a/(2*c) * np.log(2*c*xs) + Delta_test) * pow(xs, -(m+1)), ls="-.", label="%s" % Delta_test)
        plt.plot(xs, np.cos(c*xs + a/(2*c) * np.log(2*c*xs) + Delta) * pow(xs, -(m+1)), lw=1, color="blue", label="asymptotic $\delta=%s$" % Delta)

            
        plt.legend()
        ##
        plt.draw()
        plt.ioff()
        plt.show()
        
    if Nx != None:
        return Delta, xi_range, solution
        
    if count_nodes == True:
        return Delta, nodes
    else:
        return Delta

    
def bound_eigenenergies(R, Za, Zb, energy_search_range, Lfunc):
    ##
    plt.ion()
    ##
    a = R*(Zb+Za)
    b = R*(Zb-Za)
    
    def f(c2):
        (m,n,L0) = Lfunc(c2)
        parity = pow(-1,n) #pow(-1,n-m)
        L_search_range = np.linspace(L0-1.0, L0+1.0, 10)        
        def fL(L):
            mismatch = angular_equation_S(m,L,a,b,c2,parity,count_nodes=False, plot=False)
            return mismatch
        LrootsP = find_roots(fL, L_search_range)
        L = LrootsP[0]
        
        mismatch = radial_equation_R(m,L,a,b,c2, count_nodes=False)
        return mismatch
    c2_search_range = energy_search_range * R**2/2.0
    c2_roots = find_roots(f, c2_search_range)
    # count nodes
    node_counts = []
    for c2 in c2_roots:
        (m,n,L) = Lfunc(c2)
        
        mismatch, nodes = radial_equation_R(m,L,a,b,c2,count_nodes=True, plot=True)
        node_counts.append(nodes)
        
    Enuc = Za*Zb / R
    energies = np.array(c2_roots) * 2.0/R**2 #+ Enuc
    print "Found %s eigenenergies" % len(energies)
    print "Energies   : %s Hartree" % energies
    print "Node Counts: %s" % node_counts
    ###
    plt.ioff()
    plt.cla()
    for c2 in c2_roots:
        (m,n,L) = Lfunc(c2)
        
        mismatch = radial_equation_R(m,L,a,b,c2,count_nodes=False, plot=True)
    ###
    #plt.show()
    plt.draw()
    return node_counts, energies

class DiscreteSpectrum:
    def __init__(self, Lsep):
        self.Lsep = Lsep
        self.energy_file = os.path.join(hmi_directory, "discrete_energies.dat")
    def tabulate_discrete_energies(self, mmax=0, nmax=3):
        fh = open(self.energy_file, "w")
        print>>fh, "# M N Q    discrete electronic energies in Hartree"
        self.Lsep.load_separation_constants()
        R, Za, Zb = self.Lsep.R, self.Lsep.Za, self.Lsep.Zb
        c2_min, c2_max = self.Lsep.c2_range
        c2_max = min(c2_max, 0.0)
        energy_range = np.linspace(c2_min, c2_max, 100) * 2.0/R**2
        for m in range(0, mmax+1):
            print "**** m = %s ****" % m
            for n in range(0, nmax+1):
                print "**** n = %s ****" % n
                Lfunc = self.Lsep.L_interpolated(m,n)
                node_counts, energies = bound_eigenenergies(R, Za, Zb, energy_range, Lfunc)
                for nodes, en in zip(node_counts, energies):
                    print>>fh, "%d %d %d   %20.10f" % (m,n,nodes, en)
        fh.close()
    def binding_curve(self):
        fh = open("binding_curve.dat", "w")
        print>>fh, "# R / bohr      Etot / Hartree"
        self.Lsep.load_separation_constants()
        R, Za, Zb = self.Lsep.R, self.Lsep.Za, self.Lsep.Zb
        for R in np.linspace(1.0, 5.0, 10):
            c2_min, c2_max = self.Lsep.c2_range
            c2_max = min(c2_max, 0.0)
            energy_range = np.linspace(c2_min, c2_max, 100) * 2.0/R**2
            Lfunc = self.Lsep.L_interpolated(0,0)
            node_counts, energies = bound_eigenenergies(R, Za, Zb, energy_range, Lfunc)
            Enuc = Za*Zb / R
            print>>fh, "%s %s" % (R, energies[0] )# + Enuc)
            fh.flush()
        fh.close()
            
    def load_discrete_energies(self):
        data = np.loadtxt(self.energy_file)
        self.ms = np.asarray(data[:,0], dtype=int)
        self.ns = np.asarray(data[:,1], dtype=int)
        self.qs = np.asarray(data[:,2], dtype=int)
        self.ens = np.asarray(data[:,3], dtype=float)
        self.energies = {}
        for (m,n,q,en) in zip(self.ms, self.ns, self.qs, self.ens):
            self.energies[(m,n,q)] = en
        
        
########## compute wavefunctions ############
#
# wavefunction of a single electron in the potential of two positive point charges
# separated by a distance R.
#
class DimerWavefunctions:
    def __init__(self, R, Za,Zb, plot=False):
        """
        Parameters
        ----------
        R    : separation between atoms (in bohr)
        Za,Zb: atomic numbers of atoms A and B 
        """
        self.Lsep = SeparationConstants(R, Za, Zb)
        self.Lsep.load_separation_constants()
        self.spectrum = DiscreteSpectrum(self.Lsep)
        self.spectrum.load_discrete_energies()
        # 
        self.plot = plot
    def getBoundOrbital(self, m,n,trig,q):
        """
        Paramters:
        ==========
        m    : number of angular nodes in P(phi)
               m=0 (sigma), m=1 (pi), m=2 (delta), m=3 (phi)
        n    : number of angular nodes in S(eta)
        trig : functional form of P(phi) ('sin' or 'cos'), for m > 0
               both are valid solutions
        q    : number of radial nodes in R(xi)

        Returns:
        ========
        S,R,P: functions that calculate the angular and radial parts of the wavefunction
               psi(eta,xi,phi) = S(eta) * R(xi) * P(phi).
               The functions are first evaluated on an equidistant grid and then fitted
               to splines.
        """
        D, Za, Zb = self.Lsep.R, self.Lsep.Za, self.Lsep.Zb
        a = D*(Zb+Za)
        b = D*(Zb-Za)

        Lfunc = self.Lsep.L_interpolated(m,n)
        parity = pow(-1,n) #pow(-1,n-m)
        energy_search_range = np.linspace(-3.0, 0.0, 100)
        def findL(L0, c2):
            L_search_range = np.linspace(L0-3.0, L0+3.0, 20)        
            def fL(L):
                mismatch = angular_equation_S(m,L,a,b,c2,parity,count_nodes=False, plot=True)
                return mismatch
            Lroots = find_roots(fL, L_search_range)
            # count nodes
            Lnodes = []
            for L0 in Lroots:
                mismatch, nodes = angular_equation_S(m,L0,a,b,c2,parity, count_nodes=True)
                Lnodes.append( nodes )
            # select separation constant with the correct node count
            for L,nL in zip(Lroots, Lnodes):
                if nL == n:
                    break
            else:
                raise Exception("No root with n=%d nodes found.\n Lroots=%s  Lnodes=%s" % (n, Lroots, Lnodes))
            return L
        
        def f(c2):
            m,n,L0 = Lfunc(c2)
            L = findL(L0, c2)
            mismatch = radial_equation_R(m,L,a,b,c2, count_nodes=False)
            return mismatch
        c2_search_range = energy_search_range * D**2/2.0
        c2_roots = find_roots(f, c2_search_range)
        # count nodes
        node_counts = []
        for c2 in c2_roots:
            m,n,L0 = Lfunc(c2)
            L = findL(L0, c2)
            mismatch, nodes = radial_equation_R(m,L,a,b,c2,count_nodes=True, plot=True)
            node_counts.append(nodes)
            if nodes == q:
                break
        else:
            print "node_counts = %s" % node_counts
            raise Exception("No root with q=%d nodes found" % q)
        # c2 is the eigenenergy and L the separation constant
        en = c2 * 2.0/D**2
        eta_range, f = angular_equation_S(m,L,a,b,c2,parity,plot=False, Nx=400)
        S = f * pow(abs(1.0 - eta_range**2),m/2.0)
        xi_range, g = radial_equation_R(m,L,a,b,c2,parity, plot=False, Nx=400)
        R = g * pow(xi_range**2-1.0,m/2.0)
        # compute normalization constant
        deta = eta_range[1]-eta_range[0]
        dxi = xi_range[1]-xi_range[0]

        nS2 = np.sum(S**2*deta)
        neta2S2 = np.sum(eta_range**2 * S**2 * deta)
        nR2 = np.sum(R**2*dxi)
        nxi2R2 = np.sum(xi_range**2 * R**2 *dxi)
        nrm2 = D**3 / 4.0 * np.pi * (nS2 * nxi2R2 - neta2S2 * nR2)
        #print "norm^2 = %s" % nrm2
        # normalize radial wavefunction
        R /= np.sqrt(nrm2)
        # 
        Sfunc = interp1d(eta_range, S, kind='cubic',
                         fill_value=0) #"extrapolate")
        Rfunc = interp1d(xi_range, R, kind='cubic',
                         bounds_error=False, fill_value=0.0)
        
        # for m=0, only P(phi) = cos(0*phi)=1 is a valid, normalizable solution
        if m == 0:
            assert trig == 'cos'
        
        def Pfunc(phi):
            if trig == 'sin':
                return np.sin(m*phi)
            if trig == 'cos':
                return np.cos(m*phi)

        ### DEBUG
        from HMI import create_radial_function_R, create_angular_function_S, create_azimuthal_function_P

        Rfunc_num = create_radial_function_R(m,L,a,b,c2, norm2=nrm2)
        Sfunc_num = create_angular_function_S(m,L,a,b,c2,parity)
        Pfunc_num = create_azimuthal_function_P(m, trig)

        """
        plt.cla()
        plt.plot(xi_range, Rfunc(xi_range), label="R (interpolated)", ls="-", color="black")
        plt.plot(xi_range, Rfunc_num(xi_range), label="R (numerical)", ls="-.", color="blue")
        plt.legend()
        plt.show()

        plt.cla()
        plt.plot(eta_range, Sfunc(eta_range), label="S (interpolated)", ls="-", color="black")
        plt.plot(eta_range, Sfunc_num(eta_range), label="S (numerical)", ls="-.", color="blue")
        plt.legend()
        plt.show()
        """
        ###
        
        def wavefunction(grid, dV):
            x,y,z = grid
            # convert cartesian coordinates to spheroidal coordinates
            rA = np.sqrt(x**2 + y**2 + (z+D/2.0)**2)
            rB = np.sqrt(x**2 + y**2 + (z-D/2.0)**2)
            xi = (rA+rB)/D
            eta = (rA-rB)/D
            phi = np.arctan2(x,y)

            #wfn = Rfunc(xi) * Sfunc(eta) * Pfunc(phi)
            ### DEBUG
            wfn = Rfunc_num(xi) * Sfunc_num(eta) * Pfunc_num(phi)
            ###
            return wfn

        if self.plot == True:
            plt.cla()
            plt.ioff()
            plt.title("Angular wavefunction S")
            plt.xlabel("$\eta$")
            plt.ylabel("$S(\eta)$")
            plt.plot(eta_range, S)
            plt.show()
            plt.cla()
            plt.title("Radial wavefunction R")
            plt.xlabel("$\\xi$")
            plt.ylabel("$R(\\xi)$")
            plt.plot(xi_range, R)
            plt.show()

        #return en,(Rfunc, Sfunc, Pfunc), wavefunction
        ### DEBUG
        return en,(Rfunc_num, Sfunc_num, Pfunc_num), wavefunction
        ###
    
    def getContinuumOrbital(self, m,n, trig, E):
        """
        Parameters
        ----------
        m   :  number of angular nodes in P(phi),
               m=0 (sigma), m=1 (pi), m=2 (delta), m=3 (phi)
        n   :  number of angular nodes in S(eta)
        trig:  select azimuthal function P ('sin' or 'cos'),
               for m > 0, two solutions exist for P(phi), one is sin(m*phi)
               the other is cos(m*phi)
        E   :  energy of continuum orbital, 
               R(xi) has infinitely many nodes
        """
        D, Za, Zb = self.Lsep.R, self.Lsep.Za, self.Lsep.Zb
        a = D*(Zb+Za)
        b = D*(Zb-Za)

        c2 = E * D**2/2.0
        # determine separation constant L(c2)
        Lfunc = self.Lsep.L_interpolated(m,n)
        parity = pow(-1,n) #pow(-1,n-m)
        def findL(L0, c2):
            L_search_range = np.linspace(L0-1.5, L0+1.5, 10)        
            def fL(L):
                mismatch = angular_equation_S(m,L,a,b,c2,parity,count_nodes=False, plot=False)
                return mismatch
            Lroots = find_roots(fL, L_search_range)
            # count nodes
            Lnodes = []
            for L0 in Lroots:
                mismatch, nodes = angular_equation_S(m,L0,a,b,c2,parity, count_nodes=True)
                Lnodes.append( nodes )
            # select separation constant with the correct node count
            for L,nL in zip(Lroots, Lnodes):
                if nL == n:
                    break
            else:
                print "node counts = %s" % Lnodes
                raise Exception("No root with n=%d nodes found" % n)
            return L
        m,n,L0 = Lfunc(c2)
        L = findL(L0, c2)
        # compute wavefunctions
        eta_range, f = angular_equation_S(m,L,a,b,c2,parity,plot=False, Nx=400)
        S = f * pow(abs(1.0 - eta_range**2),m/2.0)
        Delta, xi_range, g = radial_equation_R_oscillating(m,L,a,b,c2,parity, plot=self.plot, Nx=80000)
        R = g * pow(xi_range**2-1.0,m/2.0)
        # normalize S
        # compute normalization constant
        deta = eta_range[1]-eta_range[0]
        nS2 = np.sum(S**2*deta)
        normS = np.sqrt(nS2)
        S /= normS
        # R is not square integrable
        # 
        Sfunc = interp1d(eta_range, S, kind='cubic',
                         fill_value="extrapolate")
        Rfunc = interp1d(xi_range, R, kind='cubic',
                         bounds_error=False, fill_value=0.0)

        # for m=0, only P(phi) = cos(0*phi)=1 is a valid, normalizable solution
        if m == 0:
            assert trig == 'cos'
        
        def Pfunc(phi):
            if trig == 'sin':
                return np.sin(m*phi)
            if trig == 'cos':
                return np.cos(m*phi)

        ### DEBUG
        from HMI import create_radial_function_R, create_angular_function_S, create_azimuthal_function_P

        Rfunc_num = create_radial_function_R(m,L,a,b,c2)
        Sfunc_num = create_angular_function_S(m,L,a,b,c2,parity,n=20+self.Lsep.nmax, normS=normS)
        Pfunc_num = create_azimuthal_function_P(m, trig)

        """
        plt.cla()
        plt.plot(xi_range, Rfunc(xi_range), label="R (interpolated)", ls="-", color="black")
        plt.plot(xi_range, Rfunc_num(xi_range), label="R (numerical)", ls="-.", color="blue")
        plt.legend()
        plt.show()

        plt.cla()
        plt.plot(eta_range, Sfunc(eta_range), label="S (interpolated)", ls="-", color="black")
        plt.plot(eta_range, Sfunc_num(eta_range), label="S (numerical)", ls="-.", color="blue")
        plt.legend()
        plt.show()
        """
        ###
        
        def wavefunction(grid, dV):
            x,y,z = grid
            # convert cartesian coordinates to spheroidal coordinates
            rA = np.sqrt(x**2 + y**2 + (z+D/2.0)**2)
            rB = np.sqrt(x**2 + y**2 + (z-D/2.0)**2)
            xi = (rA+rB)/D
            eta = (rA-rB)/D
            phi = np.arctan2(x,y)

            wfn = Rfunc(xi) * Sfunc(eta) * Pfunc(phi)
            ### DEBUG
            #wfn = Rfunc_num(xi) * Sfunc_num(eta) * Pfunc_num(phi)
            ###
            return wfn

        if self.plot == True:
            plt.cla()
            plt.ioff()
            plt.title("Angular wavefunction S")
            plt.xlabel("$\eta$")
            plt.ylabel("$S(\eta)$")
            plt.plot(eta_range, S)
            plt.show()
            plt.cla()
            plt.title("Radial wavefunction R")
            plt.xlabel("$\\xi$")
            plt.ylabel("$R(\\xi)$")
            plt.plot(xi_range, R)
            plt.show()
            
        return Delta, (Rfunc, Sfunc, Pfunc), wavefunction
        ### DEBUG
        #return Delta, (Rfunc_num, Sfunc_num, Pfunc_num), wavefunction
        ###
    def transition_dipoles(self, R1,S1,P1, R2,S2,P2):
        """
        For the transformation from prolate spheroidal coordinates to cartesian coordinates
        see "http://mathworld.wolfram.com/ProlateSpheroidalCoordinates.html"
        """
        # grids for integration and differentials
        eta = np.linspace(-1.0, 1.0, 1000)
        deta = eta[1]-eta[0]
        # for transition dipoles between two continuum orbitals the range of xi
        # might not be enough
        xi = np.linspace(1.0, 50.0, 10000)
        dxi = xi[1]-xi[0]
        phi = np.linspace(0.0, 2.0*np.pi, 1000)
        dphi = phi[1]-phi[0]
        D = self.Lsep.R
        # X-component
        tdipX = D**4/16.0 * np.sum(P1(phi)*P2(phi) * np.sin(phi) * dphi) \
            * (
                + np.sum(xi**2 * R1(xi)*R2(xi) * np.sqrt(xi**2-1.0) * dxi) \
                 *np.sum(S1(eta)*S2(eta) * np.sqrt(1-eta**2) * deta) \
                - np.sum(R1(xi)*R2(xi)*np.sqrt(xi**2-1) * dxi) \
                 *np.sum(eta**2*S1(eta)*S2(eta)*np.sqrt(1-eta**2) * deta) )
        # Y-component
        tdipY = D**4/16.0 * np.sum(P1(phi)*P2(phi) * np.cos(phi) * dphi) \
            * (
                + np.sum(xi**2 * R1(xi)*R2(xi) * np.sqrt(xi**2-1.0) * dxi) \
                 *np.sum(S1(eta)*S2(eta) * np.sqrt(1-eta**2) * deta) \
                - np.sum(R1(xi)*R2(xi)*np.sqrt(xi**2-1) * dxi) \
                 *np.sum(eta**2*S1(eta)*S2(eta)*np.sqrt(1-eta**2) * deta) )
        # Z-component
        tdipZ = D**4/16.0 * np.sum(P1(phi)*P2(phi) * dphi) \
            * (
                + np.sum(xi**3*R1(xi)*R2(xi) * dxi) \
                 *np.sum(eta*S1(eta)*S2(eta) * deta) \
                - np.sum(xi*R1(xi)*R2(xi) * dxi) \
                 *np.sum(eta**3*S1(eta)*S2(eta) * deta) )
        
        return np.array([tdipX,tdipY,tdipZ])
    
def test():
    R = 2.0
    Za = 1.0
    Zb = 1.0
    """
    nc2 = 200
    energy_range = np.linspace(-35.0, 20.0, nc2) * 2.0/R**2
    Lsep = SeparationConstants(R, Za, Zb)
    Lsep.tabulate_separation_constants(energy_range, nmax=10, mmax=3)
    Lsep.load_separation_constants()

    discrete_spectrum = DiscreteSpectrum(Lsep)
    #discrete_spectrum.tabulate_discrete_energies()
    #discrete_spectrum.load_discrete_energies()
    discrete_spectrum.binding_curve()
    #Lfunc = Lsep.L_interpolated(0,0)
    #xbound_eigenenergies(R, Za, Zb, energy_range, Lfunc)
    """

    wfn = DimerWavefunctions(R,Za,Zb)
    atomlist = [(1,(0,0,-R/2.0)), (1,(0,0,+R/2.0))]

    m,n,q = 0,0,0
    trig = 'cos'
    energy,(Rfunc1,Sfunc1,Pfunc1),wavefunction1 = wfn.getBoundOrbital(m,n,trig,q)
    Cube.function_to_cubefile(atomlist, wavefunction1, filename="/tmp/h2+_bound_orbital_%d_%d_%s_%d.cube" % (m,n,trig,q))

    """
    m,n = 0,1
    trig = 'cos'
    Es = np.linspace(0.01, 15.0, 2)
    phase_shifts = []
    transition_dipoles = []
    for E in Es:
        Delta, (Rfunc2,Sfunc2,Pfunc2),wavefunction2 = wfn.getContinuumOrbital(m,n,trig,E)
        phase_shifts.append(Delta)
        tdip = wfn.transition_dipoles(Rfunc1,Sfunc1,Pfunc1, Rfunc2,Sfunc2,Pfunc2)
        transition_dipoles.append(tdip)
    plt.cla()

    plt.xlabel("Energy / Hartree")
    plt.ylabel("Phase Shift")
    plt.plot(Es, phase_shifts)

    plt.cla()
    transition_dipoles = np.array(transition_dipoles)
    xyz2label = ["X", "Y", "Z"]
    for xyz in range(0, 3):
        plt.plot(Es, transition_dipoles[:,xyz], label="%s" % xyz2label[xyz])
    plt.legend()
    plt.ioff()
    plt.savefig("transition_dipoles.png")
    plt.show()
    """
    m,n,E = 0,1, 1.0/27.211
    trig = 'cos'
    Delta, (Rfunc2,Sfunc2,Pfunc2),wavefunction2 = wfn.getContinuumOrbital(m,n,trig,E)
    Cube.function_to_cubefile(atomlist, wavefunction2, filename="/tmp/h2+_continuum_orbital_%d_%d_%s.cube" % (m,n,str(E).replace(".", "p")), dbuff=15.0)

    plt.ioff()
    plt.show()
    
if __name__ == "__main__":
    R = 2.0
    Za = 1.0
    Zb = 1.0
    wfn = DimerWavefunctions(R,Za,Zb)
    atomlist = [(1,(0,0,-R/2.0)), (1,(0,0,+R/2.0))]
    """
    # compute the bound orbitals without any radial nor angular nodes
    m,n,q = 0,0,0
    trig = 'cos'
    energy,(Rfunc1,Sfunc1,Pfunc1),wavefunction1 = wfn.getBoundOrbital(m,n,trig,q)
    print "Energy: %s Hartree" % energy
    """
    # compute the continuum orbital with PKE=5 eV
    m,n,E = 0,1, 5.0/27.211
    trig = 'cos'
    Delta, (Rfunc2,Sfunc2,Pfunc2),wavefunction2 = wfn.getContinuumOrbital(m,n,trig,E)
    print "Phase shift: %s" % Delta

    Cube.function_to_cubefile(atomlist, wavefunction2, filename="/tmp/h2+_continuum_orbital_%d_%d_%s.cube" % (m,n,str(E).replace(".", "p")), dbuff=15.0, ppb=2.5)
    
    from DFTB.Scattering import PAD
    PAD.asymptotic_density(wavefunction2, 20.0, E)
    
    plt.show()

    # build LCAO of two atomic s-waves with PKE=5 eV
    from DFTB.Scattering.SlakoScattering import AtomicScatteringBasisSet
    bs = AtomicScatteringBasisSet(atomlist, E)
    lcao_continuum = np.array([
        +1.0, 0.0,0.0,0.0,  0.0,0.0,0.0,0.0,0.0,
        -1.0, 0.0,0.0,0.0,  0.0,0.0,0.0,0.0,0.0,
                      ])
    lcao_continuum /= la.norm(lcao_continuum)
    Cube.orbital2grid(atomlist, bs.bfs, lcao_continuum, \
                      filename="/tmp/h2+_lcao_continuum_orbital_%s.cube" % str(E).replace(".", "p"), dbuff=15.0, ppb=2.5)

