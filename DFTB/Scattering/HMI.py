#!/usr/bin/env python
"""
exact wave functions for the hydrogen molecular ion (HMI, H2+)

References
==========
[1] L.Ponomarev, L.Somov, 
    "The Wave Functions of Continuum for the Two-Center Problem in Quantum Mechanics",
    J. Comput. Phys. 20, 183-195 (1976)

"""
import numpy as np
from scipy.integrate import ode
import sys

import os.path
# Tables with separation constants 
# are loaded from the directory of the current file .
hmi_directory = os.path.dirname(os.path.abspath(__file__))

################### RADIAL WAVEFUNCTION R(xi) ##################

def radial_equation_g_series_expansion(m,L,a,b,c2, xs, n=20):
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

    Optional
    --------
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

       p_i d_(i+1)  +  q_i d_i  +  r_i d_(i-1)  +  s_i d_(i-2) = 0

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

def radial_equation_g_numerical_integration(m,L,a,b,c2, xs, info={}):
    """
    solve radial equation for g(x) for x > 1 by numerical integration. 
    Depending on whether c2 < 0.0 (bound state) or c2 >= 0.0 (continuum state)
    different integration schemes are used:

    case c2 < 0.0:
      To avoid explosion of the wavefunction for large x, we integrated from 
      the inside to an intermediate point xmatch (giving fl(x)) and from the outside 
      to the same point (giving fr(x)) and scale the outer wavefunction such that both
      solutions match (fl(x) = fr(x)). If the 1st derivative also matches, the solution
      is a bound eigenfunction.

    case c2 > 0.0:
      In this case the wavefunction oscillates so that we can simply integrate from small
      to large x.

    Parameters
    ----------
    m      :  number of nodes in angular wavefunction S(eta)
    L      :  value of separation constant
    a      :  R*(Za+Zb) = 2*D, where D is the distance between the two protons
    b      :  R*(Za-Zb) = 0, has to be zero
    c2     :  related to energy E of solution, c^2 = 1/2 E D^2
    xs     :  1d numpy array with grid on which g(x) should be evaluated

    Optional
    --------
    info   :  dictionary, during the calculation `info` is filled with additional
              information such as the number of nodes (in info['num_nodes']),
              the derivative mismatch (in info['mismatch']) or the phase shift
              (in info['phase_shift']).

    Returns
    -------
    gs     :  1d numpy array with g(xs)
    gs_der :  1d numpy array with derivative g'(xs)

    """
    assert b == 0.0, "Code does not work for Za != Zb"

    # define differential equation 
    def g(x, y):
        # y1 = g(x),  y2 = g'(x)
        y1,y2 = y
        y1deriv = y2
        y2deriv = -(a*x + c2*x**2 + m*(m+1) - L)/(x**2-1.0) * y1 - (2*(m+1)*x)/(x**2-1.0) * y2
        return [y1deriv, y2deriv]

    def jac(x, y):
        # dg/dy
        return [[1.0, 0.0],
                [-(a*x + c2*x**2 + m*(m+1) - L)/(x**2-1.0), - (2*(m+1)*x)/(x**2-1.0)]]

    I = ode(g, jac)
    I.set_integrator('lsoda', with_jacobian=True,
                     atol=[1.0e-16, 1.0e-16], nsteps=100000000000)

    # flatten input array, original shape is restored at the end
    xs_flat = xs.flatten()
    
    # sort `xs_flat` in increasing order
    sort_index = np.argsort(xs_flat)
    xs_sorted = xs_flat[sort_index]

    # split xs into different regions:
    #  * region I contains the interval [1,x0small]:
    #    around x = 1, the jacobian becomes singular so that we have to use the series
    #    expansion
    #  * region II contains the interval [x0small, x0match]
    #  * region III contains the integrals [x0small, infinity]
    #    In region II and III the solution is obtained by numerical integraion.

    # The splitting points are rather arbitrary
    x0small = 1.05
    x0match = 5.0

    # region I
    xsI = xs_sorted[xs_sorted <= x0small]
    gsI, gsI_der = radial_equation_g_series_expansion(m,L,a,b,c2,xsI)

    # In region II and III the integration scheme depends on type of state we
    # need to calculated: a bound state which decays to 0 for x -> +infinity
    # or a continuum state, which oscillates for x -> +infinity
    if c2 < 0.0:
        ## bound state

        # region II
        xsII = xs_sorted[(x0small < xs_sorted) & (xs_sorted <= x0match)]
        # starting point for outward integration xl -> xmatch
        xl = x0small
        yl = radial_equation_g_series_expansion(m,L,a,b,c2,x0small)
        I.set_initial_value(yl,xl)
        # integrate outward
        gsII = np.zeros(xsII.shape)
        gsII_der = np.zeros(xsII.shape)
        for i in range(0, len(xsII)):
            I.integrate(xsII[i])
            # save values of g(xsII[i]) and g'(xsII[i])
            gsII[i] = I.y[0]
            gsII_der[i] = I.y[1]

        # integrate to matching point
        I.integrate(x0match)
        # g(xmatch), g'(xmatch) from the left
        yl_match = I.y

        # region III
        xsIII = xs_sorted[x0match < xs_sorted]
        # starting point for inward integration is where exp(-c*x) > machine precision
        c = np.sqrt(abs(c2))
        xr = -np.log(1.0e-14)/c

        yr = [np.exp(-c*xr), -c*np.exp(-c*xr)]
        I.set_initial_value(yr,xr)
        # integrate inward
        gsIII = np.zeros(xsIII.shape)
        gsIII_der = np.zeros(xsIII.shape)
        for i in range(len(xsIII)-1, 0-1,-1):
            if xsIII[i] > xr:
                # for very large values of x, g(x) is zero for negative energies
                continue
            I.integrate(xsIII[i])
            # save values of g(xsIII[i]) and g'(xsIII[i])
            gsIII[i] = I.y[0]
            gsIII_der[i] = I.y[1]

        # integrate to matching point
        I.integrate(x0match)
        # g(xmatch), g'(xmatch) from the right
        yr_match = I.y

        # scale right solution
        scale = yl_match[0] / yr_match[0]
        gsIII *= scale
        gsIII_der *= scale
        # mismatch of derivative at xmatch
        mismatch = yr_match[1] - yl_match[1]

        # combine solutions for regions I, II and III
        gs_sorted = np.zeros(xs_flat.shape)
        gs_sorted[xs_sorted <= x0small] = gsI
        gs_sorted[(x0small < xs_sorted) & (xs_sorted <= x0match)] = gsII
        gs_sorted[x0match < xs_sorted] = gsIII
        # do the same for derivatives
        gs_der_sorted = np.zeros(xs_flat.shape)
        gs_der_sorted[xs_sorted <= x0small] = gsI_der
        gs_der_sorted[(x0small < xs_sorted) & (xs_sorted <= x0match)] = gsII_der
        gs_der_sorted[x0match < xs_sorted] = gsIII_der

        # return additional information via the dictionary `info`
        info['mismatch'] = mismatch
        # phase shift is not defined
        info['phase_shift'] = None
        
    else:
        # unbound continuum state
        # Since there is no risk of the wavefunction blowing up at large xi,
        # we can treat region II and III as a single region
        xsIIandIII = xs_sorted[x0small < xs_sorted]
        # starting point for outward integration xl -> xmatch
        xl = x0small
        yl = radial_equation_g_series_expansion(m,L,a,b,c2,x0small)
        I.set_initial_value(yl,xl)
        # integrate outward
        gsIIandIII = np.zeros(xsIIandIII.shape)
        gsIIandIII_der = np.zeros(xsIIandIII.shape)

        for i in range(0, len(xsIIandIII)):
            I.integrate(xsIIandIII[i])
            # save values of g(xsIIandIII[i]) and g'(xsIIandIII[i])
            gsIIandIII[i] = I.y[0]
            gsIIandIII_der[i] = I.y[1]

        # To obtain the phase shift `shift` and the normalization factor `scale`
        # we need to match the continuum solution to the asymptotic oscillating solution
        # at a practical infinity `xinf`.
        c = np.sqrt(abs(c2))
        wavelength = 2*np.pi/c
        # `xinf` is chosen to be at least 100 wavelenghts or 10000.0
        xinf = max(10*wavelength, 10000.0)
        # or at least the largest xi-value encountered in the input array
        xinf = max(xinf, gsIIandIII[-1])
        if xinf > gsIIandIII[-1]:
            # integrate to `xinf` if we are not there already
            I.integrate(xinf)
        ginf = I.y[0]       # g(xinf)
        ginf_der = I.y[1]   # g'(xinf)
        # compute phase shift
        shift = np.arctan2(-ginf_der, (c+a/(2*c*xinf))*ginf) - c*xinf - a/(2*c)*np.log(2*c*xinf)
        # phase shifts are only defined modulo 2 pi
        shift = shift % (2.0*np.pi)
        # scale constant from approximate asymptotic solution with the correct
        # normalization
        ginf_norm = np.cos(c * xinf + a/(2*c) * np.log(2*c*xinf) + shift) * pow(xinf,-(m+1))
        scale = ginf_norm / ginf
            
        # combine solutions for regions I and II & III
        gs_sorted = np.zeros(xs_flat.shape)
        gs_sorted[xs_sorted <= x0small] = gsI
        gs_sorted[x0small < xs_sorted]  = gsIIandIII
        # do the same for derivatives
        gs_der_sorted = np.zeros(xs_flat.shape)
        gs_der_sorted[xs_sorted <= x0small] = gsI_der
        gs_der_sorted[x0small < xs_sorted]  = gsIIandIII_der

        # apply scale factor
        gs_sorted *= scale
        gs_der_sorted *= scale
        
        # return additional information via the dictionary `info`
        # at positive energy, it's always possible to match the inner and outer solution
        info['mismatch'] = 0.0      
        info['phase_shift'] = shift
        
    # 'unsort' gs, bring values g(xi) in the same order as xi in xs
    gs = np.zeros(xs_flat.shape)
    gs[sort_index] = gs_sorted
    # same for derivatives
    gs_der = np.zeros(xs_flat.shape)
    gs_der[sort_index] = gs_der_sorted

    # count number of nodes
    # When g(x) goes through zero at x0, we have g(x0-eps) * g(x0+eps) < 0.0,
    # so we need to count the number of sign changes
    info['radial_num_nodes'] = np.count_nonzero( gs[1:] * gs[:-1] < 0.0 )

    # bring output array into the same shape as the input array
    gs = np.reshape(gs, xs.shape)
    gs_der = np.reshape(gs_der, xs.shape)
    
    return gs, gs_der
    
def create_radial_function_R(m,L,a,b,c2, norm2=1.0, info={}):
    """
    generate a radial function R(xi) for specific parameters m,L,...

    Parameters
    ----------
    see `radial_equation_f_numerical_integration()`

    Optional
    --------
    norm2 :  norm of radial wavefunction, used for normalization,
             R -> R/sqrt(norm2)
    info  :  dictionary, during the calculation inside Rfunc()  `info` is filled 
             with additional information such as the number of nodes 
             (in info['radial_num_nodes']) or the derivative mismatch 
             (in info['mismatch']).

    Returns
    -------
    Rfunc :   callable Rfunc(xi) for evaluating the radial function on a numpy grid
    """
    def Rfunc(xs):
        gs, gs_der = radial_equation_g_numerical_integration(m,L,a,b,c2, xs, info=info)
        # R(xs)
        R = gs * (xs**2 - 1.0)**(0.5*m)
        # normalize R if optional normalization constant is provided
        R /= np.sqrt(norm2)

        return R

    return Rfunc


################ ANGULAR WAVEFUNCTION S(eta) #########################
    
def angular_equation_f_series_expansion(m,L,a,b,c2, sigma, parity, etas, n=20):
    """
    solve angular equation for f(eta) around eta=sigma (where sigma=+1 or -1) by a
    series expansion of length n.

    Parameters
    ----------
    m      :  number of nodes in angular wavefunction S(eta)
    L      :  value of separation constant
    a      :  R*(Za+Zb) = 2*D, where D is the distance between the two protons
    b      :  R*(Za-Zb) = 0, has to be zero
    c2     :  related to energy E of solution, c^2 = 1/2 E D^2
    etas   :  1d numpy array with grid on which f(eta) should be evaluated, 
              the points should lie close to eta=sigma for the solution to be valid
    sigma  :  point around which solution is expanded, +1 or -1
    parity :  initial value, f(sigma) = parity, +1 or -1

    Optional
    --------
    n      :  degree of polynomial in expansion

    Returns
    -------
    fs     :  1d numpy array with f(etas)
    fs_der :  1d numpy array with derivative f'(etas)

    The differential equation to be solve is:

       (1-eta^2) f''(eta) - 2*(m+1) eta f'(eta) - [m*(m+1) - L + c^2*eta^2] f(eta) = 0

    The angular wavefunction S(eta) is related to f(eta) by:

       S(eta) = f(eta) * (1-eta^2)^(m/2)

    The ansatz for f(eta) around eta=sigma is given by

       f(eta) = sum_i^n d_i (1-sigma*eta)^i

    Substituting this ansatz into the differential equation and equating coefficients
    leads again to a four term recurrence relation for the d's:
    
       p_i d_(i+1)  +  q_i d_i  +  r_i d_(i-1)  +  s_i d_(i-2) = 0

    with the coefficients

       p_i = 2*(i+1)*(i+m+1)
       q_i = -c^2 + L - m*(m+1) - i*(2*m+i+1)
       r_i = 2*c^2
       s_i = -c^2

    and the initial conditions
       
       d_(-2) = 0 ,  d_(-1) = 0,  d_0 = sigma      
    """
    assert b == 0.0, "Code does not work for Za != Zb"
    # compute coefficients of 4 term recurrence
    p = np.zeros(n+1)
    q = np.zeros(n+1)
    r = np.zeros(n+1)
    s = np.zeros(n+1)
    for i in range(0, n+1):
        p[i] = 2*(i+1)*(i+m+1)
        q[i] = -c2+L-m*(m+1) - i*(2*m+i+1)
        r[i] = 2*c2
        s[i] = -c2
    # initialize recurrence relation
    d = np.zeros(n+1+2)
    d[-2] = 0.0
    d[-1] = 0.0
    d[0]  = parity
    # iterate recurrence relation
    for i in range(0,n):
        d[i+1] = (-q[i]*d[i]-r[i]*d[i-1]-s[i]*d[i-2])/p[i]
        
    # remove entries for d_(-2) and d_(-1)
    d = d[0:-2]
        
    # evaluate the polynomial
    # f(eta) = sum_i=0^n d_i (1-sigma*eta)^i
    omes = 1.0-sigma*etas
    # poly1d needs the coefficients in reverse order
    poly = np.poly1d(d[::-1])
    # derivative
    poly_der = np.polyder(poly, m=1)
    # evaluate polynomial
    fs = np.polyval(poly, omes)
    # evaluate derivative of polynomial
    # The minus sign comes from the chain rule
    #  d/d(x) = -sigma* d/d(1-sigma*x)
    fs_der = -sigma*np.polyval(poly_der, omes)
    
    return fs, fs_der

def create_angular_function_S(m,L,a,b,c2,parity, info={}, n=20, normS=1.0):
    """
    generate an angular function S(eta) for specific parameters m,L,...

    Parameters
    ----------
    see `angular_equation_f_series_expansion()`

    Optional
    --------
    info   :  dictionary, during the calculation inside Sfunc()  `info` is filled 
             with additional information such as the number of nodes 
             (in info['angular_num_nodes']) or the derivative mismatch (in info['mismatch']).
    n      :  degree of polynomial in series expansion
    normS  :  normalization constant, S --> S/normS

    Returns
    -------
    Sfunc :   callable Sfunc(eta) for evaluating the angular function on a numpy grid
    """
    def Sfunc(etas):
        # series expansion around eta=-1
        etas_left = etas[etas <= 0.0]
        fs_left, fs_der_left   = angular_equation_f_series_expansion(m,L,a,b,c2,-1, +1,
                                                                     etas_left, n=n)
        # ... and around eta = +1
        etas_right  = etas[etas > 0.0]        
        fs_right, fs_der_right = angular_equation_f_series_expansion(m,L,a,b,c2,+1, parity,
                                                                     etas_right, n=n)
        
        # combine both expansions of f(eta)
        fs = np.zeros(etas.shape)
        fs[etas <= 0.0] = fs_left
        fs[etas > 0.0 ] = fs_right
        # S(eta)
        S = fs * ( 1.0 - etas**2 )**(0.5*m)
        # normalize S if normalization constant is provided
        S /= normS

        # additional information:
        # compute mismatch of derivatives at eta = 0
        f0_left, f0_der_left = angular_equation_f_series_expansion(m,L,a,b,c2,-1, +1,
                                                                     0.0, n=n)
        f0_right, f0_der_right = angular_equation_f_series_expansion(m,L,a,b,c2,+1, parity,
                                                                     0.0, n=n)

        #print "f'(->0)= %s  f'(0<-)= %s" % (f0_der_left, f0_der_right)
        info['mismatch'] = f0_der_right - f0_der_left
        # count number of nodes
        # When f(x) goes through zero at eta0, we have f(x0-eps) * f(x0+eps) < 0.0,
        # so we need to count the number of sign changes
        info['angular_num_nodes'] = np.count_nonzero( fs[1:] * fs[:-1] < 0.0 )

        return S

    return Sfunc

def zero_mismatch_f(m,L,a,b,c2, n=20):
    """
    compute the mismatch of f(eta) and its first derivative f'(eta)
    at eta=0:

       mismatch  = f(eta=0,sigma=+1) - f(eta=0,sigma=-1)
 
                 = 2 sum d 
                      i   i

       mismatch_deriv = f'(eta=0,sigma=+1) - f'(eta=0,sigma=-1)
                
                      = 2 sum i d  
                           i     i

    The coefficients d_i are given by a four-term recurrence

    Parameters
    ----------
    m      :  number of nodes in angular wavefunction S(eta)
    L      :  value of separation constant
    a      :  R*(Za+Zb) = 2*D, where D is the distance between the two protons
    b      :  R*(Za-Zb) = 0, has to be zero
    c2     :  related to energy E of solution, c^2 = 1/2 E D^2

    Optional
    --------
    n      :  degree of polynomial in expansion

    Returns
    -------
    mismatch, mismatch_deriv : floats, jump in f and f' at eta=0
    """
    assert b == 0.0, "Code does not work for Za != Zb"
    # compute coefficients of 4 term recurrence
    p = np.zeros(n+1)
    q = np.zeros(n+1)
    r = np.zeros(n+1)
    s = np.zeros(n+1)
    for i in range(0, n+1):
        p[i] = 2*(i+1)*(i+m+1)
        q[i] = -c2+L-m*(m+1) - i*(2*m+i+1)
        r[i] = 2*c2
        s[i] = -c2
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

    # evaluate mismatch of f at eta=0
    mismatch = np.sum(d)

    # evaluate jump in derivative f' at eta=0
    mismatch_deriv = 0.0
    for i in range(0, n):
        mismatch_deriv += i*d[i]

    
    return mismatch, mismatch_deriv

############## AZIMUTHAL WAVEFUNCTION ##################

def create_azimuthal_function_P(m,trig):
    """
    Parameters
    ----------
    m          : integer, number of nodes in P(phi), 0 < phi < 2*pi
    trig       : 'sin' or 'cos',
                 for m > 0, two solutions are possible, sin(m*phi) and cos(m*phi)
    """
    # consistency checks
    assert trig in ['sin', 'cos']
    if trig == "sin":
        assert m != 0, "m = 0 and P(phi) = sin(m*phi) leads to a wavefunction that is zero everywhere!"
        
    def Pfunc(phi):
        if trig == "sin":
            return np.sin(m*phi)
        else:
            return np.cos(m*phi)

    return Pfunc


############## COMBINED WAVEFUNCTION R*S*P ##############

def create_wavefunction(m,L,a,b,D, c2, parity, trig, norm2=1.0, info={}):
    """
    create a wavefunction of the hydrogen molecular ion (HMI) by solving

              __2     Z1      Z2
       ( -1/2 \/   - ----  - ----  -  E ) psi(x,y,z) = 0
                      r1      r2

    for an electron in the field of two nuclei with charges Z1, Z2 separated
    by the distance `D`. 

    Note that the parameters depend on each other, so `L`,`a`,`b`,`D` and `c2`
    cannot be chosen independently of each other. 
    For the definition of the parameters see Ref. [1].
    
    Parameters
    ==========
    m          :  int, azimuthal quantum number
    L          :  separation constant L(D,c^2)
    a          :  D*(Z1+Z2), should be 2*D, since the algorithm expects Za==Zb
    b          :  D*(Z1-Z2), should be 0
    D          :  separation between ions (in bohr)
    c2         :  c^2 = 1/2*E*D^2 where E is the energy of the wavefunction
    parity     :  parity of the solution
    trig       :  'cos' or 'sin', for m == 0, only 'cos' is allowed

    Optional
    ========
    norm2  :  norm of radial wavefunction, used for normalization,
              R -> R/sqrt(norm2)
    info   :  dictionary, during the calculation `info` is filled with additional
              information such as the number of nodes (in info['num_nodes']),
              the derivative mismatch (in info['mismatch']) or the phase shift
              (in info['phase_shift']).

    Returns
    =======
    wfn   :  callable, wfn(x,y,z) evaluates the wavefunction on any grid
    """
    # radial part R(xi)
    Rfunc = create_radial_function_R(m,L,a,b,c2, norm2=norm2, info=info)
    # angular part S(eta)
    Sfunc = create_angular_function_S(m,L,a,b,c2, parity, info=info)
    # azimuthal part P(phi)
    Pfunc = create_azimuthal_function_P(m,trig)

    def wavefunction(x,y,z):
        # convert cartesian coordinates to spheroidal coordinates
        rA = np.sqrt(x**2 + y**2 + (z+D/2.0)**2)
        rB = np.sqrt(x**2 + y**2 + (z-D/2.0)**2)
        xi = (rA+rB)/D
        eta = (rA-rB)/D
        phi = np.arctan2(x,y)
        
        wfn = Rfunc(xi) * Sfunc(eta) * Pfunc(phi)
        return wfn

    return wavefunction


############### SEPARATION CONSTANTS L(c^2) ###############

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

def find_roots(f, x_search_range, eps_conv=1.0e-12):
    """
    finds the roots of f(x) by bisection. 

    Parameters:
    -----------
    f              : scalar function whose roots should be determined
    x_search_range : array with interval boundaries. 
                     Each interval may contain at most one root
    eps_conv       : Iteration stops if  |f(x0)| < esp_conv

    Returns:
    --------
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
                    break
                if abs(a-b) < eps_conv:
                    #print "No root despite sign change???"
                    x0 = (a+b)/2.0
                    if abs(fm) < 1.0e-3:
                        roots.append( x0 )
                    break
        x_last = x2
        f_last = f2

    return roots


class SeparationConstants:
    """
    The separation constant L(c^2) is tabulated for different values of 
    
     0 <= n <= nmax           number of roots in angular function S(eta)
     0 <= m <= mmax           number of roots in azimuthal function P(phi)
    
    The function L(c^2) is expressed as a polynomial

                              k
      L(c^2) = sum    a  (c^2)
                  k    k
 
    Only the coefficients a_k are written to the file `separation_constants.dat`, 
    one column of coefficients for each tuple (n,m).
    """               
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

        nc2 = len(c2s)
        self.deg = nc2 # degree of fitting polynomial

        Ls = np.zeros((nc2,nmax+1,mmax+1))
        for m in range(0, mmax+1):
            print "*** M = %d ***" % m
            # find separation constant L(c^2) for each c^2
            for i,c2 in enumerate(c2s):
                print " c2 = 1/2 R^2 * energy = %s" % c2

                # values of separation constants for which
                # S(eta) has a continuous derivative at eta=0
                Lroots = []
                # number of nodes in continuous S(eta)
                Lnodes = []

                # Since L(c^2=0) = (m+n)*(m+n+1),
                # the minimal value of L(c2=0) is assumed
                # when n=0 and the maximum one when n=nmax.
                # Also L(c^2) is apparently a monotonically increasing function
                # of c^2 and we assume that
                #     L(c^2=cmin^2) < L(c^2=0) - |cmin^2|
                #     L(c^2=cmax^2) < L(c^2=0) + cmax^2
                Lmin = m*(m+1) - abs(c2s.min())
                Lmax = (m+nmax)*(m+nmax+1) + c2s.max()
                nL = 50 * (nmax+1)
                L_search_range = np.linspace(Lmin, Lmax, nL)
                print "search range for L : %d equidistant points in [%s, %s]" % (nL, Lmin, Lmax)
                
                # Since the number of nodes `n` is not known in advance,
                # we have to consider both parities p=(-1)^(m+n)
                
                for parity in [+1,-1]:
                    print " parity= %d" % parity
                    def mismatch_func(L):
                        # At the correct L the function f(eta) should be continuous
                        # and continuously differentiable everywhere.
                        # 1) continuity of f
                        #        mismatch  = f(eta=0,sigma=+1) - f(eta=0,sigma=-1)
                        # 2) contunuity of f'
                        #   mismatch_deriv = f'(eta=0,sigma=+1) - f'(eta=0,sigma=-1)
                        mismatch, mismatch_deriv = zero_mismatch_f(m,L,a,b,c2,n=20+nmax)

                        # If parity==-1, the left and right derivatives f' are equal at eta=0
                        # simply because of symmetry. On the otherhand if parity==+1, the
                        # left and right values of f are equal at eta=0 because of symmetry.
                        # Therefore, we need to choose a different criterion for the mismatch
                        # depending on the parity.
                        if parity == -1:
                            return mismatch
                        else:
                            return mismatch_deriv
                        
                    # search range where we are looking for roots
                    # of f(L0) = 0 for L0 in [Lmin,Lmax]
                                        
                    LrootsP = find_roots(mismatch_func, L_search_range)
                    print "Found %s roots" % len(LrootsP)
                    
                    # count nodes in S(eta)
                    # `etas` grid has to be fine enough so that we do not miss any node
                    etas = np.linspace(-1.0,1.0, 1000+100*nmax)
                    LnodesP = []
                    for L0 in LrootsP:
                        # `info` dictionary is used to return additional information after calling `S`
                        info = {}
                        S = create_angular_function_S(m,L0,a,b,c2,parity,info=info, n=20+nmax)
                        # evaluate S(eta) 
                        S(etas)
                        nodes = info['angular_num_nodes']
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
            # save coefficients of polynomial to file
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


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from hydrogen_molecular_ion import SeparationConstants
    
    R = 2.0
    Za = 1
    Zb = 1

    a = R*(Za+Zb)
    b = R*(Zb-Za)
    
    # ground state
    m,n = 0,0
    E = -1.1026353820

    c2_gs = 0.5*E*R**2

    # separation constant 
    Lsep = SeparationConstants(R, Za, Zb)
    Lsep.load_separation_constants()

    Lfunc = Lsep.L_interpolated(m,n)
    mL,nL,L = Lfunc(c2_gs)

    info = {}
    
    for c2 in np.linspace(c2_gs-0.1, c2_gs+5.0, 20):
        Rfunc = create_radial_function_R(m,L,a,b,c2, norm2=1.0, info=info)
        
        xs = np.linspace(1.0, 10.0, 1000)
        plt.plot(xs, Rfunc(xs), color="black")
        print "c2= %s E= %s  mismatch = %s  phase shift = %s" % (c2, 2*c2/R**2, info['mismatch'], info['phase_shift'], info['radial_num_nodes'])

    #plt.show()
    
    """
    plt.cla()
    Sfunc = create_angular_function_S(m,L,a,b,c2_gs, (-1)**(m+n), info=info)
    etas = np.linspace(-1.0, 1.0, 1000)
    plt.plot(etas, Sfunc(etas), color="black")
    print "mismatch= %s   nr.of nodes= %s" % (info['mismatch'], info['angular_num_nodes'])
    plt.show()
    """
