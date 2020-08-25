"""
implementation of Wigner-3j-symbols 
    (l1  l2  l3)
    (m1  m2  m3)
"""
import numpy as np

def fact(i):
    "Normal factorial"
    val = 1
    while (i>1):
        val = i*val
        i = i-1
    return float(val)

def invfact(i):
    """inverse of factorial, 1/i!"""
    if (i < 0):
        return 0.0
    else:
        return (1.0/fact(i))

def fact2(i):
    "Double factorial (!!) function = 1*3*5*...*i"
    val = 1
    while (i>0):
        val = i*val
        i = i-2
    return float(val)

def ksummation(j1, m1, j2, m2, J, M):
    kmax = min(j1+j2-J, j1-m1, j2+m2)
    ksum = 0.0
    for k in xrange(0, kmax+1):
        ksum += pow(-1.0,k) * invfact(k) * invfact(j1+j2-J-k)\
                  * invfact(j1-m1-k) * invfact(j2+m2-k)\
                  * invfact(J-j2+m1+k) * invfact(J-j1-m2+k)
    return(ksum)

def triangular_inequality(j1, m1, j2, m2, J, M):
    """check if coupling of moment is impossible or not"""
    if (    
        (j1 >= 0) and (j2 >= 0) and (J >= 0)
        and (j1+j2 >= J >= abs(j2-j1)) \
        and (j1 >= m1 >= -j1) \
        and (j2 >= m2 >= -j2) \
        and (J >= M >= -J) \
        and (M == m1+m2) ):
        return(1)
    else:
        return(0)

def CGC_no_triangular_check(j1, m1, j2, m2, J, M):
    rad1 = (2.0*J+1.0) \
             * fact(J+j1-j2) * fact(J+j2-j1) * fact(j1+j2-J) \
             / fact(j1+j2+J+1)
    rad2 = fact(J+M) * fact(J-M) \
            * fact(j1-m1) * fact(j1+m1) * fact(j2-m2) * fact(j2+m2)
    ksum = ksummation(j1, m1, j2, m2, J, M)
    cgc = np.sqrt(rad1*rad2)*ksum
    return(cgc)

def CGC(j1, m1, j2, m2, J, M):
    """compute Clebsch-Gordan coefficients for coupling angular momenta (j1,m1) and (j2,m2) to
    (J,M) using Wigner's explicit formula"""
    return( triangular_inequality(j1, m1, j2, m2, J, M) \
                * CGC_no_triangular_check(j1, m1, j2, m2, J, M) )

"""WARNING: Wigner3j symbols give infs for l larger than 50"""
def __Wigner3J(l1,m1,l2,m2,l3,m3):
    if l3 < 0:
        return 0
    return( pow(-1.0, l1-l2-m3)/np.sqrt(2.0*l3+1.0) * CGC(l1,m1,l2,m2,l3,-m3) )

from DFTB.Scattering.FunctionCache import FunctionCache
Wigner3J = FunctionCache(__Wigner3J)

class WignerD:
    """Wigner D-matrix"""
    def __init__(self, l, mp, m, conj=False):
        # conj is true if complex conjugate of Wigner D-matrix is meant
        self.fac = 1.0
        self.l = l
        self.mp = mp
        self.m = m
        if (conj == True):
            self.conjugate()
        if (abs(self.mp) > self.l) or (abs(self.m) > self.l):
            self.fac = 0.0
    def conjugate(self):
        # D^l_(mp,m)* = (-1)^(mp-m) D^l_(-mp,-m)
        self.fac *= pow(-1.0,self.mp-self.m)
        self.mp *= -1
        self.m *= -1
    def __hash__(self):
        return tuple((self.l, self.mp, self.m)).__hash__()
    def __str__(self):
        text = "WignerD(l=%s,m'=%s,m=%s)" % (self.l, self.mp, self.m)
        return text

def Kronecker(a,b):
    if a == b:
        return 1
    else:
        return 0

def __average_over_Euler_angles(Dlist):
    """compute the average of a product of several Wigner D-matrices over all Euler angles.
    For averages of products of 2 and 3 D-matrices formulae 4.6.1 and 4.6.2 in Edmonds
    "Angular Momentum in Quantum Mechanics" are employed. For 4 and more D-matrices,
    the product of two D-matrices is reduced to a sum over D-matrices according to
    equation 4.3.2 and then this function is called recursively on the product with one
    factor less."""
    n = len(Dlist)
    if n == 1:
        D1 = Dlist[0]
        if D1.l == 0 and D1.mp == 0 and D1.m == 0:
            return D1.fac
        else:
            return 0
    elif n == 2:
        D1, D2 = Dlist
        """In formula 4.6.1 the first D-matrix is complex conjugated, whereas
        all matrices in Dtuple are not conjugated, therefore the minus signs."""
        return D1.fac * D2.fac * pow(-1,D1.m-D1.mp) \
            * Kronecker(-D1.mp, D2.mp) \
            * Kronecker(-D1.m, D2.m) \
            * Kronecker(D1.l, D2.l) / (2.0*D1.l + 1.0)
    elif n == 3:
        D1, D2, D3 = Dlist
        return D1.fac * D2.fac * D3.fac \
            * Wigner3J(D1.l, D1.mp, D2.l, D2.mp, D3.l, D3.mp) * Wigner3J(D1.l, D1.m, D2.l, D2.m, D3.l, D3.m)
    else:
        D1, D2 = Dlist[0], Dlist[1]
        """convert the product of the first 2 D-matrices into a sum 
        over D-matrices multiplied by Wigner 3J-symbols"""
        Dsum = 0
        for j in xrange(abs(D1.l-D2.l), D1.l+D2.l+1):
            Dsum += (2.0*j+1.0)* D1.fac * D2.fac \
                * pow(-1, -D1.mp-D2.mp+D1.m+D2.m) \
                * Wigner3J(D1.l,D1.mp, D2.l,D2.mp, j, -D1.mp-D2.mp) \
                * Wigner3J(D1.l,D1.m , D2.l,D2.m , j, -D1.m -D2.m) \
                * __average_over_Euler_angles([WignerD(j, D1.mp+D2.mp, D1.m+D2.m)] + Dlist[2:])
        return Dsum

def bincoeff(n,k):
    if k>n or k<0 or n<0:
        return 0
    else:
        return fact(n)/(fact(k)*fact(n-k))

def __WignerSmallD(l,m1,m2,beta):
    """
    real d^(l)_(m1,m2)(beta) matrices
    calculated using formula (4.1.15) in Edmonds
    """
    a = np.sqrt(fact(l+m1)*fact(l-m1)/(fact(l+m2)*fact(l-m2)))
    acc = 0.0
    for s in range(max(-(m1+m2),0),l-m2+1):
        acc += bincoeff(l+m2,l-m1-s)*bincoeff(l-m2,s)*pow(-1,l-m1-s)\
              *pow(np.cos(beta/2),2*s+m1+m2)*pow(np.sin(beta/2),2*l-2*s-m1-m2)
    return a*acc

def __WignerSmallD_pi2(l,m1,m2):
    """
    evaluate d^(l)_(m1,m2)(pi/2)
    """
    a = np.sqrt(fact(l+m1)*fact(l+m2)*fact(l-m1)*fact(l-m2))/pow(2,l)*pow(-1,l-m1)
    acc = 0.0
    for s in range(max(-(m1+m2),0),l-m2+1):
        acc += pow(-1,s)*(invfact(l-m1-s)*invfact(l-m2-s)*invfact(m1+m2+s)*invfact(s))
    return a*acc

def WignerSmallD(l,m1,m2,beta):
    """
    real d^(l)_(m1,m2)(beta) matrices
    calculated from (4.5.2) in Edmonds
    """
    acc = 0.0+0.0j
    for mp in range(-l,l+1):
        acc += np.exp(1.0j*m1*np.pi/2.0) * __WignerSmallD(l,mp,m1,np.pi/2.0) \
                  * np.exp(-1.0j*mp*beta) \
             * __WignerSmallD(l,mp,m2,np.pi/2.0) * np.exp(-1.0j*m2*np.pi/2.0)
    return acc.real

average_over_Euler_angles = __average_over_Euler_angles
#average_over_Euler_angles = FunctionCache(__average_over_Euler_angles, \
#                                              commutative_args=True, unpack_args=True)
        
def assert_eql(x,y):
    if (abs(x-y) < 1.0e-10):
        print "%s == %s" % (x,y)
    else:
        raise Exception("%s != %s" % (x,y))

def test_Wigner3J():
    assert_eql( Wigner3J(2,2,1,-1,1,-1), 1.0/np.sqrt(5.0) )
    assert_eql( Wigner3J(2,0,1,0,1,0), np.sqrt(2.0/15.0) )
    assert_eql( Wigner3J(1,0,1,0,1,0), 0.0 )
    assert_eql( Wigner3J(2,0,1,1,1,-1), 1.0/np.sqrt(30.0) )
    assert_eql( Wigner3J(2,-1,1,0,1,1), -1.0/np.sqrt(10.0) )
    assert_eql( Wigner3J(1,-1,1,0,1,1), 1.0/np.sqrt(6.0) )
    assert_eql( Wigner3J(0,0,1,0,1,0), -1.0/np.sqrt(3.0) )
    assert_eql( Wigner3J(0,0,1,-1,1,1), 1.0/np.sqrt(3.0) )

def test_WignerD_average():
    assert_eql( __average_over_Euler_angles(\
            [WignerD(1,-1,-1), WignerD(1,+1,+1), WignerD(2,0,0), WignerD(0,0,0)]),\
                    1.0/30.0 )
    assert_eql( __average_over_Euler_angles(\
            [WignerD(7,-2,5), WignerD(2,-1,2), WignerD(3,1,0), WignerD(0,0,0)]),\
                    0.0 )
    assert_eql( __average_over_Euler_angles(\
            [WignerD(7,-2,2), WignerD(2,-2,2), WignerD(5,4,-4), WignerD(1,0,0)]),\
                    - 2.0/(3.0*3003.0) - 11.0/(3.0*2730.0) )

def test_WignerSmallD():
    theta = np.pi/np.sqrt(2.4345345)
    for l in range(0,5):
        for m1 in range(-l,l+1):
            for m2 in range(-l,l+1):
                assert_eql(__WignerSmallD(l,m1,m2,theta), WignerSmallD(l,m1,m2,theta))
                assert_eql(__WignerSmallD_pi2(l,m1,m2), WignerSmallD(l,m1,m2,np.pi/2.0))

if __name__ == "__main__":
    test_Wigner3J()
    test_WignerD_average()
    test_WignerSmallD()
    Wigner3J.statistics()
