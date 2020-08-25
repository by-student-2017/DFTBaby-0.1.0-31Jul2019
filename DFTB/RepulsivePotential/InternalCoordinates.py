# NOT VERY USEFUL!
"""
Find internal coordinates such that the translational and rotational degrees of freedom
are separated from the other degrees of freedom. The 3N-6 remaining degrees of freedom
are represented by relative distances between the atoms and angles.
"""
from numpy import zeros, dot, array, cross, arctan2, arccos, identity, cos, sin, sqrt, pi, hstack, reshape
from numpy.linalg import norm, inv
from DFTB import XYZ
from DFTB.AtomicData import atom_names, hartree_to_eV, bohr_to_angs

def tripod2EulerAngles(xaxis,yaxis,zaxis):
    """
    Find the Euler angles a,b,g that would rotate the right-handed orthonormal
    set of axes x,y,z into the unrotated set of axes [1,0,0],[0,1,0],[0,0,1].

    see section "Geometric derivation" in https://en.wikipedia.org/wiki/Euler_angles

    Parameters:
    ===========
    xaxis,yaxis,zaxis: 3 vectors specifying the axes
    
    Returns:
    ========
    a,b,g: Euler angles such that R(a,b,g)*[1,0,0] = x and similarly for the y- and z-axis.
    """
    assert abs(dot(xaxis,yaxis)) < 1.0e-10
    assert abs(dot(yaxis,zaxis)) < 1.0e-10
    assert abs(dot(xaxis,zaxis)) < 1.0e-10
    assert abs(norm(xaxis) - 1.0) < 1.0e-10
    assert abs(norm(yaxis) - 1.0) < 1.0e-10
    assert abs(norm(zaxis) - 1.0) < 1.0e-10

    b = arccos(zaxis[2])
    # nodal line
    N = cross(array([0,0,1]), zaxis)
    N /= norm(N)
    a = arctan2(N[1],N[0])-pi/2.0 # projection of nodal line on space fixed x- and y-axis
    g = arctan2(dot(xaxis,N), dot(yaxis,N))

    return (a,b,g)

def EulerAngles2Rotation(a,b,g):
    """
    construct the rotation matrix around the axes x,y,z that are fixed in space:
        Rz(a)*Ry(b)*Rz(g)

    Parameters:
    ===========
    a,b,g: Euler angles

    Returns:
    ========
    3x3 rotation matrix
    """
    def Rz(phi):
        """clock-wise rotation by angle phi around the z-axis"""
        R = identity(3)
        R[0,0] = cos(phi)
        R[0,1] = sin(phi)
        R[1,0] = -sin(phi)
        R[1,1] = cos(phi)
        return R
    def Ry(phi):
        """clock-wise rotation by angle phi around the y-axis"""
        R = identity(3)
        R[0,0] = cos(phi)
        R[0,2] = -sin(phi)
        R[2,0] = sin(phi)
        R[2,2] = cos(phi)
        return R
    return dot(Rz(-a),dot(Ry(-b),Rz(-g)))
    
def construct_tripod(rA,rB,rC):
    """
    construct a right-handed orthonormal coordinate system from three
    position vectors. 

    The first axis lies along the vector joining rA and rB. 
    The second axis is perpendicular the the first axis
    and spans the plane defined by the three points. The third axis forms
    a right hand tripod with the first two axes.

    Parameters:
    ===========
    rA, rB, rC:  tuple of 3 3D numpy arrays

    Returns:
    ========
    n, a, b: tuple of 3 3D numpy arrays, normalized axes
    """
    # construct tripod n,a,b
    N = rB-rA
    n = N/norm(N)
    A = rC-rB - dot(rC-rB,n)*n
    a = A/norm(A)
    b = cross(n,a)
    return n,a,b
    
def cartesian2dihedral4pts(tup):
    """
    Given 4 cartesian position vectors rA, rB, rC, rD  describe the position
    rD by the distances of point D to points A and B and by the angle
    between D and the normal to the plane spanned by rA,rB,rC.

    Parameters:
    ===========
    (rA, rB, rC, rD): tuple of 4 cartesian 3D vectors
    
    Returns:
    ========
    (rA, rB, rC, (AD, BD, angle))
    where rA, rB and rC are the same as in the input
    and AD and BD are the distances between D and A and B.
    """
    rA, rB, rC, rD = tup
    AB = norm(rB-rA)
    AD = norm(rD-rA)
    BD = norm(rD-rB)
    n,a,b = construct_tripod(rA,rB,rC)
    """
    - n is the unit vector that connects A and B
    - a,b are orthogonal unit vectors that span the plane
    orthogonal to n.
    """
    x = 1.0/(2.0*AB)*(AD**2 - BD**2 - AB**2)
    AX = AB + x
    rX = rA + AX*n
    """
    rX is the position where the vector (rB-rA) intersects the plane 
    orthogonal to n that contains the point D. 
    """
    vDX = rD-rX
    dx = vDX/norm(vDX)
    """
    dx is the unit vector pointing from X to D
    """
    dihedral = arctan2(dot(dx,a), dot(dx,b))
    """
    Since D lies in the plane spanned by a and b, its position can
    be writtten as 
       rD = rX + sin(dihedral)*a + cos(dihedral)*b
    The dihedral angle (which differs from the usual definition) can be obtained
    as the tangent of the coordinates of dx = (rD-rX) onto relative to the axes a,b
    """
    internal_D_coordinates = array([AD,BD,dihedral])
    return (rA,rB,rC,internal_D_coordinates)

def dihedral2cartesian4pts(tup):
    """
    undo the coordinate transformation performed by 'cartesian2dihedral4pts'

    Parameters:
    ===========
    (rA,rB,rC,(AD,BD,angle))
    where rA, rB, rC are the coordiantes of the three reference points
    and AD = |rD-rA|, BD = |rD-rB| and angle is the angle between D and
    the normal of the plane plane spanned by rA, rB, rC.

    Returns:
    ========
    cartesian coordinates of all four points
    (rA,rB,rC,rD)
    """
    rA, rB, rC, internal_D_coordinates = tup
    AD,BD,dihedral = internal_D_coordinates
    n,a,b = construct_tripod(rA,rB,rC)
    AB = norm(rB-rA)
    x = 1.0/(2.0*AB)*(AD**2 - BD**2 - AB**2)
    AX = AB + x
    rX = rA + AX*n
    """
    print "abs(AX) = %s" % abs(AX)
    print "rA = %s" % rA
    print "rB = %s" % rB
    print "AB = %s" % AB
    print "x = %s" % x
    print "AD = %s" % AD
    """
    assert AD >= abs(AX)
    XD = sqrt(pow(AD,2) - pow(AX,2))
    rD = rX + XD*(sin(dihedral)*a + cos(dihedral)*b)
    return (rA,rB,rC,rD)

def cartesian2internal(cartesian_coordinates):
    """
    transform list of cartesian atom positions into internal coordinates
    where each atom location is specified in terms of its distance to 
    two other atoms and one angle.

    Parameters:
    ===========
    cartesian_coordinates: list of 3D numpy arrays with positions of N atoms
        [array(x1,y1,z1), array(x2,y2,z2), ..., array(xN,yN,zN)]

    Returns:
    ========
    internal coordinates
    """
    pos = cartesian_coordinates
    if len(pos) < 3:
        raise Exception("Diatomic molecules have only 1 internal degree of freedom.")
    # check that three consecutive atoms do not lie
    # on the same line as then the dihedral angle is
    # not well defined
    for i in range(2, len(pos)):
        rAB = pos[i-1] - pos[i-2]
        rBC = pos[i] - pos[i-1]
        if abs(abs(dot(rAB,rBC)/(norm(rAB)*norm(rBC)))-1.0) < 1.0e-10:
            raise Exception("atoms %s,%s and %s lie on a straight line. => No dihedral angle can be defined." % (i-2,i-1,i))
    # find internal coordinates of atoms N-1,N-2,...,3
    for i in range(len(pos)-1,2,-1):
        pos[i-3], pos[i-2], pos[i-1], pos[i] = \
            cartesian2dihedral4pts((pos[i-3], pos[i-2], pos[i-1], pos[i]))
    # internal coordinates of atoms 2,1,0
    z,x,y = construct_tripod(pos[0], pos[1], pos[2])
    a,b,g = tripod2EulerAngles(x,y,z)
    r10 = pos[1]-pos[0]
    d01 = norm(r10)
    r20 = pos[2]-pos[0]
    d02 = norm(r20)
    alpha = arctan2(dot(r20,x), dot(r20,z))
    # first 9 internal coordinates are: origin (pos[0] = origin), Euler angles, binding angle
    # between first three atoms and distances between first three atoms
    pos[1] = array([a,b,g])
    pos[2] = array([alpha,d01,d02])
    return pos

def internal2cartesian(internal_coordinates):
    """
    transform internal coordinates back to cartesian coordinates
    """
    ipos = internal_coordinates
    # find cartesian coordinates of atoms 0,1,2
    origin = ipos[0]
    a,b,g = ipos[1]
    # alpha is the angle at atom0 between the vector atom0 -> atom1 and atom0 -> atom2
    # d01 is the distance between atom0 and atom1
    # d02 is the distance between atom0 and atom2
    alpha, d01,d02 = ipos[2]
    Rot = EulerAngles2Rotation(a,b,g)
    # positions of the first 3 atoms in the unrotated, unshifted frame
    pos0 = array([0,0,0])
    pos1 = array([0,0,d01])
    pos2 = d02*array([sin(alpha), 0.0, cos(alpha)])
    # translate origin of molecule and apply rotation
    ipos[0] = dot(Rot,pos0) + origin
    ipos[1] = dot(Rot,pos1) + origin
    ipos[2] = dot(Rot,pos2) + origin
    # transform internal coordinates of atoms 3,4,...,N-1
    # to cartesian coordinates
    for i in range(3,len(ipos)):
        ipos[i-3], ipos[i-2], ipos[i-1], ipos[i] = dihedral2cartesian4pts((ipos[i-3], ipos[i-2], ipos[i-1], ipos[i]))
    return ipos

# convert back and forth between list of atomic positions
# and one vector with all positions in a row
def _lst2vec(pos):
    return hstack(pos).copy()
def _vec2lst(vec):
    return list(reshape(vec, (len(vec)/3, 3)).copy())
    

class InternalCoordinates:
    def __init__(self, atomlist):
        self.atomlist = atomlist
        # just check that transformation actually works
        pos = [array(posi) for (Zi,posi) in self.atomlist]
        ipos = self.getGeneralizedCoordinates()
        print "Convert back to cartesian coordinates"
        rpos = internal2cartesian(ipos)
        print "Recovered cartesian coordinates:"
        for p in rpos:
            print p
        assert sum([sum(abs(rposi - posi)) for rposi,posi in zip(rpos,pos)]) < 1.0e-10
    def getGeneralizedCoordinates(self):
        pos = [array(posi) for (Zi,posi) in self.atomlist]
        print "cartesian coordinates:"
        for p in pos:
            print p
        # internal coordinates + 3 translational + 3 rotational degrees of freedom
        ipos = cartesian2internal(pos)
        print "internal coordinates:"
        for p in ipos:
            print p
        return ipos
    def getJacobian(self, h=1.0e-10):
        """
        numerically find Jacobian matrix for transformation from internal to cartesian coordinates

        J_ij = d( x_i )/d( chi_j )

        where x_1,x_2,...,x_3N are the cartesian coordinates of N atoms
        and chi_1,chi_2,...,chi_3N are the internal coordinates

        Parameters:
        ===========
        h: step for difference quotient

        Returns:
        ========
        3Nx3N numpy array with J_ij
        """
        pos = [array(posi) for (Zi,posi) in self.atomlist]
        x = _lst2vec(pos)
        ipos = cartesian2internal(pos)
        Jacobian = zeros((3*len(pos),3*len(pos)))
        for j in range(0, 3*len(pos)): # iterate over atoms
            # symmetric difference quotient
            chiplus = _lst2vec(ipos)
            chiplus[j] += h
            chiminus = _lst2vec(ipos)
            chiminus[j] -= h

            xplus = _lst2vec(internal2cartesian(_vec2lst(chiplus)))
            xminus = _lst2vec(internal2cartesian(_vec2lst(chiminus)))
            Jacobian[:,j] = (xplus - xminus) / (2*h)
            # forward difference quotient
            # Jacobian[:,j] = (xplus - x) / h

        print "Jacobian:"
        from DFTB.utils import annotated_matrix
        print annotated_matrix(Jacobian,["x_%s" % i for i in range(0, 3*len(pos))], ["q_%s" % j for j in range(0, 3*len(pos))])
        return Jacobian
    def getGeneralizedForces(self, cartesian_forces):
        """
        transform forces into internal coordinates using Jacobian

        FQ_i = dV/dq_i = sum_j dV/dx_j*dx_j/dq_i = sum_j J_ji*F_j

        F_j are cartesian forces, FQ_i are generalized forces

        Parameters:
        ===========
        cartesian_forces: list of forces on each atom
           [(Z1,[F1_x,F1_y, F1_z]), ..., (ZN,[FN_x,FN_y,FN_z)]
        """
        forces = XYZ.atomlist2vector(cartesian_forces)
        print "Cartesian forces:"
        print forces
        J = self.getJacobian()
        Q = dot(J.transpose(),forces)
        print "Generalized forces:"
        print Q
        return Q
    def getPairForces(self, cartesian_forces):
        """
        find generalized forces that are associated with 
        bond lengths

        Parameters:
        ===========
        cartesian_forces: list of forces on each atom
           [(Z1,[F1_x,F1_y, F1_z]), ..., (ZN,[FN_x,FN_y,FN_z)]
          
        Returns:
        ========
        bond_lengths: dictionary bl
           bl[(i,j)] is the bond length between atom i and atom j
           Only bond lengths for which forces are available are returned.
        pair_forces: dictionary FQ
           Q[(i,j)] is the force acting on atom i due to atom j. It is 
           a scalar quantity, the force vector points along the bond.
           Forces are not available for all pairs (i,j).
        """
        Q = _lst2vec(self.getGeneralizedCoordinates())
        print Q
        FQ = self.getGeneralizedForces(cartesian_forces)
        print FQ
        bond_lengths = {}
        pair_forces = {}
        # first 6 generalized force components belong to the translational
        # and rotational degrees of freedom
        # 7th component belongs to binding angle of first atom pair
        pair_forces[(0,1)] = FQ[7]
        bond_lengths[(0,1)] = Q[7]
        pair_forces[(0,2)] = FQ[8]
        bond_lengths[(0,2)] = Q[8]
        for i in range(3,len(FQ)/3):
            pair_forces[(i-3,i)] = FQ[3*i]
            bond_lengths[(i-3,i)] = Q[3*i]
            pair_forces[(i-2,i)] = FQ[3*i+1]
            bond_lengths[(i-2,i)] = Q[3*i+1]
            # FQ[3*i+2] belongs to dihedral angle

        import string
        print "             Generalized Forces along Bonds:"
        print "             ==============================="
        print "        atom i - atom j         bond length         force F_ij [hartree/bohr]             force F_ij [eV/AA]"
        for (i,j) in pair_forces:
            print "           %s-  %s   %s %s %s" % \
                (string.ljust("%s%d" % (atom_names[self.atomlist[i][0]-1], i+1), 4), \
                 string.ljust("%s%d" % (atom_names[self.atomlist[j][0]-1], j+1), 4), \
                 string.rjust("%.7f bohr" % bond_lengths[(i,j)], 20), \
                 string.rjust("%.7f hartree/bohr" % pair_forces[(i,j)], 30), \
                 string.rjust("%.7f eV/AA" % (pair_forces[(i,j)]*hartree_to_eV/bohr_to_angs), 30))
        return bond_lengths, pair_forces

##### TESTS #######

def test_Euler_angles():
    from numpy.random import rand
    for i in range(0, 1000):
        a = 2.0*pi*(rand()-0.5)
        b = pi*rand()
        g = 2.0*pi*(rand()-0.5)

        print "original Euler angles: %s" % array([a,b,g])
        Rot = EulerAngles2Rotation(a,b,g)
        x = dot(Rot, array([1,0,0]))
        y = dot(Rot, array([0,1,0]))
        z = dot(Rot, array([0,0,1]))
        ra,rb,rg = tripod2EulerAngles(x,y,z)
        print "recovered Euler angles: %s" % array([ra,rb,rg])
        print "ra-a = %s pi" % ((ra-a)/pi)
        assert abs(rb-b) < 1.0e-10
        assert abs(rg-g) < 1.0e-10
        assert abs(sin(ra-a)) < 1.0e-10

def test_dihedral_transformation():
    from numpy.random import rand

    for i in range(0, 1000):
        rA = 10.0*(rand(3)-0.5)
        rB = 10.0*(rand(3)-0.5)
        rC = 10.0*(rand(3)-0.5)
        rD = 10.0*(rand(3)-0.5)
        
        cartesian = (rA,rB,rC,rD)
        print "Original cartesian:"
        print cartesian
        internal = cartesian2dihedral4pts(cartesian)
        print "Internal coordinates:"
        print internal
        cartesian_rec = dihedral2cartesian4pts(internal)
        print "Recovered cartesian:"
        print cartesian_rec
        assert sum(abs(cartesian_rec[-1] - cartesian[-1])) < 1.0e-10


if __name__ == "__main__":
    import sys
    from DFTB import XYZ
    import Gaussian
    from numpy.random import rand
#    test_Euler_angles()
#    test_dihedral_transformation()
    log_file = sys.argv[1]
    atoms = Gaussian.read_geometry(log_file)
    atomlist = []
    for k in atoms.keys():
        atomlist.append(atoms[k])
    forces = Gaussian.read_forces(log_file)
    forcelist = []
    for k in atoms.keys():
        forcelist.append(forces[k])

    IC = InternalCoordinates(atomlist)
    print IC.getJacobian()
#    IC.getGeneralizedForces(forcelist)
    print IC.getPairForces(forcelist)
#    print trihedron2EulerAngles(array([1.0,0.0,0]),array([0,1.0,0]),array([0,0,1]))
