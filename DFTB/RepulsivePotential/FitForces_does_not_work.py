#    DOES NOT WORK YET !
from DFTB.AtomicData import atom_names, atom_masses
from Gaussian import read_geometry, read_forces
from numpy import array, dot, cross, zeros, vstack, concatenate
from numpy.linalg import solve, norm, det, pinv

def index2pair(m,N):
    c = 0
    for i in range(0,N):
        for j in range(i+1,N):
            if m == c:
                return (i,j)
            c += 1
    
def pair2index(k,l,N):
    assert k != l
    c = 0
    for i in range(0,N):
        for j in range(i+1,N):
            if (i == k and j == l) or (i == l and j == k):
                return c
            c += 1

def pair_potential_derivatives(atoms, forces, ZA, ZB):
    """
    For a given structure with forces collect pairs {R_AB, V'(R_AB)} for 
    fitting the repulsive potential. The derivative as found by solving
    a system of linear equations that relates the total forces on each atom
    to the derivative of the potential energy with respect to interatomic 
    distances.
    """
    def delta(a,b):
        if a == b:
            return 1.0
        else:
            return 0.0

    N = len(atoms)
    dim = N*(N-1)/2
    A = zeros((dim,dim))
    b = zeros(dim)
    print "N*(N-1)/2 = %s" % dim
    for i in range(0,N):
        for k in range(0,N):
            if i >= k:
                continue
#            (Zi,ri) = atoms[i]
            (Zk,rk) = atoms[k]
            atnumi, Fi = forces[i]
            m = pair2index(i,k,N)
            print "m = %s" % m
            print "Fi = %s, rk = %s" % (Fi, rk)
#            b[m] = -dot(Fi,ri-rk)
            b[m] = -dot(Fi,rk)
            for j in range(0, N):
                for l in range(0,N):
                    if j == l:
                        continue
                    (Zj,rj) = atoms[j]
                    (Zl,rl) = atoms[l]
                    n = pair2index(j,l,N)
#                    A[m,n] += delta(i,l)*dot(rl-rj,ri-rk)/norm(rl-rj)
                    A[m,n] += delta(i,l)*dot(rl-rj,rk)/norm(rl-rj)


    print "A"
    print A.tolist()
    print "b"
    print b
    from numpy.linalg import det
    print "det(A) = %s" % det(A)
    x = solve(A,b)
    print "x"
    print x
    all_forces = zeros(3)
    all_angmom = zeros(3)
    # reconstruct forces
    for i in range(0, N):
        Fi_rec = 0.0
        (Zi,ri) = atoms[i]
        for j in range(0, N):
            if i != j:
                (Zj,rj) = atoms[j]
                Fi_rec -= x[pair2index(i,j,N)]*(ri-rj)/norm(ri-rj)
        atnumi, Fi = forces[i]
        print "%s Fi = %s Fi_rec = %s |Fi|=%s" % (i,Fi, Fi_rec,norm(Fi))
        assert norm(Fi-Fi_rec) < 1.0e-8
        all_forces += Fi
        all_angmom += cross(ri,Fi)
    print "sum of forces Ftot = %s" % all_forces
    print "total angular momentum = %s" % all_angmom
    # find derivatives w/r/t distances between atom pairs of certain type
    Vderiv = []
    for i in range(0, N):
        for j in range(0, N):
            if i == j:
                continue
            Zi,ri = atoms[i]
            Zj,rj = atoms[j]
            
            if (Zi == ZA and Zj == ZB):
                m = pair2index(i,j,N)
                Vderiv.append( (norm(ri-rj), x[m], i, j) )
    return Vderiv

def select_internal_coordinates(atoms, Nselect):
    """
    select N atom pairs, whose relative distances will
    be used as internal coordinates.

    rij = |r_Ai - r_Bj|

    Parameters:
    ===========
    atoms: dictionary with atom positions atoms[i] = (Zi,(xi,yi,zi))

    Returns:
    ========
    A,B: two lists with indeces to the first and second atoms of the atom pairs
    """
    N = len(atoms)
    counter = 0
    A = []
    B = []
    for i in range(0, N):
        for j in range(i+1, N):

            if A.count(i) >= 3 or B.count(j) >= 3:
                continue

            A.append(i)
            B.append(j)
            counter += 1 
            if counter == Nselect:
                print "len(A) = %s len(B) = %s  Nselect = %s" % (len(A), len(B), Nselect)
                print A
                print B
                assert len(A) == Nselect
                assert len(B) == Nselect
                return (A,B)
            
def distance_potential_derivatives(atoms, forces, ZA, ZB):
    def delta(a,b):
        if a == b:
            return 1.0
        else:
            return 0.0
    N = len(atoms)
    f = 3*N-6
    g = N-2
    A,B = select_internal_coordinates(atoms, f)
    print "As internal coordinates we use the distances between the following atom pairs"
    for j in range(0, f):
        print " %s - %s" % (A[j], B[j])
    Amat = zeros((3*N,3*N))
    b = zeros(3*N)

    for i in range(0, 3*N):
        ic = int(i/3)
        coord = i%3 # 0->x, 1->y, 2->z
        Zi, posi = atoms[ic]
        mi = atom_masses[atom_names[Zi-1]]
        for j in range(0, 3*N-6):
            ZAj, posAj = atoms[A[j]]
            ZBj, posBj = atoms[B[j]]
            rAjBj = norm(posAj-posBj)
            assert abs(rAjBj) > 1.0e-10
            fac = 1.0/rAjBj * (delta(ic,A[j]) - delta(ic,B[j]))
            Amat[i,j] = (posAj[coord]-posBj[coord])*fac
            if j == f-1:
                print "i = %s ic = %s coord = %s A[j=%s] = %s B[j=%s] = %s fac = %s Amat[%s,%s] = %s posAj[%s] = %s posBj[%s] = %s" % (i, ic, coord, j, A[j], j, B[j], fac, i, j, Amat[i,j], coord, posAj[coord], coord, posBj[coord])
        # The molecular potential is invariant with respect
        # to 6 coordinates:
        #  3 translational degrees of freedom
        #  3 rotational degrees of freedom
        Amat[i,j+1] = delta(coord,0)
        Amat[i,j+2] = delta(coord,1)
        Amat[i,j+3] = delta(coord,2)
        Amat[i,j+4] = ( posi[1]*delta(coord,0) + posi[0]*delta(coord,1) )
        Amat[i,j+5] = ( posi[2]*delta(coord,0) + posi[0]*delta(coord,2) )
        Amat[i,j+6] = ( posi[2]*delta(coord,1) + posi[1]*delta(coord,2) )

        Zic, Fic = forces[ic]
        b[i]   = -Fic[coord]

    print "Amat.shape = %s" % str(Amat.shape)
    print "b.shape = %s" % str(b.shape)
    print "Amat = \n%s" % Amat.round(decimals=3)
    print "det(Amat) = %s" % det(Amat)
#    assert abs(det(Amat)) > 1.0e-9
#    u = solve(Amat, b)
    Apinv = pinv(Amat)
    u = dot(Apinv,b)

    print "u = \n%s" % u
    # select derivatives w/r/t distance between atoms of certain elements
    Vderiv = []
    for j in range(0, f):
        ZAj, posAj = atoms[A[j]]
        ZBj, posBj = atoms[B[j]]
        dj = norm(posAj-posBj)
        if (ZAj == ZA and ZBj == ZB) or (ZAj == ZB and ZBj == ZA):
            Vderiv.append( (dj, u[j], A[j], B[j]) )
    return Vderiv


if __name__ == "__main__":
    import sys
    from matplotlib.pyplot import plot, show, xlabel, ylabel, title
    RVp_pairs = []
    if len(sys.argv) < 3:
        print "Usage: %s Z1 Z2 <gaussian log file>" % sys.argv[0]
        print "extract pairs {R_12, V'(R_12)} for atom pair Z1-Z2 from the gaussian log files"
        print "provided as input to stdin."
        exit(-1)
    Z1, Z2 = int(sys.argv[1]), int(sys.argv[2])
    if len(sys.argv) > 3:
        log_files = [sys.argv[3]]
    else:
        # read file names from stdin
        log_files = sys.stdin.readlines()
    for log_file in log_files:
        atoms = read_geometry(log_file)
        forces = read_forces(log_file)
        print "atoms = %s" % atoms
        print "forces = %s" % forces
#        RVp_pairs += distance_potential_derivatives(atoms, forces, Z1, Z2)
        RVp_pairs += pair_potential_derivatives(atoms, forces, Z1, Z2)
        #RVp_pairs = set(RVp_pairs)
        

    for tup in RVp_pairs:
        print tup
    title("%s - %s" % (atom_names[Z1-1], atom_names[Z2-1]))
    xlabel("interatomic distance R_AB in bohr")
    ylabel("V'(R_AB) in hartree/bohr")
    plot([pair[0] for pair in RVp_pairs], [pair[1] for pair in RVp_pairs], "o")
    show()

