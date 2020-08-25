"""
read formatted G09 checkpoint files
"""
import numpy as np

def read_float_block(fh, N):
    """
    read a block of N floats separated into 5 columns
    """
    nrow = N/5
    if N % 5 != 0:
        nrow += 1
    data = []
    for i in range(0, nrow):
        l = fh.readline().replace("D", "E")
        parts = l.strip().split()
        data += map(float, parts)
    assert len(data) == N
    return np.array(data, dtype=float)

def read_int_block(fh, N):
    """
    read a block of N integers separated into 6 columns
    """
    nrow = N/6
    if N % 6 != 0:
        nrow += 1
    data = []
    for i in range(0, nrow):
        l = fh.readline()
        parts = l.strip().split()
        data += map(int, parts)
    assert len(data) == N
    return np.array(data, dtype=int)


class _Block(object):
    def __init__(self, args, fh):
        pass
    def getName(self):
        return self.__class__.__name__.replace("_Block", "")
    def getPy(self):
        """returns a representation that can be used with python"""
        pass

class _IgnoreBlock(_Block):
    pass

class _Charge_Block(_Block):
    def __init__(self, args, fh):
#        print "args = %s" % args
        assert args[0] == "I"
        self.charge = int(args[1])
    def getPy(self):
        return self.charge

class int_Block(_Block):
    def __init__(self, args, fh):
        assert args[0] == "I"
        N = int(args[2])
        self.data = read_int_block(fh, N)
    def getPy(self):
        return self.data
    
class float_Block(_Block):
    def __init__(self, args, fh):
        assert args[0] == "R"
        N = int(args[2])
        self.data = read_float_block(fh, N)
    def getPy(self):
        return self.data
    
##########################################
class _Atomic_numbers_Block(int_Block):
    pass

class _Nuclear_charges_Block(float_Block):
    pass

class _Int_Atom_Types_Block(int_Block):
    pass

class _MM_charges_Block(float_Block):
    pass

class _Current_cartesian_coordinates_Block(float_Block):
    pass

class _Mulliken_Charges_Block(float_Block):
    pass

class _Cartesian_Gradient_Block(float_Block):
    def getPy(self):
        return self.data

class  _Cartesian_Force_Constants_Block(float_Block):
    def getPy(self):
        # Hessian contains lower triangular matrix
        # with N*(N+1)/2 entries where N=Nat*3
        M = len(self.data)
        N = int( 0.5 * (np.sqrt(1.0 + 8.0*M) - 1.0) )
        Nat = int( N/3.0 )
        assert (3*Nat)*(3*Nat+1)/2 == M

        Hess = np.zeros((N,N))
        for j in range(1, N+1):
            for i in range(1,j+1):
                indx = (j-1)*j/2+i-1
                hij = self.data[indx]
                Hess[i-1,j-1] = hij
                # hessian is symmetric
                Hess[j-1,i-1] = Hess[i-1,j-1]
        return Hess

class _Dipole_Moment_Block(float_Block):
    pass

class _Dipole_Derivatives_Block(float_Block):
    pass

###########################################
class single_float_Block(_Block):
    def __init__(self, args, fh):
        assert args[0] == "R"
        self.f = float(args[1])
    def getPy(self):
        return self.f

class _Total_Energy_Block(single_float_Block):
    pass

class _IRC_point_Block(float_Block):
    def __init__(self, args, fh):
        assert args[-3] == "R"
        self.quantity = args[1]
        N = int(args[-1])
        self.data = read_float_block(fh, N)
    def getName(self):
        return "_IRC_point_%s" % self.quantity
    def getPy(self):
        return self.data
        
class _IRC_Number_of_geometries_Block(int_Block):
    pass

def get_block_type_args(l):
    parts = l.split(" ")
    block_type = ""
    for ip,p in enumerate(parts):
        if p == "":
            args = " ".join(parts[ip:]).split()
            break
        else:
            block_type += "_" + p
    # replace reserved characters
    block_type = block_type.replace("-", "_").replace("=", "_").replace(")","_").replace("(","_").replace("/","_")
    return block_type, args

def parseCheckpointFile(filename, convert=True):
    Data = {}
    """A dictonary containing all the blocks read."""
    fh = open(filename, 'r')
    block = _IgnoreBlock([], fh)
    while True:
        line = fh.readline()
        if line == "":
            # end of file
            break
        if line[0].isupper():
            """A capital letter starts a block of information. 
            The name of the block is used to find the correct derived class 
            (by appending _Block). """
            block_type, args = get_block_type_args(line)
            try:
                block = eval(block_type + "_Block")(args, fh)
                """call the constructor of a class that bears the same name as the current block"""
                Data[block.getName()] = block
            except NameError:
#                print "Cannot read blocks of type %s" % block_type
                block = _IgnoreBlock(args, fh)
    fh.close()
    PyData = {}
    if convert == True:
        """convert blocks into format suitable for PyQuante"""
        for (block_type, block) in Data.iteritems():
            PyData[block_type] = block.getPy()
        return(PyData)
    else:
        return Data

if __name__ == "__main__":
    import sys

    usage = "python %s <Gaussian checkpoint file>" % sys.argv[0]
    if len(sys.argv) < 2:
        print usage
        exit(-1)

    chk_file = sys.argv[1]
    Data = parseCheckpointFile(chk_file)
    print Data.keys()
