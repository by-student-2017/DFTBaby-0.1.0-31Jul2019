"""
odd bits and pieces of code
"""
from numpy import zeros, argsort
import numpy as np
import inspect

# for python >= 2.7 this function is added in itertools
def combinations_with_replacement(iterable, r):
    # combinations_with_replacement('ABC', 2) --> AA AB AC BB BC CC
    pool = tuple(iterable)
    n = len(pool)
    if not n and r:
        return
    indices = [0] * r
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != n - 1:
                break
        else:
            return
        indices[i:] = [indices[i] + 1] * (r - i)
        yield tuple(pool[i] for i in indices)

# for scipy >= v0.11 this function is added to scipy.linalg
# This implementation is stolen from scipy v0.11 linalg/special_matrices.py
def block_diag(*arrs):
    """
    Create a block diagonal matrix from provided arrays.

    Given the inputs `A`, `B` and `C`, the output will have these
    arrays arranged on the diagonal::

        [[A, 0, 0],
         [0, B, 0],
         [0, 0, C]]

    Parameters
    ----------
    A, B, C, ... : array_like, up to 2-D
        Input arrays.  A 1-D array or array_like sequence of length `n`is
        treated as a 2-D array with shape ``(1,n)``.

    Returns
    -------
    D : ndarray
        Array with `A`, `B`, `C`, ... on the diagonal.  `D` has the
        same dtype as `A`.

    Notes
    -----
    If all the input arrays are square, the output is known as a
    block diagonal matrix.

    Examples
    --------
    >>> from scipy.linalg import block_diag
    >>> A = [[1, 0],
    ...      [0, 1]]
    >>> B = [[3, 4, 5],
    ...      [6, 7, 8]]
    >>> C = [[7]]
    >>> block_diag(A, B, C)
    [[1 0 0 0 0 0]
     [0 1 0 0 0 0]
     [0 0 3 4 5 0]
     [0 0 6 7 8 0]
     [0 0 0 0 0 7]]
    >>> block_diag(1.0, [2, 3], [[4, 5], [6, 7]])
    array([[ 1.,  0.,  0.,  0.,  0.],
           [ 0.,  2.,  3.,  0.,  0.],
           [ 0.,  0.,  0.,  4.,  5.],
           [ 0.,  0.,  0.,  6.,  7.]])

    """
    if arrs == ():
        arrs = ([],)
    arrs = [np.atleast_2d(a) for a in arrs]

    bad_args = [k for k in range(len(arrs)) if arrs[k].ndim > 2]
    if bad_args:
        raise ValueError("arguments in the following positions have dimension "
                            "greater than 2: %s" % bad_args)

    shapes = np.array([a.shape for a in arrs])
    out = np.zeros(np.sum(shapes, axis=0), dtype=arrs[0].dtype)

    r, c = 0, 0
    for i, (rr, cc) in enumerate(shapes):
        out[r:r + rr, c:c + cc] = arrs[i]
        r += rr
        c += cc
    return out

def argsort_2d(arr):
    """
    for a 2-dimensional array return two lists of indeces
    row_sort and col_sort such that arr[row_sort[i],col_sort[i]] is the
    i-th lowest element in arr.

    Parameters:
    ===========
    arr: 2D numpy array
    
    Returns:
    ========
    row_sort: 1D numpy array with row indeces
    col_sort= 1D numpy array with column indeces
    """
    assert len(arr.shape) == 2
    nrows,ncols = arr.shape
    sort_indx = argsort(arr, axis=None)
    row_sort = sort_indx / ncols
    col_sort = sort_indx % ncols
    
    return row_sort, col_sort

def annotated_matrix(M, row_labels, col_labels, format="%.4f", colwidth=10, block_len=10):
    """
    format a matrix such that columns and rows are labeled. If rows are too
    long they are split into blocks.

    Parameters:
    ===========
    M: 2d numpy array with shape mxn
    row_labels: list of m labels for rows
        a label "-" creates a horizontal separation line
    col_labels: list of n labels for columns

    Returns:
    ========
    string with formatted matrix
    """
    import string
    from math import ceil
    m,n = M.shape

    nblocks = int(ceil(n/float(block_len)))
    txt = ""
    for b in range(0, nblocks):
        txt += " "*(colwidth+1) + "|"
        for col in range(b*block_len, min(n,(b+1)*block_len)):
            txt += string.center(col_labels[col], colwidth+1) + "|"
        txt += "\n" + "-"*(colwidth+2)*(min(block_len,n)+1) + "\n"

        nr_sep = 0 # count the separation lines to keep track
                   # of how many data lines were printed
        for row in range(0,len(row_labels)):
            if row_labels[row] == "-":
                txt += "-"*(min(block_len,n)+1)*(colwidth+2)
                nr_sep += 1
            else:
                txt += string.center(row_labels[row], colwidth+1) + "|"
                for col in range(b*block_len, min(n,(b+1)*block_len)):
                    txt += string.center(format % M[row-nr_sep,col], colwidth+1) + "|"
            txt += "\n"
        txt += "\n"
    return txt


class dotdic(dict):
    """
    overload dictionary to allow accessing data by .-notation
    e.g.
    >> d = dotdic()
    >> d["bla"] = 1
    >> print d.bla
    """
    def __getattr__(self, name):
        return self[name]
    def __setattr__(self, name, value):
        self[name] = value


def numerical_gradient(f,x0,h=1.0e-5):
    """
    compute gradient of f at x0 by numerical differentiation

    Parameters:
    ===========
    f: scalar function of a 1D vector x
    x0: 1D numpy array

    Returns:
    ========
    1D numpy array with difference quotient 
       df/dx_i = (f(x0 + h*ei) - f(x0))/h 
    """
    n = len(x0)
    f0 = f(x0)
    dfdx = zeros(n)
    for i in range(0, n):
        print "numerical gradient: %d of %d" % (i,n)
        ei = zeros(n)
        ei[i] = 1.0 # unit vector
#        # forward gradient
#        x_hi = x0 + h*ei
#        dfdx[i] = (f(x_hi) - f0)/h
        # symmetric gradient
        x_mhi = x0 - h*ei
        x_phi = x0 + h*ei
        dfdx[i] = (f(x_phi) - f(x_mhi))/(2.0*h)
    return dfdx

def numerical_hessian(f,x0,h=1.0e-8):
    """
    compute matrix of second derivatives of f at f0 by finite differences

    Parameters:
    ===========
    f: scalar function of a 1D vector x
    x0: 1D numpy array

    Returns:
    ========
    2D numpy array with difference quotient 
       H_ij = d^2f/(dx_i dx_j)(x0)

    Warning: This is probably not the most efficient way to compute second derivatives,
    since f(x) is evaluated several times at the same x.
    """
    n = len(x0)
    hessian = zeros((n,n))
    f0 = f(x0)
    for i in range(0, n):
        ei = zeros(n)
        ei[i] = 1.0
        for j in range(i, n):
            if i != j:
                ej = zeros(n)
                ej[j] = 1.0
                x_pipj = x0 + h*(ei + ej)
                x_pimj = x0 + h*(ei - ej)
                x_mipj = x0 + h*(-ei + ej)
                x_mimj = x0 - h*(ei + ej)

                hessian[i,j] = ( f(x_pipj) - f(x_pimj) - f(x_mipj) + f(x_mimj) )/(4*h*h)
                hessian[j,i] = hessian[i,j]
        # i == j
        x_pi = x0 + h*ei
        x_mi = x0 - h*ei
        hessian[i,i] = ( f(x_pi) - 2*f0 + f(x_mi) )/(h*h)
    return hessian

def numerical_hessian_G(grad,x0,h=1.0e-8):
    """
    compute hessian by numerical differentiation of gradients

    Parameters:
    ===========
    grad: grad(x) computes the gradient at position x
    """
    n = len(x0)
    hessian = zeros((n,n))
    g0 = grad(x0)
    for i in range(0, n):
        ei = zeros(n)
        ei[i] = 1.0
        x_phi = x0 + h*ei
        hessian[i,:] = (grad(x_phi) - g0)/h
    # hessian should be symmetric
    
    return hessian
    

def call_with_opts_from_dict(func):
    """
    call function func with its keywords replaced by values in dictionary.
    Avoids raising an error if dictionary contains keys that are not keywords
    in func.

    stolen from http://bytes.com/topic/python/answers/170128-expanding-dictionary-function-arguments
    """
    argnames = func.func_code.co_varnames[hasattr(func, 'im_func'):func.func_code.co_argcount]
    ndefaults = len(func.func_defaults or ())
    if ndefaults:
        optnames = argnames[-ndefaults:]
        argnames = argnames[:-ndefaults]
    else:
        optnames = []
    def _f(*args, **opts):
        try:
            actualargs = args #[args[argname] for argname in argnames]
            actualopts = {}
            for io,optname in enumerate(optnames):
                if optname in opts.keys(): 
                    actualopts[optname] = opts[optname]
                else:
                    actualopts[optname] = func.func_defaults[io]
#            print "Calling %s with arguments = %s and options = %s" % (func.func_name, actualargs, actualopts)
        except TypeError: raise TypeError, '%s(...) requires arg(s) %r'%(
            func.func_name, [argname for argname in argnames])
        return func(*actualargs, **actualopts)
    
    _f.func_name = func.func_name
    return _f

import ConfigParser
from optparse import OptionParser, OptionGroup

class OptionParserFuncWrapper:
    """
    Extract the arguments and keywords of a function and use them to
    build an OptionParser object. Each keyword translates into one command option.
    """
    def __init__(self, funcs, usage, section_headers=['DFTBaby'], ignore_unknown_options=False, verbose=1):
        """
        Parameters:
        ===========
        funcs: single function or list of functions whose keywords will be exported as options
        usage: text

        WARNING: functions must only contain keyword arguments
        """
        # read values from configuration file
        config = ConfigParser.ConfigParser()
        conffiles = config.read("dftbaby.cfg")
        if len(conffiles) > 0:
            #print "configuration read from %s" % conffiles[0]
            pass
        
        self.parser = OptionParser(usage)

        if ignore_unknown_options == True:
            self.parser.error = self.parser_error
            
        # The options are divided into different sections depending
        # on the prefix before the dot, so 'Convergence.niterations' 
        # would be in the section 'Convergence', while 'parameter_set'
        # would be in the main section.
        self.option_groups = {"": self.parser}
        if type(funcs) != type([]):
            # not a list? create one with single argument
            funcs = [funcs]
        # remember which parameters belong to which function
        self.func_parameters = {}
        for func in funcs:
            fvars = []
            # total number of variables, required + keywords
            varnames = inspect.getargspec(func)[0]
            nvars = len(varnames)
            nreq = nvars - len(func.func_defaults)
            ivar = 0
            for var in varnames[nreq:nvars]:
                fvars.append(var)
                # extract help from docstring of function
                help_text=""
                found = 0
                for l in func.__doc__.split('\n'):
                    if ":" in l:
                        if found == 1:
                            # stop at the next keyword
                            break
                    if "%s:" % var in l:
                        v,help_text = l.strip().split(":")
                        found = 1
                    else:
                        if found == 1:
                            help_text += "%s, " % l.strip()
                # determine into which section this option is placed
                if "." in v:
                    section, option = v.split(".")
                    option_group = self.option_groups.get(section, None)
                    # add new group of options if it does not exist already
                    if option_group == None:
                        option_group = OptionGroup(self.parser, section)
                        self.parser.add_option_group(option_group)
                        self.option_groups[section] = option_group
                else:
                    section, option = "", var
                    option_group = self.parser
                default = func.func_defaults[ivar]
                if type(default) == type(1):
                    numtype = "int"
                elif type(default) == type(1.0):
                    numtype = "float"
                elif type(default) == type([]):
                    # no lists as parameters
                    ivar += 1
                    continue
                else:
                    numtype = "str"
                for section_header in section_headers:
                    # try all sections until an option with this name is found
                    try:
                        default = config.get(section_header, option)
                        if verbose > 0:
                            print "Option %s => %s" % (option, default)
                    except ConfigParser.NoOptionError as e:
                        continue
                    except ConfigParser.NoSectionError as e:
                        # go to next section
                        continue
                    break
                option_group.add_option("--%s" % option, dest=option, type=numtype,\
                                    default=default, help=help_text + " [default: %default]")
                ivar += 1
            self.func_parameters[unique_func_name(func)] = fvars
    def convert_types(self, options):
        """
        Perform type conversions that are not done automatically by the OptionParser class:
         - convert string "None" to None
         - evaluate lists 
        """
        opts = {}
        for k,v in options.__dict__.iteritems():
            if v == "None" or v == "none":
#                print "found None"
                opts[k] = None
            elif type(v) == type("") and v[0] == "[" and v[-1] == "]":
#                print "found list"
                opts[k] = eval(v)
            elif type(v) == type(""):
                try:
                    opts[k] = eval(v)
                except:
                    opts[k] = v
            else:
                # no conversion needed
                opts[k] = v
        return opts
    def parse_args(self, f=None):
        """
        parse arguments and options and return those ones
        in a dictionary that are keywords to the function f.
        If f is None, return all parameters.
        """
        options, args = self.parser.parse_args()
        opts = self.convert_types(options)

        if f != None:
            # filter those options that are valid arguments to function f
            opts = dict([(k,v) for k,v in opts.iteritems() if k in self.func_parameters[unique_func_name(f)]])
        return opts, args
    
    def parser_error(self, msg):
        print msg
        # overrides OptionParser.error
        if "no such option" in msg:
            pass
        else:
            print msg
            exit(-1)
            
def unique_func_name(f):
    """
    build a unique function name as
      <module name>_<func_name>
    """
    unique_name = inspect.getmodule(f).__name__ + "_" + f.func_name
    return unique_name
            
if __name__ == "__main__":
    def bla(x,y,z=100,w=1000):
        print "x = %s, y = %s, z = %s, w = %s" % (x,y,z,w)
    opts = {'w': -1000, 'test':100}
    call_with_opts_from_dict(bla)(1,2,**opts)
