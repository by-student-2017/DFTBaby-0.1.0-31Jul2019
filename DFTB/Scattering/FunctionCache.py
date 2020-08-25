
class FunctionCache:
    """Results are saved so that they can be retrieved from
    the cache if the function is called with the same arguments
    again."""
    def __init__(self, func, commutative_args=False, unpack_args=False):
        """arguments to the function func have to be iterable"""
        self.cache = {}
        self.func = func
        self.nr_cache_accesses = 0
        self.nr_new_computations = 0
        self.commutative_args = commutative_args
        """If the arguments can be reordered without changing the result such as in the 
        example below, where sum(i,j) = sum(j,i) the values only have to be cached for
        one particular order."""
        self.unpack_args = unpack_args
        """The argument list is contained in a list itself. For instance, if Sum(i,j)
        is called instead as Sum([i,j])."""
    def __call__(self, *arglist):
        if self.unpack_args:   # ugly
            args = arglist[0]  # ugly
        else:                  # ugly   
            args = arglist     # ugly
        if self.commutative_args:
            key = list(args)
            key.sort()
            key = tuple(key)
        else:
            key = tuple(args)
        if self.cache.has_key(key):
            self.nr_cache_accesses += 1
            return self.cache[key]
        else:
            val = self.func(*arglist)
            self.cache[key] = val
            self.nr_new_computations += 1
            return val
    def statistics(self):
        print "nr. of cache accesses = %s" % self.nr_cache_accesses
        print "nr. of new computations = %s" % self.nr_new_computations

if __name__ == "__main__":
    def Sum(a,b):
        return(a+b)

    Sum = FunctionCache(Sum, commutative_args=True)
    """override the definition of Sum"""
    for i in xrange(1, 11):
        for j in xrange(1, 11):
            print Sum(i,j)
    Sum.statistics()
