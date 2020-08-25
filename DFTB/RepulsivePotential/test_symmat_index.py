from pairwise_decomposition import symmat_index
import sys

dim = int(sys.argv[1])
print "number of unique entries: %d" % ((dim*dim - dim)/2)
for i in range(0, dim):
    for j in range(0, dim):
        if i == j:
            entry = " - "
        else:
            entry = "%.2d " % symmat_index(i,j,dim)
        print entry,
    print ""

for i in range(0, dim):
    for j in range(0, dim):
        if i == j:
            entry = " -- "
        else:
            entry = "%d%d " % (i,j) #symmat_index(i,j,dim)
        print entry,
    print ""
