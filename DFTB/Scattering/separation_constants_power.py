
def separation_constants_series(m,n):
    # 21.7.5 in Abramovitz/Stegun
    l0 = n*(n+1)
    l2 = 0.5 * (1.0 - (2.0*m-1.0)*(2.0*m+1.0)/float((2.0*n-1.0)*(2.0*n+3)))
    l4 = (-(n-m+1)*(n-m+2)*(n+m+1)*(n+m+2))/float(2*(2*n+1)*(2*n+3)**3*(2*n+5)) \
         + ((n-m-1)*(n-m)*(n+m-1)*(n+m))/float(2*(2*n-3)*(2*n-1)**3*(2*n+1))
    l6 = (4*m**2-1)*( (n-m+1)*(n-m+2)*(n+m+1)*(n+m+2)/float((2*n-1)*(2*n+1)*(2*n+3)**5*(2*n+5)*(2*n+7)) \
                     -(n-m-1)*(n-m)*(n+m-1)*(n+m)/float( (2*n-5)*(2*n-3)*(2*n-1)**5*(2*n+1)*(2*n+3) )
                    )
    A = (n-m-1)*(n-m)*(n+m-1)*(n+m)/float((2*n-5)**2*(2*n-3)*(2*n-1)**7*(2*n+1)*(2*n+3)**2) \
        -(n-m+1)*(n-m+2)*(n+m+1)*(n+m+2)/float((2*n-1)**2*(2*n+1)*(2*n+3)**7*(2*n+5)*(2*n+7)**2)
    B = (n-m-3)*(n-m-2)*(n-m-1)*(n-m)*(n+m-3)*(n+m-2)*(n+m-1)*(n+m)/float((2*n-7)*(2*n-5)**2*(2*n-3)**3*(2*n-1)**4*(2*n+1)) \
        -(n-m+1)*(n-m+2)*(n-m+3)*(n-m+4)*(n+m+1)*(n+m+2)*(n+m+3)*(n+m+4)/float((2*n+1)*(2*n+3)**4*(2*n+5)**3*(2*n+7)**2*(2*n+9))
    C = (n-m+1)**2*(n-m+2)**2*(n+m+1)**2*(n+m+2)**2/float((2*n+1)**2*(2*n+3)**7*(2*n+5)**2) \
        -(n-m-1)**2*(n-m)**2*(n+m-1)**2*(n+m)**2/float((2*n-3)**2*(2*n-1)**7*(2*n+1)**2)
    D = (n-m-1)*(n-m)*(n-m+1)*(n-m+2)*(n+m-1)*(n+m)*(n+m+1)*(n+m+2)/float((2*n-3)*(2*n-1)**4*(2*n+1)**2*(2*n+3)**4*(2*n+5))
    
    l8 = 2*(4*m**2-1)**2*A + 1/16.0 * B + 1/8.0 * C + 1/2.0 * D

    return [l8,l6,l4,l2,l0]

if __name__ == "__main__":
    for m in range(0, 3):
        for n in range(0,10):
            print "%d %d  " % (m,n),
            for l in separation_constants_series(m,n):
                print "  %20.10f" % l,
            print ""
            
                
