#!/usr/bin/env python
"""
average total cross section and anisotropy parameter beta incoherently over different ionization channels I,
that have the same ionization energy.

  <sigma> = sum_I sigma_I

  <beta> = (sum_I simga_I*beta_I)/<sigma>
"""

import numpy as np
import sys
import os

if __name__ == "__main__":
    import optparse
    
    usage  = "%s <list of .dat files>\n" % os.path.basename(sys.argv[0])
    usage += " average cross sections and anisotropy parameters incoherently over different ionization channels.\n"
    usage += " Each .dat file should contain the columns \n"
    usage += "   PKE/eV     sigma    beta1   beta2   beta3   beta4\n"
    usage += " with the angular distribution for one ionization channel\n"

    parser = optparse.OptionParser(usage)
    parser.add_option("--avg_file", dest="avg_file", help="Channel-averaged angular distribution is written to this file [default: %default]", default="betas_avg.dat")

    (opts,args) = parser.parse_args()
    
    if len(args) < 1:
        print usage
        exit(-1)
    paths = args

    data_list = []
    for i,pad_file in enumerate(paths):
        data = np.loadtxt(pad_file)
        data_list.append(data)

    data_avg = np.zeros(data_list[0].shape)
    # PKE axis
    data_avg[:,0] = data_list[0][:,0]
    
    for data in data_list:
        sigma = data[:,1]
        data_avg[:,1] += sigma
        for k in range(1,5):
            data_avg[:,1+k] += sigma*data[:,1+k]
    for k in range(1,5):
        data_avg[:,1+k] /= data_avg[:,1]

    fh = open(opts.avg_file, "w")
    print>>fh, "# PKE/eV     sigma    beta1   beta2   beta3   beta4"
    np.savetxt(fh, data_avg)
    fh.close()

    print "Average written to '%s'." % opts.avg_file

    
