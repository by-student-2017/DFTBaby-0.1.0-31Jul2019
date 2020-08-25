"""
repulsive potentials were fitted automatically
"""

atompairs = {}

def load_atompairs(atpairs, missing_reppots="error"):
    """import all atom pairs that are needed automatically"""
    from DFTB.AtomicData import atom_names
    for (Zi,Zj) in atpairs:
        atI, atJ = atom_names[Zi-1], atom_names[Zj-1]
        try:
            atompairs[(Zi,Zj)] = __import__("DFTB.RepulsivePotential.reppot_tables.%s_%s" % (atI,atJ), fromlist=['Z1','Z2'])
        except ImportError as e:
            if missing_reppots == "dummy":
                print "WARNING: Repulsive potential for atom pair %s-%s not found!" % (atI,atJ)
                print "         A dummy potential with Vrep=0 is loaded instead. Optimizations and"
                print "         dynamics simulations with dummy potentials are bound to produce nonsense."
                atompairs[(Zi,Zj)] = __import__("DFTB.RepulsivePotential.reppot_tables.dummy", fromlist=['Z1','Z2'])
                # set correct atomic numbers
                atompairs[(Zi,Zj)].Z1 = Zi
                atompairs[(Zi,Zj)].Z1 = Zj
            else:
                raise e
            
__all__ = ["atompairs"]
