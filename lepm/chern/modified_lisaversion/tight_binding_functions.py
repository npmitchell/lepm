from numpy import *

def vec(ang, bl=1):
    return [bl*cos(ang), bl*sin(ang)]

def ang_fac(ang):
    return exp(2*1j*ang)

def calc_matrix(angs, num_nei, pin, delta = 1, bls = -1, tvals = -1) :return lambda k : make_M(k,angs, num_nei, pin, delta, bls, tvals)
#num_nei is number of neighbors of each sublattice kind.  in the honeycomb lattice it'd be three

def make_M(k, angs, num_neis, pin, delta, bls, tvals):

    #print 'delta is ', delta 
    num_sites = len(angs)
    
    
    M = zeros([num_sites, num_sites], dtype = 'complex')

    
    for i in range(num_sites):
        index = i%(num_sites)
        angs_for_row = angs[index]
        bls_for_row = bls[index]
        num_neis_row = num_neis[index]
        num_bonds = len(angs[index])
        
        tv= tvals[index]
        num_bonds = sum(tv)
        fill_count = 0
        for j in range(num_sites):
            nnj = num_neis_row[j]

            
            for l in range(nnj):
                
                M[i,j] += tv[fill_count]*exp(1j* dot(k, vec(angs_for_row[fill_count], bls_for_row[fill_count])))
                fill_count = fill_count +1
            
    
    return M
