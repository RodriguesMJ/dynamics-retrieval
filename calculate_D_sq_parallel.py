# -*- coding: utf-8 -*-
import getopt
import sys
import numpy
import time
import joblib

import settings_rho_light as settings
 
def f(myArguments):
    
    try:
        optionPairs, leftOver = getopt.getopt(myArguments, "h", ["index="])
    except getopt.GetoptError:
        print 'Usage: python ....py --index <index>'
        sys.exit(2)   
    for option, value in optionPairs:
        if option == '-h':
            print 'Usage: python ....py --index <index>'
            sys.exit()
        elif option == "--index":
            loop_idx = int(value)

    print 'loop_idx: ', loop_idx
    
    q = settings.q    
    datatype = settings.datatype
    results_path = settings.results_path
    
    step = settings.paral_step 
    
    d_sq = joblib.load('%s/d_sq.jbl'%results_path)   
    print 'd_sq: ', d_sq.shape
    
    S = d_sq.shape[0]
    s = S-q
    
    print S, ' samples'

    print 'Calculate D_sq, index: ', loop_idx
    
    q_start = loop_idx * step
    q_end   = q_start  + step
    if q_end > q:
        q_end = q
    print 'q_start: ', q_start, 'q_end: ', q_end
    
    D_sq = numpy.zeros((s+1,s+1), dtype=datatype) # will need to be s,s
    starttime = time.time()

    for i in range(q_start, q_end):
        if (i%100==0):
            print i, '/', q
        term = d_sq[i:i+s+1, i:i+s+1]
        D_sq = D_sq + term
        
    print 'Time: ', time.time() - starttime   
    
    print 'D_sq: ', D_sq.shape
    joblib.dump(D_sq, '%s/D_sq_loop_idx_%d.jbl'%(results_path,loop_idx))
    
    print 'Time: ', time.time() - starttime   
    
if __name__== "__main__":
    
    f(sys.argv[1:])