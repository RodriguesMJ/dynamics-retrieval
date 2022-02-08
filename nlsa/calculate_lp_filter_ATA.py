# -*- coding: utf-8 -*-
import argparse
import importlib
import numpy
import time
import joblib

 
def f(loop_idx, settings):
    results_path = settings.results_path
    f_max = settings.f_max
    n_lp_vectors = 2*f_max + 1
    ATA_lp_filter = numpy.zeros((n_lp_vectors, n_lp_vectors))
    print ATA_lp_filter.shape, ATA_lp_filter.dtype
    start = time.time()
    for i in range(loop_idx, loop_idx+1):#(n_lp_vectors):
        print i
        ai = joblib.load('%s/aj/a_%d.jbl'%(results_path,i))  
        for j in range(i, n_lp_vectors):            
            aj = joblib.load('%s/aj/a_%d.jbl'%(results_path,j))       
            ATA_ij = numpy.inner(ai, aj)
            ATA_lp_filter[i,j] = ATA_ij
            ATA_lp_filter[j,i] = ATA_ij
    print 'Time:', time.time() - start
    joblib.dump(ATA_lp_filter, '%s/ATA_lp_filter/chunck_%d.jbl'%(results_path, loop_idx))
    
def main(args=None):
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("worker_ID", help="Worker ID")
    parser.add_argument("settings", help="settings file")
    args = parser.parse_args(args)

    print 'Worker ID: ', args.worker_ID
    
    # Dynamic import based on the command line argument
    settings = importlib.import_module(args.settings)

    #print("Label: %s"%settings.label)
    
    f(int(args.worker_ID), settings)


if __name__ == "__main__":
    main()