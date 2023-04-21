# -*- coding: utf-8 -*-
import numpy
import time
import joblib
import argparse
import importlib


def f(loop_idx, settings):
    results_path = settings.results_path
    ATA_lp_filter = joblib.load('%s/ATA_lp_filter.jbl'%results_path)
    Q_lp_filter   = joblib.load('%s/F_on_qr.jbl'%results_path)
    
    s = Q_lp_filter.shape[0]
    step = settings.paral_step_lp_filter_Dsq 
    
    i_start = loop_idx*step
    i_end   = (loop_idx+1)*step
    if i_end > s:
        i_end = s
    print 'i_start, i_end:', i_start, i_end

    f_max = settings.f_max_considered    
    n_lp_vectors = 2*f_max + 1
    
    ATA_lp_filter = ATA_lp_filter[0:n_lp_vectors, 0:n_lp_vectors]
    Q_lp_filter = Q_lp_filter[:, 0:n_lp_vectors]
    
    print 'ATA_lp_filter:', ATA_lp_filter.shape
    print 'Q_lp_filter:', Q_lp_filter.shape
    
    D_sq = numpy.zeros((s,s), dtype=settings.datatype)
    start = time.time()
    for i in range(i_start, i_end):
        for j in range(i,s):
            delta_i_j = Q_lp_filter.T[:,i]-Q_lp_filter.T[:,j]
            tmp = numpy.matmul(ATA_lp_filter, delta_i_j)
            D_sq_ij = numpy.matmul(delta_i_j.T, tmp)
            D_sq[i,j] = D_sq_ij
            D_sq[j,i] = D_sq_ij
                            
    print 'Time:', time.time() - start
    joblib.dump(D_sq, '%s/D_sq_lp_filtered_fmax_%d_chunck_%d.jbl'%(results_path, 
                                                                   f_max, 
                                                                   loop_idx))
    
def main(args=None):
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("worker_ID", help="Worker ID")
    parser.add_argument("settings",  help="settings file")
    args = parser.parse_args(args)

    print 'Worker ID: ', args.worker_ID
    
    # Dynamic import based on the command line argument
    settings = importlib.import_module(args.settings)
    #print("Label: %s"%settings.label)
    
    f(int(args.worker_ID), settings)


if __name__ == "__main__":
    main()