# -*- coding: utf-8 -*-
import joblib
import numpy
import matplotlib
matplotlib.use('Agg') # Force matplotlib to not use any Xwindows backend.
import matplotlib.pyplot


def plot(k_miller, T, data_path, label, corr_label):
    Is_avg = []
    for i in range(40):
        idxs = numpy.argwhere(k_miller==i)[:,0]
        #print idxs.shape
        T_temp = T[idxs, :]
        #print T_temp.shape
        avg = T_temp.sum()/T_temp.nnz
        Is_avg.append(avg)
            
    matplotlib.pyplot.figure(figsize=(40, 10))  
    matplotlib.pyplot.plot(range(len(Is_avg)), Is_avg, '-o')
    matplotlib.pyplot.xlabel(r'k'),
    matplotlib.pyplot.ylabel(r'<I>_k')
    matplotlib.pyplot.savefig('%s/%s_Is%s_vs_k.png'%(data_path, label, corr_label))  
    matplotlib.pyplot.close()
    
def main(settings):
    label = settings.label    
    data_path = settings.data_path
    datatype = settings.datatype
    fraction = settings.t_correction_factor
    t_d_y = settings.t_d_y
    print 'label: ', label
    print data_path
    print 'Datatype: ', datatype
    print 'Fraction: ', fraction
    print 't_d_y:', t_d_y
    
    A = 2 * fraction * fraction - 2 * fraction + 1
    B = 2 * fraction * (1 - fraction)
    print 'A: ', A
    print 'B: ', B
    print 'A+B: ', A+B
    
    k_miller = joblib.load('%s/miller_k_%s.jbl'%(data_path, label))
    print 'N. Bragg pts:', k_miller.shape[0]
    
    T_sparse = joblib.load('%s/T_sparse_%s.jbl'%(data_path, label))            # sparse.csr.csr_matrix
    
    print 'T_sparse: ', T_sparse.shape, T_sparse.dtype 
    m = T_sparse.shape[0]
    S = T_sparse.shape[1]
    print m, ' Bragg pts, ', S, ' samples.'    
    
    factors = 1.0/( A + B * numpy.cos( 2 * numpy.pi * k_miller * t_d_y ) )
    print 'Correction factors: ', factors.shape
    factors_rep = numpy.repeat(factors, S, axis=1)
    factors_rep = numpy.asarray(factors_rep)
    print 'Correction factors: ', factors_rep.shape
    T_corr = T_sparse.multiply(factors)                                        # sparse.coo.coo_matrix
    T_corr = T_corr.tocsr()  
    T_corr = T_corr.astype(datatype)
    print 'T_corr: ', T_corr.shape, T_corr.dtype                                                 # sparse.csr.csr_matrix
    print 'nnz before correction: ', T_sparse.nnz
    print 'nnz after correction: ', T_corr.nnz
    joblib.dump(T_corr, '%s/T_sparse_corr_%s.jbl'%(data_path, label))
    
    plot(k_miller, T_sparse, data_path, label, '')
    plot(k_miller, T_corr,   data_path, label, '_corr')