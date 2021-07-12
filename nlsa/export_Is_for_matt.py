# -*- coding: utf-8 -*-
import os
import joblib
import numpy



def get_merged_data(x, mask, datatype):
    N = mask.sum(axis=1)
    print 'N: ', N.shape
    
    I_avg = (x.sum(axis=1))/N
    I_avg = numpy.asarray(I_avg, dtype=datatype)
    print 'I_avg:', I_avg.shape, I_avg.dtype
    
    I_avg_rep = I_avg[:, numpy.newaxis]
    I_avg_rep = numpy.repeat(I_avg_rep, x.shape[1], axis = 1)
    print 'x:', x.shape, x.dtype
    print 'I_avg_rep:', I_avg_rep.shape, I_avg_rep.dtype
    
    dI_sq = (x-I_avg_rep)*(x-I_avg_rep)
    print 'dI_sq:', dI_sq.shape
    
    s_dI_sq = dI_sq.sum(axis=1)
    
    # Crystfel definition
    sigI = (numpy.sqrt(s_dI_sq))/N
    sigI = numpy.asarray(sigI, dtype=datatype)
    print 'sigI:', sigI.shape, sigI.dtype
    
    return I_avg, sigI



def get_bin(t_uniform, t_left, t_right, x, mask, datatype):
    idxs_r = numpy.argwhere(t_uniform < t_right)
    idxs_l = numpy.argwhere(t_uniform > t_left)
    idxs = numpy.intersect1d(idxs_r, idxs_l)
    print idxs.shape, ' in bin', t_left, 'to', t_right, 'fs'   
    T = x[:, idxs]
    M = mask[:, idxs]
    print 'Bin matrices: ', T.shape, M.shape    
    I_avg, sigI = get_merged_data(T, M, datatype)
    nnans = numpy.count_nonzero(numpy.isnan(I_avg))
    print 'n of nans in I_avg: ', nnans
    nnans = numpy.count_nonzero(numpy.isnan(sigI))
    print 'n of nans in sigI: ', nnans
    I_avg[numpy.isnan(I_avg)] = 0
    sigI[numpy.isnan(sigI)] = 0
    return I_avg, sigI
    
    
def export_merged_data_light(settings):
    data_path = settings.data_path
    data_file = settings.data_file
    results_path = settings.results_path
    datatype = settings.datatype
    label = settings.label
    
    mult_factor = 100
    
    out_path = '%s/merged'%results_path
    if not os.path.exists(out_path):
        os.mkdir(out_path)
        
    # EXTRACT DATA
    T_sparse = joblib.load(data_file)
    print 'T_sparse: ', T_sparse.shape
    print 'T_sparse nonzero: ', T_sparse.count_nonzero()
    x = T_sparse[:,:].todense()
    print 'x: ', x.shape, x.dtype
    x = numpy.asarray(x, dtype=datatype)
    print 'x: ', x.shape, x.dtype
    nnans = numpy.count_nonzero(numpy.isnan(x))
    print 'n of nans in x: ', nnans
    
    x = mult_factor*x
    
    M_sparse = joblib.load('%s/M_sparse_%s.jbl'%(data_path, label))
    print 'M_sparse: ', M_sparse.shape
    print 'M_sparse nonzero: ', M_sparse.count_nonzero()
    mask = M_sparse[:,:].todense()
    print 'mask: ', mask.shape, mask.dtype
    mask = numpy.asarray(mask, dtype=numpy.uint8)
    print 'mask: ', mask.shape, mask.dtype
    
    t_uniform = joblib.load('%s/t_uniform_%s.jbl'%(data_path, label))
    t_uniform = t_uniform.flatten()
    print 't_uniform: ', t_uniform.shape
    
    miller_h = joblib.load('%s/miller_h_%s.jbl'%(data_path, label))
    miller_k = joblib.load('%s/miller_k_%s.jbl'%(data_path, label))
    miller_l = joblib.load('%s/miller_l_%s.jbl'%(data_path, label))
    
    out = numpy.zeros((miller_h.shape[0], 5))
    out[:, 0] = miller_h.flatten()
    out[:, 1] = miller_k.flatten()
    out[:, 2] = miller_l.flatten()
    
    #NEGATIVE DELAYS
    print '\nNEGATIVE DELAYS'
    t_left = -331
    t_right = -185
    I_avg, sigI = get_bin(t_uniform, t_left, t_right, x, mask, datatype)    
    out[:, 3] = I_avg
    out[:, 4] = sigI        
    f_out = '%s/%s_merged_%d_to_%d_fs.txt'%(out_path, label, t_left, t_right)
    numpy.savetxt(f_out, out, fmt='%6d%6d%6d%15.2f%15.2f')  
    
    # POSITIVE DELAYS
    print '\nPOSITIVE DELAYS (ULTRASHORT)'
    t_left = -185
    t_right = 22
    I_avg, sigI = get_bin(t_uniform, t_left, t_right, x, mask, datatype)
    out[:, 3] = I_avg
    out[:, 4] = sigI        
    f_out = '%s/%s_merged_%d_to_%d_fs.txt'%(out_path, label, t_left, t_right)
    numpy.savetxt(f_out, out, fmt='%6d%6d%6d%15.2f%15.2f')  
    
    # POSITIVE DELAYS 
    print '\nPOSITIVE DELAYS'
    t_left = 22
    t_right = 381
    I_avg, sigI = get_bin(t_uniform, t_left, t_right, x, mask, datatype)
    out[:, 3] = I_avg
    out[:, 4] = sigI        
    f_out = '%s/%s_merged_%d_to_%d_fs.txt'%(out_path, label, t_left, t_right)
    numpy.savetxt(f_out, out, fmt='%6d%6d%6d%15.2f%15.2f')  


   
def export_merged_data_dark(settings):    
    data_path = settings.data_path
    data_file = settings.data_file
    results_path = settings.results_path
    datatype = settings.datatype
    label = settings.label
    
    mult_factor = 100
    
    out_path = '%s/merged'%results_path
    if not os.path.exists(out_path):
        os.mkdir(out_path)
       
    # EXTRACT DATA
    T_sparse = joblib.load(data_file)
    print 'T_sparse: ', T_sparse.shape
    print 'T_sparse nonzero: ', T_sparse.count_nonzero()
    x = T_sparse[:,:].todense()
    print 'x: ', x.shape, x.dtype
    x = numpy.asarray(x, dtype=datatype)
    print 'x: ', x.shape, x.dtype
    nnans = numpy.count_nonzero(numpy.isnan(x))
    print 'n of nans in x: ', nnans
    
    x = mult_factor*x
    
    M_sparse = joblib.load('%s/M_sparse_%s.jbl'%(data_path, label))
    print 'M_sparse: ', M_sparse.shape
    print 'M_sparse nonzero: ', M_sparse.count_nonzero()
    mask = M_sparse[:,:].todense()
    print 'mask: ', mask.shape, mask.dtype
    mask = numpy.asarray(mask, dtype=numpy.uint8)
    print 'mask: ', mask.shape, mask.dtype
    
    I_avg, sigI = get_merged_data(x, mask, datatype)
    
    miller_h = joblib.load('%s/miller_h_%s.jbl'%(data_path, label))
    miller_k = joblib.load('%s/miller_k_%s.jbl'%(data_path, label))
    miller_l = joblib.load('%s/miller_l_%s.jbl'%(data_path, label))
    
    out = numpy.zeros((miller_h.shape[0], 5))
    out[:, 0] = miller_h.flatten()
    out[:, 1] = miller_k.flatten()
    out[:, 2] = miller_l.flatten()
    out[:, 3] = I_avg
    out[:, 4] = sigI
    
    f_out = '%s/%s_merged.txt'%(out_path, label)
    numpy.savetxt(f_out, out, fmt='%6d%6d%6d%15.2f%15.2f')  