# -*- coding: utf-8 -*-
import joblib
import numpy
import numpy.linalg

def SVD_f(A):
    print 'SVD'
    U, S, VH = numpy.linalg.svd(A, full_matrices=False)
    print 'done'
    return U, S, VH

def SVD_f_manual(A):
    print 'SVD_manual'
    AtA = numpy.matmul(A.T, A)
    print 'A.T A: ', AtA.shape
    print 'Eigendecompose'
    evals_AtA, evecs_AtA = numpy.linalg.eigh(AtA)
    print 'Done'
    evals_AtA[numpy.argwhere(evals_AtA<0)]=0  
    S = numpy.sqrt(evals_AtA)
    VH = evecs_AtA.T
    U_temp = numpy.matmul(evecs_AtA, numpy.diag(1.0/S))
    U = numpy.matmul(A, U_temp)     
    return U, S, VH

def sorting(U, S, VH):
    print 'Sorting'
    sort_idxs = numpy.argsort(S)[::-1]
    S_sorted = S[sort_idxs]
    VH_sorted = VH[sort_idxs,:]
    U_sorted = U[:,sort_idxs]
    return U_sorted, S_sorted, VH_sorted

def project_chronos(VH, Phi):
    VT_final = numpy.matmul(VH, Phi.T)
    return VT_final

def project_chronos_loop(V_temp, Phi):
    s = Phi.shape[0]
    nmodes = Phi.shape[1]
    V_temp_T = V_temp.T    
    print 's, nmodes: ', s, nmodes    
    VT_final = numpy.zeros((s,nmodes))
    for k in range(nmodes):
        print k
        for i in range(s):
            VT_final[i,k] = numpy.sum(numpy.multiply(Phi[i,:], V_temp_T[:,k]))
    VT_final = VT_final.T
    print 'VT_final_loop: ', VT_final.shape
    return VT_final

def main(settings):    
    print '\n****** RUNNING SVD ******'
    
    results_path = settings.results_path
    toproject = settings.toproject
    datatype = settings.datatype

    A = joblib.load('%s/A_parallel.jbl'%results_path)
    print 'Loaded'
    U, S, VH = SVD_f_manual(A)
    U, S, VH = sorting(U, S, VH)
    
    print 'Done'
    print 'U: ', U.shape
    print 'S: ', S.shape
    print 'VH: ', VH.shape

    joblib.dump(U[:,0:20], '%s/U.jbl'%results_path)
    joblib.dump(S, '%s/S.jbl'%results_path)
    joblib.dump(VH, '%s/VH.jbl'%results_path)
        
    evecs = joblib.load('%s/evecs_sorted.jbl'%(results_path))

    nmodes = VH.shape[0]
    print 'nmodes: ', nmodes

    Phi = numpy.zeros((evecs.shape[0], nmodes), dtype=datatype)
    for j in range(nmodes):
        ev_toproject = toproject[j]
        print 'Evec: ', ev_toproject
        Phi[:, j] = evecs[:, ev_toproject]
    
    VT_final = project_chronos(VH, Phi)    
    print 'VT_final: ', VT_final.shape
    joblib.dump(VT_final, '%s/VT_final.jbl'%results_path)
