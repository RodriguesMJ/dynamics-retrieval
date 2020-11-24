# -*- coding: utf-8 -*-
import joblib
import numpy
import numpy.linalg

import settings_rho_light as settings


def SVD_f(A):
    print 'SVD'
    U, S, VH = numpy.linalg.svd(A, full_matrices=False)
    return U, S, VH

def SVD_f_manual(A):
    print 'SVD_manual'
    AtA = numpy.matmul(A.T, A)
    print 'A.T A: ', AtA.shape
    print 'Eigendecompose'
    evals_AtA, evecs_AtA = numpy.linalg.eig(AtA)
    print 'Done'    
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

if __name__ == '__main__':
    
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

    joblib.dump(U, '%s/U.jbl'%results_path)
    joblib.dump(S, '%s/S.jbl'%results_path)
    joblib.dump(VH, '%s/VH.jbl'%results_path)
    
#    U_m, S_m, VH_m = SVD_f_manual(A)
#    U_m, S_m, VH_m = sorting(U_m, S_m, VH_m)
#    
#    print 'Done'
#    print 'U: ', U_m.shape
#    print 'S: ', S_m.shape
#    print 'VH: ', VH_m.shape
#    
#    diff = abs(U)-abs(U_m)
#    print numpy.amax(abs(diff))
#    diff = S-S_m
#    print numpy.amax(abs(diff))
#    diff = abs(VH)-abs(VH_m)
#    print numpy.amax(abs(diff))
        
    P_evecs_norm = joblib.load('%s/P_sym_ARPACK_evecs_normalised.jbl'%(results_path))
#    
#    VH = joblib.load('%s/VH.jbl'%results_path)
    nmodes = VH.shape[0]
    print 'nmodes: ', nmodes

    #Phi = P_evecs_norm[:,0:nmodes]  
    Phi = numpy.zeros((P_evecs_norm.shape[0], nmodes), dtype=datatype)
    for j in range(nmodes):
        ev_toproject = toproject[j]
        print 'Evec: ', ev_toproject
        Phi[:, j] = P_evecs_norm[:, ev_toproject]
    
    VT_final = project_chronos(VH, Phi)    
    print 'VT_final: ', VT_final.shape
    joblib.dump(VT_final, '%s/VT_final.jbl'%results_path)
    
#    VT_final_loop = project_chronos(VH, Phi)    
#    print 'VT_final (loop): ', VT_final_loop.shape
#    joblib.dump(VT_final_loop, '%s/VT_final_loop.jbl'%results_path)
#    
#    diff = abs(VT_final)-abs(VT_final_loop)
#    print numpy.amax(abs(diff))