# -*- coding: utf-8 -*-
import numpy
import pickle
import time

import settings

results_path = settings.results_path
b = settings.b
datatype = settings.datatype
q = settings.q

f = open('%s/T_anomaly.pkl'%(results_path), 'rb')
x = pickle.load(f)
f.close()

dx = x[:,1:]-x[:,:-1]
print x.shape, dx.shape

f = open('%s/D_all_bigdata.pkl'%results_path, 'rb')
D_all = pickle.load(f)
f.close()

f = open('%s/D_bigdata.pkl'%results_path, 'rb')
D = pickle.load(f)
f.close()

f = open('%s/N_bigdata.pkl'%results_path, 'rb')
N = pickle.load(f)
f.close()

f = open('%s/velocities_bigdata.pkl'%results_path, 'rb')
v = pickle.load(f)
f.close()

s = D.shape[0]


#cosTheta = numpy.zeros((s, b), dtype=datatype)
#
#for i in range(s):
#    v_i = v[i]
#    for j in range(b):
#        idx_nn = N[i,j]
#        D_i_nn = D[i,j]
#        cosine = 0
#        for k in range(q):
#            cosine = cosine + numpy.inner(x[i+1+k,:]-x[i+k,:], x[i+1+k,:]-x[idx_nn+1+k,:])
#        cosine = cosine  / (v_i * D_i_nn)
#        
#        cosTheta[i,j] = cosine

cosTheta = numpy.nan*numpy.ones((s, s), dtype=datatype)
starttime = time.time()
for i in range(s):
    v_i = v[i]
    print i
    for j in range(b):
        print i, j
        
        if i != j:
            idx_nn = N[i,j]
            D_i_nn = D_all[i,idx_nn]
#            test = D[i,j]
#            if abs(D_i_nn - test) > 0.00001:
#                print 'Problem'
            #cosine = 0 # i, nn
            cosine_2 = 0
            for k in range(q):
#                cosine = cosine + numpy.inner(x[:,i+1+k]-x[:,i+k], 
#                                              x[:,i+1+k]-x[:,idx_nn+1+k])
                cosine_2 = cosine_2 + numpy.inner(dx[:,i+k], 
                                                  x[:,i+1+k]-x[:,idx_nn+1+k])

            #cosine = cosine  / (v_i * D_i_nn)
            cosine_2 = cosine_2  / (v_i * D_i_nn)
            #print cosine - cosine_2
           
            cosTheta[i,idx_nn] = cosine_2
            
            
            v_nn = v[idx_nn]
            D_nn_i = D_all[idx_nn,i]
#            if abs(D_nn_i - D_i_nn) > 0.00001:
#                print 'Problem'
            #cosine = 0 # nn, i
            cosine_2 = 0
            for k in range(q):
#                cosine = cosine + numpy.inner(x[:,idx_nn+1+k]-x[:,idx_nn+k], 
#                                              x[:,idx_nn+1+k]-x[:,i+1+k])
                cosine_2 = cosine_2 + numpy.inner(dx[:,idx_nn+k], 
                                              x[:,idx_nn+1+k]-x[:,i+1+k])
            #cosine = cosine  / (v_nn * D_nn_i)
            cosine_2 = cosine_2  / (v_nn * D_nn_i)
            #print cosine - cosine_2
           
            cosTheta[idx_nn,i] = cosine_2
            
            
print time.time() - starttime
           
f = open('%s/cosines.pkl'%results_path, 'wb')
pickle.dump(cosTheta, f)
f.close()


#cosTheta = numpy.ones((s, s), dtype=datatype)
#
#for i in range(s):
#    v_i = v[i]
#    print i
#    for j in range(s):
#        if (j%500 == 0):
#            print i, j, '/', s
#        if i != j:
#        
#            D_i_j = D[i,j]
#            cosine = 0
#            for k in range(q):
#                cosine = cosine + numpy.inner(x[i+1+k,:]-x[i+k,:], 
#                                              x[i+1+k,:]-x[j+1+k,:])
#            #print cosine, v_i*D_i_j
#            cosine = cosine  / (v_i * D_i_j)
#            #print cosine
#            
#            cosTheta[i,j] = cosine