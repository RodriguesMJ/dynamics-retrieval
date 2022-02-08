# -*- coding: utf-8 -*-
import matplotlib
matplotlib.use('Agg') # Force matplotlib to not use any Xwindows backend.
matplotlib.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot
import joblib
import numpy
import os

def normalise(evecs, mu):
    norm = numpy.asarray([mu, ]*evecs.shape[1])
    norm = norm.T    
    norm = numpy.sqrt(norm)
    evecs_norm = evecs / norm
    return evecs_norm

def get_mu_SFX(evecs):
    mu = evecs[:,0]*evecs[:,0]
    return mu

def main(settings):
    
    #label = settings.eigenlabel  
    results_path = settings.results_path
    
#    if label == '_sym_ARPACK':
#        mu_label = '_sym'
#    else:
#        mu_label = label
        
    #mu = joblib.load('%s/mu_P%s.jbl'%(results_path, mu_label))
    evecs = numpy.real(joblib.load('%s/P_evecs_sorted.jbl'%(results_path)))
    
#    s = evecs.shape[0]
#    print 'Size: ', s
#    
#    print 'Test: Evecs * Evecs.T'
#    test = numpy.matmul(evecs, evecs.T)
#    diff = test - numpy.eye(s)
#    print '%.4f  %.4f'%(numpy.amax(diff), numpy.amin(diff))
#    
#    print 'Test: Evecs.T * Evecs'
#    test = numpy.matmul(evecs.T, evecs)
#    diff = test - numpy.eye(test.shape[0])
#    print '%.4f  %.4f'%(numpy.amax(diff), numpy.amin(diff))
#    
#    print 'Test: diag(mu) * Evecs * Evecs.T'
#    test = numpy.matmul(numpy.diag(mu), numpy.matmul(evecs, evecs.T))
#    diff = test - numpy.eye(s)
#    print '%.4f  %.4f'%(numpy.amax(diff), numpy.amin(diff))
#    
#    evecs_norm = normalise(evecs, mu)
#    
#    print 'Test: diag(mu) * Evecs_norm * Evecs_norm.T'
#    test = numpy.matmul(numpy.diag(mu), 
#                        numpy.matmul(evecs_norm, evecs_norm.T))
#    diff = test - numpy.eye(s)
#    print '%.4f  %.4f'%(numpy.amax(diff), numpy.amin(diff))

    
    mu = get_mu_SFX(evecs)
    evecs_norm = normalise(evecs, mu)
    
    print '\nSave normalised eigenvectors'
    joblib.dump(evecs_norm, '%s/P_evecs_normalised.jbl'%(results_path))
    
    print '\nSave mu'
    joblib.dump(mu, '%s/mu.jbl'%(results_path))
    
#    P_sym = joblib.load('%s/P_sym.jbl'%results_path)
#    mu_T = mu.T
#    dot = numpy.dot(mu_T, P_sym)
#    diff = dot - mu_T
#    print numpy.amax(diff), numpy.amin(diff)

# #def main_part_1(settings):
# import settings_bR_light as settings

# results_path = settings.results_path

# figure_path = '%s/P_evecs'%(results_path)
# if not os.path.exists(figure_path):
#     os.mkdir(figure_path)

# print '\nImport the eigenvectors of P'
# evecs = numpy.real(joblib.load('%s/P%s_evecs_sorted.jbl'%(results_path, settings.eigenlabel)))
# print 'evecs: ', evecs.shape, evecs.dtype

# print '\nPlot evecs'
# for i in range(evecs.shape[1]):
#     phi = evecs[:,i]
#     matplotlib.pyplot.figure(figsize=(30,10))
#     ax = matplotlib.pyplot.gca()
#     ax.tick_params(axis='x', labelsize=25)
#     ax.tick_params(axis='y', labelsize=25)
#     matplotlib.pyplot.plot(range(evecs.shape[0]), phi, 'o-', color='indigo', markersize=2)
#     matplotlib.pyplot.savefig('%s/P%s_evec_%d.png'%(figure_path, settings.eigenlabel, i), dpi=2*96)
#     matplotlib.pyplot.close()
    
# evals_sorted = joblib.load('%s/P%s_evals_sorted.jbl'%(results_path, settings.eigenlabel))    
# matplotlib.pyplot.scatter(range(evals_sorted.shape[0]), evals_sorted, c='b')
# matplotlib.pyplot.axhline(y=0, xmin=0, xmax=1, c='k')
# matplotlib.pyplot.savefig('%s/P_eigenvalues.png'%(figure_path))
# matplotlib.pyplot.close()

# print '\nTest: evecs.T * evecs'
# test = numpy.matmul(evecs.T, evecs)
# diff = test - numpy.eye(test.shape[0])
# print '%.4f  %.4f'%(numpy.amax(diff), numpy.amin(diff))

# matplotlib.pyplot.imshow(test, cmap='jet')
# matplotlib.pyplot.colorbar()
# matplotlib.pyplot.savefig('%s/evecs_P_T_evecs_P.png'%results_path)
# matplotlib.pyplot.close()

# print '\nTest: evec_0 * evec_0.T'  
# e_0 = evecs[:,0]
# print e_0.shape  
# test = numpy.matmul(e_0[:, numpy.newaxis], e_0[numpy.newaxis, :])
# print test.shape
# print '%.4f  %.4f'%(numpy.amax(test), numpy.amin(test))
# print '%.4f  %.4f'%(numpy.amax(numpy.diag(test)), numpy.amin(numpy.diag(test)))
    
    # print '\nGet the eigenvectors of P'
    # Q = joblib.load('%s/Q.jbl'%results_path)
    # print 'Q: ', Q.shape, Q.dtype
    # sqrt_Q = numpy.sqrt(Q)
    # inv_sqrt_Q = 1.0/sqrt_Q
    # print 'inv_sqrt_Q: ', inv_sqrt_Q.shape, inv_sqrt_Q.dtype
    # evecs_P = numpy.matmul(numpy.diag(inv_sqrt_Q), evecs_Psym)
    # print 'evecs_P: ', evecs_P.shape, evecs_P.dtype
    
    # joblib.dump(evecs_P, '%s/P_evecs_sorted.jbl'%(results_path))
    
    # print '\nTest: evecs_P.T * evecs_P'
    # test = numpy.matmul(evecs_P.T, evecs_P)
    # diff = test - numpy.eye(test.shape[0])
    # print '%.4f  %.4f'%(numpy.amax(diff), numpy.amin(diff))
    # print diff
    
    # print '\nPlot evecs_P'
    # for i in range(evecs_P.shape[1]):
    #     phi = evecs_P[:,i]
    #     matplotlib.pyplot.figure(figsize=(30,10))
    #     ax = matplotlib.pyplot.gca()
    #     ax.tick_params(axis='x', labelsize=25)
    #     ax.tick_params(axis='y', labelsize=25)
    #     matplotlib.pyplot.plot(range(evecs_P.shape[0]), phi, 'bo-', markersize=2)
    #     matplotlib.pyplot.savefig('%s/P_evec_%d.png'%(figure_path, i), dpi=2*96)
    #     matplotlib.pyplot.close()
     
# print 'Check that evecs are eigenvectors of P:'
#P = joblib.load('%s/P.jbl'%results_path)
# left = numpy.matmul(P, evecs)
# right = numpy.matmul(evecs, numpy.diag(evals_sorted))
# diff = left - right
# print '%.4f  %.4f'%(numpy.amax(diff), numpy.amin(diff))
    
# print '\nGet mu'
# mu = get_mu_SFX(evecs)
# # Q = joblib.load('%s/Q.jbl'%results_path)
# # print 'Q: ', Q.shape, Q.dtype
# # sum_Q = Q.sum()
# # mu = Q/sum_Q
# # print 'mu: ', mu.shape, mu.dtype
# # print 'Check mu P = mu:'
# # diff = numpy.matmul(mu.T, P) - mu.T
# # print 'diff: ', diff.shape, diff.dtype
# # print '%.4f  %.4f'%(numpy.amax(diff), numpy.amin(diff))

# joblib.dump(mu, '%s/mu.jbl'%(results_path))

# matplotlib.pyplot.figure(figsize=(30,10))
# ax = matplotlib.pyplot.gca()
# ax.tick_params(axis='x', labelsize=25)
# ax.tick_params(axis='y', labelsize=25)
# matplotlib.pyplot.plot(range(mu.shape[0]), mu, 'bo-', markersize=2)
# matplotlib.pyplot.savefig('%s/mu.png'%(figure_path), dpi=2*96)
# matplotlib.pyplot.close()

# evecs_norm = normalise(evecs, mu)
# print '\nPlot evecs'
# for i in range(evecs.shape[1]):
#     phi = evecs_norm[:,i]
#     matplotlib.pyplot.figure(figsize=(30,10))
#     ax = matplotlib.pyplot.gca()
#     ax.tick_params(axis='x', labelsize=25)
#     ax.tick_params(axis='y', labelsize=25)
#     matplotlib.pyplot.plot(range(evecs.shape[0]), phi, 'o-', color='indigo', markersize=2)
#     matplotlib.pyplot.savefig('%s/P%s_evec_%d_norm.png'%(figure_path, settings.eigenlabel, i), dpi=2*96)
#     matplotlib.pyplot.close()


# print '\nCheck: mu Phi_norm Phi_norm.T'
# test = numpy.matmul(numpy.diag(mu), evecs_norm)
# test = numpy.matmul(test, evecs_norm.T)
#     #test = test - numpy.eye(test.shape[0])
# print 'test: ', test.shape, test.dtype
# print '%.4f  %.4f'%(numpy.amax(test), numpy.amin(test))
# print '%.4f  %.4f'%(numpy.amax(numpy.diag(test)), numpy.amin(numpy.diag(test)))
# matplotlib.pyplot.imshow(test, cmap='jet')
# matplotlib.pyplot.colorbar()
# matplotlib.pyplot.savefig('%s/mu_Phi_norm_Phi_norm_T.png'%results_path)
# matplotlib.pyplot.close()   

# def main_part_2(settings): 
#     results_path = settings.results_path
    
#     mu = joblib.load('%s/mu.jbl'%(results_path))
#     #evecs_P = joblib.load('%s/P_evecs_sorted.jbl'%(results_path))
#     evecs_P = numpy.real(joblib.load('%s/P%s_evecs_sorted.jbl'%(results_path, settings.eigenlabel)))
#     print 'mu', numpy.amax(mu), numpy.amin(mu)
    
#     print '\nCheck: mu Phi Phi.T'
#     test = numpy.matmul(numpy.diag(mu), evecs_P)
#     test = numpy.matmul(test, evecs_P.T)
#     #test = test - numpy.eye(test.shape[0])
#     print 'test: ', test.shape, test.dtype
#     print '%.4f  %.4f'%(numpy.amax(test), numpy.amin(test))
#     print '%.4f  %.4f'%(numpy.amax(numpy.diag(test)), numpy.amin(numpy.diag(test)))
   




