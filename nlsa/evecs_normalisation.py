# -*- coding: utf-8 -*-
import joblib
import numpy

import settings_rho_light as settings


def normalise(evecs, mu):
    norm = numpy.asarray([mu, ]*evecs.shape[1])
    norm = norm.T    
    norm = numpy.sqrt(norm)
    evecs_norm = evecs / norm
    return evecs_norm

def get_mu_SFX(evecs):
    mu = evecs[:,0]*evecs[:,0]
    return mu

if __name__ == '__main__':
    
    label = '_sym_ARPACK'    
    results_path = settings.results_path
    
#    if label == '_sym_ARPACK':
#        mu_label = '_sym'
#    else:
#        mu_label = label
        
    #mu = joblib.load('%s/mu_P%s.jbl'%(results_path, mu_label))
    evecs = numpy.real(joblib.load('%s/P%s_evecs_sorted.jbl'%(results_path, label)))
    
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
    joblib.dump(evecs_norm, '%s/P%s_evecs_normalised.jbl'%(results_path, label))
    
    print '\nSave mu'
    joblib.dump(mu, '%s/mu.jbl'%(results_path))
    
#    P_sym = joblib.load('%s/P_sym.jbl'%results_path)
#    mu_T = mu.T
#    dot = numpy.dot(mu_T, P_sym)
#    diff = dot - mu_T
#    print numpy.amax(diff), numpy.amin(diff)