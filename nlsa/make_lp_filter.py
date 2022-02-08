# -*- coding: utf-8 -*-

import numpy
import matplotlib.pyplot
import scipy.linalg
import joblib



def plot(settings, M, label):
       
    fig = matplotlib.pyplot.figure(figsize=(35,40))
    ax = fig.add_subplot(6, 1, 1)
    ax.plot(range(M.shape[0]), M[:,0], color='m')    
    for j in range(1, 6):
        ax = fig.add_subplot(6, 1, j+1)
        ax.plot(range(M.shape[0]), M[:,2*j-1], color='m')
        ax.plot(range(M.shape[0]), M[:,2*j],   color='b')
        ax.text(0.01, 0.1, 'j=%d'%j, fontsize=26, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
    matplotlib.pyplot.savefig('%s/lp_filter_functions%s_0to5.png'%(settings.results_path, label))
    matplotlib.pyplot.close()  
    
    fig = matplotlib.pyplot.figure(figsize=(35,40))
    ax = fig.add_subplot(11, 1, 1)
    ax.plot(range(M.shape[0]), M[:,0], color='m')    
    for j in range(10, 101, 10):
        ax = fig.add_subplot(11, 1, j/10+1)
        ax.plot(range(M.shape[0]), M[:,2*j-1], color='m')
        ax.plot(range(M.shape[0]), M[:,2*j],   color='b')
        ax.text(0.01, 0.1, 'j=%d'%j, fontsize=26, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
    matplotlib.pyplot.savefig('%s/lp_filter_functions%s_0to100.png'%(settings.results_path, label))
    matplotlib.pyplot.close()  
    
    # fig = matplotlib.pyplot.figure(figsize=(35,40))
    # ax = fig.add_subplot(11, 1, 1)
    # ax.plot(range(M.shape[0]), M[:,0], color='m')    
    # for j in range(20, 201, 20):
    #     ax = fig.add_subplot(11, 1, j/20+1)
    #     ax.plot(range(M.shape[0]), M[:,2*j-1], color='m')
    #     ax.plot(range(M.shape[0]), M[:,2*j],   color='b')
    #     ax.text(0.01, 0.1, 'j=%d'%j, fontsize=26, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
    # matplotlib.pyplot.savefig('%s/lp_filter_functions%s_0to200.png'%(settings.results_path, label))
    # matplotlib.pyplot.close()  
    
    
    
def get_F(settings):
    S = settings.S
    q = settings.q  
    s = S-q
    
    t = numpy.asarray(range(s))
    T = 1*(s)
    omega = 2*numpy.pi / T
        
    # Make filter matrix F
    f_max = settings.f_max
    F = numpy.zeros((s,(2*f_max)+1))
    
    for i in range(0, f_max+1):
        if i == 0:
            lp_filter_cos_i = numpy.cos(i*omega*t)
            F[:,i] = lp_filter_cos_i
        else:
            lp_filter_cos_i = numpy.cos(i*omega*t)
            lp_filter_sin_i = numpy.sin(i*omega*t)
            F[:,2*i-1] = lp_filter_cos_i
            F[:,2*i]   = lp_filter_sin_i   
    plot(settings, F, '')       
    return F   


def normalise(v):
    mod_sq = numpy.inner(v,v)
    v = v/numpy.sqrt(mod_sq)
    return v

def on(settings, Z):    
    Z[:,0] = normalise(Z[:,0])
    #print 'Normalisation: %0.2f'%numpy.inner(Z[:,0], Z[:,0])
    
    for i in range(1, Z.shape[1]):
        #print 'i =', i
        for j in range(0, i):
            Z[:,i] = Z[:,i] - numpy.inner(Z[:,i], Z[:,j])*Z[:,j]
        Z[:,i] = normalise(Z[:,i])
        #print 'Normalisation: %0.2f'%numpy.inner(Z[:,i], Z[:,i])
        # for j in range(0, i):            
        #     print 'Orthogonality: %0.2f'%numpy.inner(Z[:,i], Z[:,j])
            
    plot(settings, Z, '_o_n')
    return Z

def on_modified(settings, Z):    
    Z[:,0] = normalise(Z[:,0])
    print 'Normalisation: %0.2f'%numpy.inner(Z[:,0], Z[:,0])
    
    for start in range(1, Z.shape[1]):
        print 'start =', start
        
        for j in range(start, Z.shape[1]):
            Z[:,j] = Z[:,j] - numpy.inner(Z[:,j], Z[:,start-1])*Z[:,start-1]
        Z[:,start] = normalise(Z[:,start])
        print 'Normalisation: %0.2f'%numpy.inner(Z[:,start], Z[:,start])
        for j in range(0, start):            
            print 'Orthogonality: %0.2f'%numpy.inner(Z[:,start], Z[:,j])
            
    plot(settings, Z, '_o_n_modified')
    return Z

def on_qr(settings, Z):
    q_ortho, r = scipy.linalg.qr(Z, mode='economic')
    plot(settings, q_ortho, '_qr')
    return q_ortho, r

def check_on(Z):
    diff = numpy.matmul(Z.T, Z) - numpy.eye(Z.shape[1])
    return diff

def main(settings):
    
    results_path = settings.results_path
    
    F = get_F(settings)
    
    d = check_on(F)
    print numpy.amax(abs(d))
    
    #F_on = on(settings, F)
    Q, R = on_qr(settings, F)
    
    d = check_on(Q)
    print numpy.amax(abs(d))
    
    joblib.dump(Q, '%s/F_on_qr.jbl'%results_path)
    
def check(settings):
    results_path = settings.results_path
    Q = joblib.load('%s/F_on_qr.jbl'%results_path)
    d = check_on(Q)
    print numpy.amax(abs(d))
    plot(settings, Q, '_qr_check')
    