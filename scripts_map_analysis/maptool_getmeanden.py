# -*- coding: utf-8 -*-
#import scipy.io
import numpy
import joblib
import h5py

def get_meanden(mode):
    
    radius = 1.7
    distance = 0.2
    sigmacutoffs = [3.0, 3.5, 4.0, 4.5]
    

    here = '/das/work/p18/p18594/cecilia-offline/NLSA/data_rho_2/results_NLSA/map_analysis';
    fn = '%s/results_m_0_%d/mapd0.mat'%(here, mode)

    #Calculate mean positive densities
    print 'Loading'
    
    f = h5py.File(fn, 'r') 
    mapd0 = numpy.asarray(f['/mapd0'], dtype=numpy.float32)
    print 'mapd0 :', mapd0.shape, mapd0.dtype
    
    
    for sigmacutoff in sigmacutoffs:
        print 'Sigma cutoff: ', sigmacutoff
        meanposden = numpy.mean(numpy.where(mapd0 <  sigmacutoff, 0, mapd0), axis=0)
        meannegden = numpy.mean(numpy.where(mapd0 > -sigmacutoff, 0, mapd0), axis=0)
        joblib.dump(meanposden, 
                    '%s/results_m_0_%d/meanposden_m_0_%d_radius_%.1f_dist_%.1f_sigcutoff_%.1f.jbl'
                    %(here, mode, mode, radius, distance, sigmacutoff))
        joblib.dump(meannegden, 
                    '%s/results_m_0_%d/meannegden_m_0_%d_radius_%.1f_dist_%.1f_sigcutoff_%.1f.jbl'
                    %(here, mode, mode, radius, distance, sigmacutoff))
        print 'meanposden: ', meanposden.shape, meanposden.dtype
        print 'meannegden: ', meannegden.shape, meannegden.dtype
    
if __name__ == "__main__":
    m = 2
    get_meanden(m)