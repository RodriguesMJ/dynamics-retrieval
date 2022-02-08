# -*- coding: utf-8 -*-
import numpy
import time
import joblib
import matplotlib.pyplot

 
def main(settings):
    results_path = settings.results_path
    f_max = settings.f_max
    n_lp_vectors = 2*f_max + 1
    ATA_lp_filter = numpy.zeros((n_lp_vectors, n_lp_vectors))
    print ATA_lp_filter.shape, ATA_lp_filter.dtype
    start = time.time()
    for i in range(0, n_lp_vectors):
        print i
        chunck = joblib.load('%s/ATA_lp_filter/chunck_%d.jbl'%(results_path, i))
        ATA_lp_filter += chunck
    print 'Time:', time.time() - start
    joblib.dump(ATA_lp_filter, '%s/ATA_lp_filter.jbl'%(results_path))
    matplotlib.pyplot.imshow(numpy.log10(abs(ATA_lp_filter)))
    matplotlib.pyplot.savefig('%s/log_10_abs_ATA_lp_filter.png'%(results_path))
    
#test = joblib.load('/das/work/p17/p17491/Cecilia_Casadei/NLSA/data_bR_2/results_NLSA/bR_light_lp_filter/ATA_lp_filter.jbl')