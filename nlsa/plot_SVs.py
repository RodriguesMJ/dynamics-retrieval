# -*- coding: utf-8 -*-
import joblib
import matplotlib.pyplot
import os
import numpy

def main(settings):

    results_path = settings.results_path
        
    S = joblib.load('%s/S.jbl'%results_path)
    nmodes = S.shape[0]
    print 'nmodes: ', nmodes
    print S
    
    out_folder = '%s/chronos'%results_path
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    
    matplotlib.pyplot.figure()
    matplotlib.pyplot.scatter(range(nmodes), S, s=1)
    matplotlib.pyplot.savefig('%s/SVs.png'%out_folder, dpi=96*2)
    matplotlib.pyplot.close()
    
    s0 = S[0]
    S_norm = S/s0
    
    matplotlib.pyplot.figure()
    matplotlib.pyplot.scatter(range(nmodes), numpy.log10(S_norm), s=1)
    matplotlib.pyplot.savefig('%s/SVs_norm_log.png'%out_folder, dpi=96*2)
    matplotlib.pyplot.close()