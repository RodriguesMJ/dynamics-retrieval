# -*- coding: utf-8 -*-
import joblib
import matplotlib
matplotlib.use('Agg') # Force matplotlib to not use any Xwindows backend.
matplotlib.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot
import os

import settings_rho_light as settings


print '****** RUNNING plot_P_evecs ******'
results_path = settings.results_path
l = settings.l
label = '_sym_ARPACK'

evals_sorted = joblib.load('%s/P%s_evals_sorted.jbl'%(results_path, label))
evecs_sorted = joblib.load('%s/P%s_evecs_normalised.jbl'%(results_path, label))

figure_path = '%s/P%s_evecs'%(results_path, label)
if not  os.path.exists(figure_path):
    os.mkdir(figure_path)

s = evecs_sorted.shape[0]
for i in range(l):
    print i
    phi = evecs_sorted[:,i]
    matplotlib.pyplot.figure(figsize=(30,10))
    ax = matplotlib.pyplot.gca()
    ax.tick_params(axis='x', labelsize=25)
    ax.tick_params(axis='y', labelsize=25)
    matplotlib.pyplot.plot(range(s), phi, 'o-', markersize=2)
    matplotlib.pyplot.savefig('%s/P%s_evec_%d_normalised.png'%(figure_path, label, i), dpi=2*96)
    matplotlib.pyplot.close()
    
#for i in range(l):
#    print i
#    phi = evecs_sorted[:,i]
#    matplotlib.pyplot.figure(figsize=(30,10))
#    #matplotlib.pyplot.plot(range(1200), phi[0:1200], 'o-', markersize=1)
#    matplotlib.pyplot.plot(range(s), phi[0:s], 'o-', markersize=1)
#    matplotlib.pyplot.savefig('%s/P%s_evec_%d.png'%(figure_path, label, i), dpi=2*96)
#    matplotlib.pyplot.close()    
    
matplotlib.pyplot.scatter(range(l), evals_sorted[0:l])
matplotlib.pyplot.savefig('%s/P%s_eigenvalues.png'%(figure_path, label))
matplotlib.pyplot.close()