# -*- coding: utf-8 -*-
import joblib
import numpy
import os

import settings_NP as settings

results_path = settings.results_path


mu_P_sym = joblib.load('%s/mu_P_sym.jbl'%results_path)

if numpy.all(mu_P_sym > 0):
    print 'All positive'
if numpy.all(mu_P_sym <= 0):
    print 'All negative - change sign'
    mu_P_sym = -mu_P_sym
    
os.system('rm %s/mu_P_sym.jbl'%results_path)
    
joblib.dump(mu_P_sym, '%s/mu_P_sym.jbl'%results_path)


#evals_P_sym_left = joblib.load('%s/evals_P_sym_left.jbl'%results_path)
#print evals_P_sym_left[0], evals_P_sym_left[-1]