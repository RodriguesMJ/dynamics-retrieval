# -*- coding: utf-8 -*-
import pickle
import numpy

import settings

results_path = settings.results_path

f = open('%s/mu.pkl'%results_path, 'rb')
mu = pickle.load(f)
f.close()


f = open('%s/mu_P_sym.pkl'%results_path, 'rb')
mu_P_sym = pickle.load(f)
f.close()

diff = mu - mu_P_sym
print numpy.nanmax(diff), numpy.nanmin(diff)
