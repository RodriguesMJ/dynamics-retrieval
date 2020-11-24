# -*- coding: utf-8 -*-
import joblib
import numpy
import matplotlib.pyplot

import settings_rho_light as settings
results_path = settings.results_path
label = settings.label

D = joblib.load('%s/D.jbl'%results_path)
D_sq = D*D

log_epsilon_list = numpy.linspace(7, 11, num=50)

epsilons = []
sigmas = []
values = []
for log_epsilon in log_epsilon_list:
    print log_epsilon
    epsilon = 10**log_epsilon
    sigma = numpy.sqrt(2*epsilon)
    
    exponent = D_sq/(2*epsilon)
    W = numpy.exp(-exponent)
    value =  W.sum()
    
    epsilons.append(epsilon)
    sigmas.append(sigma)
    values.append(value)

values = numpy.asarray(values)    
log_values = numpy.log10(values)

max_val = numpy.amax(log_values)
min_val = numpy.amin(log_values)
opt_val = (max_val+min_val)/2

diff = abs(log_values - opt_val)
idx = numpy.argmin(diff)
print idx

epsilon_opt = epsilons[idx]
sigma_opt = sigmas[idx]
log_epsilon_opt = log_epsilon_list[idx]
print 'Epsilon_opt = (sigma_opt**2) / 2: ', epsilon_opt
print 'Log(epsilon_opt): ', log_epsilon_opt
print 'Sigma_opt: ', sigma_opt

matplotlib.pyplot.plot(log_epsilon_list, log_values)
matplotlib.pyplot.xlabel(r"log$_{10}\epsilon$")
matplotlib.pyplot.ylabel(r"log$_{10}\sum W_{ij}$")
matplotlib.pyplot.axvline(x=log_epsilon_opt, ymin=0, ymax=1)
matplotlib.pyplot.savefig('%s/bilog_%s.png'%(results_path, label))
matplotlib.pyplot.close()

# bR light - MATLAB
# sigma_opt ~ 7000
# epsilon_opt = (sigma**2)/2 = 24500000
# log10(epsilon_opt) = 7.38