# -*- coding: utf-8 -*-
import numpy

label = 'light'

root_folder = '/das/work/p18/p18594/cecilia-offline/NLSA/data_rho_2'
data_path = '%s/data_converted_%s'%(root_folder, label)
results_path = '%s/results_NLSA/rho_%s'%(root_folder, label)

datatype = numpy.float32

t_d_y = 0.244
t_correction_factor = 0.174
# #q = 16384   # Concatenation n.
# #paral_step = 1000 # with 17 proc

# q = 32768
# paral_step = 1000 # with 33 proc

# #
# b = 15000
# #
# #n = 2
# #sigma_opt = 46843.5 #47388.4 
# #sigma_sq = (n*sigma_opt)**2
# #
# #l = 20     # N. diffusion map eigenvectors
# #
# #nmodes = 3 # N. SVD modes   max nmodes = l
# #toproject = [0, 18, 19]
# #paral_step_A = 1400 # with 12 proc
# #
# #ncopies = 1000
# #paral_step_reconstruction = 12000 