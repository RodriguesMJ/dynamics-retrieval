# -*- coding: utf-8 -*-
import numpy

label = 'alldark'

root_folder = '/das/work/p18/p18594/cecilia-offline/NLSA/data_rho_2'
data_path = '%s/data_converted_%s'%(root_folder, label)
results_path = '%s/results_NLSA/rho_%s'%(root_folder, label)

datatype = numpy.float32

t_d_y = 0.244
t_correction_factor = 0.174
# q = 4096   # Concatenation n.
# paral_step = 1000 # with 5 proc


# b = 2000    # Nearest neighbor n.


# n = 2
# sigma_opt = 124559.6
# sigma_sq = (n*sigma_opt)**2

# l = 10     # N. diffusion map eigenvectors

# nmodes = 4 # N. SVD modes   max nmodes = l
# toproject = [0, 7, 8, 9]
# paral_step_A = 400


# ncopies = 1000
# paral_step_reconstruction = 2000