# -*- coding: utf-8 -*-
import numpy

label = 'light'

root_folder = '/das/work/p17/p17491/Cecilia_Casadei/NLSA/data_bR_2'
data_path = '%s/converted_data_%s'%(root_folder, label)
results_path = '%s/results_NLSA/bR_%s'%(root_folder, label)
data_file = '%s/T_sparse_%s.jbl'%(data_path, label)

datatype = numpy.float32

q = 16384   # Concatenation n.
paral_step = 1000 # with 17 proc
n_workers = 17

n = 116732 # S-q+1 
# #b = 1000   # Nearest neighbor n.
# b = 5000



# n = 4
# sigma_opt = 6574.4 
# sigma_sq = (n*sigma_opt)**2


# l = 10     # N. diffusion map eigenvectors

# nmodes = 6 # N. SVD modes   max nmodes = l
# toproject = [0, 1, 2, 7, 8, 9]
# paral_step_A = 1400 # with 12 proc

# ncopies = 1000
# paral_step_reconstruction = 10000 