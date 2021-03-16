# -*- coding: utf-8 -*-
import numpy

results_path = '../data_bR/bR_light_q_16384_b_5000'
label = 'light'

q = 16384   # Concatenation n.

#b = 1000    # Nearest neighbor n.
b = 5000

datatype = numpy.float32

paral_step = 1000 # with 17 proc

n = 4
sigma_opt = 6574.4 
sigma_sq = (n*sigma_opt)**2


l = 10     # N. diffusion map eigenvectors

nmodes = 6 # N. SVD modes   max nmodes = l
toproject = [0, 1, 2, 7, 8, 9]
paral_step_A = 1400 # with 12 proc

ncopies = 1000
paral_step_reconstruction = 10000 