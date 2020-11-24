# -*- coding: utf-8 -*-
import numpy

label = 'dark'
results_path = '../data_bR/bR_dark_q_16384_b_5000'

q = 16384   # Concatenation n.

#b = 1000    # Nearest neighbor n.
b = 5000

datatype = numpy.float32

#paral_step = 1000 # with 17 proc
paral_step = 1500 # with 11 proc
paral_step_A = 1000

n = 4
sigma_opt = 3118.1 
sigma_sq = (n*sigma_opt)**2

l = 10     # N. diffusion map eigenvectors
nmodes = 3 # N. SVD modes   max nmodes = l

ncopies = 1000
paral_step_reconstruction = 10000 # With 9 processors