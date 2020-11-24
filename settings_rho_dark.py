# -*- coding: utf-8 -*-
import numpy

label = 'dark'
results_path = '../data_rho/rho_%s'%label

datatype = numpy.float32
q = 4096   # Concatenation n.
paral_step = 1000 # with 5 proc


b = 2000    # Nearest neighbor n.


n = 2
sigma_opt = 124559.6
sigma_sq = (n*sigma_opt)**2

l = 10     # N. diffusion map eigenvectors

nmodes = 4 # N. SVD modes   max nmodes = l
toproject = [0, 7, 8, 9]
paral_step_A = 400


ncopies = 1000
paral_step_reconstruction = 2000