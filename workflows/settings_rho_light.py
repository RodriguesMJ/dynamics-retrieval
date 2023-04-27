# -*- coding: utf-8 -*-
import numpy

label = "light"

root_folder = "/das/work/p18/p18594/cecilia-offline/NLSA/data_rho_2"
data_path = "%s/data_converted_%s" % (root_folder, label)
results_path = "%s/results_NLSA/rho_%s" % (root_folder, label)
data_file = "%s/T_sparse_corr_%s.jbl" % (data_path, label)

datatype = numpy.float32

# Translational disorder correction parameters
t_d_y = 0.244
t_correction_factor = 0.174

# Calculation of distances between supervectors
q = 30000  # Concatenation n
paral_step = 1000  # with 30 proc
n_workers = 30
n = 183558  # N. samples, for merging D_sq
b = 15000  # Nearest neighbor n.

n = 2
sigma_opt = 1119.0
sigma_sq = (n * sigma_opt) ** 2

eigenlabel = "_sym_ARPACK"
l = 10  # N. diffusion map eigenvectors

nmodes = 3  # N. SVD modes   max nmodes = l
toproject = [0, 8, 9]
n_workers_A = 30
paral_step_A = 1000

ncopies = 1000
paral_step_reconstruction = 18000
n_workers_reconstruction = 11
modes_to_reconstruct = [1, 2]  # [0, 1, 2]
