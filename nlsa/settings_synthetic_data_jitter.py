# -*- coding: utf-8 -*-
import numpy
import math

jitter_factor = 1.0
if jitter_factor == 0.3:
    test_n = 5   
if jitter_factor == 1.0:
    test_n = 6
if jitter_factor == 0.1:
    test_n = 7
if jitter_factor == 0.5:
    test_n = 8
root_f = '../../synthetic_data_jitter/test%d'%test_n

datatype = numpy.float64

# BUILD MODEL DATA
m = 7000
S = 30000
T_model = 26000 #S-q+1
tc = float(S)/2

#results_path = '%s/binning'%root_f
############
### LPSA ###
############

#PARAS SCANS
# results_path = '%s/LPSA_para_search'%root_f
# f_max_q_scan = 100
# q_f_max_scan = 2001

#SELECTED VALUES
q = 4001
f_max = 100
f_max_considered = f_max

# #A
# data_file = '%s/x.jbl'%results_path
# paral_step_A = 500
# n_workers_A = int(math.ceil(float(q)/paral_step_A))

#RECONSTRUCTION
modes_to_reconstruct = range(20) 
p = 0
results_path = '%s/LPSA_para_search/f_max_%d_q_%d/reconstruction_p_%d/x_r_SVD'%(root_f, f_max, q, p)
# STANDARD RECONSTRUCTION 
# p = (q-1)/2
# ncopies = q
# paral_step_reconstruction = 2000
# n_workers_reconstruction = int(math.ceil(float(S-q+1-ncopies+1)/paral_step_reconstruction))

# ############
# ### NLSA ###
# ############

# # D_sq
# q = 4001
# paral_step = 100
# n_workers = int(math.ceil(float(q)/paral_step))
# results_path = '%s/NLSA'%(root_f)
# b = 3000
# #results_path = '%s/NLSA/q_%d'%(root_f, q)
# log10eps = 1.0
# sigma_sq = 2*10**log10eps

# l = 50
# # # results_path = '%s/binning'%root_f
# results_path = '%s/NLSA/q_%d/b_%d/log10eps_%0.1f'%(root_f, q, b, log10eps)

# #data_file = '%s/x.jbl'%results_path
# paral_step_A = 200
# n_workers_A = int(math.ceil(float(q)/paral_step_A))
# nmodes = l 
# toproject = range(nmodes) 

# # # #RECONSTRUCTION
# modes_to_reconstruct = range(20) 
# p = 0
# #results_path = '%s/NLSA/q_%d/b_%d/log10eps_%0.1f/reconstruction_p_%d'%(root_f, q, b, log10eps, p)
# # # STANDARD RECONSTRUCTION 
# #p = (q-1)/2
# ncopies = q
# paral_step_reconstruction = 4000
# #results_path = '%s/NLSA/q_%d/b_%d/log10eps_%0.1f/reconstruction_p_%d'%(root_f, q, b, log10eps, p)
# # # # # # ## ???
# # # # # # #n_workers_reconstruction = int(math.ceil(float(S-q-ncopies+1)/paral_step_reconstruction))
# n_workers_reconstruction = int(math.ceil(float(S-q+1-ncopies+1)/paral_step_reconstruction))