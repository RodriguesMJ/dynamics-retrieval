# -*- coding: utf-8 -*-
import numpy
import math

# #f_max_considered = 5

results_path = '../../synthetic_data_spiral_3D/test2'#'/ssa'#'nlsa/b_time_nns' 
m = 3
S = 800 # test1 400, from test2 800
q = 100 #100 #test1 40, from test2 100 


data_file = '%s/x.jbl'%results_path
datatype = numpy.float64

"""
# D_sq
paral_step = 20
n_workers = int(math.ceil(float(q)/paral_step))
print n_workers


b = 50
# # # # # # # # # f_max = 100

# # # # # # # # # n_workers_aj = (2*f_max) + 1
    
# # # # # # # # # paral_step_lp_filter_Dsq = 1000
# # # # # # # # # n_workers_lp_filter_Dsq = int(math.ceil((S-q)/paral_step_lp_filter_Dsq)) #26

log10eps = 1.5
sigma_sq = 2*10**log10eps

l = 31
nmodes = l #7 
toproject = range(nmodes) #[0, 1, 2, 3, 4, 29, 30] 
paral_step_A = 20
n_workers_A = int(math.ceil(float(q)/paral_step_A))
print n_workers_A
"""
ncopies = q
modes_to_reconstruct = range(10)

paral_step_reconstruction = 100
n_workers_reconstruction = int(math.ceil(float(S-q-ncopies+1)/paral_step_reconstruction))
print n_workers_reconstruction
