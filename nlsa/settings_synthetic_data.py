# -*- coding: utf-8 -*-
import numpy
import math



results_path = '../../synthetic_data_4/test5/ssa/q_4000'#'nlsa/b_4000_eu_nns' 
m = 7000
S = 30000
q = 4000

# D_sq
paral_step = 400
n_workers = int(math.ceil(float(q)/paral_step))

data_file = '%s/x.jbl'%results_path
datatype = numpy.float64

b = 3000 #1500

# # # # # # # # n_workers_aj = (2*f_max) + 1
    
# # # # # # # # paral_step_lp_filter_Dsq = 1000
# # # # # # # # n_workers_lp_filter_Dsq = int(math.ceil((S-q)/paral_step_lp_filter_Dsq)) #26

log10eps = 6.6
sigma_sq = 2*10**log10eps

l = 31
nmodes = l #7 
toproject = range(nmodes) #[0, 1, 2, 3, 4, 29, 30] 
paral_step_A = 400
n_workers_A = int(math.ceil(float(q)/paral_step_A))

ncopies = q
modes_to_reconstruct = range(20) 

paral_step_reconstruction = 1000
n_workers_reconstruction = int(math.ceil(float(S-q-ncopies+1)/paral_step_reconstruction))

# f_max = 100
# f_max_considered = 100