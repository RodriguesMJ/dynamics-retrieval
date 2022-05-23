# -*- coding: utf-8 -*-
import numpy
import math

m = 7000
S = 30000
q = 4001

T = S-q+1
tc = float(S)/2

jitter_factor = 1.0

datatype = numpy.float64
#results_path = '../../synthetic_data_jitter/test6/binning'

f_max = 100
f_max_considered = f_max

p = (q-1)/2
#p = 0
results_path = '../../synthetic_data_jitter/test6/LPSA_para_search/f_max_%d_q_%d/reconstruction_p_%d/x_r_SVD'%(f_max, q, p)





# data_file = '%s/x.jbl'%results_path


# paral_step_A = 500
# n_workers_A = int(math.ceil(float(q)/paral_step_A))


modes_to_reconstruct = range(4) 

# STANDARD RECONSTRUCTION
ncopies = q
paral_step_reconstruction = 2000
n_workers_reconstruction = int(math.ceil(float(S-q-ncopies+1)/paral_step_reconstruction))

