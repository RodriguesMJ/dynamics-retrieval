# -*- coding: utf-8 -*-
import os

### DENSE ### (results_test_15)

# EXTRACT DATA
flag = 0
if flag == 1:
    os.system("python extract_T_anomaly_matrix_NPacific.py")

# CALCULATE D, N, v
flag = 0
if flag == 1:
    os.system("python calculate_distances.py")

# CALCULATE AND SYMMETRISE TRANSITION MATRIX
flag = 0
if flag == 1:
    os.system("python transition_matrix.py")

# CALCULATE PROBABILITY MATRIX
flag = 0
if flag == 1:
    os.system("python probability_matrix.py")
    # os.system('python mu_P_sym_sign.py')

# EIGENDECOMPOSE
flag = 0
if flag == 1:
    os.system("python eigendecompose.py")

# flag = 0
# if flag == 1:
#     os.system('python evecs_normalisation.py')

flag = 0
if flag == 1:
    os.system("python plot_P_evecs.py")

flag = 0
if flag == 1:
    os.system("python calculate_A.py")
    # os.system('sbatch run_parallel_A.sh')
    # import util_merge_A
    # nproc = 5
    # util_merge_A.f(nproc)

flag = 0
if flag == 1:
    os.system("python SVD.py")

flag = 0
if flag == 1:
    os.system("python plot_chronos.py")

flag = 1
if flag == 1:
    os.system("python reconstruction.py")


flag = 0
if flag == 1:
    os.system("python reconstruct_on_ocean_grid.py")

flag = 0
if flag == 1:
    os.system("python show_movie_NP.py")
