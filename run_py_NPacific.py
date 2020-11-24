# -*- coding: utf-8 -*-
import os

### DENSE ### (results_test_15)

# EXTRACT DATA
flag = 0
if flag == 1:
    os.system('python extract_T_anomaly_matrix_NPacific.py')   
    
# CALCULATE D, N, v  
flag = 0
if flag == 1:  
    os.system('python calculate_distances.py')
    
# CALCULATE AND SYMMETRISE TRANSITION MATRIX
flag = 0
if flag == 1:
    os.system('python transition_matrix.py')
    
# CALCULATE PROBABILITY MATRIX
flag = 0
if flag == 1:
    os.system('python probability_matrix.py')
    #os.system('python mu_P_sym_sign.py')

# EIGENDECOMPOSE
flag = 0
if flag == 1:    
    os.system('python eigendecompose.py')
 
flag = 0
if flag == 1:           
    os.system('python evecs_normalisation.py')    
    
flag = 0
if flag == 1:      
    os.system('python plot_P_evecs.py')

flag = 0
if flag == 1:
    os.system('python calculate_A.py')
    #os.system('sbatch run_parallel_A.sh')
    #import util_merge_A
    #nproc = 5
    #util_merge_A.f(nproc)
    
flag = 0
if flag == 1:
    os.system('python SVD.py')
    
flag = 0
if flag == 1:
    os.system('python plot_chronos.py')   
    
flag = 1
if flag == 1:
    #os.system('python reconstruction.py') 
    
    # OR
    
    # STEP 1: middle + forward in chuncks
    #os.system('sbatch run_parallel_reconstruction.sh')
    
    # STEP 2: merge chuncks
    #import util_merge_x_r
    #nproc = 9   
    #for mode in range(7):
        #util_merge_x_r.f(nproc, mode) 
        
    # STEP 3: backward
    import reconstruction
    reconstruction.reconstruct_unwrap_loop_chunck_bwd()
    
    # STEP 4: append backward to middle + forward
    import util_append_bwd_reconstruction
    mode = 0
    util_append_bwd_reconstruction.f(mode)
    
    
flag = 0
if flag == 1:
    os.system('python reconstruct_on_ocean_grid.py')   

flag = 0
if flag == 1:
    os.system('python show_movie_NP.py')   
        
### SPARSE ### (results_test_16)

# EXTRACT DATA
#os.system('python extract_T_anomaly_matrix_NPacific.py')    
# MAKE SPARSE   
#os.system('python make_sparse.py')
# CALCULATE D, N, v    
#os.system('python calculate_distances.py')



#TO DO
# CALCULATE AND SYMMETRISE TRANSITION MATRIX
#os.system('python transition_matrix.py')
#os.system('python probability_matrix.py')
#os.system('python mu_P_sym_sign.py')
#os.system('python eigendecompose.py')
#os.system('python plot_P_evecs.py')
#os.system('python evecs_normalisation.py')
#os.system('python calculate_A.py')
#os.system('python SVD.py')
#os.system('python plot_SVs.py')
#os.system('python project_chronos.py')
#os.system('python plot_chronos.py'