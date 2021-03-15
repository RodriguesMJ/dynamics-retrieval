# -*- coding: utf-8 -*-
import os

### SPARSE ###

# Data conversion with convert.py
flag = 0
if flag == 1:
    os.system('python convert.py')    # sbatch -p day run.sh


# CALCULATE d_sq
flag = 0
if flag == 1:    
    import calculate_distances_utilities
    calculate_distances_utilities.calculate_d_sq_SFX()

# CALCULATE D_sq    
flag = 0
if flag == 1:
    os.system('sbatch run_parallel.sh')
    
flag = 1
if flag == 1:    
    import calculate_distances_utilities
    #calculate_distances_utilities.merge_D_sq()
    calculate_distances_utilities.sort_D_sq()

flag = 0
if flag ==1:
    os.system('python get_epsilon.py')
    
flag = 0
if flag ==1:
    os.system('python transition_matrix.py')
    
flag = 0
if flag == 1:
    os.system('python probability_matrix.py')
        
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
    os.system('sbatch run_parallel_A.sh')
    
flag = 0
if flag == 1:
    import util_merge_A
    nproc = 12
    util_merge_A.f(nproc)  

flag = 0
if flag == 1:    
    os.system('python SVD.py')

flag = 0
if flag == 1:      
    os.system('python plot_SVs.py')
    os.system('python plot_chronos.py')
    
flag = 0
if flag == 1:
    os.system('sbatch run_parallel_reconstruction.sh')
    
flag = 0
if flag == 1:
    import util_merge_x_r
    nproc = 11  
    for mode in range(1,3):
        util_merge_x_r.f(nproc, mode) 

flag = 0
if flag == 1:
    import reconstruction
    reconstruction.reconstruct_unwrap_loop_chunck_bwd()

flag = 0
if flag == 1:
    import util_append_bwd_reconstruction
    for mode in range(0, 3):
        util_append_bwd_reconstruction.f(mode)
        
flag = 0
if flag == 1:    
    os.system('python export_Is.py')        