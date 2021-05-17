#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

# USE: sbatch -p day run.sh
# FOR PARALLEL STEPS USE:
# conda activate myenv_nlsa
# python run_py_rho.py

import settings_rho_dark as settings

### SPARSE ###

# Data conversion with convert.py
flag = 0
if flag == 1:
    import nlsa.convert
    nlsa.convert.main(settings)

# Apply translational disorder correction
flag = 0
if flag == 1:
    import nlsa.t_disorder_correct
    nlsa.t_disorder_correct.main(settings)

# CALCULATE d_sq
flag = 0
if flag == 1:    
    import nlsa.calculate_distances_utilities
    nlsa.calculate_distances_utilities.calculate_d_sq_SFX_steps(settings)
    
    # Only for small datasets eg dark
    #nlsa.calculate_distances_utilities.calculate_d_sq_SFX(settings)
    #nlsa.calculate_distances_utilities.compare(settings)

# CALCULATE D_sq    
flag = 0
if flag == 1:
    end_worker = settings.n_workers - 1
    os.system('sbatch -p day -t 1-00:00:00 --mem=350G --array=0-%d ../scripts_parallel_submission/run_parallel.sh %s'
              %(end_worker, settings.__name__)) 
    
    
flag = 0
if flag == 1:    
    import nlsa.calculate_distances_utilities
    nlsa.calculate_distances_utilities.merge_D_sq(settings)

flag = 0
if flag == 1:    
    import nlsa.calculate_distances_utilities
    nlsa.calculate_distances_utilities.sort_D_sq(settings)

flag = 0
if flag ==1:
    import nlsa.get_epsilon
    nlsa.get_epsilon.main(settings)
    
flag = 0
if flag ==1:
    import nlsa.transition_matrix
    nlsa.transition_matrix.main(settings)
   
flag = 0
if flag == 1:
    import nlsa.probability_matrix
    nlsa.probability_matrix.main(settings)
        
flag = 0
if flag == 1:
    import nlsa.eigendecompose
    nlsa.eigendecompose.main(settings)   

flag = 1
if flag == 1:
    import nlsa.evecs_normalisation
    nlsa.evecs_normalisation.main(settings)
    
flag = 1
if flag == 1:  
    import nlsa.plot_P_evecs
    nlsa.plot_P_evecs.main(settings)

# flag = 0
# if flag == 1:
#     os.system('sbatch run_parallel_A.sh')
    
# flag = 0
# if flag == 1:
#     import util_merge_A
#     nproc = 12
#     util_merge_A.f(nproc)  

# flag = 0
# if flag == 1:    
#     os.system('python SVD.py')

# flag = 0
# if flag == 1:      
#     os.system('python plot_SVs.py')
#     os.system('python plot_chronos.py')
    
# flag = 0
# if flag == 1:
#     os.system('sbatch run_parallel_reconstruction.sh')
    
# flag = 0
# if flag == 1:
#     import util_merge_x_r
#     nproc = 11  
#     for mode in range(1,3):
#         util_merge_x_r.f(nproc, mode) 

# flag = 0
# if flag == 1:
#     import reconstruction
#     reconstruction.reconstruct_unwrap_loop_chunck_bwd()

# flag = 0
# if flag == 1:
#     import util_append_bwd_reconstruction
#     for mode in range(0, 3):
#         util_append_bwd_reconstruction.f(mode)
        
# flag = 0
# if flag == 1:    
#     os.system('python export_Is.py')        