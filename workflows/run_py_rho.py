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

flag = 0
if flag == 1:
    import nlsa.evecs_normalisation
    nlsa.evecs_normalisation.main(settings)
    
flag = 0
if flag == 1:  
    import nlsa.plot_P_evecs
    nlsa.plot_P_evecs.main(settings)

flag = 0
if flag == 1:
    end_worker = settings.n_workers_A - 1
    #os.system('sbatch -p day -t 1-00:00:00 --mem=350G --array=0-%d ../scripts_parallel_submission/run_parallel_A.sh %s'
    #          %(end_worker, settings.__name__)) 
    os.system('sbatch -p hour --array=0-%d ../scripts_parallel_submission/run_parallel_A.sh %s'
              %(end_worker, settings.__name__))
        
flag = 0
if flag == 1:
    import nlsa.util_merge_A
    nlsa.util_merge_A.main(settings)  

flag = 0
if flag == 1:  
    import nlsa.SVD
    nlsa.SVD.main(settings)

flag = 0
if flag == 1:  
    import nlsa.plot_SVs 
    import nlsa.plot_chronos
    nlsa.plot_SVs.main(settings)
    nlsa.plot_chronos.main(settings)
    
flag = 0
if flag == 1:
    end_worker = settings.n_workers_reconstruction - 1
    #os.system('sbatch -p day -t 1-00:00:00 --mem=350G --array=0-%d ../scripts_parallel_submission/run_parallel_A.sh %s'
    #          %(end_worker, settings.__name__)) 
    os.system('sbatch -p day -t 1-00:00:00 --array=0-%d ../scripts_parallel_submission/run_parallel_reconstruction.sh %s'
              %(end_worker, settings.__name__))    

flag = 0
if flag == 1:
    import nlsa.util_merge_x_r
    for mode in range(0, settings.nmodes):
        nlsa.util_merge_x_r.f(settings, mode) 

flag = 0
if flag == 1:
    import nlsa.reconstruction
    nlsa.reconstruction.reconstruct_unwrap_loop_chunck_bwd(settings)

flag = 0
if flag == 1:
    import nlsa.util_append_bwd_reconstruction
    for mode in range(0, settings.nmodes):
        nlsa.util_append_bwd_reconstruction.f(settings, mode)
        
flag = 1
if flag == 1:  
    import nlsa.export_Is
    for mode in range(1, settings.nmodes):
        nlsa.export_Is.get_Is(settings, mode)    