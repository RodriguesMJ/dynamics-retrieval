#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

# USE: sbatch -p day run.sh
# FOR PARALLEL STEPS USE:
# conda activate myenv_nlsa
# python run_py_rho.py

import settings_bR_light as settings

### SPARSE ###

# Data conversion with convert.py
flag = 0
if flag == 1:
    import dynamics_retrieval.convert
    nlsa.convert.main(settings)

# Apply translational disorder correction
# FOR RHODOPSIN
flag = 0
if flag == 1:
    import dynamics_retrieval.t_disorder_correct
    nlsa.t_disorder_correct.main(settings)

# Make_lp_filter
flag = 0
if flag == 1:
    import dynamics_retrieval.make_lp_filter
    nlsa.make_lp_filter.main(settings)
    nlsa.make_lp_filter.check(settings)
flag = 0
if flag == 1:
    end_worker = settings.n_workers_aj - 1
    os.system('sbatch -p day -t 1-00:00:00 --mem=350G --array=0-%d ../scripts_parallel_submission/run_parallel_aj.sh %s'
              %(end_worker, settings.__name__))
flag = 0
if flag == 1:
    end_worker = settings.n_workers_aj - 1
    os.system('sbatch -p day -t 1-00:00:00 --array=0-%d ../scripts_parallel_submission/run_parallel_ATA_lp_filter.sh %s'
              %(end_worker, settings.__name__))
flag = 0
if flag == 1:
    import dynamics_retrieval.calculate_lp_filter_ATA_merge
    nlsa.calculate_lp_filter_ATA_merge.main(settings)
flag = 0
if flag == 1:
    os.system('sbatch -p week -t 8-00:00:00 --mem=350G --array=0-14 ../scripts_parallel_submission/run_parallel_Dsq_lp_filter.sh %s'
              %(settings.__name__))
flag = 0
if flag == 1:
    import dynamics_retrieval.util_merge_D_sq
    nlsa.util_merge_D_sq.f(settings)
flag = 1
if flag == 1:
    import dynamics_retrieval.plot_distance_distributions
    # nlsa.plot_distance_distributions.plot_distributions(settings)
    # nlsa.plot_distance_distributions.plot_D_matrix(settings)
    # nlsa.plot_distance_distributions.plot_D_0j(settings)
    #nlsa.plot_distance_distributions.plot_D_0j_recap_lin(settings)
    nlsa.plot_distance_distributions.plot_D_submatrix(settings)

# CALCULATE d_sq
flag = 0
if flag == 1:
    import dynamics_retrieval.calculate_distances_utilities
    nlsa.calculate_distances_utilities.calculate_d_sq_SFX_steps(settings)
    nlsa.calculate_distances_utilities.calculate_d_sq_SFX_element_n(settings)
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
    import dynamics_retrieval.calculate_distances_utilities
    nlsa.calculate_distances_utilities.merge_D_sq(settings)

# CALCULATE N_D_sq_elements
flag = 0
if flag == 1:
    end_worker = settings.n_workers - 1
    os.system('sbatch -p day -t 1-00:00:00 --mem=350G --array=0-%d ../scripts_parallel_submission/run_parallel_n_Dsq_elements.sh %s'
              %(end_worker, settings.__name__))

flag = 0
if flag == 1:
    import dynamics_retrieval.calculate_distances_utilities
    nlsa.calculate_distances_utilities.merge_N_D_sq_elements(settings)

flag = 0
if flag == 1:
    import dynamics_retrieval.get_epsilon
    #nlsa.get_epsilon.get_hist(settings)
    #nlsa.get_epsilon.get_epsilon_curve(settings)

    #nlsa.get_epsilon.get_N_Dsq_elements_distribution(settings)
    nlsa.get_epsilon.get_distributions(settings)
    #nlsa.get_epsilon.test(settings)


flag = 0
if flag == 1:
    import dynamics_retrieval.calculate_distances_utilities
    nlsa.calculate_distances_utilities.sort_D_sq(settings)


flag = 0
if flag == 1:
    import dynamics_retrieval.get_epsilon
    nlsa.get_epsilon.get_hist_b_nns(settings)
    nlsa.get_epsilon.get_epsilon_curve_b_nns(settings)


flag = 0
if flag ==1:
    import dynamics_retrieval.transition_matrix
    nlsa.transition_matrix.main(settings)

flag = 0
if flag == 1:
    import dynamics_retrieval.probability_matrix
    nlsa.probability_matrix.main(settings)

flag = 0
if flag == 1:
    import dynamics_retrieval.eigendecompose
    nlsa.eigendecompose.main(settings)

# flag = 0
# if flag ==1:
#     import dynamics_retrieval.transition_matrix
#     nlsa.transition_matrix.check(settings)


flag = 0
if flag == 1:
    import dynamics_retrieval.evecs_normalisation
    nlsa.evecs_normalisation.main_part_1(settings)
    #nlsa.evecs_normalisation.main_part_2(settings)

flag = 0
if flag == 1:
    import dynamics_retrieval.plot_P_evecs
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
    import dynamics_retrieval.util_merge_A
    nlsa.util_merge_A.main(settings)

flag = 0
if flag == 1:
    import dynamics_retrieval.SVD
    nlsa.SVD.main(settings)

flag = 0
if flag == 1:
    import dynamics_retrieval.plot_SVs
    import dynamics_retrieval.plot_chronos
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
    import dynamics_retrieval.util_merge_x_r
    for mode in range(0, settings.nmodes):
        nlsa.util_merge_x_r.f(settings, mode)

flag = 0
if flag == 1:
    import dynamics_retrieval.reconstruction
    nlsa.reconstruction.reconstruct_unwrap_loop_chunck_bwd(settings)

flag = 0
if flag == 1:
    import dynamics_retrieval.util_append_bwd_reconstruction
    for mode in range(0, settings.nmodes):
        nlsa.util_append_bwd_reconstruction.f(settings, mode)

flag = 0
if flag == 1:
    import dynamics_retrieval.export_Is
    for mode in [1]:#range(1, settings.nmodes):
        nlsa.export_Is.get_Is(settings, mode)

flag = 0
if flag == 1:
    import dynamics_retrieval.export_Is
    nlsa.export_Is.export_merged_data_light(settings)