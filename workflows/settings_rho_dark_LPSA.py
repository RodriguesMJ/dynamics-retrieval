# -*- coding: utf-8 -*-
import joblib
import numpy

label = "alldark"

root_folder = "/das/work/p17/p17491/Cecilia_Casadei/NLSA/data_rho/LPSA"
data_path = "%s/converted_data_%s" % (root_folder, label)
# results_path = '%s/results_LPSA/bR_%s_dI_mirrored'%(root_folder, label)


datatype = numpy.float64

f_alpha = 0.01
f_beta = 0.17
f_gamma = 1 - f_alpha - f_beta

t_d_y = 0.245

# S = 119507
# m = 22727

f_max_q_scan = 20
q_f_max_scan = 15001
