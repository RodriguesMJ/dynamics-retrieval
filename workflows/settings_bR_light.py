# -*- coding: utf-8 -*-
import numpy
import joblib

label = 'light'

root_folder = '/das/work/p17/p17491/Cecilia_Casadei/NLSA/data_bR_2'
#data_path = '%s/converted_data_%s'%(root_folder, label)
results_path = '%s/results_LPSA/bR_%s_dI_mirrored'%(root_folder, label)


datatype = numpy.float64

S = 119507
m = 22727

f_max_q_scan = 20
q_f_max_scan = 15001   