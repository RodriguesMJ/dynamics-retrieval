# -*- coding: utf-8 -*-

import numpy


import settings_bR_dark as settings

folder = settings.results_path

#I_py = numpy.loadtxt('%s/reconstructed_intensities_mode_0_2/bR_light_mode_0_2_timestep_100.txt'%folder)
I_py = numpy.loadtxt('%s/reconstructed_intensities_mode_0/bR_dark_mode_0_avg.txt'%folder)
I_py = I_py[:,3]

#I_m = numpy.loadtxt('../../TR-SFX/bR_TR_SFX/NLSA_test/reconstruction/reconstructed_intensities_step_50/light_mode-1-3_hklIsigI_100.hkl')
I_m = numpy.loadtxt('../../TR-SFX/bR_TR_SFX/NLSA_test/reconstruction/reconstructed_intensities_bins/mult_dark_nlsa_rec_mode-1_hklIsigI_avg.hkl')
I_m = I_m[:,3]

ratio = I_py/I_m

nan_idxs = numpy.argwhere(numpy.isnan(ratio))


ratio_clean = [ratio[i,] for i in range(ratio.shape[0]) if not (   numpy.isnan(ratio[i,])
                                                                or numpy.isinf(ratio[i,]) )]

print numpy.mean(ratio_clean)
print numpy.std(ratio_clean)