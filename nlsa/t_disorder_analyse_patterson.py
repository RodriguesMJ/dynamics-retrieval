# -*- coding: utf-8 -*-

# Produce and visualise patterson maps
# ./generate_mtzs_rho
# cctbx.patterson_map blahblah.mtz
# mapslicer blahblah_patt.ccp4

### To use this code:
# source activate myenv_gemmi
# python t_disorder_analyse_patterson
# source deactivate

import gemmi
import numpy
import matplotlib
from matplotlib import pyplot

#ccp4 = gemmi.read_ccp4_map('./rho_dark_ortho_patt.ccp4')
#ccp4.setup()
#arr = numpy.array(ccp4.grid, copy=False)
#print(ccp4.grid.unit_cell.a, ccp4.grid.unit_cell.b)
#x = numpy.linspace(0, ccp4.grid.unit_cell.a, num=arr.shape[0], endpoint=False)
#y = numpy.linspace(0, ccp4.grid.unit_cell.b, num=arr.shape[1], endpoint=False)
#X, Y = numpy.meshgrid(x, y, indexing='ij')
#pyplot.contourf(X, Y, arr[:,:,0], vmin=0, vmax=50)
#pyplot.gca().set_aspect('equal', adjustable='box')
#pyplot.colorbar()
#pyplot.savefig('./rho_dark_ortho_patt_z0.png')
#pyplot.close()
#
#pyplot.plot(y,arr[0,:,0])
#pyplot.savefig('./rho_dark_ortho_patt_x0_z0.png')
#pyplot.close()

#path = '/das/work/p18/p18594/cecilia-offline/NLSA/data_rho_2/translation_corr_det'
path = '/das/work/p17/p17491/Cecilia_Casadei/rho_translation_correction_paper'
label = 'swissfel_combined_dark_xsphere_precise_1ps-dark'
outfolder = '%s/method_bf'%path
#label = 'rho'

fractions = numpy.arange(0.18, 0.24, 0.01)
colors = matplotlib.pylab.cm.viridis(numpy.linspace(0,1,7)) 

pyplot.figure(figsize=(14, 14))
pyplot.ylim([-20,+50])

fn = '%s/%s_patt.ccp4'%(outfolder, label)
ccp4 = gemmi.read_ccp4_map(fn)
ccp4.setup()
arr = numpy.array(ccp4.grid, copy=False)
y = numpy.linspace(0, ccp4.grid.unit_cell.b, num=arr.shape[1], endpoint=False)
pyplot.plot(y,arr[0,:,0],label='uncorrected',c=colors[0])

for idx, fraction in enumerate(fractions):
    print(fraction)
    fn = '%s/%s_corrected_frac_%.2f_patt.ccp4'%(outfolder, label, fraction)
    ccp4 = gemmi.read_ccp4_map(fn)
    ccp4.setup()
    arr = numpy.array(ccp4.grid, copy=False)
    y = numpy.linspace(0, ccp4.grid.unit_cell.b, num=arr.shape[1], endpoint=False)
    pyplot.plot(y,arr[0,:,0],label=r'$\alpha=%.2f$'%fraction,c=colors[idx+1])
matplotlib.pyplot.gca().tick_params(axis='both', labelsize=28)    
matplotlib.pyplot.legend(frameon=False, fontsize=34) 
matplotlib.pyplot.xlabel(r'$y \;\; [\AA]$', fontsize=34),
matplotlib.pyplot.ylabel('Patterson map value', fontsize=34, rotation=90, labelpad=4)
        
pyplot.tight_layout()
pyplot.savefig('%s/%s_patt_x0_z0_frac_range.pdf'%(outfolder, label),dpi=96*2)
pyplot.close()