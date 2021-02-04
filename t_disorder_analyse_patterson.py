# -*- coding: utf-8 -*-

# Produce andvisualise patterson maps
# cctbx.patterson_map blahblah.mtz
# mapslicer blahblah_patt.ccp4

### To use this code:
# source activate myenv_gemmi
# python analyse_patterson
# source deactivate


import gemmi
import numpy
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

#label = 'rho_dark_ortho'
label = '348394_pushres1.9'
outfolder = './I_correction_method_2'

fractions = numpy.arange(0.19, 0.26, 0.01)

pyplot.figure()
pyplot.ylim([-20,+50])

fn = '%s/%s_corrected_frac_0.01_patt.ccp4'%(outfolder, label)
ccp4 = gemmi.read_ccp4_map(fn)
ccp4.setup()
arr = numpy.array(ccp4.grid, copy=False)
y = numpy.linspace(0, ccp4.grid.unit_cell.b, num=arr.shape[1], endpoint=False)
pyplot.plot(y,arr[0,:,0],label='original')

for fraction in fractions:
    print(fraction)
    fn = '%s/%s_corrected_frac_%.2f_patt.ccp4'%(outfolder, label, fraction)
    ccp4 = gemmi.read_ccp4_map(fn)
    ccp4.setup()
    arr = numpy.array(ccp4.grid, copy=False)
    y = numpy.linspace(0, ccp4.grid.unit_cell.b, num=arr.shape[1], endpoint=False)
    pyplot.plot(y,arr[0,:,0],label='%.2f'%fraction)
    
pyplot.legend()
pyplot.savefig('%s/%s_patt_x0_z0_frac_0p19_0p26.png'%(outfolder, label))
pyplot.close()