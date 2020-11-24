# -*- coding: utf-8 -*-
import pickle
import numpy
import matplotlib.pylab as pl
import os
import matplotlib

import settings

results_path = settings.results_path
l = settings.l


f = open('%s/T_anomaly.pkl'%(results_path), 'rb')
x = pickle.load(f)
f.close()
x = x[:, 1200:1200+2000]
print 'Data: ', x.shape

# Exclude last latitude point and last longitude point
movie_combined = numpy.zeros((x.shape[0],x.shape[1]))
for mode in range(l):
    print mode
    f = open('%s/movie_mode_%d.pkl'%(results_path, mode), 'rb')
    movie_1D = pickle.load(f)
    f.close()
    movie_1D = movie_1D[:, 1176:1176+2000]
    print movie_1D.shape
    movie_combined = movie_combined + movie_1D
    diff = movie_combined - x
    diff = abs(diff)
    print  numpy.average(diff)


#outdir = '%s/movie_modes_5_8'%results_path
#if not os.path.exists(outdir):
#    os.mkdir(outdir)
#    
##pl.ion() 
##pl.imshow(movie_combined[:,:,0], vmin=-0.00013, vmax=0.00013, origin='lower', cmap='jet')   
#pl.imshow(movie_combined[:,:,0], vmin=-0.2, vmax=0.2, origin='lower', cmap='jet') 
#pl.colorbar()    
#
#n = -1                      
#for i in range(0,2400):#1176(n_time_pts):          
#    pl.cla()                          
#    #pl.imshow(movie_combined[:,:,i], vmin=-0.00013, vmax=0.00013, origin='lower', cmap='jet')
#    pl.imshow(movie_combined[:,:,i], vmin=-0.2, vmax=0.2, origin='lower', cmap='jet')
#    if i%12 == 0:
#        n = n+1
#    pl.title('Year: %d   Month: %d'%(n, i%12))
#    #pl.draw()
#    pl.savefig('%s/Modes_5_8_index_%05d.png'%(outdir, i))
#    #pl.pause(0.002)                    
##pl.ioff()                             
#print('Done!')  
#
#
## Look for 1st frame in PNAS movies
##idxs_candidates = []
##for i in range(4800):
##    img = movie_combined[:,:,i]
##    a = img[7:8,8:11]
##    b = img[8:9,10:12]
##    c = img[7,11]
##    if numpy.all([a<-0.15]) and numpy.all([b>0.15]) and c<img[8,11] and img[7,12]<img[7,11]:
##        idxs_candidates.append(i)
##        matplotlib.pyplot.figure()
##        matplotlib.pyplot.imshow(img, origin='lower', vmin=-0.2, vmax=+0.2, cmap='jet')
##        matplotlib.pyplot.colorbar()
##        matplotlib.pyplot.savefig('%s/candidates_modes_9_10/%05d.png'%(results_path,i))
##        matplotlib.pyplot.close()
##        
