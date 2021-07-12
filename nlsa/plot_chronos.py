# -*- coding: utf-8 -*-
import joblib
import matplotlib.pyplot
import os
import numpy  

def plot(settings):
    results_path = settings.results_path
    label = ''
    
    VT_final = joblib.load('%s/VT_final%s.jbl'%(results_path,  label))
    
    nmodes = VT_final.shape[0]
    s = VT_final.shape[1]
    
    print 'nmodes: ', nmodes
    print 's: ', s
    
    out_folder = '%s/chronos%s'%(results_path, label)
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    
    for i in range(nmodes):
        print i
        chrono = VT_final[i,:]
        matplotlib.pyplot.figure(figsize=(30,10))
        matplotlib.pyplot.plot(range(s), chrono, 'o-', markersize=8)
        #matplotlib.pyplot.plot(range(120), chrono[0:120], 'o-', markersize=8)
        ax = matplotlib.pyplot.gca()
        ax.tick_params(axis='x', labelsize=25)
        ax.tick_params(axis='y', labelsize=25)
        matplotlib.pyplot.savefig('%s/chrono%s_%d.png'%(out_folder, label, i), dpi=2*96)
        matplotlib.pyplot.close()
        
#        matplotlib.pyplot.figure(figsize=(2048/96, 512/96), dpi=96)
#        matplotlib.pyplot.plot(range(s), chrono, 'o', markersize=2, c='b', zorder=0)
#        #matplotlib.pyplot.scatter(i, chrono[i], c='c', s=150, zorder=10)
#        
#        ax = matplotlib.pyplot.gca()
#        ax.spines['bottom'].set_position('zero')
#        ax.tick_params(axis='x', which='major', labelsize=20)
#        ax.tick_params(axis='y', which='major', labelsize=0)
#        ax.spines['right'].set_visible(False)
#        ax.spines['top'].set_visible(False)
#        ax.spines['left'].set_visible(False)
#        ax.tick_params(axis='x', colors='magenta')
#        matplotlib.pyplot.tick_params(
#                axis='y',          # changes apply to the y-axis
#                which='both',      # both major and minor ticks are affected
#                left=False,        # ticks along the left edge are off
#                ) #
#        
#        matplotlib.pyplot.savefig('%schrono%s_%d.png'%(out_folder, label, i), dpi=2*96)
#        print 'Saved'
#        matplotlib.pyplot.close()
        #os.system('convert %s -resize 2048x512 %s'%(figname, rs_figname))
    
def plot_abs(settings):
    results_path = settings.results_path
    label = ''
    
    VT_final = joblib.load('%s/VT_final%s.jbl'%(results_path,  label))
    
    nmodes = VT_final.shape[0]
    s = VT_final.shape[1]
    
    print 'nmodes: ', nmodes
    print 's: ', s
    
    out_folder = '%s/chronos%s'%(results_path, label)
    if not os.path.exists(out_folder):
        os.mkdir(out_folder)
    
    for i in range(nmodes):
        print i
        chrono = VT_final[i,:]
        matplotlib.pyplot.figure(figsize=(40,10))
        
        matplotlib.pyplot.plot(range(29000, 29000+s-1000), abs(chrono)[0:-1000], 'o-', c='b', markersize=8)
        #matplotlib.pyplot.plot(range(120), chrono[0:120], 'o-', markersize=8)
        ax = matplotlib.pyplot.gca()
        ax.tick_params(axis='x', labelsize=25)
        ax.tick_params(axis='y', labelsize=25)
        matplotlib.pyplot.savefig('%s/abs_chrono%s_%d.png'%(out_folder, label, i), dpi=2*96)
        matplotlib.pyplot.close() 
# def plot_chrono_movie(mode):
#     results_path = settings.results_path
#     label = ''    
#     VT_final = joblib.load('%s/VT_final%s.jbl'%(results_path,  label))  
#     s = VT_final.shape[1]   
#     chrono =  VT_final[mode,:]
#     print 'mode: ', mode
#     print 's: ', s
    
#     out_folder = '%s/chronos%s_%d_movie_noaxis'%(results_path, label, mode)
#     if not os.path.exists(out_folder):
#         os.mkdir(out_folder)
                
#     my_dpi=96
#     for i in range(100, 99200, 100):   #99200
#         print mode, i
#         figname = '%s/chrono%s_%d_timepoint_%0.5d.png'%(out_folder, label, mode, i)
#         rs_figname = '%s/rs_chrono%s_%d_timepoint_%0.5d.png'%(out_folder, label, mode, i)
#         matplotlib.pyplot.figure(figsize=(2048/my_dpi, 512/my_dpi), dpi=my_dpi)
#         matplotlib.pyplot.plot(range(s), chrono, 'o', markersize=2, c='b', zorder=0)
#         matplotlib.pyplot.scatter(i, chrono[i], c='c', s=150, zorder=10)
        
#         ax = matplotlib.pyplot.gca()
#         ax.spines['bottom'].set_position('zero')
#         ax.tick_params(axis='x', which='major', labelsize=20)
#         ax.tick_params(axis='y', which='major', labelsize=0)
#         ax.spines['right'].set_visible(False)
#         ax.spines['top'].set_visible(False)
#         ax.spines['left'].set_visible(False)
#         ax.tick_params(axis='x', colors='magenta')
#         matplotlib.pyplot.tick_params(
#                 axis='y',          # changes apply to the y-axis
#                 which='both',      # both major and minor ticks are affected
#                 left=False,        # ticks along the left edge are off
#                 ) #
        
#         matplotlib.pyplot.savefig(figname, dpi=my_dpi)
#         matplotlib.pyplot.close()
#         os.system('convert %s -resize 2048x512 %s'%(figname, rs_figname))

  
# def plot_topos():
#     results_path = settings.results_path  
#     q = settings.q
#     U = joblib.load('%s/U.jbl'%(results_path))
#     print U.shape
#     print len(numpy.argwhere(U[:,3])<0)    
#     m = float(U.shape[0])/q
#     print m
#     m = int(m)
#     for i in range(U.shape[1]):
#         topo = U[0:m, i]
#         matplotlib.pyplot.plot(range(m), topo)
#         matplotlib.pyplot.savefig('%s/topo_%d.png'%(results_path, i))
#         matplotlib.pyplot.close()



def main(settings):   
    plot(settings)
    plot_abs(settings)
