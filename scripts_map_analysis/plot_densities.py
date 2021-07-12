import matplotlib
matplotlib.use('Agg') # Force matplotlib to not use any Xwindows backend.
matplotlib.rcParams['agg.path.chunksize'] = 100000000
import matplotlib.pyplot
import numpy
import joblib
from mtspec import mtspec   
import os

#matplotlib.rcParams['agg.path.chunksize'] = 10000 



def plot(neg_integral, pos_integral, nmaps, path, label):
    matplotlib.pyplot.figure(figsize=(40,10))
    matplotlib.pyplot.gca().tick_params(axis='x', labelsize=25) 
    matplotlib.pyplot.gca().tick_params(axis='y', labelsize=0) 
    matplotlib.pyplot.scatter(range(nmaps), neg_integral, c='m', s=5)
    matplotlib.pyplot.scatter(range(nmaps), pos_integral, c='b', s=5)
    matplotlib.pyplot.axhline(y=0, xmin=0, xmax=1, c='k')
    matplotlib.pyplot.savefig('%s/%s.png'%(path, label))
    matplotlib.pyplot.close()
    
    fig = matplotlib.pyplot.figure(figsize=(40,10))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()
    ax1.tick_params(labelsize=23) 
    ax2.tick_params(labelsize=23) 
    
    X = range(nmaps)
    ax1.scatter(X, neg_integral, c='m', s=5)
    ax1.scatter(X, pos_integral, c='b', s=5)
    matplotlib.pyplot.axhline(y=0, xmin=0, xmax=1, c='k')
    ax1.set_xlabel(r"Time steps / 100", fontsize=30)
    
    new_tick_locations = numpy.array([0, 435.0, 690.0, 1050.0, X[-1]])
    for i in new_tick_locations:
        matplotlib.pyplot.axvline(x=i, ymin=0, ymax=1, c='k')

    def tick_function(X):
        step_fs = 100*(385.0+335.0)/213557
        V = (X-435)*step_fs
        return ["%.3f" % z for z in V]

    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(new_tick_locations)
    ax2.set_xticklabels(tick_function(new_tick_locations))
    ax2.set_xlabel(r"Time [fs]", fontsize=30)
    matplotlib.pyplot.savefig('%s/timeaxis_%s.png'%(path, label))
    matplotlib.pyplot.close()



def plot_zoom(neg_integral, pos_integral, nmaps, path, label):
    # matplotlib.pyplot.figure(figsize=(40,10))
    # matplotlib.pyplot.gca().tick_params(axis='x', labelsize=25) 
    # matplotlib.pyplot.gca().tick_params(axis='y', labelsize=0) 
    # start = 435
    # end = 1050
    # matplotlib.pyplot.scatter(range(start, end), neg_integral[start:end], c='m', s=5)
    # matplotlib.pyplot.scatter(range(start, end), pos_integral[start:end], c='b', s=5)
    # matplotlib.pyplot.axhline(y=0, xmin=0, xmax=1, c='k')
    # matplotlib.pyplot.savefig('%s/%s_zoom_435_1050.png'%(path, label))
    # matplotlib.pyplot.close()
    
    if 'm_0_2' in label:
        fig = matplotlib.pyplot.figure(figsize=(40,10))
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twiny()
        ax1.tick_params(labelsize=23) 
        ax2.tick_params(labelsize=23) 
        
        start = 435
        end = 1050
        ax1.scatter(range(start, end), neg_integral[start:end], c='m', s=5)
        ax1.scatter(range(start, end), pos_integral[start:end], c='b', s=5)
        matplotlib.pyplot.axhline(y=0, xmin=0, xmax=1, c='k')
        ax1.set_xlabel(r"Time steps / 100", fontsize=30)
        
        new_tick_locations = numpy.array([435.0, 690.0, 1050.0])
        for i in new_tick_locations:
            matplotlib.pyplot.axvline(x=i, ymin=0, ymax=1, c='k')
    
        def tick_function(X):
            step_fs = 100*(385.0+335.0)/213557
            V = (X-435)*step_fs
            return ["%.3f" % z for z in V]
    
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_xticks(new_tick_locations)
        ax2.set_xticklabels(tick_function(new_tick_locations))
        ax2.set_xlabel(r"Time [fs]", fontsize=30)
        matplotlib.pyplot.savefig('%s/%s_zoom_%d_%d.png'%(path, label, start, end))
        matplotlib.pyplot.close()
        
    if 'm_0_1' in label:
        fig = matplotlib.pyplot.figure(figsize=(40,10))
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twiny()
        ax1.tick_params(labelsize=23) 
        ax2.tick_params(labelsize=23) 
        
        start = 1500
        end = neg_integral.shape[0]
        ax1.scatter(range(start, end), neg_integral[start:end], c='m', s=5)
        ax1.scatter(range(start, end), pos_integral[start:end], c='b', s=5)
        matplotlib.pyplot.axhline(y=0, xmin=0, xmax=1, c='k')
        ax1.set_xlabel(r"Time steps / 100", fontsize=30)
        
        new_tick_locations = numpy.array([start, end])
        for i in new_tick_locations:
            matplotlib.pyplot.axvline(x=i, ymin=0, ymax=1, c='k')
    
        def tick_function(X):
            step_fs = 100*(385.0+335.0)/213557
            V = (X-435)*step_fs
            return ["%.3f" % z for z in V]
    
        ax2.set_xlim(ax1.get_xlim())
        ax2.set_xticks(new_tick_locations)
        ax2.set_xticklabels(tick_function(new_tick_locations))
        ax2.set_xlabel(r"Time [fs]", fontsize=30)
        matplotlib.pyplot.savefig('%s/%s_zoom_%d_%d.png'%(path, label, start, end))
        matplotlib.pyplot.close()
        
def multitape(neg_integral, pos_integral, nmaps, path, label):
    def tick_function(X):        
        step_fs = 100*(385.0+335.0)/213557
        V = (X+start-t0)*step_fs
        return ["%.3f" % z for z in V]
    def tick_function_THz_to_wavens(f_THz):
        c = 33.35641
        return c*f_THz
        
    t0 = 435
    fcrit_value = 0.85
    
    if 'm_0_2' in label: 
        m = 2          
        start_end_list = [[550, 650]]#[[690, 800]]#[[435, 690], [500, 690], [690, 1050]]
    if 'm_0_1' in label: 
        m = 1
        start_end_list = [[1050, 1450], [1050, 2100]]#[[500, 1000]]#, [1780, 1950]]      
    
    for start_end in start_end_list:
        
        start = start_end[0]
        end = start_end[1]
        path_figs = '%s/multitape_m_0_%d_fcrit_%.2f_range_%d_%d'%(path, m, fcrit_value, start, end)
        
        if not os.path.exists(path_figs):
            os.mkdir(path_figs)
        
        data = neg_integral[start:end]
        delta_s = (10**(-15))*100*(385.0+335.0)/213557 #s
        xmax = 200 # THz, for figures
        
        fig = matplotlib.pyplot.figure(figsize=(35,40))
        
        # TIME DOMAIUN
        ax1 = fig.add_subplot(6, 1, 1)
        ax1.set_xlim(start, end)
        ax1.plot(range(start, end), data, color='m')
        matplotlib.pyplot.setp(ax1.get_xticklabels(), rotation=45, fontsize=25)
        matplotlib.pyplot.setp(ax1.get_yticklabels(), rotation=0, fontsize=25)
        ax1.set_xlabel(r"Frame n. / 100", fontsize=30)
        ax1.text(0.01, 
                  0.95, 
                  'Modes: %s + %s, %s sigma, %s'%(label[2], label[4], label[36:39], label[40:44]), 
                  transform=ax1.transAxes, 
                  fontsize=30,
                  verticalalignment='top')#, bbox=props)
     
        ax1_t = ax1.twiny()
        new_tick_locations = numpy.array([start-start, (start-start+end-start)/2, end-start])
        ax1_t.set_xticks(new_tick_locations)        
        ax1_t.set_xticklabels(tick_function(new_tick_locations))
        matplotlib.pyplot.setp(ax1_t.get_xticklabels(), rotation=0, fontsize=25)
        ax1_t.set_xlabel(r"Time [fs]", fontsize=30)
         
        # MULTITAPER ANALYSIS, WITHOUT F-STATS
        spec, freq, jackknife, _, _ = mtspec(data=data, 
                                              delta=delta_s, 
                                              time_bandwidth=3.5,
                                              number_of_tapers=4, 
                                              nfft=2*data.shape[0], 
                                              statistics=True)
        
        # MULTITAPER, LOG-SCALE
        ax2 = fig.add_subplot(6, 1, 2)
        ax2.plot(freq/(10**12), spec, color='black')  
        ax2.set_yscale('log')      
        matplotlib.pyplot.xticks(numpy.arange(0, xmax, 5.0))
        matplotlib.pyplot.setp(ax2.get_xticklabels(), rotation=45, fontsize=25)
        matplotlib.pyplot.setp(ax2.get_yticklabels(), rotation=0, fontsize=25)
        ax2.fill_between(freq/(10**12), jackknife[:, 0], jackknife[:, 1],
                          color="blue", alpha=0.3)
        ax2.set_xlim(0, xmax)
        
        # MULTITAPER, LINEAR SCALE
        ax3 = fig.add_subplot(6, 1, 3)
        ax3.plot(freq/(10**12), spec, color='black')        
        matplotlib.pyplot.xticks(numpy.arange(0, xmax, 5.0))
        matplotlib.pyplot.setp(ax3.get_xticklabels(), rotation=45, fontsize=25)
        matplotlib.pyplot.setp(ax3.get_yticklabels(), rotation=0, fontsize=25)
        ax3.fill_between(freq/(10**12), jackknife[:, 0], jackknife[:, 1],
                          color="blue", alpha=0.3)
        ax3.set_xlim(0, xmax)
        
        # MULTITAPER ANALYSIS, WITH F-STATS
        spec, freq, jackknife, fstatistics, _ = mtspec(data=data, 
                                              delta=delta_s, 
                                              time_bandwidth=3.5,
                                              number_of_tapers=4, 
                                              nfft=2*data.shape[0], 
                                              statistics=True, 
                                              rshape=0, 
                                              fcrit=fcrit_value)
        
        # F-STATS
        ax4 = fig.add_subplot(6, 1, 4)
        ax4.set_ylabel("F-stats for periodic lines", fontsize=30)
        ax4.set_xlim(0, xmax)
        print fstatistics.shape
        idx = numpy.where(abs(freq/(10**12) - xmax) < 10)
        
        idx = idx[0][-1]
        print idx
        ax4.set_ylim(0, 1.1*numpy.amax(fstatistics[0:idx]))
        ax4.plot(freq/(10**12), fstatistics, color="c")
        #ax4.set_xlabel(r"Frequency [THz]", fontsize=30)
        matplotlib.pyplot.xticks(numpy.arange(0, xmax, 5.0))
        matplotlib.pyplot.setp(ax4.get_xticklabels(), rotation=45, fontsize=25)
        matplotlib.pyplot.setp(ax4.get_yticklabels(), rotation=0,  fontsize=25)
        
        # Plot the confidence intervals.
        for p in [100*fcrit_value]:
            y = numpy.percentile(fstatistics, p)
            matplotlib.pyplot.hlines(y, 0, xmax, linestyles="--", color="0.2")
            matplotlib.pyplot.text(x=1, y=y+0.2, s="%i %%"%p, ha="left", fontsize=25)
        
        # MULTITAPER WITH F-STATS, LOG-SCALE
        ax5 = fig.add_subplot(6, 1, 5)
        ax5.set_yscale('log')
        ax5.plot(freq/(10**12), spec, color='b')        
        matplotlib.pyplot.xticks(numpy.arange(0, xmax, 5.0))
        matplotlib.pyplot.setp(ax5.get_xticklabels(), rotation=45, fontsize=25)
        matplotlib.pyplot.setp(ax5.get_yticklabels(), rotation=0, fontsize=25)
        ax5.set_facecolor((0.1, 1.0, 0.7, 0.1))
        ax5.set_xlim(0, xmax)
        
    
     
        # MULTITAPER WITH F-STATS, LINEAR SCALE
        ax6 = fig.add_subplot(6, 1, 6)
        ax6.plot(freq/(10**12), spec, color='b')        
        ax6.set_xlabel(r"Frequency [THz]", fontsize=30)    
        matplotlib.pyplot.xticks(numpy.arange(0, xmax, 5.0))
        matplotlib.pyplot.setp(ax6.get_xticklabels(), rotation=45, fontsize=25)
        matplotlib.pyplot.setp(ax6.get_yticklabels(), rotation=0, fontsize=25)
        ax6.set_facecolor((0.1, 1.0, 0.7, 0.1))
        ax6.set_xlim(0, xmax)
        
        ax6_t = ax6.twiny()
        ax6_t.set_xticks(ax6.get_xticks()) 
        labels = []
        for i in tick_function_THz_to_wavens(ax6.get_xticks()):
            labels.append('%.1f'%i)
        ax6_t.set_xticklabels(labels)
        matplotlib.pyplot.setp(ax6_t.get_xticklabels(), rotation=45, fontsize=25)
        ax6_t.set_xlabel(r"Frequency [cm-1]", fontsize=30)
        
        matplotlib.pyplot.tight_layout()
        matplotlib.pyplot.savefig('%s/multitape_%s_zoom_%d_%d_F_stats.png'%(path_figs, label, start, end))
        matplotlib.pyplot.close()
    


def multitape_zoom(neg_integral, pos_integral, nmaps, path, label):
    step = 10
    t0_idx = 43500
    t_idx_left  = 50000 #105000#178000
    t_idx_right = 99990 #145000#195000
    def tick_function(X):        
        step_fs = (385.0+335.0)/213557
        V = (X*step+t_idx_left-t0_idx)*step_fs
        return ["%.3f" % z for z in V]
    def tick_function_THz_to_wavens(f_THz):
        c = 33.35641
        return c*f_THz
            
    fcrit_value = 0.80
    
    if 'm_0_2' in label: 
        m = 2          
    if 'm_0_1' in label: 
        m = 1
        
    path_figs = '%s/multitape_m_0_%d_fcrit_%.2f_range_%d_%d'%(path, m, fcrit_value, t_idx_left, t_idx_right)
    
    if not os.path.exists(path_figs):
        os.mkdir(path_figs)
    
    data = neg_integral[:]
    delta_s = (10**(-15))*step*(385.0+335.0)/213557 #s
    xmax = 200 # THz, for figures
    
    fig = matplotlib.pyplot.figure(figsize=(35,40))
    
    # TIME DOMAIUN
    ax1 = fig.add_subplot(6, 1, 1)
    ax1.set_xlim(t_idx_left/step, t_idx_right/step)
    print data.shape
    print t_idx_left/10, (t_idx_right+step)/step
    ax1.plot(range(t_idx_left/step, (t_idx_right+step)/step), data, color='m')
    matplotlib.pyplot.setp(ax1.get_xticklabels(), rotation=45, fontsize=25)
    matplotlib.pyplot.setp(ax1.get_yticklabels(), rotation=0, fontsize=25)
    ax1.set_xlabel(r"Frame n. / %d"%step, fontsize=30)
    ax1.text(0.01, 
              0.95, 
              'Modes: %s + %s, %s sigma, %s'%(label[2], label[4], label[36:39], label[40:44]), 
              transform=ax1.transAxes, 
              fontsize=30,
              verticalalignment='top')#, bbox=props)
 
    ax1_t = ax1.twiny()
    new_tick_locations = numpy.array([(t_idx_left-t_idx_left)/step, (t_idx_left-t_idx_left+t_idx_right-t_idx_left)/(2*step), (t_idx_right-t_idx_left)/step])
    ax1_t.set_xticks(new_tick_locations)        
    ax1_t.set_xticklabels(tick_function(new_tick_locations))
    matplotlib.pyplot.setp(ax1_t.get_xticklabels(), rotation=0, fontsize=25)
    ax1_t.set_xlabel(r"Time [fs]", fontsize=30)
    
    # MULTITAPER ANALYSIS, WITHOUT F-STATS
    spec, freq, jackknife, _, _ = mtspec(data=data, 
                                          delta=delta_s, 
                                          time_bandwidth=3.5,
                                          number_of_tapers=4, 
                                          nfft=2*data.shape[0], 
                                          statistics=True)
    
    # MULTITAPER, LOG-SCALE
    ax2 = fig.add_subplot(6, 1, 2)
    ax2.plot(freq/(10**12), spec, color='black')  
    ax2.set_yscale('log')      
    matplotlib.pyplot.xticks(numpy.arange(0, xmax, 5.0))
    matplotlib.pyplot.setp(ax2.get_xticklabels(), rotation=45, fontsize=25)
    matplotlib.pyplot.setp(ax2.get_yticklabels(), rotation=0, fontsize=25)
    ax2.fill_between(freq/(10**12), jackknife[:, 0], jackknife[:, 1],
                      color="blue", alpha=0.3)
    ax2.set_xlim(0, xmax)
    
    # MULTITAPER, LINEAR SCALE
    ax3 = fig.add_subplot(6, 1, 3)
    ax3.plot(freq/(10**12), spec, color='black')        
    matplotlib.pyplot.xticks(numpy.arange(0, xmax, 5.0))
    matplotlib.pyplot.setp(ax3.get_xticklabels(), rotation=45, fontsize=25)
    matplotlib.pyplot.setp(ax3.get_yticklabels(), rotation=0, fontsize=25)
    ax3.fill_between(freq/(10**12), jackknife[:, 0], jackknife[:, 1],
                      color="blue", alpha=0.3)
    ax3.set_xlim(0, xmax)
    
    # MULTITAPER ANALYSIS, WITH F-STATS
    spec, freq, jackknife, fstatistics, _ = mtspec(data=data, 
                                          delta=delta_s, 
                                          time_bandwidth=3.5,
                                          number_of_tapers=4, 
                                          nfft=2*data.shape[0], 
                                          statistics=True, 
                                          rshape=0, 
                                          fcrit=fcrit_value)
    
    # F-STATS
    ax4 = fig.add_subplot(6, 1, 4)
    ax4.set_ylabel("F-stats for periodic lines", fontsize=30)
    ax4.set_xlim(0, xmax)
    print fstatistics.shape
    idx = numpy.where(abs(freq/(10**12) - xmax) < 10)
    
    idx = idx[0][-1]
    print idx
    ax4.set_ylim(0, 1.1*numpy.amax(fstatistics[0:idx]))
    ax4.plot(freq/(10**12), fstatistics, color="c")
    #ax4.set_xlabel(r"Frequency [THz]", fontsize=30)
    matplotlib.pyplot.xticks(numpy.arange(0, xmax, 5.0))
    matplotlib.pyplot.setp(ax4.get_xticklabels(), rotation=45, fontsize=25)
    matplotlib.pyplot.setp(ax4.get_yticklabels(), rotation=0,  fontsize=25)
    
    # Plot the confidence intervals.
    for p in [100*fcrit_value]:
        y = numpy.percentile(fstatistics, p)
        matplotlib.pyplot.hlines(y, 0, xmax, linestyles="--", color="0.2")
        matplotlib.pyplot.text(x=1, y=y+0.2, s="%i %%"%p, ha="left", fontsize=25)
    
    # MULTITAPER WITH F-STATS, LOG-SCALE
    ax5 = fig.add_subplot(6, 1, 5)
    ax5.set_yscale('log')
    ax5.plot(freq/(10**12), spec, color='b')        
    matplotlib.pyplot.xticks(numpy.arange(0, xmax, 5.0))
    matplotlib.pyplot.setp(ax5.get_xticklabels(), rotation=45, fontsize=25)
    matplotlib.pyplot.setp(ax5.get_yticklabels(), rotation=0, fontsize=25)
    ax5.set_facecolor((0.1, 1.0, 0.7, 0.1))
    ax5.set_xlim(0, xmax)
     
    # MULTITAPER WITH F-STATS, LINEAR SCALE
    ax6 = fig.add_subplot(6, 1, 6)
    ax6.plot(freq/(10**12), spec, color='b')        
    ax6.set_xlabel(r"Frequency [THz]", fontsize=30)    
    matplotlib.pyplot.xticks(numpy.arange(0, xmax, 5.0))
    matplotlib.pyplot.setp(ax6.get_xticklabels(), rotation=45, fontsize=25)
    matplotlib.pyplot.setp(ax6.get_yticklabels(), rotation=0, fontsize=25)
    ax6.set_facecolor((0.1, 1.0, 0.7, 0.1))
    ax6.set_xlim(0, xmax)
    
    ax6_t = ax6.twiny()
    ax6_t.set_xticks(ax6.get_xticks()) 
    labels = []
    for i in tick_function_THz_to_wavens(ax6.get_xticks()):
        labels.append('%.1f'%i)
    ax6_t.set_xticklabels(labels)
    matplotlib.pyplot.setp(ax6_t.get_xticklabels(), rotation=45, fontsize=25)
    ax6_t.set_xlabel(r"Frequency [cm-1]", fontsize=30)
    
    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.savefig('%s/multitape_%s_zoom_%d_%d_F_stats.png'%(path_figs, label, t_idx_left, t_idx_right))
    matplotlib.pyplot.close()   
    
    
    
def plot_whole_density(meanposden, meannegden, path, label):
    whole_posden = numpy.sum(meanposden, axis=0)
    whole_negden = numpy.sum(meannegden, axis=0)
    nmaps = meannegden.shape[1]
    plot(whole_negden, whole_posden, nmaps, path, label)
    

def plot_chain_density(meanposden, meannegden, path, label, start, end):
    whole_posden = numpy.sum(meanposden[start:end+1,:], axis=0)
    whole_negden = numpy.sum(meannegden[start:end+1,:], axis=0)
    nmaps = meannegden.shape[1]
    plot(whole_negden, whole_posden, nmaps, path, label)



def plot_atoms_density(meanposden, meannegden, path, label, atoms, pospeak_idx):    
    
    print 'meanposden: ', meanposden.shape
    print 'meannegden: ', meannegden.shape
    
    #whole_posden = meanposden[pospeak_idx, :]
    whole_posden = numpy.sum(meanposden[atoms,:], axis=0)
    whole_negden = numpy.sum(meannegden[atoms,:], axis=0)
    nmaps = meannegden.shape[1]
    #plot(whole_negden, whole_posden, nmaps, path, label)
    #plot_zoom(whole_negden, whole_posden, nmaps, path, label)
    #multitape(whole_negden, whole_posden, nmaps, path, label)
    multitape_zoom(whole_negden, whole_posden, nmaps, path, label)

            
if __name__ == '__main__':
    radius = 1.7
    distance = 0.2
    sigcutoffs = [2.5, 3.0, 3.5, 4.0, 4.5]
    modes = [1]
    
    for mode in modes:
        for sigcutoff in sigcutoffs:
            # path = '/das/work/p18/p18594/cecilia-offline/NLSA/data_rho_2/results_NLSA/map_analysis/results_m_0_%d'%mode
            # label = 'm_0_%d_radius_%.1f_dist_%.1f_sigcutoff_%.1f'%(mode, radius, distance, sigcutoff)
            # fn_p = '%s/meanposden_%s.jbl'%(path, label)
            # fn_n = '%s/meannegden_%s.jbl'%(path, label)
            # meanposden = joblib.load(fn_p)
            # meannegden = joblib.load(fn_n)
            
            # flag = 0
            # if flag == 1:
            #     plot_whole_density(meanposden, meannegden, path, label)
            
            
            # flag = 0
            # if flag == 1:
            #     # RET A
            #     RETA_start = 2421
            #     RETA_end = 2440+1
            #     RETA_label = '%s_RETA'%label
            #     plot_chain_density(meanposden, meannegden, path, RETA_label, RETA_start, RETA_end)
                
            #     # RET B
            #     RETB_start = 4851
            #     RETB_end = 4870+1
            #     RETB_label = '%s_RETB'%label
            #     plot_chain_density(meanposden, meannegden, path, RETB_label, RETB_start, RETB_end)
            
            
            path = '/das/work/p18/p18594/cecilia-offline/NLSA/data_rho_2/results_NLSA/map_analysis/results_m_0_%d_step_10_range_50000_99990'%mode
            label = 'm_0_%d_radius_%.1f_dist_%.1f_sigcutoff_%.1f'%(mode, radius, distance, sigcutoff)
            fn_p = '%s/meanposden_%s.jbl'%(path, label)
            fn_n = '%s/meannegden_%s.jbl'%(path, label)
            meanposden = joblib.load(fn_p)
            meannegden = joblib.load(fn_n)
            
            flag = 1
            if flag == 1:
                #RET A C11, C12, C20
                RETA_atoms = [2423, 2424, 2433]
                RETA_pospeak_idx = 0
                RETA_label = '%s_RETA_neg_C11_C12_C20_pos_peakcentered'%label
                plot_atoms_density(meanposden, meannegden, path, RETA_label, RETA_atoms, RETA_pospeak_idx)
                
                #RET B C11, C12, C20
                RETB_atoms = [4853, 4854, 4863]
                RETB_pospeak_idx = 1
                RETB_label = '%s_RETB_neg_C11_C12_C20_pos_peakcentered'%label
                plot_atoms_density(meanposden, meannegden, path, RETB_label, RETB_atoms, RETB_pospeak_idx)