# -*- coding: utf-8 -*-
import joblib
import numpy
import matplotlib
matplotlib.use('Agg') # Force matplotlib to not use any Xwindows backend.
matplotlib.rcParams['agg.path.chunksize'] = 10000
import matplotlib.pyplot


def plot_distributions(settings):

    results_path = settings.results_path
    f_max = settings.f_max_considered
    
    D_sq = joblib.load('%s/D_sq_lp_filtered_fmax_%d.jbl'%(results_path, f_max))   
    print 'D_sq: ', D_sq.shape, D_sq.dtype 
    
    idxs = numpy.triu_indices_from(D_sq)    
    print 'idxs: ', len(idxs), len(idxs[0]), len(idxs[1])
    
    D_flat = numpy.sqrt(D_sq[idxs].flatten())
    print 'D flattened: ', D_flat.shape
    
    [p99] = numpy.percentile(D_flat, [99])
    mybins = numpy.linspace(0, p99, num=400, endpoint=True)    
    
    matplotlib.pyplot.figure(figsize=(10,10))
    matplotlib.pyplot.xlim((0, p99))
    matplotlib.pyplot.hist(D_flat,bins=mybins,color='b')
    matplotlib.pyplot.savefig('%s/D_lp_filtered_fmax_%d_unique_hist.png'%(results_path, f_max))
    matplotlib.pyplot.close()
    
    D_nn_r = numpy.sqrt(numpy.diag(D_sq, k=1 )[1:])
    print 'D_nn_r', D_nn_r.shape
    D_nn_l = numpy.sqrt(numpy.diag(D_sq, k=-1)[:-1])
    print 'D_nn_l', D_nn_l.shape
    
    avg_nn_D = (D_nn_r+D_nn_l)/2
    print 'avg_nn_D', avg_nn_D.shape    # S-q-1 = s-1
    
    fig = matplotlib.pyplot.figure(figsize=(40,30))
    
    ax1 = fig.add_subplot(311)
    ax1.set_xlim((0, p99))
    ax1.tick_params(axis='both', which='major', labelsize=26)
    ax1.tick_params(axis='both', which='minor', labelsize=26)
    ax1.hist(D_flat,bins=mybins,color='b')
    
    ax2 = fig.add_subplot(312)
    ax2.set_xlim((0, p99))
    ax2.tick_params(axis='both', which='major', labelsize=26)
    ax2.tick_params(axis='both', which='minor', labelsize=26)
    ax2.hist(avg_nn_D,bins=mybins,color='b')
    
    ax3 = fig.add_subplot(313)
    ax3.set_xlim((0, p99))
    ax3.tick_params(axis='both', which='major', labelsize=26)
    ax3.tick_params(axis='both', which='minor', labelsize=26)
    ax3.set_yscale('log', nonposy='clip')
    ax3.hist(D_flat,bins=mybins,color='b')
    
    matplotlib.pyplot.savefig('%s/D_lp_filtered_fmax_%d_unique_and_nns_hist_log.png'%(results_path, f_max))
    matplotlib.pyplot.close()


def plot_d_0j(settings): 
    results_path = settings.results_path
    idxs = [0, 5000, 15000]
    d_sq = joblib.load('%s/d_sq.jbl'%(results_path)) 
    print numpy.amax(d_sq), numpy.amin(d_sq)
    print numpy.amax(numpy.diag(d_sq)), numpy.amin(numpy.diag(d_sq))
    
    numpy.fill_diagonal(d_sq, 0)
    print numpy.amax(d_sq), numpy.amin(d_sq)
    print numpy.amax(numpy.diag(d_sq)), numpy.amin(numpy.diag(d_sq))
    
    d = numpy.sqrt(d_sq)
    
    for idx in idxs:
    
        d_0j = d[idx,:]
        print 'd_0j: ', d_0j.shape, d_0j.dtype   
        
        matplotlib.pyplot.figure(figsize=(20,10))
        matplotlib.pyplot.scatter(range(d_0j.shape[0]), d_0j, c='b', s=1)        
        ax = matplotlib.pyplot.gca()
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.tick_params(axis='both', which='minor', labelsize=20)
        matplotlib.pyplot.savefig('%s/dist_%d_j.png'%(results_path, idx))
        matplotlib.pyplot.close()

def plot_D_0j(settings): 
    results_path = settings.results_path
    idxs = [0, 5000, 15000]
    d_sq = joblib.load('%s/D_sq_normalised.jbl'%(results_path)) 
    print numpy.amax(d_sq), numpy.amin(d_sq)
    print 'Diag:', numpy.amax(numpy.diag(d_sq)), numpy.amin(numpy.diag(d_sq))
    
    numpy.fill_diagonal(d_sq, 0)
    print numpy.amax(d_sq), numpy.amin(d_sq)
    print 'Diag:', numpy.amax(numpy.diag(d_sq)), numpy.amin(numpy.diag(d_sq))
    
    d = numpy.sqrt(d_sq)
    
    for idx in idxs:
    
        d_0j = d[idx,:]
        print 'd_0j: ', d_0j.shape, d_0j.dtype   
        
        matplotlib.pyplot.figure(figsize=(20,10))
        matplotlib.pyplot.plot(range(d_0j.shape[0]), d_0j, c='b', ms=1)        
        ax = matplotlib.pyplot.gca()
        ax.tick_params(axis='both', which='major', labelsize=38)
        ax.tick_params(axis='both', which='minor', labelsize=38)
        #ax.set_ylim([140, 180])
        matplotlib.pyplot.savefig('%s/D_%d_j.png'%(results_path, idx))
        matplotlib.pyplot.close()
        
def plot_D_0j_zoom(settings): 
    results_path = settings.results_path
    idxs = [0, 5000, 15000]
    d_sq = joblib.load('%s/D_sq_parallel.jbl'%(results_path)) 
    print numpy.amax(d_sq), numpy.amin(d_sq)
    print 'Diag:', numpy.amax(numpy.diag(d_sq)), numpy.amin(numpy.diag(d_sq))
    
    numpy.fill_diagonal(d_sq, 0)
    print numpy.amax(d_sq), numpy.amin(d_sq)
    print 'Diag:', numpy.amax(numpy.diag(d_sq)), numpy.amin(numpy.diag(d_sq))
    
    d = numpy.sqrt(d_sq)
    
    for idx in idxs:
    
        d_0j = d[idx,:]
        print 'd_0j: ', d_0j.shape, d_0j.dtype   
        
        matplotlib.pyplot.figure(figsize=(20,10))
        matplotlib.pyplot.scatter(range(d_0j.shape[0]), d_0j, c='b', s=1)        
        ax = matplotlib.pyplot.gca()
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.tick_params(axis='both', which='minor', labelsize=20)
        ax.set_ylim([50, 70])
        matplotlib.pyplot.savefig('%s/D_%d_j_zoom.png'%(results_path, idx))
        matplotlib.pyplot.close()
                        
            
def plot_D_0j_lp_filter(settings): 
    results_path = settings.results_path
    idxs = [0, 5000, 15000]
    f_max = settings.f_max_considered
    
    D_sq = joblib.load('%s/D_sq_lp_filtered_fmax_%d.jbl'%(results_path, f_max))
    print numpy.amax(D_sq), numpy.amin(D_sq)
    print 'Diag:', numpy.amax(numpy.diag(D_sq)), numpy.amin(numpy.diag(D_sq))
   
    for idx in idxs:        
        D_0j = numpy.sqrt(D_sq[idx,:])           
        print 'D_0j: ', D_0j.shape, D_0j.dtype               
        matplotlib.pyplot.figure(figsize=(20,10))
        matplotlib.pyplot.scatter(range(D_0j.shape[0]), D_0j, c='b', s=1)   
        ax = matplotlib.pyplot.gca()
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.tick_params(axis='both', which='minor', labelsize=20)
        matplotlib.pyplot.savefig('%s/D_%d_j_lp_filtered_fmax_%d.png'%(results_path, idx, f_max))
        matplotlib.pyplot.close()

    
def plot_D_0j_lp_filter_zoom(settings): 
    results_path = settings.results_path
    idxs = [0, 5000, 15000]
    f_max = settings.f_max_considered
    
    D_sq = joblib.load('%s/D_sq_lp_filtered_fmax_%d.jbl'%(results_path, f_max))
    print numpy.amax(D_sq), numpy.amin(D_sq)
    print 'Diag:', numpy.amax(numpy.diag(D_sq)), numpy.amin(numpy.diag(D_sq))
   
    for idx in idxs:        
        D_0j = numpy.sqrt(D_sq[idx,:])           
        print 'D_0j: ', D_0j.shape, D_0j.dtype               
        matplotlib.pyplot.figure(figsize=(20,10))
        matplotlib.pyplot.scatter(range(D_0j.shape[0]), D_0j, c='b', s=1)   
        ax = matplotlib.pyplot.gca()
        ax.tick_params(axis='both', which='major', labelsize=20)
        ax.tick_params(axis='both', which='minor', labelsize=20)
        ax.set_ylim([125, 175])
        matplotlib.pyplot.savefig('%s/D_%d_j_lp_filtered_fmax_%d_zoom.png'%(results_path, idx, f_max))
        matplotlib.pyplot.close()
  
    
def plot_D_0j_recap_log(settings): 
    results_path = settings.results_path
    f_max_lst = [10, 25, 50, 75, 100, 150, 200]
    fig = matplotlib.pyplot.figure(figsize=(20,70))    

    for i, f_max in enumerate(f_max_lst):
        
        D_0j = numpy.sqrt(joblib.load('%s/D_sq_lp_filtered_fmax_%d.jbl'%(results_path, f_max))[0,:])    
        print 'D_0j: ', D_0j.shape, D_0j.dtype   
                
        ax = fig.add_subplot(7,1,i+1)
        ax.scatter(range(D_0j.shape[0]), numpy.log10(D_0j), c='b', s=1)  
        ax.tick_params(axis='both', which='major', labelsize=26)
        ax.tick_params(axis='both', which='minor', labelsize=26)
        
        new_tick_locations = numpy.array([10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 110000])

        def tick_function(X):
            V = 1/(1+X)
            V = (X-16079)*(1000.0/97508)
            return ["%.1f" % z for z in V]
        ax2 = ax.twiny()

        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(new_tick_locations)
        ax2.set_xticklabels(tick_function(new_tick_locations))
        ax2.tick_params(axis='both', which='major', labelsize=26)
        ax2.tick_params(axis='both', which='minor', labelsize=26)
        
        ax.text(0.01, 0.1, 'jmax = %d'%f_max, fontsize=26, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.savefig('%s/D_0j_lp_filtered_log.png'%(results_path))
    matplotlib.pyplot.close()
    
def plot_D_0j_recap_lin(settings): 
    results_path = settings.results_path
    f_max_lst = [10, 25, 50, 75, 100, 150, 200]
    fig = matplotlib.pyplot.figure(figsize=(20,70))    

    for i, f_max in enumerate(f_max_lst):
        idx = 100000 #0
        D_0j = numpy.sqrt(joblib.load('%s/D_sq_lp_filtered_fmax_%d.jbl'%(results_path, f_max))[idx,:])    
        print 'D_0j: ', D_0j.shape, D_0j.dtype   
                
        ax = fig.add_subplot(7,1,i+1)
        ax.scatter(range(D_0j.shape[0]), D_0j, c='b', s=1)  
        ax.tick_params(axis='both', which='major', labelsize=26)
        ax.tick_params(axis='both', which='minor', labelsize=26)
        
        new_tick_locations = numpy.array([10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 110000])

        def tick_function(X):
            V = 1/(1+X)
            V = (X-16079)*(1000.0/97508)
            return ["%.1f" % z for z in V]
        ax2 = ax.twiny()

        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(new_tick_locations)
        ax2.set_xticklabels(tick_function(new_tick_locations))
        ax2.tick_params(axis='both', which='major', labelsize=26)
        ax2.tick_params(axis='both', which='minor', labelsize=26)
        
        ax.text(0.01, 0.1, 'jmax = %d'%f_max, fontsize=26, horizontalalignment='left', verticalalignment='center', transform=ax.transAxes)
    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.savefig('%s/D_%d_j_lp_filtered_lin.png'%(results_path, idx))
    matplotlib.pyplot.close()
        
def plot_D_matrix(settings):
    results_path = settings.results_path
    f_max = settings.f_max_considered
    D = numpy.sqrt(joblib.load('%s/D_sq_lp_filtered_fmax_%d.jbl'%(results_path, f_max)))    
    print 'D: ', D.shape, D.dtype
    
    D_flat = D.flatten()
    print 'D flattened: ', D_flat.shape
    
    [p99] = numpy.percentile(D_flat, [99])
    D = numpy.asarray(D, dtype=numpy.float32)
    #matplotlib.pyplot.imshow(numpy.log10(D+1), cmap='jet', vmin=6.6, vmax=numpy.log10(p99))
    matplotlib.pyplot.imshow(numpy.log10(D+1), cmap='jet', vmin=0, vmax=numpy.log10(p99))
    matplotlib.pyplot.colorbar()
    matplotlib.pyplot.savefig('%s/D_lp_filtered_fmax_%d.png'%(results_path, f_max))
    matplotlib.pyplot.close()
    
def plot_D_submatrix(settings):
    results_path = settings.results_path
    f_max = settings.f_max_considered
    row_min = 10000
    row_max = 11000
    col_min = 11000
    col_max = 12000
    D = numpy.sqrt(joblib.load('%s/D_sq_lp_filtered_fmax_%d.jbl'%(results_path, f_max))[row_min:row_max, col_min:col_max])    
    print 'D: ', D.shape, D.dtype
    
    D_flat = D.flatten()
    print 'D flattened: ', D_flat.shape
    
    #[p99] = numpy.percentile(D_flat, [99])
    D = numpy.asarray(D, dtype=numpy.float32)
    matplotlib.pyplot.imshow(numpy.log10(D+1), cmap='jet', extent=[col_min,col_max,row_max,row_min])
    matplotlib.pyplot.colorbar()
    matplotlib.pyplot.savefig('%s/D_submatrix_%d_%d_%d_%d_lp_filtered_fmax_%d.png'%(results_path, row_min, row_max, col_min, col_max, f_max))
    matplotlib.pyplot.close()