# -*- coding: utf-8 -*-

### TEST SPECTRUM
# import spectrum
# import matplotlib
# import numpy

# # data = spectrum.data_cosine(N=1024, A=0.1, sampling=1024, freq=200)
# # #matplotlib.pyplot.scatter(range(data.shape[0]), data)
# # p = spectrum.Periodogram(data, sampling=1024)
# # p.run()
# # p.plot(marker='o')

# data = spectrum.data_cosine(N=2048, A=0.1, sampling=1024, freq=200)
# #matplotlib.pyplot.scatter(range(data.shape[0]), data)
# # If you already have the DPSS windows
# #[tapers, eigen] = dpss(2048, 2.5, 4)
# #res = pmtm(data, e=eigen, v=tapers, show=False)
# # You do not need to compute the DPSS before end
# #res = pmtm(data, NW=2.5, show=False)

# [sk, weights, evs] = spectrum.pmtm(data, NW=2.5, k=4, show=True)
# weighted_sk = numpy.real(sk.T * weights)
# test = numpy.mean(weighted_sk, axis=1)
# matplotlib.pyplot.scatter(range(test.shape[0]), test)
# #matplotlib.pyplot.xlim(300, 500)

# ### TEST mtspec WITH GEOLOGY EXAMPLE
import matplotlib.pyplot
import numpy
from mtspec import mtspec
from mtspec.util import _load_mtdata

# data = _load_mtdata('v22_174_series.dat.gz')
# print data.shape

# # Calculate the spectral estimation.
# spec, freq, jackknife, _, _ = mtspec(
#     data=data, delta=4930.0, time_bandwidth=3.5,
#     number_of_tapers=5, nfft=312, statistics=True)

# fig = plt.figure()
# ax1 = fig.add_subplot(2, 1, 1)
# ax1.plot(np.arange(len(data)) * 4.930, data, color='black') # Plot in thousands of years.
# ax1.set_xlim(0, 800)
# ax1.set_ylim(-1.0, 1.0)
# ax1.set_xlabel("Time [1000 years]")
# ax1.set_ylabel("Change in $\delta^{18}O$")
# ax2 = fig.add_subplot(2, 1, 2)
# ax2.set_yscale('log')
# freq *= 1E6 # Convert frequency to Ma.
# ax2.plot(freq, spec, color='black')
# ax2.fill_between(freq, jackknife[:, 0], jackknife[:, 1], color="red", alpha=0.3)
# ax2.set_xlim(freq[0], freq[-1])
# ax2.set_ylim(0.1E1, 1E5)
# ax2.set_xlabel("Frequency [c Ma$^{-1}]$")
# ax2.set_ylabel("Power Spectral Density ($\delta^{18}O/ca^{-1}$)")
# plt.tight_layout()
# plt.show()

### TEST mtspec WITH SUM OF SINES
print '\nTEST: Sum of sines'
start = 0 #s
stop  = 2 #s
n = 500 #500000
step = float(stop-start)/n 
print 'Time range [s]:', start, '-', stop
print 'N samples:', n
print 'Sampling period [s]:', step
print 'Sampling frequency [Hz]:', 1/step, '-> Nyquist frequency [Hz]:', 1/(2*step)
t = numpy.arange(start, stop, step=step)

phase = numpy.pi/4
f_1 = 15 # Hz
data_1 = numpy.sin(2*numpy.pi*f_1*t + 2*numpy.pi*t*t)#30*t)
# plt.figure()
# plt.gca().set_xlabel("Time [s]")
# plt.plot(t, data_1)

#f_2 = 40 # Hz
#data_2 = numpy.sin(2*numpy.pi*f_2*t)
# plt.figure()
# plt.gca().set_xlabel("Time [s]")
# plt.plot(t, data_2)

#f_3 = 3 # Hz
#data_3 = numpy.sin(2*numpy.pi*f_3*t)
# plt.figure()
# plt.gca().set_xlabel("Time [s]")
# plt.plot(t, data_3)

numpy.random.seed(42)
A = 0.0
randomlist = A*numpy.random.randn(t.shape[0])
data = randomlist + data_1 #+ data_2  #+ data_3 + randomlist
#print 'Components at', f_1, f_2, 'Hz, dephased by', phase, ' plus gaussian noise with amplitude', A
#print 'Components at', f_1, f_2, f_3, 'Hz, plus gaussian noise with amplitude', A

# plt.figure()
# plt.gca().set_xlabel("Time [s]")
# plt.plot(t, data)

# spec, freq, jackknife, fstatistics, _ = mtspec(
#     data=data, delta=step, time_bandwidth=3,
#     number_of_tapers=4, nfft=2*n, statistics=True, rshape=0, fcrit=0.9)

# fig = plt.figure()
# ax1 = fig.add_subplot(2, 1, 1)
# ax1.plot(t, data, color='black')
# ax2 = fig.add_subplot(2, 1, 2)
# ax2.set_yscale('log')
# ax2.plot(freq, spec, color='black')
# ax2.fill_between(freq, jackknife[:, 0], jackknife[:, 1],
#                   color="red", alpha=0.3)
# ax2.set_xlim(0, 80)
# ax2.set_ylim(1E-6, 1E0)
# plt.tight_layout()
# plt.show()



# BY INCREASING n THE NYQUIST FREQUENCY INCREASES, AND ALSO THE S/N OF PSD

fig = matplotlib.pyplot.figure(figsize=(35,40))
        
# TIME DOMAIUN
ax1 = fig.add_subplot(6, 1, 1)
#ax1.set_xlim(start, end)
ax1.plot(t, data, color='m')
matplotlib.pyplot.setp(ax1.get_xticklabels(), rotation=45, fontsize=25)
matplotlib.pyplot.setp(ax1.get_yticklabels(), rotation=0, fontsize=25)
ax1.set_xlabel(r"Time [s]", fontsize=30)
ax1.text(0.01, 
         0.95, 
         'Component at %.1f Hz, with phase quadratic in time, plus gaussian noise with amplitude %.1f\nnpoints: %d, sampling frequency: %.1f Hz '%(f_1, A, n, 1.0/step),
         #'Components at %.1f %.1f Hz, dephased by %.1f rad, plus gaussian noise with amplitude %.1f\nnpoints: %d, sampling frequency: %.1f Hz '%(f_1, f_2, phase, A, n, 1.0/step), 
         #'Components at %.1f %.1f %.1f Hz, plus gaussian noise with amplitude %.1f\nnpoints: %d, sampling frequency: %.1f Hz '%(f_1, f_2, f_3, A, n, 1.0/step), 
         #'Only gaussian noise'
         transform=ax1.transAxes, 
         fontsize=30,
         verticalalignment='top')#, bbox=props)
     
        
# MULTITAPER ANALYSIS, WITHOUT F-STATS
spec, freq, jackknife, _, _ = mtspec(data=data, 
                                     delta=step, 
                                     time_bandwidth=3,
                                     number_of_tapers=4, 
                                     nfft=2*data.shape[0], 
                                     statistics=True)

print 'Freq min, max [Hz]:', freq[0], freq[-1]
print 'Freq step:', freq[1]-freq[0]

xmax = 80

# MULTITAPER, LOG-SCALE
ax2 = fig.add_subplot(6, 1, 2)
ax2.plot(freq, spec, color='black')  
ax2.set_yscale('log')      
matplotlib.pyplot.xticks(numpy.arange(0, xmax, 5.0))
matplotlib.pyplot.setp(ax2.get_xticklabels(), rotation=45, fontsize=25)
matplotlib.pyplot.setp(ax2.get_yticklabels(), rotation=0, fontsize=25)
ax2.fill_between(freq, jackknife[:, 0], jackknife[:, 1],
                  color="blue", alpha=0.3)
ax2.set_xlim(0, xmax)
        
# MULTITAPER, LINEAR SCALE
ax3 = fig.add_subplot(6, 1, 3)
ax3.plot(freq, spec, color='black')        
matplotlib.pyplot.xticks(numpy.arange(0, xmax, 5.0))
matplotlib.pyplot.setp(ax3.get_xticklabels(), rotation=45, fontsize=25)
matplotlib.pyplot.setp(ax3.get_yticklabels(), rotation=0, fontsize=25)
ax3.fill_between(freq, jackknife[:, 0], jackknife[:, 1],
                  color="blue", alpha=0.3)
ax3.set_xlim(0, xmax)

# MULTITAPER ANALYSIS, WITH F-STATS
spec, freq, jackknife, fstatistics, _ = mtspec(data=data, 
                                      delta=step, 
                                      time_bandwidth=3.5,
                                      number_of_tapers=4, 
                                      nfft=2*data.shape[0], 
                                      statistics=True, 
                                      rshape=0, 
                                      fcrit=0.90)

# F-STATS
ax4 = fig.add_subplot(6, 1, 4)
ax4.set_ylabel("F-stats for periodic lines", fontsize=30)
ax4.set_xlim(0, xmax)
idx = numpy.where(abs(freq - xmax) < 10)
idx = idx[0][-1]
print idx
ax4.set_ylim(0, 1.1*numpy.amax(fstatistics[0:idx]))
ax4.plot(freq, fstatistics, color="c")
#ax4.set_xlabel(r"Frequency [THz]", fontsize=30)
matplotlib.pyplot.xticks(numpy.arange(0, xmax, 5.0))
matplotlib.pyplot.setp(ax4.get_xticklabels(), rotation=45, fontsize=25)
matplotlib.pyplot.setp(ax4.get_yticklabels(), rotation=0,  fontsize=25)

# Plot the confidence intervals.
for p in [100*0.90]:
    y = numpy.percentile(fstatistics, p)
    matplotlib.pyplot.hlines(y, 0, xmax, linestyles="--", color="0.2")
    matplotlib.pyplot.text(x=1, y=y+0.2, s="%i %%"%p, ha="left", fontsize=25)

# MULTITAPER WITH F-STATS, LOG-SCALE
ax5 = fig.add_subplot(6, 1, 5)
ax5.set_yscale('log')
ax5.plot(freq, spec, color='b')        
matplotlib.pyplot.xticks(numpy.arange(0, xmax, 5.0))
matplotlib.pyplot.setp(ax5.get_xticklabels(), rotation=45, fontsize=25)
matplotlib.pyplot.setp(ax5.get_yticklabels(), rotation=0, fontsize=25)
ax5.set_facecolor((0.1, 1.0, 0.7, 0.1))
ax5.set_xlim(0, xmax)
 
# MULTITAPER WITH F-STATS, LINEAR SCALE
ax6 = fig.add_subplot(6, 1, 6)
ax6.plot(freq, spec, color='b')        
ax6.set_xlabel(r"Frequency [Hz]", fontsize=30)    
matplotlib.pyplot.xticks(numpy.arange(0, xmax, 5.0))
matplotlib.pyplot.setp(ax6.get_xticklabels(), rotation=45, fontsize=25)
matplotlib.pyplot.setp(ax6.get_yticklabels(), rotation=0, fontsize=25)
ax6.set_facecolor((0.1, 1.0, 0.7, 0.1))
ax6.set_xlim(0, xmax)

matplotlib.pyplot.tight_layout()
#matplotlib.pyplot.savefig('./test_multitape_sines_plus_noise_A_%.1f_n_%d.png'%(A, n))
matplotlib.pyplot.savefig('./test_multitape_sine_plus_noise_A_%.1f_n_%d_quadraticphase.png'%(A, n))
#matplotlib.pyplot.savefig('./test_multitape_only_noise_n_%d.png'%(n))
matplotlib.pyplot.close()
    