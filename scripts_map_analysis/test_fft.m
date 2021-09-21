clear all;

sampling_frequency = 1000;
sampling_period = 1/sampling_frequency;
n_points = 150;
t_vector = sampling_period*(0:n_points-1);

% SIGNAL
S = 0.7*sin(2*pi*50*t_vector) + sin(2*pi*120*t_vector);

%plot(t_vector,S)
plot(t_vector(1:100),S(1:100))
title('Signal')
xlabel('t (s)')
ylabel('X(t)')

n_points_fft = 1*n_points;
Y = fft(S, n_points_fft);

plot((0:n_points_fft-1),Y)
title('fft')
xlabel('frequency (vector index)')
ylabel('fft')

P2 = abs(Y/n_points_fft);

plot((0:n_points_fft-1),P2)
title('Two-sided spectrum P2')
ylabel('P2')
xlabel('frequency (vector index)')

P1 = P2(1:n_points_fft/2+1);
P1(2:end-1) = 2*P1(2:end-1);

f = sampling_frequency*(0:(n_points_fft/2))/n_points_fft;


plot(f,P1) 
title('Single-Sided Amplitude Spectrum of X(t)')
xlabel('f (Hz)')
ylabel('|P1(f)|')

