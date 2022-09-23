% script_generate_uniform_delay_gaussian
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% C. Casadei, Jun-2022
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;

% SIGMA VALUE FOR TIMESTAMP GAUSSIAN MODELING
sigma = 15.0                                                               %#ok<NOPTS>   
FWHM = 2.355 * sigma                                                       %#ok<NOPTS>

% LOAD DATA
path = './';
folder = '';
fn = 'data_bR_light_int_SCL_nS219559_nBrg53620';          
data_file = [path, folder, fn, '.mat'];
load(data_file, 'M_scl', ...
                'T_scl', ...
                'miller_h', ...
                'miller_k', ...
                'miller_l')
            
% LOAD ALL LIGHT DATA TIMESTAMPS
fn = 'sorted_info';
time_delays_file = [path, folder, fn, '.mat']; 
load(time_delays_file, 'timestamps_light_sort');

% LOAD METADATA
load(time_delays_file, 'scales_dark', ...
                       'reflection_ns_dark', ...
                       'uniqueID_dark')
load(time_delays_file, 'scales_light_sort', ...
                       'reflection_ns_light_sort', ...
                       'uniqueID_light_sort')


% BUILD G MATRIX (N_timestamps X 2500)
% EACH ROW IS A GAUSSIAN [-1000 fs, +1500 fs] 
% CENTERED ON THE CORRESPONDING TIMESTAMP
N_ts = length(timestamps_light_sort);
t = linspace(-1000, 1500, 2500);
G = zeros(N_ts, length(t));
N = 1/(sigma*sqrt(2*pi));
for i =1:N_ts
    t0 = timestamps_light_sort(i);
	G(i,:) = N * exp(-(0.5*(t-t0).*(t-t0))/(sigma*sigma));
end

% SUM ALL THE GAUSSIANS
S = sum(G);
plot(t,S);
saveas(gcf, 'frame_distribution.jpg')
close(gcf);

