% script_generate_uniform_delay_gaussian
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This code generates a uniform distribution for delay time
% by modeling each timestamp as a gaussian distribution centered on the
% value from the timing tool.
% C. Casadei, April-2019, February-2021
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear;

% SIGMA VALUE FOR TIMESTAMP GAUSSIAN MODELING
sigma = 5.0                                                                %#ok<NOPTS>   
FWHM = 2.355 * sigma                                                       %#ok<NOPTS>

% LOAD DATA
path = '/das/work/p18/p18594/cecilia-offline/NLSA/data_rho_2/data_extraction/';
folder = '';
fn = 'data_rho_light_sorted_SCL_nS278263_nBrg74887';          
data_file = [path, folder, fn, '.mat'];
load(data_file, 'M_scl', ...
                'T_scl', ...
                'miller_h', ...
                'miller_k', ...
                'miller_l')
            
% LOAD ALL LIGHT DATA TIMESTAMPS
fn = [path, folder, 'timestamps_light_sorted.mat']; 
load(fn, 'timestamps_light_sort');

fn = [path, folder, 'drls_reflection_n_light_sorted.mat']; 
load(fn, 'drls_light_sort', ...
         'reflection_ns_light_sort')
     
fn = [path, folder, 'partialator_OK_light_sorted.mat'];      
load(fn, 'scales_light_sort', ...
         'uniqueID_light_sort')                  

timestamps_light = timestamps_light_sort;
clear timestamps_light_sort
drls_light = drls_light_sort;
clear drls_light_sort
reflection_ns_light = reflection_ns_light_sort;
clear reflection_ns_light_sort
scales_light = scales_light_sort;
clear scales_light_sort
uniqueID_light = uniqueID_light_sort;
clear uniqueID_light_sort

% ORIGINAL DISTRIBUTION                   
N_ts = length(timestamps_light);
t = linspace(-800, 800, 1600);
G = zeros(N_ts, length(t));
N = 1/(sigma*sqrt(2*pi));
for i =1:N_ts
    t0 = timestamps_light(i);
	G(i,:) = N * exp(-(0.5*(t-t0).*(t-t0))/(sigma*sigma));
end
S = sum(G);
plot(t,S);
hold;
% saveas(gcf, 'sum_gaussians_all.jpg')
% close(gcf);

% % SELECT ONLY TIMESTAMPS IN RANGE:
% idx_ts_range = find(timestamps_light < 465 & ...
%                     timestamps_light > -335);
idx_ts_range = find(timestamps_light < 385 & ...
                    timestamps_light > -335);
                
timestamps = timestamps_light(idx_ts_range);
M_temp = M_scl(idx_ts_range, :);                
T_temp = T_scl(idx_ts_range, :);               
drls_light_temp = drls_light(idx_ts_range);     
scales_light_temp = scales_light(idx_ts_range, :);   
reflection_ns_light_temp = reflection_ns_light(idx_ts_range); 
uniqueID_light_temp = uniqueID_light(idx_ts_range);            

% BUILD G MATRIX (N_timestamps X 2000)
% EACH ROW IS A GAUSSIAN [-1000 fs, +1000 fs] 
% CENTERED ON THE CORRESPONDING TIMESTAMP
N_ts = length(timestamps);
t = linspace(-800, 800, 1600);
G = zeros(N_ts, length(t));
N = 1/(sigma*sqrt(2*pi));
for i =1:N_ts
    t0 = timestamps(i);
	G(i,:) = N * exp(-(0.5*(t-t0).*(t-t0))/(sigma*sigma));
end

% SUM ALL THE GAUSSIANS
% S = sum(G);
% plot(t,S);
% saveas(gcf, 'sum_gaussians_all.jpg')
% close(gcf);

% FIND SUM MINIMUM VALUE IN [0, 410] fs
% t1 = 0.0;
% t2 = 3900.0;
% [Min1, I_t1] = min(abs(t-t1));
% [Min2, I_t2] = min(abs(t-t2));
% % v = min(S(I_t1:I_t2));
v = 290; 
% SHUFFLE THE TIMESTAMPS
s = RandStream('mlfg6331_64'); 
order = randperm(s, N_ts);

thr_factors = 10; %[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20];
for j=1:length(thr_factors)
    j
    % ADD THE TIMESTAMP-CENTRED GAUSSIANS ONE AFTER THE OTHER 
    % GENERATING A FLAT DISTRIBUTION
    thr_factor = thr_factors(j)                                            %#ok<NOPTS>
    thr = N/thr_factor;
    
    S_uniform   = zeros(1, length(t));
    S_discarded = zeros(1, length(t));
    t_uniform   = [];
    idx_uniform = [];
    n_uniform   = 0;
    n_discarded = 0;

    top = 0;
    for i=1:N_ts
         idx = order(i);
         t0 = timestamps(idx);
         S_uniform = S_uniform + G(idx, :);
         if (~isempty(find(S_uniform > v, 1)) && max(S_uniform - top)>thr)
             % discard
             S_uniform   = S_uniform - G(idx, :); 
             S_discarded = S_discarded + G(idx, :);
             n_discarded = n_discarded + 1;
         else
             % keep
             idx_uniform = [idx_uniform; idx];                             %#ok<AGROW>
             t_uniform   = [t_uniform; t0];                                %#ok<AGROW>
             n_uniform   = n_uniform + 1;
         end
         top = v + max(S_uniform - v);
    end
    
    % PLOT
%     plot(t, S_uniform+S_discarded);
%     hold;
    plot(t, S_discarded);
    plot(t, S_uniform);
    fn = sprintf('Sum_gaussians_sigma_%d_fs_n_in_%d_thr_factor_%d.jpg', ...
                  int16(sigma), ...
                  n_uniform, ...
                  thr_factor)                                              %#ok<NOPTS>
    saveas(gcf, fn);
    close(gcf);

    % SELECT DATA CONTRIBUTING TO UNIFORM DISTRIBUTION
    M = M_temp(idx_uniform, :);
    T = T_temp(idx_uniform, :);    
    drls_light_uniform = drls_light_temp(idx_uniform);     
    scales_light_uniform = scales_light_temp(idx_uniform, :);   
    reflection_ns_light_uniform = reflection_ns_light_temp(idx_uniform); 
    uniqueID_light_uniform = uniqueID_light_temp(idx_uniform);  
    
    % SORT!
    [t_uniform, I] = sort(t_uniform);
    
    M = M(I,:);
    T = T(I,:);   
    drls_light_uniform = drls_light_uniform(I);
    scales_light_uniform = scales_light_uniform(I, :);
    reflection_ns_light_uniform = reflection_ns_light_uniform(I);
    uniqueID_light_uniform = uniqueID_light_uniform(I);
    
    % SAVE
    fn = sprintf('data_rho_light_SCL_unifdelay_nS%d_nBrg%d', ...
                  n_uniform, ...
                  size(M, 2));         
    flat_data_file = [path, folder, fn, '.mat'];
    save(flat_data_file, ...
        'T', ...
        'M', ...
        'miller_h', ...
        'miller_k', ...
        'miller_l', ...
        't_uniform', ...
        '-v7.3');
    
%     fn = 'sortedInfo_dark';
%     dark_fn = [path, folder, fn, '.mat'];
%     save(dark_fn, ...
%          'drls_dark', ...
%          'scales_dark', ...
%          'reflection_ns_dark', ...
%          'uniqueID_dark', ...
%          '-v7.3');
     
    fn = sprintf('metadata_light_uniform_nS%d', n_uniform);
    light_fn = [path, folder, fn, '.mat'];
    save(light_fn, ...
         'drls_light_uniform', ...
         'scales_light_uniform', ...
         'reflection_ns_light_uniform', ...
         'uniqueID_light_uniform', ...
         't_uniform', ...
         '-v7.3');

end