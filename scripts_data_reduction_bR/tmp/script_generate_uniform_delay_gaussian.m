% script_generate_uniform_delay_gaussian
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This code generates a uniform distribution for delay time
% by modeling each timestamp as a gaussian distribution centered on the
% value from the timing tool.
% C. Casadei, April-2019
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear;

% SIGMA VALUE FOR TIMESTAMP GAUSSIAN MODELING
sigma = 40.0                                                               %#ok<NOPTS>   
FWHM = 2.355 * sigma                                                       %#ok<NOPTS>

% LOAD DATA
path = '/das/work/p17/p17491/Cecilia_Casadei/TR-SFX/bR_TR_SFX/LCLS_2017_cxilp4115/';
folder = 'test_running_times/data_reduction_running_times/';
fn = 'data_bR_test_light_sorted_int_DRL_SCL_nS74437_nBrg52219';          
data_file = [path, folder, fn, '.mat'];
load(data_file, 'M_drl_scl', ...
                'T_drl_scl', ...
                'miller_h', ...
                'miller_k', ...
                'miller_l')
            
% LOAD ALL LIGHT DATA TIMESTAMPS
fn = 'sorted_info';
time_delays_file = [path, folder, fn, '.mat']; 
load(time_delays_file, 'timestamps_light_sort');

% LOAD METADATA
load(time_delays_file, 'drls_dark', ...
                       'scales_dark', ...
                       'reflection_ns_dark', ...
                       'uniqueID_dark')
load(time_delays_file, 'drls_light_sort', ...
                       'scales_light_sort', ...
                       'reflection_ns_light_sort', ...
                       'uniqueID_light_sort')

% SELECT ONLY TIMESTAMPS BELOW 1500 fs
idx_ts_fsRange = find(timestamps_light_sort < 1500);
timestamps = timestamps_light_sort(idx_ts_fsRange);
M_temp = M_drl_scl(idx_ts_fsRange, :);                
T_temp = T_drl_scl(idx_ts_fsRange, :);               
drls_light_temp = drls_light_sort(idx_ts_fsRange);     
scales_light_temp = scales_light_sort(idx_ts_fsRange, :);   
reflection_ns_light_temp = reflection_ns_light_sort(idx_ts_fsRange); 
uniqueID_light_temp = uniqueID_light_sort(idx_ts_fsRange);            

% BUILD G MATRIX (N_timestamps X 2500)
% EACH ROW IS A GAUSSIAN [-1000 fs, +1500 fs] 
% CENTERED ON THE CORRESPONDING TIMESTAMP
N_ts = length(timestamps);
t = linspace(-1000, 1500, 2500);
G = zeros(N_ts, length(t));
N = 1/(sigma*sqrt(2*pi));
for i =1:N_ts
    t0 = timestamps(i);
	G(i,:) = N * exp(-(0.5*(t-t0).*(t-t0))/(sigma*sigma));
end

% SUM ALL THE GAUSSIANS
S = sum(G);
plot(t,S);
saveas(gcf, 'sum_gaussians.jpg')
close(gcf);

% FIND SUM MINIMUM VALUE IN [0, 1000] fs
t1 = 0.0;
t2 = 1000.0;
[Min1, I_t1] = min(abs(t-t1));
[Min2, I_t2] = min(abs(t-t2));
v = min(S(I_t1:I_t2));

% SHUFFLE THE TIMESTAMPS
order = randperm(N_ts);

thr_factors = 8; %[1, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 20];
for j=1:length(thr_factors)
    
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
             idx_uniform = [idx_uniform; idx];                                 %#ok<AGROW>
             t_uniform   = [t_uniform; t0];                                      %#ok<AGROW>
             n_uniform   = n_uniform + 1;
         end
         top = v + max(S_uniform - v);
    end
    
    % PLOT
    plot(t, S_uniform+S_discarded);
    hold;
    plot(t, S_discarded);
    plot(t, S_uniform);
    fn = sprintf('Sum_gaussians_sigma_%d_fs_n_in_%d_thr_factor_%d.jpg', ...
                  int16(sigma), ...
                  n_uniform, ...
                  thr_factor)                                              %#ok<NOPTS>
    saveas(gcf, fn);
    close(gcf);
    
    % COUNT NEGATIVE TIMESTAMPS
    N_neg      = length(find(t_uniform<0))                                        %#ok<NOPTS>
    N_neg_peak = length(find(t_uniform<-400))                                %#ok<NOPTS>
    
    % SELECT DATA CONTRIBUTING TO UNIFORM DISTRIBUTION
    M_drl_scl = M_temp(idx_uniform, :);
    T_drl_scl = T_temp(idx_uniform, :);    
    drls_light_uniform = drls_light_temp(idx_uniform);     
    scales_light_uniform = scales_light_temp(idx_uniform, :);   
    reflection_ns_light_uniform = reflection_ns_light_temp(idx_uniform); 
    uniqueID_light_uniform = uniqueID_light_temp(idx_uniform);  
    
    % SORT!
    [t_uniform, I] = sort(t_uniform);
    
    M_drl_scl = M_drl_scl(I,:);
    T_drl_scl = T_drl_scl(I,:);   
    drls_light_uniform = drls_light_uniform(I);
    scales_light_uniform = scales_light_uniform(I, :);
    reflection_ns_light_uniform = reflection_ns_light_uniform(I);
    uniqueID_light_uniform = uniqueID_light_uniform(I);
    
    % SAVE
    fn = sprintf('data_bR_test_light_int_unifdelay_gauss_sigma_%dfs_DRL_SCL_nS%d_nBrg%d', ...
                  int16(sigma), ... 
                  n_uniform, ...
                  size(M_drl_scl, 2));         
    flat_data_file = [path, folder, fn, '.mat'];
    save(flat_data_file, ...
        'T_drl_scl', ...
        'M_drl_scl', ...
        'miller_h', ...
        'miller_k', ...
        'miller_l', ...
        't_uniform', ...
        '-v7.3');
    
    fn = 'sortedInfo_dark';
    dark_fn = [path, folder, fn, '.mat'];
    save(dark_fn, ...
         'drls_dark', ...
         'scales_dark', ...
         'reflection_ns_dark', ...
         'uniqueID_dark', ...
         '-v7.3');
     
    fn = 'sortedInfo_light';
    dark_fn = [path, folder, fn, '.mat'];
    save(dark_fn, ...
         'drls_light_uniform', ...
         'scales_light_uniform', ...
         'reflection_ns_light_uniform', ...
         'uniqueID_light_uniform', ...
         't_uniform', ...
         '-v7.3');

end