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

% SELECT ONLY TIMESTAMPS BELOW 1500 fs
%idx_ts_fsRange = find(timestamps_light_sort < 1500);
%timestamps = timestamps_light_sort(idx_ts_fsRange);
%M_temp = M_drl_scl(idx_ts_fsRange, :);                
%T_temp = T_drl_scl(idx_ts_fsRange, :);               
%drls_light_temp = drls_light_sort(idx_ts_fsRange);     
%scales_light_temp = scales_light_sort(idx_ts_fsRange, :);   
%reflection_ns_light_temp = reflection_ns_light_sort(idx_ts_fsRange); 
%uniqueID_light_temp = uniqueID_light_sort(idx_ts_fsRange);            

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
         t0 = timestamps_light_sort(idx);
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
    M_uniform = M_scl(idx_uniform, :);
    T_uniform = T_scl(idx_uniform, :);      
    scales_light_uniform = scales_light_sort(idx_uniform, :);   
    reflection_ns_light_uniform = reflection_ns_light_sort(idx_uniform); 
    uniqueID_light_uniform = uniqueID_light_sort(idx_uniform);  
    
    % SORT!
    [t_uniform, I] = sort(t_uniform);
    
    M_uniform = M_uniform(I,:);
    T_uniform = T_uniform(I,:);   
    scales_light_uniform = scales_light_uniform(I, :);
    reflection_ns_light_uniform = reflection_ns_light_uniform(I);
    uniqueID_light_uniform = uniqueID_light_uniform(I);
    
    % SAVE
    fn = sprintf('data_bR_light_int_unifdelay_SCL_nS%d_nBrg%d', ...
                  n_uniform, ...
                  size(M_uniform, 2));         
    flat_data_file = [path, folder, fn, '.mat'];
    save(flat_data_file, ...
        'T_uniform', ...
        'M_uniform', ...
        'miller_h', ...
        'miller_k', ...
        'miller_l', ...
        't_uniform', ...
        '-v7.3');
    
    fn = 'sortedInfo_dark';
    dark_fn = [path, folder, fn, '.mat'];
    save(dark_fn, ...
         'scales_dark', ...
         'reflection_ns_dark', ...
         'uniqueID_dark', ...
         '-v7.3');
     
    fn = 'sortedInfo_light';
    light_fn = [path, folder, fn, '.mat'];
    save(light_fn, ...
         'scales_light_uniform', ...
         'reflection_ns_light_uniform', ...
         'uniqueID_light_uniform', ...
         '-v7.3');

end