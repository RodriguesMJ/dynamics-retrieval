% script_time_sort
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script separates dark and light (class-1) data.
% Light data are ordered according to ascending timing tool measurents.
% Cecilia C., March-12-2019
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all

% LOAD TIMESTAMPS 
path = '/das/work/p18/p18594/cecilia-offline/NLSA/data_rho_2/data_extraction/';
folder = '';
fn = 'timestamps';
time_delays_file = [path, folder, fn, '.mat']; 
load(time_delays_file, 'timestamps_light');

% LOAD UNIQUE IDs AND SCALING PARAMETERS
fn = 'partialator_OK_light';
param_file = [path, folder, fn, '.mat'];
load(param_file, 'uniqueID_light');
load(param_file, 'scales_light');

% LOAD DRLs AND REFLECTION #
fn = 'drls_reflection_n_light';
data_file = [path, folder, fn, '.mat'];
load(data_file, 'drls_light');
load(data_file, 'reflection_ns_light');

n_scaled = length(uniqueID_light); % LIGHT 

% Separation
% idxs_dark_temp = find(isnan(timestamps)); % not necessarily dark
% idxs_light = find(~isnan(timestamps));    % light
% uniqueID_dark_temp = uniqueID(idxs_dark_temp);
% uniqueID_light = uniqueID(idxs_light);
% scales_dark_temp = scales(idxs_dark_temp, :);
% scales_light = scales(idxs_light, :);
% drls_dark_temp = drls(idxs_dark_temp);
% drls_light = drls(idxs_light);
% reflection_ns_dark_temp = reflection_ns(idxs_dark_temp);
% reflection_ns_light = reflection_ns(idxs_light);
% 
% timestamps_light = timestamps(idxs_light);

% Sorting
[timestamps_light_sort, I] = sort(timestamps_light, 'ascend'); 
uniqueID_light_sort = uniqueID_light(I);
scales_light_sort = scales_light(I,:);
drls_light_sort = drls_light(I);
reflection_ns_light_sort = reflection_ns_light(I);

n_nans = sum(isnan(timestamps_light_sort));

timestamps_light_sort = timestamps_light_sort(1:end-n_nans);
uniqueID_light_sort = uniqueID_light_sort(1:end-n_nans);
scales_light_sort = scales_light_sort(1:end-n_nans,:);
drls_light_sort = drls_light_sort(1:end-n_nans);
reflection_ns_light_sort = reflection_ns_light_sort(1:end-n_nans);

n_nans_final = sum(isnan(timestamps_light_sort));

fn = [path, folder, 'partialator_OK_light_sorted.mat'];
save(fn, 'uniqueID_light_sort', ...
         'scales_light_sort');
fn = [path, folder, 'drls_reflection_n_light_sorted.mat'];     
save(fn, 'drls_light_sort', ...
         'reflection_ns_light_sort');
fn = [path, folder, 'timestamps_light_sorted.mat'];     
save(fn, 'timestamps_light_sort');