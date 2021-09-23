% script_time_sort
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script separates dark and light (class-1) data.
% Light data are ordered according to ascending timing tool measurents.
% Cecilia C., March-12-2019
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all

% LOAD TIMESTAMPS 
path = './';
folder = '';
fn = 'timestamps';
time_delays_file = [path, folder, fn, '.mat']; 
load(time_delays_file, 'timestamps_scaled_frames');

% LOAD UNIQUE IDs AND SCALING PARAMETERS
fn = 'partialator_OK';
param_file = [path, folder, fn, '.mat'];
load(param_file, 'uniqueID');
load(param_file, 'scales');

% LOAD REFLECTION #
fn = 'reflection_n';
data_file = [path, folder, fn, '.mat'];
load(data_file, 'reflection_ns');

n_scaled = length(uniqueID); % LIGHT AND DARK

idxs_dark = find(isnan(timestamps_scaled_frames));
idxs_light = find(~isnan(timestamps_scaled_frames));
uniqueID_dark = uniqueID(idxs_dark);
uniqueID_light = uniqueID(idxs_light);
scales_dark = scales(idxs_dark, :);
scales_light = scales(idxs_light, :);
reflection_ns_dark = reflection_ns(idxs_dark);
reflection_ns_light = reflection_ns(idxs_light);

timestamps_scaledframes_light = timestamps_scaled_frames(idxs_light);

[timestamps_light_sort, I] = sort(timestamps_scaledframes_light, 'ascend'); 
uniqueID_light_sort = uniqueID_light(I);
scales_light_sort = scales_light(I,:);
reflection_ns_light_sort = reflection_ns_light(I);

fn = [path, folder, 'sorted_info.mat'];
save(fn, 'uniqueID_dark', 'uniqueID_light_sort', ...
         'scales_dark', 'scales_light_sort', ...
         'reflection_ns_dark', 'reflection_ns_light_sort', ...
         'timestamps_light_sort', ...
         '-v7.3');