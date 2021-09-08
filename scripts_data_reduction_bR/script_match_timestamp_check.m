% script_match_timestamp
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script checks that all class1 (light) frames have a timestamp 
% and that all dark frames don't.
% Cecilia C., March-4-2019
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear all

% LOAD TIMESTAMPS 
path = './';
folder = '';
fn = 'timestamps';
time_delays_file = [path, folder, fn, '.mat']; 
load(time_delays_file, 'timestamps_scaled_frames');

% LOAD UNIQUE IDs
fn = 'partialator_OK';
param_file = [path, folder, fn, '.mat'];
load(param_file, 'uniqueID');

n_scaled = length(uniqueID); % LIGHT AND DARK

n_good_dark = 0;
n_bad_dark = 0;
n_good_light = 0;
n_bad_light = 0;
n = 0;
for k=1:n_scaled
    ts = timestamps_scaled_frames(k);
    ID = uniqueID{k};
    n = n+1;
    %if ~contains(ID, 'class1')
    if strfind(ID, 'class1')
        % IT IS class1 i.e. light
        if isnan(ts)
            n_bad_light = n_bad_light + 1;
        else
            n_good_light = n_good_light + 1;
        end
        
    else        
        % IT IS DARK
        if isnan(ts)
            n_good_dark = n_good_dark + 1;
        else
            n_bad_dark = n_bad_dark + 1;
        end        
    end
end