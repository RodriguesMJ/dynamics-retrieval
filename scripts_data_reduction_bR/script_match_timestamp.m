% script_match_timestamp
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script matches scalable frames with their timestamp.
% Dark frames (class 4) do not have a timestamp.
% Cecilia C., March-4-2019
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear

% LOAD TIMESTAMPS 
path = './';
folder = '';
fn = 'timestamps_all';
time_delays_file = [path, folder, fn, '.mat']; 
load(time_delays_file, 'timestamps');
load(time_delays_file, 'uniqueID_timed');

% LOAD UNIQUE IDs OF SCALABLE FILES
fn = 'partialator_OK';
param_file = [path, folder, fn, '.mat'];
load(param_file, 'uniqueID')

% MATCH SCALABLE FILES WITH THEIR TIMESTAMPS
n_scaled = length(uniqueID); % LIGHT AND DARK
timestamps_scaled_frames = NaN(n_scaled, 1);

n_timed = 0;
m_problem = 0;

for k=1:n_scaled
   k
   uniqueID_scaled = uniqueID{k};
   idx = find(strcmp(uniqueID_timed, uniqueID_scaled));
   if length(idx) > 0                                                      %#ok<ISMT>
      ts = timestamps(idx(1));
      timestamps_scaled_frames(k) = ts;
      n_timed = n_timed + 1;
   end
   if length(idx) > 1
      m_problem = m_problem + 1;
   end
end   

% SAVE
fn = 'timestamps';
filename_out = [path, folder, fn, '.mat'];
save(filename_out, 'timestamps_scaled_frames', ...
                   'n_timed', ...
                   'm_problem', ...
                   '-v7.3');