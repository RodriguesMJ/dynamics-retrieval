% script_read_timestamp
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script reads timestamps from time delays file.
% Cecilia C., March-4-2019
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all

% LOAD TIMESTAMPS FILE
path = '/das/work/p17/p17491/Cecilia_Casadei/TR-SFX/bR_TR_SFX/LCLS_2017_cxilp4115/';
folder = 'test_running_times/data_reduction_running_times/';
fn = 'time_delays_full_path';
time_delays_file = [path, folder, fn, '.txt']; 
str_time_delays = fileread(time_delays_file);

% SPLIT TIME DELAYS FILE BY LINE
parts_time_delays = strtrim(regexp(str_time_delays, '\n', 'split'));  

n_time_delays = length(parts_time_delays)-1;

timestamps = zeros(n_time_delays, 1);
uniqueID_timed = cell(n_time_delays, 1);

for k=1:n_time_delays
   
     % EXTRACT UNIQUE IDs AND TIMESTAMPS
     time_delay_line = parts_time_delays{k};
     time_delay_line = regexp(time_delay_line, '\s+', 'split');
     filename = time_delay_line{1};
     event = time_delay_line{2};
     timestamp = str2double(time_delay_line{3});
     uniqueID = [filename, '.cxi_event_', event];
     
     timestamps(k) = timestamp;
     uniqueID_timed{k} = uniqueID;
end   

% SAVE
fn = 'timestamps_all';
filename_out = [path, folder, fn, '.mat'];
save(filename_out,'timestamps', 'uniqueID_timed', '-v7.3');