% script_read_timestamp
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script reads timestamps from time delays file.
% Cecilia C., March-4-2019
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all

labels = {'1_6', ...
          '7_12', ...
          '13_18', ...
          '19_19', ...
          '20_23'};
      
      
for i=1:5
    
    label = char(labels(i))     
    
    % LOAD TIMESTAMPS FILE
    path = '/das/work/p18/p18594/cecilia-offline/NLSA/data_rho_2/data_extraction';
    folder = '/timing_data';
    fn = '_uniqueIDs_ts';
    time_delays_file = [path, folder, '/scans_', label, fn, '.txt']; 
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
         uniqueID = time_delay_line{1};
         timestamp = str2double(time_delay_line{2});

         timestamps(k) = timestamp;
         uniqueID_timed{k} = uniqueID;
    end   

    % SAVE
    fn = 'timestamps_scans_';
    filename_out = [path, '/', fn, label, '_all.mat'];
    save(filename_out, 'timestamps', 'uniqueID_timed', '-v7.3');
end