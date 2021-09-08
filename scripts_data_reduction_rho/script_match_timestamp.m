% script_match_timestamp
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script matches scalable frames with their timestamp.
% Dark frames (class 4) do not have a timestamp.
% Cecilia C., March-4-2019
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear
labels = {'1_6', ...
          '7_12', ...
          '13_18', ...
          '19_19', ...
          '20_23'};

for i=1:5
    
    label = char(labels(i))           

    % LOAD TIMESTAMPS 
    path = '/das/work/p18/p18594/cecilia-offline/NLSA/data_rho_2/data_extraction';
    folder = '/';
    fn = 'timestamps_scans_';
    time_delays_file = [path, folder, fn, label, '_all.mat']; 
    load(time_delays_file, 'timestamps');
    load(time_delays_file, 'uniqueID_timed');

    % LOAD UNIQUE IDs OF SCALABLE FILES
    fn = 'partialator_OK_scans_';
    param_file = [path, folder, fn, label, '_light.mat'];
    load(param_file, 'uniqueID')

    % MATCH SCALABLE FILES WITH THEIR TIMESTAMPS
    n_scaled = length(uniqueID); % LIGHT SCALED
    timestamps_scaled_frames = NaN(n_scaled, 1);

    n_timed = 0;
    m_problem = 0;

    for k=1:n_scaled
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
    fn = 'timestamps_scans_';
    filename_out = [path, folder, fn, label, '.mat'];
    save(filename_out, 'timestamps_scaled_frames', ...
                       'n_timed', ...
                       'm_problem', ...
                       '-v7.3');
               
end