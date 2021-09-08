% script_reavd_stream_params
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script reads scalable frames stream file positions and extracts
% information from the stream file: diffraction resolution limit (drl) and
% number of integrated reflections.
% Cecilia C., March-4-2019
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all
labels = {'nlsa_scan_1_6_light', ...
          'nlsa_scan_7_12_light', ...
          'nlsa_scan_13_18_light', ...
          'nlsa_scan_19_19_light', ...
          'nlsa_scan_20_23_light', ...
          'alldark'};
      
labels_partialator = {'scans_1_6_light', ...
                      'scans_7_12_light', ...
                      'scans_13_18_light', ...
                      'scans_19_light', ...
                      'scans_20_23_light', ...
                      'alldark'};
                  
for i=1:6
    label = char(labels(i))
    label_partialator = char(labels_partialator(i))

    % LOAD POSITIONS IN STREAM FILE (ONLY SCALABLE FRAMES)
    path = '/das/work/p18/p18594/cecilia-offline/NLSA/data_rho_2/data_extraction';
    folder = '/';
    fn = 'stream_line_n_';
    stream_line_n_file = [path, folder, fn, label_partialator, '.mat']; 
    load(stream_line_n_file, 'stream_line');

    % N SCALABLE FRAMES
    nScaled = length(stream_line);

    % LOAD STREAM FILE
    fn = 'chunck_rho_';
    stream_file = [path, folder, fn, label, '_idxd.stream']; 
    str_stream_file = fileread(stream_file); % read entire file into one string

    % SPLIT STREAM BY LINE
    parts_stream = strtrim(regexp(str_stream_file, '\n', 'split')); 

    drls = zeros(nScaled, 1);
    reflection_ns = zeros(nScaled, 1);

    for k=1:nScaled
        % START OF FRAME IN STREAM
        stream_idx = stream_line(k);
        filename = parts_stream{stream_idx};

    % bR
    %     if strfind(filename, 'class1')
    %         idx_1 = 12;
    %         idx_2 = 16;
    %         idx_3 = 13;
    %     else
    %         idx_1 = 10;
    %         idx_2 = 13;
    %         idx_3 = 12;
    %     end

    % rho
        idx_1 = 10;
        idx_2 = 13;
        idx_3 = 13;

        % N PEAKS FOUND BY HIT FINDER
        n_peaks = str2double(parts_stream{stream_idx+idx_1}(13:end));

        % EXTRACT DIFFRACTION RESOLUTION LIMIT
        drl_line = parts_stream{stream_idx+idx_2+n_peaks+idx_3};
        drl_line = regexp(drl_line, '\s+', 'split');
        drl = str2double(drl_line{6}); % Diffraction resolution limit in A
        drls(k) = drl;

        % EXTRACT NUMBER INTEGRATED REFLECTIONS
        n_reflections_line = parts_stream{stream_idx+idx_2+n_peaks+idx_3+1};
        n_reflections_line = regexp(n_reflections_line, '\s+', 'split');
        n_reflections = str2double(n_reflections_line{3});
        reflection_ns(k) = n_reflections;
    end   

    % SAVE
    fn = 'drls_reflection_n_';
    filename_out = [path, folder, fn, label_partialator, '.mat'];
    save(filename_out, 'drls', 'reflection_ns', '-v7.3');

end