% script_read_stream_params
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script reads scalable frames stream file positions and extracts
% information from the stream file: diffraction resolution limit (drl) and
% number of integrated reflections.
% Cecilia C., March-4-2019
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all

% LOAD POSITIONS IN STREAM FILE (ONLY SCALABLE FRAMES)
path = '/das/work/p17/p17491/Cecilia_Casadei/TR-SFX/bR_TR_SFX/LCLS_2017_cxilp4115/';
folder = 'test_running_times/data_reduction_running_times/';
fn = 'stream_line_n';
stream_line_n_file = [path, folder, fn, '.mat']; 
load(stream_line_n_file, 'stream_line');

% N SCALABLE FRAMES
nScaled = length(stream_line);

% LOAD STREAM FILE
fn = 'test_light_reidxd';
stream_file = [path, folder, fn, '.stream']; 
str_stream_file = fileread(stream_file); % read entire file into one string

% SPLIT STREAM BY LINE
parts_stream = strtrim(regexp(str_stream_file, '\n', 'split')); 


drls = zeros(nScaled, 1);
reflection_ns = zeros(nScaled, 1);

for k=1:nScaled
    % START OF FRAME IN STREAM
    stream_idx = stream_line(k)
    filename = parts_stream{stream_idx}
    
    if strfind(filename, 'class1')
        idx_1 = 12;
        idx_2 = 16;
        idx_3 = 13;
    else
        idx_1 = 10;
        idx_2 = 13;
        idx_3 = 12;
    end
    
    % N PEAKS FOUND BY HIT FINDER
    n_peaks = str2double(parts_stream{stream_idx+idx_1}(13:end))
    
    % EXTRACT DIFFRACTION RESOLUTION LIMIT
    drl_line = parts_stream{stream_idx+idx_2+n_peaks+idx_3};
    drl_line = regexp(drl_line, '\s+', 'split');
    drl = str2double(drl_line{6}) % Diffraction resolution limit in A
    drls(k) = drl;
    
    % EXTRACT NUMBER INTEGRATED REFLECTIONS
    n_reflections_line = parts_stream{stream_idx+idx_2+n_peaks+idx_3+1};
    n_reflections_line = regexp(n_reflections_line, '\s+', 'split');
    n_reflections = str2double(n_reflections_line{3})
    reflection_ns(k) = n_reflections;
end   

% SAVE
fn = 'drls_reflection_n';
filename_out = [path, folder, fn, '.mat'];
save(filename_out,'drls', 'reflection_ns', '-v7.3');