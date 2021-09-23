% script_read_stream_params
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script reads scalable frames stream file positions and extracts
% information from the stream file: diffraction resolution limit (drl) and
% number of integrated reflections.
% Cecilia C., March-4-2019
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all

% LOAD POSITIONS IN STREAM FILE (ONLY SCALABLE FRAMES)
path = './';
folder = '';
fn = 'stream_line_n';
stream_line_n_file = [path, folder, fn, '.mat']; 
load(stream_line_n_file, 'stream_line');

% N SCALABLE FRAMES
nScaled = length(stream_line);

% LOAD STREAM FILE
fn = 'all_detwinned';
stream_file = [path, folder, fn, '.stream']; 
str_stream_file = fileread(stream_file); % read entire file into one string

% SPLIT STREAM BY LINE
parts_stream = strtrim(regexp(str_stream_file, '\n', 'split')); 

reflection_ns = zeros(nScaled, 1);

for k=1:nScaled
    % START OF FRAME IN STREAM
    k
    stream_idx = stream_line(k);
    filename = parts_stream{stream_idx};
    
    idx_1 = 11; %10;
    idx_2 = 14; %13;
    idx_3 = 14;
    
    % N PEAKS FOUND BY HIT FINDER
    n_peaks = str2double(parts_stream{stream_idx+idx_1}(13:end))
    
    % EXTRACT NUMBER INTEGRATED REFLECTIONS
    n_reflections_line = parts_stream{stream_idx+idx_2+n_peaks+idx_3};
    n_reflections_line = regexp(n_reflections_line, '\s+', 'split');
    n_reflections = str2double(n_reflections_line{3})
    reflection_ns(k) = n_reflections;
end   

% SAVE
fn = 'reflection_n';
filename_out = [path, folder, fn, '.mat'];
save(filename_out,'reflection_ns', '-v7.3');