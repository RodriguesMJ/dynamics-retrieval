% script_read_stream_line
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Prepare input files with:
% grep -n 'Image filename' ___.stream > Image_filename_position.txt
% grep -n 'Event' ___.stream > Event_position.txt
%
% This script reads hit positions in stream file 
% from Image_filename_positions.txt. 
% These positions concern all hits.
% Using the unique ID, these items are matched to the (scalable) items 
% extracted from partialator.params.
% The stream file positions for only those frames that are
% scalable are saved.
% Cecilia C., March-3-2019
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all

% READ POSITIONS OF ALL THE HITS IN STREAM FILE
path = './';
folder = '';

fn = 'Image_filename';
img_filename_pos = [path, folder, fn, '_position.txt']; 

fn = 'Event';
event_pos = [path, folder, fn, '_position.txt'];

str_img_fn_pos = fileread(img_filename_pos);                               % read file into one string
str_ev_pos = fileread(event_pos);

parts_img_fn_pos = strtrim(regexp( str_img_fn_pos, '(\r|\n)+', 'split'));  %split by each line
parts_ev_pos = strtrim(regexp( str_ev_pos, '(\r|\n)+', 'split')); 

nrows = length(parts_img_fn_pos) - 1                                  %number of rows 
                                                                           %(remove 1 because of 0 in the last row)
                                         
% EXTRACT UNIQUE_ID AND STREAM LINE NUMBER FOR ALL HITS                                         
stream_line_hits = zeros(nrows, 1);                            
uniqueID_hits = cell(nrows, 1);

for k=1:nrows
     data_img_fn_pos = strtrim(regexp(parts_img_fn_pos{k}, ':', 'split')); %split by spaces
     data_ev_pos = strtrim(regexp(parts_ev_pos{k}, ':', 'split'));
     stream_line_hits(k,1) = str2double(data_img_fn_pos{1});               % line n in stream file
     f_n = data_img_fn_pos{3};
     event = data_ev_pos{3}(3:end);
     unique = [f_n, '_event_', event];
     uniqueID_hits{k,1} = unique;
end

% LOAD INFO CONCERNING ONLY SCALABLE HITS
partialator_params = [path, folder, 'partialator_OK.mat'];
load(partialator_params, 'uniqueID');

nScaled = length(uniqueID);

% USING THE UNIQUE_ID, EXTRACT STREAM LINE NUMBER FOR SCALABLE HITS
stream_line = zeros(nScaled, 1);
for j=1:nScaled
    j
    ID = uniqueID{j};
    idx = find(strcmp(uniqueID_hits, ID));
    stream_line(j) = stream_line_hits(idx);
end

% SAVE STREAM LINE N FOR SCALABLE HITS
fn = 'stream_line_n';
filename_out = [path, folder, fn, '.mat'];
save(filename_out,'stream_line','-v7.3');