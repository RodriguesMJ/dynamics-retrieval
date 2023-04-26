% script_read_stream_line
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Prepare input files with:
% grep -n 'Image filename' ___.stream > Image_filename_position.txt
% grep -n 'Event' ___.stream > Event_position.txt
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all

label = 'light';

% READ POSITIONS OF ALL THE HITS IN STREAM FILE
path = '/das/work/p17/p17491/Cecilia_Casadei/NLSA/data_rho/data_extraction';
folder = '/';

fn = 'Image_filename_position_scans_';
img_filename_pos = [path, folder, fn, label, '.txt'];

fn = 'Event_position_scans_';
event_pos = [path, folder, fn, label, '.txt'];

str_img_fn_pos = fileread(img_filename_pos);                                   %read file into one string
str_ev_pos = fileread(event_pos);

parts_img_fn_pos = strtrim(regexp( str_img_fn_pos, '(\r|\n)+', 'split'));      %split by each line
parts_ev_pos = strtrim(regexp( str_ev_pos, '(\r|\n)+', 'split')); 

nrows = length(parts_img_fn_pos) - 1;                                          %number of rows 
                                         
% EXTRACT UNIQUE_ID FOR ALL HITS                                                               
uniqueID_hits = cell(nrows, 1);

for k=1:nrows
     data_img_fn_pos = strtrim(regexp(parts_img_fn_pos{k}, ':', 'split'));     %split by spaces
     data_ev_pos = strtrim(regexp(parts_ev_pos{k}, ':', 'split'));
     f_n = data_img_fn_pos{3};
     event = data_ev_pos{3}(3:end);
     unique = [f_n, '_event_', event];
     uniqueID_hits{k,1} = unique;
end

% LOAD
metadata_fn = [path, folder, 'metadata.mat'];
% load(metadata_fn, 'uniqueID_dark_temp', ...
%                   'scales_dark_temp',...
%                   'reflection_ns_dark_temp', ...
%                   'drls_dark_temp');
% uniqueID_temp = uniqueID_dark_temp;

load(metadata_fn, 'uniqueID_light_sort', ...
                  'scales_light_sort',...
                  'reflection_ns_light_sort', ...
                  'drls_light_sort', ...
                  'timestamps_light_sort');
uniqueID_temp = uniqueID_light_sort;

n = length(uniqueID_temp);

idxs  = [];
for j=1:n
     ID = uniqueID_temp{j};
     idx = find(strcmp(uniqueID_hits, ID));
     if ~isempty(idx)
        display ('found')
        idxs = [idxs, j];  
     end
end
idxs = idxs';
size(idxs)

% uniqueID_dark = uniqueID_temp(idxs);
% scales_dark = scales_dark_temp(idxs, :);
% reflection_ns_dark = reflection_ns_dark_temp(idxs);
% drls_dark = drls_dark_temp(idxs);

uniqueID_light = uniqueID_temp(idxs);
scales_light = scales_light_sort(idxs, :);
reflection_ns_light = reflection_ns_light_sort(idxs);
drls_light = drls_light_sort(idxs);
timestamps_light = timestamps_light_sort(idxs);

% SAVE 
fn = ['metadata_', label];
filename_out = [path, folder, fn, '.mat'];
% save(filename_out, 'uniqueID_dark', ...
%                    'scales_dark', ...
%                    'reflection_ns_dark', ...
%                    'drls_dark', ...
%                    '-v7.3');

save(filename_out, 'uniqueID_light', ...
                   'scales_light', ...
                   'reflection_ns_light', ...
                   'drls_light', ...
                   'timestamps_light', ...
                   '-v7.3');