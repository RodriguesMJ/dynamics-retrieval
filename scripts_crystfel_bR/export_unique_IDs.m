clear all
fn = ['./', 'sortedInfo_dark.mat']; 
file_name = ['./', 'uniqueID_dark.txt'];
fn_open = fopen(file_name,'w');
     
% LOAD UNIQUE_IDs
load(fn, 'uniqueID_dark');
for idx=1:130000
    uniqueID_dark{idx}
    fprintf(fn_open, ...
                 '%s\n', ...
                 uniqueID_dark{idx});
end
fclose(fn_open);
clear uniqueID_dark

fn = ['./', 'sortedInfo_light.mat']; 
file_name = ['./', 'uniqueID_light.txt'];
fn_open = fopen(file_name,'w');
     
% LOAD UNIQUE_IDs
load(fn, 'uniqueID_light_uniform');
for idx=1:133115
    uniqueID_light_uniform{idx}
    fprintf(fn_open, ...
                 '%s\n', ...
                 uniqueID_light_uniform{idx});
end
fclose(fn_open);

