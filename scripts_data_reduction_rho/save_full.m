clear all
fileData1 = './light_scan_19_sorted_SCL.mat';
savepath = '../../../LPSA/scan_19/converted_data_light';

load(fileData1, 'miller_h','miller_k','miller_l');
fn = [savepath, '/miller_idxs_light.mat'];
save(fn, 'miller_h', 'miller_k', 'miller_l');

load(fileData1, 'timestamps_selected');
fn = [savepath, '/t_uniform_light.mat'];
save(fn, 'timestamps_selected');

load(fileData1, 'T_selected');
T = full(T_selected);
size(T)
fn = [savepath, '/T_light_full.mat'];
save(fn, 'T', '-v7.3');
clear T

load(fileData1, 'M_selected');
M = full(M_selected);
size(M)
fn = [savepath, '/M_light_full.mat'];
save(fn, 'M', '-v7.3');
clear M