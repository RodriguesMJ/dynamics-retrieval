%fileData1 = './data_bR_light_int_SCL_rescut_20A_to_1p8A_unifdelay_nS132885_nBrg22727.mat';
fileData1 = './data_bR_dark_int_SCL_rescut_20A_to_1p8A_nS259423_nBrg22727.mat';

load(fileData1, 'miller_h','miller_k','miller_l');
save('../converted_data_dark/miller_idxs_dark.mat', 'miller_h', 'miller_k', 'miller_l');

%load(fileData1, 't_uniform');
%save('../converted_data_light/t_uniform_light.mat', 't_uniform');

load(fileData1, 'T_scl');
size(T_scl)
T = full(T_scl);
size(T)

save('../converted_data_dark/T_dark_full.mat', 'T', '-v7.3');
clear T T_scl

load(fileData1, 'M_scl');
M = full(M_scl);
save('../converted_data_dark/M_dark_full.mat', 'M', '-v7.3');
clear M M_scl

%load('sorted_info.mat', 'timestamps_light_sort');
%save('../converted_data_light/t_light.mat', 'timestamps_light_sort');