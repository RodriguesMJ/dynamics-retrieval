%fileData1 = './original_data/data_bR_light_selected_int_unifdelay_DRL_SCL_BST_nS116552_nBrg47735.mat';
fileData1 = './data_bR_dark_selected_int_SCL_rescut_BST_nS130000_nBrg39172.mat';

load(fileData1, 'miller_h','miller_k','miller_l');
save('../converted_data_dark/miller_idxs_dark.mat', 'miller_h', 'miller_k', 'miller_l');

%load(fileData1, 't_uniform');
%save('./converted_data/t_uniform_light.mat', 't_uniform');

load(fileData1, 'T');
size(T)
T = full(T);
size(T)

save('../converted_data_dark/T_dark_full.mat', 'T', '-v7.3');
clear T

load(fileData1, 'M');
M = full(M);
save('../converted_data_dark/M_dark_full.mat', 'M', '-v7.3');
clear M
