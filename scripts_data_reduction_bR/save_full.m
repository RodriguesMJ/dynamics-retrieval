%fileData1 = './original_data/data_bR_light_selected_int_unifdelay_DRL_SCL_BST_nS116552_nBrg47735.mat';
fileData1 = './original_data/data_bR_dark_selected_int_DRL_SCL_BST_nS116000_nBrg47735.mat';

load(fileData1, 'miller_h','miller_k','miller_l');
save('./converted_data_dark/miller_idxs_dark.mat', 'miller_h', 'miller_k', 'miller_l');

%load(fileData1, 't_uniform');
%save('./converted_data/t_uniform_light.mat', 't_uniform');

load(fileData1, 'T_bst');
T = full(T_bst);
size(T_bst)
size(T)

save('./converted_data_dark/T_dark_full.mat', 'T', '-v7.3');
clear T_bst T

load(fileData1, 'M_bst');
M = full(M_bst);
save('./converted_data_dark/M_dark_full.mat', 'M', '-v7.3');
clear M_bst M
