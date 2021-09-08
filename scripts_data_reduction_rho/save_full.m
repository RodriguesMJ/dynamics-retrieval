clear all
fileData1 = './data_rho_alldark_SCL_BST_nS51395_nBrg74887.mat';

load(fileData1, 'miller_h','miller_k','miller_l');
save('../data_converted_alldark/miller_idxs_alldark.mat', 'miller_h', 'miller_k', 'miller_l');

%load(fileData1, 't_uniform');
%save('../data_converted_light/t_uniform_light.mat', 't_uniform');

load(fileData1, 'T_bst');
T = full(T_bst);
size(T_bst)
clear T_bst
size(T)

% % T(93, 1)
% % T(615, 1)
% % T(6173, 2)
% % T(5581, 3)
% % T(4417, 15)
% % T(2284, 76)
% % T(1598, 182)
% 
save('../data_converted_alldark/T_alldark_full.mat', 'T', '-v7.3');
clear T

load(fileData1, 'M_bst');
M = full(M_bst);
clear M_bst
save('../data_converted_alldark/M_alldark_full.mat', 'M', '-v7.3');
clear M