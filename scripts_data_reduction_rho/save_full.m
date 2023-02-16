clear all
fileData1 = './data_rho_light_SCL_unifdelay_nS213557_nBrg74887.mat';

load(fileData1, 'miller_h','miller_k','miller_l');
save('../../../LPSA/converted_data_light/miller_idxs_light.mat', 'miller_h', 'miller_k', 'miller_l');

load(fileData1, 't_uniform');
save('../../../LPSA/converted_data_light/t_uniform_light.mat', 't_uniform');

load(fileData1, 'T');
T = full(T);
size(T)

save('../../../LPSA/converted_data_light/T_light_full.mat', 'T', '-v7.3');
clear T

load(fileData1, 'M');
M = full(M);
size(M)
save('../../../LPSA/converted_data_light/M_light_full.mat', 'M', '-v7.3');
clear M