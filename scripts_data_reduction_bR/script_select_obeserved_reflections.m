%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Cecilia C. 11/2019
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear

% to load:
path = './';
fn = 'data_bR_light_int_unifdelay_gauss_sigma_40fs_DRL_SCL_nS116552_nBrg54396.mat'; 
fileData = [path, fn];

load(fileData, 'M_drl_scl', 'T_drl_scl', 'miller_h', 'miller_k', 'miller_l', 't_uniform');
M_light = M_drl_scl;
T_light = T_drl_scl;

clear M_drl_scl T_drl_scl

fn = 'data_bR_dark_int_DRL_SCL_nS234988_nBrg54396.mat'; 
fileData = [path, fn];

load(fileData, 'M_drl_scl', 'T_drl_scl');
M_dark = M_drl_scl;
T_dark = T_drl_scl;

clear M_drl_scl T_drl_scl

s_light = sum(M_light);
s_dark = sum(M_dark);

T_light_selected = T_light(:, (s_light>1 & s_dark>1));
M_light_selected = M_light(:, (s_light>1 & s_dark>1));

T_dark_selected = T_dark(:, (s_light>1 & s_dark>1));
M_dark_selected = M_dark(:, (s_light>1 & s_dark>1));

h_selected = miller_h(s_light>1 & s_dark>1);
k_selected = miller_k(s_light>1 & s_dark>1);
l_selected = miller_l(s_light>1 & s_dark>1);

% to save:
path = './';
fn = ['data_bR_light_selected_int_unifdelay_DRL_SCL_nS116552_nBrg', num2str(size(l_selected, 1)), '.mat']; 
fileData = [path, fn];

save(fileData, 'T_light_selected', ...
               'M_light_selected', ...
               'h_selected', ...
               'k_selected', ...
               'l_selected', ...
               't_uniform');
           
fn = ['data_bR_dark_selected_int_DRL_SCL_nS234988_nBrg', num2str(size(l_selected, 1)), '.mat']; 
fileData = [path, fn];

save(fileData, 'T_dark_selected', ...
               'M_dark_selected', ...
               'h_selected', ...
               'k_selected', ...
               'l_selected');
           


