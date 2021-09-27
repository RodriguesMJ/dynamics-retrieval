%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Cecilia C. 11/2019
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear

% to load:
path = './';
fn = 'data_bR_light_int_unifdelay_SCL_rescut_nS133115_nBrg39172.mat'; 
fileData = [path, fn];

load(fileData, 'M_scl', 'T_scl', 'miller_h', 'miller_k', 'miller_l', 't_uniform');
M_light = M_scl;
T_light = T_scl;

clear M_scl T_scl

fn = 'data_bR_dark_int_SCL_rescut_nS259423_nBrg39172.mat'; 
fileData = [path, fn];

load(fileData, 'M_scl', 'T_scl');
M_dark = M_scl;
T_dark = T_scl;

clear M_scl T_scl

s_light = sum(M_light);
s_dark  = sum(M_dark);

T_light_selected = T_light(:, (s_light>1 & s_dark>1));
M_light_selected = M_light(:, (s_light>1 & s_dark>1));

T_dark_selected  = T_dark(:, (s_light>1 & s_dark>1));
M_dark_selected  = M_dark(:, (s_light>1 & s_dark>1));

h_selected = miller_h(s_light>1 & s_dark>1);
k_selected = miller_k(s_light>1 & s_dark>1);
l_selected = miller_l(s_light>1 & s_dark>1);

% to save:
path = './';
fn = ['data_bR_light_selected_int_unifdelay_SCL_rescut_nS133115_nBrg', num2str(size(l_selected, 1)), '.mat']; 
fileData = [path, fn];

save(fileData, 'T_light_selected', ...
               'M_light_selected', ...
               'h_selected', ...
               'k_selected', ...
               'l_selected', ...
               't_uniform', ...
               '-v7.3'); 
           
fn = ['data_bR_dark_selected_int_SCL_rescut_nS259423_nBrg', num2str(size(l_selected, 1)), '.mat']; 
fileData = [path, fn];

save(fileData, 'T_dark_selected', ...
               'M_dark_selected', ...
               'h_selected', ...
               'k_selected', ...
               'l_selected', ...
               '-v7.3'); 
           


