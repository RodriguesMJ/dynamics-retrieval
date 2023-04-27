% script_boost_myData
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script boosts (normalize) the DRL-removed and scaled data via a an intensity 
% normalization approach (partially adapted from RS-RF files).
% A. Hosseini, June-2018, March-2019
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear

tic
%_________________________________________________________________________________________
% initializations:

% to load:
path = './';
%fn = 'data_bR_light_selected_int_unifdelay_SCL_rescut_nS133115_nBrg39172.mat'; 
fn = 'data_bR_dark_selected_int_SCL_rescut_nS259423_nBrg39172.mat'; 
fileData = [path, fn];

% to save:
%fn = 'data_bR_light_selected_int_unifdelay_SCL_rescut_BST_nS133115_nBrg39172.mat'; 
fn = 'data_bR_dark_selected_int_SCL_rescut_BST_nS259423_nBrg39172.mat'; 
fileData_BST = [path, fn];

load(fileData);

%%%%%%%%%%%% TO RM
T = T_dark_selected;
M = M_dark_selected;
clear T_dark_selected M_dark_selected

miller_h = h_selected;
miller_k = k_selected;
miller_l = l_selected;
%%%%%%%%%%%%%%%

[nS, nBrg] = size(T);

T_bst = T;
M_bst = M;

%_________________________________________________________________________________________
% Boosting:
for k=1:nBrg
    k
    co(k) = sum(M(:,k));                                               %#ok<SAGROW>
    if co(k)
        T_bst(:,k) = 1000.0.*T(:,k)./co(k);
    end
end


save(fileData_BST,'T_bst', ...
                  'M_bst', ...
                  'miller_h', ...
                  'miller_k', ...
                  'miller_l', ...
                  '-v7.3');  
%CMC - And t_uniform for light data

avg_before = sum(T, 'all')/sum(M, 'all')
avg_after = sum(T_bst, 'all')/sum(M_bst, 'all')