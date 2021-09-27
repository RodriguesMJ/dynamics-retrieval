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

generate_hkl_avg = false; % whether to save the averaged data or not

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
%_________________________________________________________________________________________
% % Finding average and std of "DRL'ed, scaled, boosted" data and save them as a *hkl file 
% if generate_hkl_avg
% 
% T_avg = full(sum(T_drl_scl_bst)./sum(M_drl_scl_bst));
% T_avg(isnan(T_avg)) = 0;
% T_avg(isinf(T_avg)) = 0;
% T_avg(T_avg<0) = 0;
% T_STD_new = sqrt(T_avg);
% 
% fileID = fopen(fileHKL,'w');  
% fprintf(fileID,'%3d %3d %3d %6.6f %6.6f\n',[miller_h,miller_k,miller_l,T_avg',T_STD_new']');
% fclose(fileID);
% end
% 
% toc
% EOF
