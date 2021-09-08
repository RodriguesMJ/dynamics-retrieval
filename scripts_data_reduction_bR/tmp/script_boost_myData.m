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
fn = 'data_bR_test_light_sorted_int_DRL_SCL_nS74437_nBrg52219.mat'; 
fileData = [path, fn];

% to save:
fn = 'data_bR_test_light_sorted_int_DRL_SCL_BST_nS74437_nBrg52219.mat'; 
fileData_BST = [path, fn];

%fn = '/data_bR_light_int_unifdelay_gauss_DRL_SCL_DTW_BST_AVG_nS125858_nBrg50947.hkl';  
%file_hkl = [path, fn];

generate_hkl_avg = false; % whether to save the averaged data or not

load(fileData);

%%%%%%%%%%%% TO RM
T_dtw = T_drl_scl;
M_dtw = M_drl_scl;
clear T_drl_scl M_drl_scl
%%%%%%%%%%%%%%%

[nS, nBrg]=size(T_dtw);

T_bst = T_dtw;
M_bst = M_dtw;

%_________________________________________________________________________________________
% Boosting:
for k=1:nBrg
    co(k) = sum(M_dtw(:,k));                                               %#ok<SAGROW>
    if co(k)
        T_bst(:,k) = T_dtw(:,k)./co(k);
    end
end


% save data with DRL cuts, scaled and boosted:
save(fileData_BST,'T_bst', ...
                  'M_bst', ...
                  'miller_h', ...
                  'miller_k', ...
                  'miller_l', ...
                  '-v7.3');  
%CMC - And t_uniform for light data

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
