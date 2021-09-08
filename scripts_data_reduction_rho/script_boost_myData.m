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
fn = 'data_rho_alldark_SCL_nS51395_nBrg74887.mat'; 
fileData = [path, fn];

% to save:
fn = 'data_rho_alldark_SCL_BST_nS51395_nBrg74887.mat'; 
fileData_BST = [path, fn];

load(fileData, 'T_scl', 'M_scl', 'miller_h', 'miller_k', 'miller_l');%, 't_uniform');
T = T_scl;
M = M_scl;
clear T_scl M_scl;

[nS, nBrg] = size(T);

T_bst = T;
M_bst = M;

%_________________________________________________________________________________________
% Boosting:
for k=1:nBrg
    co(k) = sum(M(:,k));                                               %#ok<SAGROW>
    if co(k)
        T_bst(:,k) = T(:,k)./co(k);
    end
end

% save data with DRL cuts, scaled and boosted:
save(fileData_BST,'T_bst', ...
                  'M_bst', ...
                  'miller_h', ...
                  'miller_k', ...
                  'miller_l', ...
                  '-v7.3');  
                  %'t_uniform', ...
                  
%CMC - And t_uniform for light data