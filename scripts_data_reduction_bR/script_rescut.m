% script_pack_myData_drl
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script generates a DRL mask from a dataset using diffraction resolution limits of 
% snapshots, and is applied on that dataset at the end.
% A. Hosseini, Dec-2017, March-2018, March-2019
% C. Casadei - modified March-21-2019
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear 

tic
%_________________________________________________________________________________________
% initializations:
n = '259423';
datatype = 'dark';

% to load:
fileData = ['data_bR_', datatype, '_int_SCL_nS', n, '_nBrg53620.mat'];          

load(fileData,'T_scl',...
              'M_scl',...
              'miller_h',...
              'miller_k',...
              'miller_l');

drl = 1.5;  % Angstrom

% lattice sizes for group P6(3):
a = 62.36;  % in Angstrom
b = 62.39;  % in Angstrom
c = 111.18; % in Angstrom

qmax = 1./drl ; % notice that diff_res_lim is in Angstrom
 
% q-vector in reciprocal space:
qvec = [miller_h./a, miller_h./(sqrt(3)*a) + 2*miller_k./(sqrt(3)*b), miller_l./c];
q2   = qvec(:,1).^2 + qvec(:,2).^2 + qvec(:,3).^2; 
q    = sqrt(q2);
 
IND = find(q < qmax);
T_scl = T_scl(:,IND);
M_scl = M_scl(:,IND);
miller_h = miller_h(IND);
miller_k = miller_k(IND);
miller_l = miller_l(IND);

nBrg = int2str(size(T_scl, 2));
fileData_rescut = ['data_bR_', datatype, '_int_SCL_rescut_nS', n, '_nBrg', nBrg, '.mat']; 

save(fileData_rescut, ...
     'T_scl', ...
     'M_scl', ...
     'miller_h', ...
     'miller_k', ...
     'miller_l', ...
     '-v7.3');  
fprintf('Data file saved.\n')
toc
% EOF