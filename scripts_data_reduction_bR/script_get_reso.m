%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
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

load(fileData,...
              'miller_h',...
              'miller_k',...
              'miller_l');

drl = 1.5;      % Angstrom
qmax = 1./drl ; % notice that diff_res_lim is in Angstrom
 
% lattice sizes for group P6(3):
a = 62.36;  % in Angstrom
b = 62.39;  % in Angstrom
c = 111.18; % in Angstrom
 
% q-vector in reciprocal space:
qvec = [miller_h./a, miller_h./(sqrt(3)*a) + 2*miller_k./(sqrt(3)*b), miller_l./c];
q2   = qvec(:,1).^2 + qvec(:,2).^2 + qvec(:,3).^2; 
q    = sqrt(q2);
 
d = 1.0./q;