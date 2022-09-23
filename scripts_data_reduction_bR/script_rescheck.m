%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear 

tic
%_________________________________________________________________________________________
% initializations:
n = '219559';
datatype = 'light';

% to load:
fileData = ['data_bR_', datatype, '_int_SCL_nS', n, '_nBrg53620.mat'];          

load(fileData,'T_scl',...
              'M_scl',...
              'miller_h',...
              'miller_k',...
              'miller_l');


% lattice sizes for group P6(3):
a = 62.36;  % in Angstrom
b = 62.39;  % in Angstrom
c = 111.18; % in Angstrom


qmin = 1.0/20;%0.6106 ; 
 

qmax = 1.0/1.8;%0.6488;

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
q = q(IND);

IND = find(q >= qmin);
T_scl = T_scl(:,IND);
M_scl = M_scl(:,IND);
miller_h = miller_h(IND);
miller_k = miller_k(IND);
miller_l = miller_l(IND);

nBrg = int2str(size(T_scl, 2))

% EOF