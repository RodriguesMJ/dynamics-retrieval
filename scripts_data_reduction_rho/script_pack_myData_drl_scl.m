% script_pack_myData_drl_scl
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script reads Partialator parameters of crystFEL (OSF & relB) and apply them 
% to the snapshots, which are already passed the DRL removal step.
% A. Hosseini, Dec-2017, May-2018, July-2018, March-2019
% Cecilia C., modified 22 March 2019
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear

%_________________________________________________________________________________________
% To load   
label = 'alldark';
nS = '51395';
nBrg = '74887';
path = '/das/work/p18/p18594/cecilia-offline/NLSA/data_rho_2/data_extraction/';
folder = '/';
fileData = [path, folder, 'data_rho_', label, '_nS', nS, '_nBrg', nBrg, '.mat'];         
scalePara_file = [path, folder, 'partialator_OK_', label, '.mat'];

% To save:
fileData_SCL = ['data_rho_', label, '_SCL_nS', nS '_nBrg', nBrg, '.mat']; 

% lattice sizes:
a = 61.45;
b = 90.81;
c = 150.82;

% load data: 
load(fileData,'T','M','miller_h','miller_k','miller_l'); 
load(scalePara_file, ['scales']);

%scales = scales_light_sort;

OSF = scales(:,1);
relB = scales(:,2);

%_________________________________________________________________________________________
% Scaling with OSF and relB parameters:

% P6_3
%qvec = [miller_h./a, miller_h./(sqrt(3)*a) + 2*miller_k./(sqrt(3)*b), miller_l./c];
% P222
qvec = [miller_h./a, miller_k./b, miller_l./c];
q2 = qvec(:,1).^2 + qvec(:,2).^2 + qvec(:,3).^2; 
%notice: sinTheta_lambda2 = q2/4 = 1/(4*d2) 

T_scl = sparse(size(T, 1), size(T, 2));
blocksize = 10000;  % for large matrix multiplication
N = blocksize;

nBrg = size(T, 2)
for i = 1 : ceil(size(T,1)/N)
  i
  if i*N > size(T,1)
    T_scl(1+(i-1)*N:end,:) = T(1+(i-1)*N:end,:) ...
                             .* repmat(exp(-OSF(1+(i-1)*N:end)),1,nBrg) ...
                             .* exp(relB(1+(i-1)*N:end) * q2'/4); 
  else
    T_scl(1+(i-1)*N:i*N,:) = T(1+(i-1)*N:i*N,:) ...
                             .* repmat(exp(-OSF(1+(i-1)*N:i*N)),1,nBrg) ...
                             .* exp(relB(1+(i-1)*N:i*N) * q2'/4); 
  end  
end

M_scl = M;

% save scaled data
save(fileData_SCL,'T_scl','M_scl','miller_h','miller_k','miller_l','-v7.3');  
%EOF