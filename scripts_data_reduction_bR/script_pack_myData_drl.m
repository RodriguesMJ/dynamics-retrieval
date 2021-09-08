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
dataForm = 'int';    % 'int' vs. 'amp'    (amplitude vs. intensity)
n = '234988';
datatype = 'dark';


%%%% CMC
push_res = 0.12; % A^{-1}
%%%% CMC

% to load:
fileData = ['data_bR_', datatype, '_int_nS', n, '_nBrg54396.mat'];          
%fileInfo = 'light_sorted_collection_time_info.mat';
fileInfo = 'sorted_info.mat';

% to save:             
fileData_DRL = ['data_bR_', datatype, '_int_DRL_nS', n, '_nBrg54396.mat']; 


load(fileData,'T',...
              'M',...
              'miller_h',...
              'miller_k',...
              'miller_l');

load(fileInfo, 'drls_dark')

drls = drls_dark;
clear drls_dark;

fprintf('%i non-zero drls.\n', length(find(drls)))
mean_drl = mean(drls);
drls(drls == 0) = mean_drl;

fileParams = matfile(fileData);
[nS , nBrg] = size(fileParams,'T');

% notice_drl = 'DRL mask removes Bragg points beyond DRL. Also, qmax=1/diff_res_lim.';
 
%_________________________________________________________________________________________
% set up the mask file:
maskDRL = ones(nS,nBrg);
fprintf('Mask initialized\n')

% lattice sizes for group P6(3):
a = 62.32; % in Angstrom
b = 62.32; % in Angstrom
c = 111.1; % in Angstrom

qmax = 1./drls + 0.12; % notice that diff_res_lim is in Angstrom
 
% q-vector in reciprocal space:
qvec = [miller_h./a, miller_h./(sqrt(3)*a) + 2*miller_k./(sqrt(3)*b), miller_l./c];
q2   = qvec(:,1).^2 + qvec(:,2).^2 + qvec(:,3).^2; 
q    = sqrt(q2);
 
% find reflections with better resolution than DRL for each snapshot:
tic
for ii = 1:nS
    if mod(ii,50) == 0
        disp(ii)
    end
    IND = find(q > qmax(ii));
    maskDRL(ii,IND) = 0;      
end
  
maskDRL(isnan(maskDRL)) = 0;
fprintf('Set nan values to 0.\n')
%maskDRL = sparse(maskDRL); % Cecilia C. March 22, 2019 - Problem making mask sparse
fprintf('Mask was kept full.\n')

%_________________________________________________________________________________________
% Applying the DRL mask:
fprintf('Start sparse matrix multiplication.\n')
M_drl = maskDRL.*M;  % Full by sparse gives sparse - Cecilia C. March 22, 2019
fprintf('DRL applied to M\n')
T_drl = maskDRL.*T;  % Full by sparse gives sparse - Cecilia C. March 22, 2019
fprintf('DRL applied to T\n')

clear maskDRL % Cecilia C. - March 22, 2019
fprintf('Memory freed\n')

%_________________________________________________________________________________________
% saving the DRL mask as well as the masked data:
% notice_mask = 'Bragg reflections beyond DRLs are removed, and qmax is q at DRL.';

%save(fileMask_DRL,'maskDRL','drls_light_sort','qmax','notice_mask', 'notice_drl','-v7.3');
%fprintf('Mask file saved.\n')
save(fileData_DRL,'T_drl','M_drl','miller_h','miller_k','miller_l','-v7.3');  
fprintf('Data file saved.\n')
toc
% EOF
