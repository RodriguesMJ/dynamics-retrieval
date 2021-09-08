% script_packing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script reads light/dark data information 
% and the redundancy file for the data as well. 
% Then it packs the data  
% in a sparse matrix and generates a mask. 
% Results are saved in a MAT file for analysis. 
% Ahmad H., Dec-07-17, updated on May-24-18 for femtosecond data
% Cecilia C., March-12-19, modified
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clc
clear
dataForm = 'int';

tic 

path = './';
folder = '';
fn = [path, folder, 'sorted_info.mat']; 

% LOAD UNIQUE_IDs
load(fn, 'uniqueID_dark');

% LOAD REFLECTION N
load(fn, 'reflection_ns_dark');

uniqueID = uniqueID_dark;
reflection_ns = reflection_ns_dark;

clear uniqueID_dark reflection_ns_dark;

num_snapshot = size(uniqueID, 1);

% % "miller_h", "miller_k", "miller_l" are Miller indices
% % they are intentionally created to have the same packing convention as "redundancy"
h_max = 60;       
k_max = 60;
l_max = 120;
num_h = h_max+1;
num_k = k_max+1;
num_l = l_max+1;
miller_h = zeros(num_h,num_k,num_l);
miller_k = zeros(num_h,num_k,num_l);
miller_l = zeros(num_h,num_k,num_l);
num_miller = num_h*num_k*num_l;
for hh=0:h_max
  for kk=0:k_max
    for ll=0:l_max
      miller_h(hh+1,kk+1,ll+1) = hh;
      miller_k(hh+1,kk+1,ll+1) = kk;
      miller_l(hh+1,kk+1,ll+1) = ll;
    end
  end
end

% "redundany" has the same packing convention 
% as "miller_h", "miller_k" and "miller_l"
% LOAD REDUNDANCY FILE
filRedundancy = 'redundancy.mat';
load(filRedundancy,'redundancy')      

% (h, 0, l) and (0, h, l) are sym equivalent.
% The n of obs is added together tom increase the redundancy of (h, 0, l)
% and the redundancy of (0, h, l) is set to zero.
for hh=1:h_max
  for ll=0:l_max
    redundancy_h0l = redundancy(hh+1, 0+1,ll+1); 
    redundancy_0hl = redundancy(0+1,hh+1,ll+1); 
    redundancy(hh+1, 0+1,ll+1) = redundancy_h0l+redundancy_0hl;
    redundancy(0+1,hh+1,ll+1) = 0; 
  end
end

active_reflection = find(redundancy(:)>0);         
num_unique_reflection = numel(active_reflection);  
% i.e., active lattice vertices (Millers) that have
% Brag spots for at least one snapshot
% (notice, each snapshot contains a subset of active spots)

% follow the Glownia convention
T = zeros(num_snapshot,num_unique_reflection);
M = zeros(num_snapshot,num_unique_reflection);

for jj=1:num_snapshot
    
  temp_vol = zeros(num_h,num_k,num_l);     % shape as redundancy file
  count = zeros(num_h,num_k,num_l);        % shape as redundancy file
  temp_vec = nan(num_miller,1);   
  
  num_reflection = reflection_ns(jj); % Bragg' spot number for each snapshot

  ID = uniqueID{jj};
  s = strfind(ID, '/');
  s = s(end);
  ID = ID(s:end);
  filename = [path, folder, 'data_hklI/', ID, '_hklI.dat'];
  
  fid_hklI = fopen(filename,'r');        
  hklI = fscanf(fid_hklI,'%f');          % read (h,k,l,I) for each snapshot 
  fclose(fid_hklI);
  hklI = reshape(hklI,4,num_reflection)';
  
  for nn=1:num_reflection      % for each Bragg spot in a typical snapshot:
    hh = hklI(nn,1);           % read h from 1st column
    kk = hklI(nn,2);           % read k from 2nd column
    ll = hklI(nn,3);           % read l from 3rd column
    intensity = hklI(nn,4);    % read I from 4th column

    if ((hh==0) && (kk>0))     % if h=0 and k>0, set h=k and set k=0   
      hh = kk;
      kk = 0;
    end % if
    
%again the same packing convention as redundancy. Like creating redundancy file,
%here we make a similar 3D lattice and add intensities of Bragg's points to
%vertices (h,k,l).

    temp_vol(hh+1,kk+1,ll+1) = temp_vol(hh+1,kk+1,ll+1)+intensity; 
    
%we count how many times a certain Bragg's point (lattice vertex) has been repeated when
%we add intensities on top of each other (needed for intensity normalization of vertices) 
    
    count(hh+1,kk+1,ll+1) = count(hh+1,kk+1,ll+1)+1;
  end % for nn=1:num_reflection
   
  count = count(:);                        % 1D
  M(jj,:) = (count(active_reflection)>0);  % make a mask for snapshot jj by counting non-zero vertices 
  count(count==0) = 1;                     % take care of Inf because of devision in next line
  temp_vec = temp_vol(:)./count;           % point-wise intensity normalization for each vertex (Bragg's spot) in snapshot jj
  T(jj,:) = temp_vec(active_reflection);   % take the above normalized 3D lattice of intensities for snapshot jj
                                           % and only consider the total active spots as a snapshot TT(jj,:)                                                               
end % for jj=1:num_snapshot
 
miller_h = miller_h(:);
miller_h = miller_h(active_reflection);
miller_k = miller_k(:);
miller_k = miller_k(active_reflection);
miller_l = miller_l(:);
miller_l = miller_l(active_reflection);
T = sparse(T);
M = sparse(M);
 
%T(T<0) = 0;
%M(T<0) = 0;
if strcmp(dataForm,'amp')
    T = sqrt(T);  % convert to amplitudes
    disp('-- data format: Amplitude --');
elseif strcmp(dataForm,'int')
    disp('-- data format: Intensity --')
end

% to save
filename = sprintf('data_bR_dark_%s_nS%d_nBrg%d.mat', ...
                   dataForm, ...
                   num_snapshot,...
                   num_unique_reflection);         

save(filename,'T','M','miller_h','miller_k','miller_l', '-v7.3')
toc