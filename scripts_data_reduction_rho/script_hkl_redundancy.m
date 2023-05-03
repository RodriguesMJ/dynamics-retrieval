% script_hkl_redundancy
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script creates a redundancy file. 
% Ourmazd Lab, UWM, Dec-2017
% Cecilia C., Mar-2019 - Aug-2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear

path = '/das/work/p18/p18594/cecilia-offline/NLSA/data_rho_2/data_extraction/';
load([path, 'partialator_OK_alldark.mat'], 'uniqueID');

fileRedundancy = [path, 'redundancy.mat'];                         
num_snapshots = length(uniqueID);

h_max = 35;
k_max = 50;
l_max = 85;
 
redundancy = zeros((h_max+1),(k_max+1),(l_max+1));
 for nn=1:num_snapshots
     nn
     ID = uniqueID{nn};
     s = strfind(ID, '/');
     s = s(end);
     ID = ID(s:end);
    
     % Load single frame reflections with:
     % no multiple observcations
     % res cutoff applied
     filename = [path, 'data_hklI_alldark_uniques_rescut', ID, '_hklI_uniques_rescut.dat'];
     fid_hklI = fopen(filename,'r');
     hklI = fscanf(fid_hklI,'%f');
     n_reflection = size(hklI, 1)/4;
     hklI = reshape(hklI,4,n_reflection)';

     for kk=1:n_reflection
       h = hklI(kk,1);
       k = hklI(kk,2);
       l = hklI(kk,3);
       redundancy(h+1,k+1,l+1) = redundancy(h+1,k+1,l+1)+1;
     end
     fclose(fid_hklI);
 end
 save(fileRedundancy,'redundancy');