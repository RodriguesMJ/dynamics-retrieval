% script_hkl_redundancy
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script creates a redundancy file. 
% Ahmad H., Dec-2017
% Cecilia C., Mar-2019
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear

path = './';
load([path, 'partialator_OK.mat'], 'uniqueID');
load([path, 'reflection_n.mat'], 'reflection_ns');

fileRedundancy = [path, 'redundancy.mat'];                         
num_snapshots = length(uniqueID);

h_max = 60;
k_max = 60;
l_max = 120;
 
redundancy = zeros((h_max+1),(k_max+1),(l_max+1));
 for nn=1:num_snapshots
     nn
     ID = uniqueID{nn};
     s = strfind(ID, '/');
     s = s(end);
     ID = ID(s:end);
     
     n_reflection = reflection_ns(nn);
     filename = [path, 'data_hklI', ID, '_hklI.dat'];
     fid_hklI = fopen(filename, 'r');
     hklI = fscanf(fid_hklI, '%f');
     hklI = reshape(hklI, 4, n_reflection)';
     for kk=1:n_reflection
       h = hklI(kk,1);
       k = hklI(kk,2);
       l = hklI(kk,3);
       redundancy(h+1,k+1,l+1) = redundancy(h+1,k+1,l+1)+1;
     end
     fclose(fid_hklI);
 end
 save(fileRedundancy,'redundancy');