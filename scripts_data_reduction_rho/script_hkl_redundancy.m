% script_hkl_redundancy
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script creates a redundancy file. 
% Ahmad H., Dec-2017
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


 
 
% return
% %-----------------------------------------------------------------------------------------
% addpath('./export_fig')
% redundancy(redundancy<1)=nan;
% for l=1:l_max
%   h = figure(1);
%   %h = figure;
%   set(h,'color','w','resize','off')
%   pos = get(h,'pos');
%   set(h,'pos',[pos(1) pos(2) pos(3) 1.2*pos(4)])
%   hsp = subplot(1,1,1);
%   pcolor([0:h_max],[0:k_max],squeeze(redundancy(:,:,l+1))')
%   axis equal
%   set(hsp,'xlim',[0 h_max],'ylim',[0 k_max])
%   set(hsp,'fontSize',15,'lineWidth',2)
%   xlabel('h','fontSize',15)
%   ylabel('k','fontSize',15)
%   title(['Redundancy,l=' num2str(l)],'fontSize',20)
%   colorbar('fontSize',15)
%   export_fig('-jpeg','-r200',['redundancy_l_' num2str(l) '.jpg'])
%   close(1)
% end
% 
% %EOF
