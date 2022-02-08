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

tic
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
       redundancy(h+1,k+1partialator-cxiklp,l+1) = redundancy(h+1,k+1,l+1)+1;
     end
     fclose(fid_hklI);
 end
 save(fileRedundancy,'redundancy');
 toc

 
 
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
