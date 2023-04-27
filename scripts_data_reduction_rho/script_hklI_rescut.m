%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Cecilia C., Aug 2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear
labels = {'1_6', ...
          '7_12', ...
          '13_18', ...
          '19_19', ...
          '20_23'};
      
for i=1:5
    
label = char(labels(i))

path = '/das/work/p18/p18594/cecilia-offline/NLSA/data_rho_2/data_extraction/';
load([path, 'partialator_OK_scans_', label, '_light.mat'], 'uniqueID');
load([path, 'drls_reflection_n_scans_', label, '_light.mat'], 'reflection_ns');

%uniqueID = uniqueID_light; 
%reflection_ns = reflection_ns_light;
                 
num_snapshots = length(uniqueID)

a = 61.45;
b = 90.81;
c = 150.82;

d_lim = 1.83; % Angstroms

 for nn=1:num_snapshots
     
     ID = uniqueID{nn};
     s = strfind(ID, '/');
     s = s(end);
     ID = ID(s:end);
     
     n_reflection = reflection_ns(nn);
     filename = [path, 'data_hklI_scans_', label, '_light', ID, '_hklI.dat'];
     fid_hklI = fopen(filename,'r');
     hklI = fscanf(fid_hklI,'%f');
     hklI = reshape(hklI,4,n_reflection)';
     hkl = hklI(:,1:3);
     [u,I,J] = unique(hkl, 'rows', 'first');
     hasDuplicates = size(u,1) < size(hkl,1);
     ixDupRows = setdiff(1:size(hkl,1), I);
     dupRowValues = hkl(ixDupRows,:);
          
     idxs_uniques = setdiff(1:n_reflection,ixDupRows)';
     
     % Remove multiple observations from same frame
     hklI_uniques = hklI(idxs_uniques,:);
     
     ds = 1.0 ./ sqrt((hklI_uniques(:,1)./a).^2 + (hklI_uniques(:,2)./b).^2 + (hklI_uniques(:,3)./c).^2);
     
     idxs = find(ds>d_lim);
     
     % Remove reflections beyon high res cutoff
     hklI_uniques_rescut = hklI_uniques(idxs, :);
     
     fn_out = [path, 'data_hklI_scans_', label, '_light_uniques_rescut', ID, '_hklI_uniques_rescut.dat'];
     writematrix(hklI_uniques_rescut, fn_out, 'Delimiter', '\t');
     fclose(fid_hklI);
 end
end