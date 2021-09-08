%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%v%%v%%%%%%%%%%%%%
% Cecilia C., Aug-2020
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear

path = '/das/work/p18/p18594/cecilia-offline/NLSA/data_rho_2/data_extraction/';

labels = {'1_6', '7_12', '13_18', '19_19', '20_23'};
n_chuncks = size(labels, 2);

n_tot_scaled = 283442;

uniqueID_light = cell(n_tot_scaled, 1);
scales_light = zeros(n_tot_scaled, 2);
timestamps_light = zeros(n_tot_scaled, 1);
drls_light = zeros(n_tot_scaled, 1);
reflection_ns_light = zeros(n_tot_scaled, 1);

n_end = 0;
for i = 1:n_chuncks
    label = char(labels(1,i))
    load([path, 'partialator_OK_scans_', label, '_light.mat'], ...
          'uniqueID', ...
          'scales');    
    load([path, 'timestamps_scans_', label, '.mat'], ...
          'timestamps_scaled_frames');
    load([path, 'drls_reflection_n_scans_', label, '_light.mat'], ...
          'drls', ...
          'reflection_ns');
                                                              
    n = size(uniqueID, 1);
    n_start = n_end;
    n_end = n_start + n;
    n_start+1                                                              %#ok<*NOPTS>
    n_end
    
    uniqueID_light(n_start+1:n_end,1) = uniqueID;
    scales_light(n_start+1:n_end,:) = scales;
    timestamps_light(n_start+1:n_end) = timestamps_scaled_frames;
    drls_light(n_start+1:n_end) = drls;
    reflection_ns_light(n_start+1:n_end) = reflection_ns;
end


save([path, 'partialator_OK_light.mat'], 'uniqueID_light', 'scales_light');
save([path, 'timestamps.mat'], 'timestamps_light');
save([path, 'drls_reflection_n_light.mat'], 'drls_light', 'reflection_ns_light');
