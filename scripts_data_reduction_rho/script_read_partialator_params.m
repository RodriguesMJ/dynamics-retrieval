% script_read_partialator_paramsv
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script reads Partialator parameters (i.e., OSF & relB) 
% for ALL (scalable) data 
% and saves them in a MAT file 
% together with corresponding filename and eventID.
% Ourmazd Lab, UWM, Dec-28-2017
% Modified by Cecilia C., March-2019
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all

% READ PARTIALATOR PARAMETER FILE
path = '/das/work/p18/p18594/cecilia-offline/NLSA/data_rho_2';
folder = '/scaling';
fn = '/partialator_OK';
filename_in = [path, folder, fn, '.params']; 
str = fileread(filename_in); % read entire file into one string

% SPLIT PARAMETER FILE LINE BY LINE
parts = strtrim(regexp( str, '(\r|\n)+', 'split'));   

nrows = length(parts) - 1;    %number of rows 
                              %(remove 1 because of 0 in the last row)
n = 51395
% DATA STRUCTURE TO STORE RESULTS                              
scales = zeros(n, 2); 
filenameID = cell(n, 1);
eventID = zeros(n, 1, 'int32');
uniqueID = cell(n, 1);

i = 0;
% EXTRACT INFORMATION FROM PARTIALATOR PARAMETER FILE
for k=1:nrows
    k
    data = strtrim(regexp( parts{k}, '\s+', 'split'));   
    
    f_n = data{6};                          % filename
    if (    contains(f_n, '/rho_dark') ...
         || contains(f_n, '/rho_alldark')) % ...
         %|| contains(f_n, '/rho_nlsa_scan_22/') ...
         %|| contains(f_n, '/rho_nlsa_scan_23/'))%...
         %|| contains(f_n, '/rho_nlsa_scan_17/') ...
         %|| contains(f_n, '/rho_nlsa_scan_18/'))
        
        i = i+1;
        scales(i,1) = str2double(data{2});      % OSF  (linear scaling factors)
        scales(i,2) = str2double(data{3});      % relB (Debye-Waller scaling factors)

        event = data{7}(3:end);                 % event number
        unique = [f_n, '_event_', event];       % combine filename and enent n

        filenameID{i,1} = f_n;
        eventID(i,1) = str2double(event);
        uniqueID{i,1} = unique;
    end
end

 i
 notice = '1st column: OSF, 2nd column: relB';
  
 % SAVE
 folder = '/data_extraction';
 filename_out = [path, folder, fn, '_alldark.mat'];
 save(filename_out, ...
      'scales', ...
      'filenameID', ...
      'eventID', ...
      'uniqueID', ...
      'notice', ...
      '-v7.3');
