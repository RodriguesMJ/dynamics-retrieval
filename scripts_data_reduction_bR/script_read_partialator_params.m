% script_read_partialator_params
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script reads Partialator parameters (i.e., OSF & relB) 
% for ALL (scalable) data 
% and saves them in a MAT file 
% together with corresponding filename and eventID.
% Ahmad H., Dec-28-2017
% Modified by Cecilia C., March-3-2019
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clear all

% READ PARTIALATOR PARAMETER FILE
path = './';
folder = '';
fn = 'partialator_OK';
filename_in = [path, folder, fn, '.params'];
str = fileread(filename_in); % read entire file into one string

% SPLIT PARAMETER FILE LINE BY LINE
parts = strtrim(regexp( str, '(\r|\n)+', 'split'));   

ncols = 2;                    %number of columns 
nrows = length(parts) - 1;    %number of rows 
                              %(remove 1 because of 0 in the last row)

% DATA STRUCTURE TO STORE RESULTS                              
scales = zeros(nrows, ncols); 
filenameID = cell(nrows, 1);
eventID = zeros(nrows, 1, 'int32');
uniqueID = cell(nrows, 1);

% EXTRACT INFORMATION FROM PARTIALATOR PARAMETER FILE
for k=1:nrows
    data = strtrim(regexp( parts{k}, '\s+', 'split'));   
    
    %partialInfo(k,1) = str2double(data{1}); % data increment
    scales(k,1) = str2double(data{2}); % OSF  (linear scaling factors)
    scales(k,2) = str2double(data{3}); % relB (Debye-Waller scaling factors)
    
    f_n = data{6};                          % filename
    event = data{7}(3:end);                 % event number
    unique = [f_n, '_event_', event];       % combine filename and enent n
    
    filenameID{k,1} = f_n;
    eventID(k,1) = str2double(event);
    uniqueID{k,1} = unique;
end
 notice = '1st column: OSF, 2nd column: relB';
  
 % SAVE
 filename_out = [path, folder, fn, '.mat'];
 save(filename_out, ...
      'scales', ...
      'filenameID', ...
      'eventID', ...
      'uniqueID', ...
      'notice', ...
      '-v7.3');
