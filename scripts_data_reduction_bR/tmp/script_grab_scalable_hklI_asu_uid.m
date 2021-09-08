path = '/das/work/p17/p17491/Cecilia_Casadei/TR-SFX/bR_TR_SFX/LCLS_2017_cxilp4115/test_running_times/data_reduction_running_times/';

directory = [path, 'data_hklI'];
if ~exist(directory,'dir')
    system(['mkdir ' directory]);
end

% LOAD STREAM FILE
fn = 'test_light_reidxd';
stream_file = [path, fn, '.stream']; 
str_stream_file = fileread(stream_file); % read entire file into one string
disp('read')

% SPLIT STREAM BY LINE
parts_stream = strtrim(regexp(str_stream_file, '\n', 'split')); 
disp('split')

% LOG CONTAINING FRAMES IDs
%snapshotInfo_file = fopen([path, 'test_snapshotInfo.dat'],'w');

% LINE NUMBERS IN STREAM
load([path, 'stream_line_n.mat'], 'stream_line');
 
% N SCALABLE FRAMES
nScalableFrames = length(stream_line);

% UNIQUE_IDs
load([path, 'partialator_test_light_OK.mat'], 'uniqueID');

for k=70001:nScalableFrames %70000%1:nScalableFrames
     k
     ID = uniqueID{k};
     line_in_stream = stream_line(k);
     disp(parts_stream{line_in_stream});
     
     if strfind(ID, 'class1')
        idx_1 = 12;
        idx_2 = 16;
        idx_3 = 13;
     else
        idx_1 = 10;
        idx_2 = 13;
        idx_3 = 12;
     end
     
     % N PEAKS FOUND BY HIT FINDER
     n_peaks = str2double(parts_stream{line_in_stream+idx_1}(13:end));
     
     % EXTRACT NUMBER INTEGRATED REFLECTIONS
     n_reflections_line = parts_stream{line_in_stream+idx_2+n_peaks+idx_3+1};
     n_reflections_line = regexp(n_reflections_line, '\s+', 'split');
     n_reflections = str2double(n_reflections_line{3});
        
     %myData = nan(n_reflections, 4);
     startline = line_in_stream+idx_2+n_peaks+idx_3+1+5;
     endline  = startline+n_reflections-1;
     
    
     s = strfind(ID, '/');
     s = s(end);
     ID = ID(s:end);
     fn = [directory, ID, '_hklI.dat'];
     file_hklI = fopen(fn,'w');
     
     n = 0;
     for line=startline:endline
         n = n+1;
         linestream = parts_stream{line};
         linestream = regexp(linestream, '\s+', 'split');
         hh = str2num(linestream{1});                                      %#ok<*ST2NM>
         kk = str2num(linestream{2});
         ll = str2num(linestream{3});
         hkl = [hh, kk, ll];
         hkl_asu = asuP6_3(hkl);
         I = str2double(linestream(4));
         fprintf(file_hklI, ...
                 '%4d%5d%5d%11.2f\n', ...
                 hkl_asu(1), ...
                 hkl_asu(2), ...
                 hkl_asu(3), ...
                 I);
     end
     fclose(file_hklI);
 end

