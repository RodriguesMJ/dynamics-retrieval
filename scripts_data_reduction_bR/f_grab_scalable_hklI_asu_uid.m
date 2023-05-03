%%%%%%%%%%%%%%%%%%%%%%
% Ourmazd Lab, UWM
% Modified, C. Casadei
%%%%%%%%%%%%%%%%%%%%%%

function f_grab_scalable_hklI_asu_uid(n)

path = './';
% LINE NUMBERS IN STREAM
load([path, 'stream_line_n.mat'], 'stream_line');
 
% N SCALABLE FRAMES
nScalableFrames = length(stream_line);

chunck = 50000;
start_n = (chunck*n)+1
end_n = chunck*(n+1)
if end_n > nScalableFrames 
    end_n = nScalableFrames
end

directory = [path, 'data_hklI'];
if ~exist(directory,'dir')
    system(['mkdir ' directory]);
end

% LOAD STREAM FILE
fn = 'all_detwinned';
stream_file = [path, fn, '.stream']; 
str_stream_file = fileread(stream_file); % read entire file into one string
disp('read')

% SPLIT STREAM BY LINE
parts_stream = strtrim(regexp(str_stream_file, '\n', 'split')); 
disp('split')

% UNIQUE_IDs
load([path, 'partialator_OK.mat'], 'uniqueID');

for k=start_n:end_n 
     k
     ID = uniqueID{k};
     line_in_stream = stream_line(k);
     disp(parts_stream{line_in_stream});
     
     idx_1 = 11;
     idx_2 = 14;
     idx_3 = 14;
     
     % N PEAKS FOUND BY HIT FINDER
     n_peaks = str2double(parts_stream{line_in_stream+idx_1}(13:end));
     
     % EXTRACT NUMBER INTEGRATED REFLECTIONS
     n_reflections_line = parts_stream{line_in_stream+idx_2+n_peaks+idx_3};
     n_reflections_line = regexp(n_reflections_line, '\s+', 'split');
     n_reflections = str2double(n_reflections_line{3});
        
     %myData = nan(n_reflections, 4);
     startline = line_in_stream+idx_2+n_peaks+idx_3+5;
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
         hh = str2num(linestream{1});                                          %#ok<*ST2NM>
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
end

