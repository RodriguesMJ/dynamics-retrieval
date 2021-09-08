% script_check_uniques
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This script compares unique reflections as determined in script_packing.m 
% with those resulting from partialator merging (filename.hkl).
% Cecilia C., April-2-2019
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% LOAD UNIQUE h, k, l SETS
path = '/Users/casadei/Documents/casadei/';
folder = 'data_reduction/';
fn = 'data_bR_light_int_sortdelay_nS243093_nBrg55129.mat';
file = [path, folder, fn];
load(file, 'miller_h', 'miller_k', 'miller_l');

% LOAD REDUNDANCY FILE
fn = 'redundancy.mat';
file = [path, folder, fn];
load(file, 'redundancy')

% LOAD MERGED INTENSITIES FILE FROM PARTIALATOR
folder = 'bR_data/';
fn = 'all_merged_partialator_edited.hkl';
file = [path, folder, fn];
f_str = fileread(file);

% SPLIT MERGED I FILE LINE BY LINE
parts = strtrim(regexp( f_str, '(\r|\n)+', 'split')); 

% STORE UNIQUE h, k, l SETS FROM PARTIALATOR
n_partialator = length(parts) - 1;
partialator_uniques = zeros(n_partialator, 4);
for i=1:n_partialator
    
    data = strtrim(regexp( parts{i}, '\s+', 'split'));   
    
    partialator_uniques(i,1) = str2double(data{1}); % h
    partialator_uniques(i,2) = str2double(data{2}); % k
    partialator_uniques(i,3) = str2double(data{3}); % l
    partialator_uniques(i,4) = str2double(data{7}); % n_spots
end   

% CHECK THAT PARTIALATOR DOES NOT CONSIDER REFLECTIONS WITH LESS THAN 2
% OBSERVATIONS
f = find(partialator_uniques(:,4)==1); % no elements

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%% MATCH UNIQUE REFLECTIONS TO THOSE FROM PARTIALATOR %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

disp('Match sets');

n = length(miller_h);
n_found = 0;
n_not_found = 0;

uniques_matches = zeros(n, 1);
partialator_uniques_matches = zeros(n_partialator, 1);
for i=1:n
    h = miller_h(i);
    k = miller_k(i);
    l = miller_l(i);
    found = 0;
    for j=1:n_partialator
        if (partialator_uniques(j,1)==h && ...
            partialator_uniques(j,2)==k && ...
            partialator_uniques(j,3)==l)
       
            found = 1;
            n_found = n_found + 1;
            uniques_matches(i,1) = 1;
            partialator_uniques_matches(j,1) = 1;
            
        end
    end
    if found == 0
        n_not_found = n_not_found + 1;
    end
        
end

% EXTRACT N MATCHED AND NON MATCHED
matched = length(find(uniques_matches==1));
non_matched = length(find(uniques_matches==0));

% CHECK THAT ALL SETS FROM PARTIALATOR ARE MATCHED
if length(find(partialator_uniques_matches==1)) == n_partialator
    disp('All sets from partialator were matched')
else
    disp('Not all sets from partialator were matched')
end

% CALCULATE THE REDUNDANCY OF EXTRA SETS
disp('Calculate redundancy of extra sets')
r_non_matched = zeros(non_matched, 4);
non_matched_idx = find(uniques_matches==0);
for i=1:non_matched
    idx = non_matched_idx(i);
    h = miller_h(idx);
    k = miller_k(idx);
    l = miller_l(idx);
    r = redundancy(h+1, k+1, l+1);
    r_non_matched(i,1) = h;
    r_non_matched(i,2) = k;
    r_non_matched(i,3) = l;
    r_non_matched(i,4) = r;
end  

f = find(r_non_matched(:,4)<2);
if length(f) == length(r_non_matched)
    disp('All non matched sets have redundancy smaller than 2')
end
% MOST NON MATCHED SETS HAVE REDUNDANCY = 1 
% (not found in partialator merged file,
% since partialator considers only spots with redundancy above 1)
% SOME HAVE REDUNDANCY OF 0
% SHOW THAT THESE ARE OF THE TYPE h,0,l
% WHICH WILL ASSUME THE REDUYNDANCY OF THEIR 0, h, l EQUIVALENT
disp('Check zero-redundancy items')
r_0 = find(r_non_matched(:,4)==0);
r_sym = zeros(length(r_0), 1);
for i=1:length(r_0)
    idx = r_0(i);
    h = r_non_matched(idx,1);
    k = r_non_matched(idx,2);
    l = r_non_matched(idx,3);
    if ~(r_non_matched(idx,4)==0)
        disp('PROBLEM')
    end
    if ~(k==0)
        disp('UNEXPECTED')
    end
    r_sym(i,1) = redundancy(0+1, h+1, l+1);
end

f = find(r_sym==1);
if length(f) == length(r_sym)
    disp('Zero-redundancy h,0,l items have one-redundancy 0,h,l eqv.')
end