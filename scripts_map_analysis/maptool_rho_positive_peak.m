% maptool_bR
% -------------------------------------------------------------------------
% written by Cecilia Wickstrand, 05-04-2020
%
% publication: 
% "A tool for visualizing protein motions in time-resolved crystallography"
% published online 01-04-2020, in Structural Dynamics (Vol.7, Issue 2).
% https://doi.org/10.1063/1.5126921 
%
% For analysis of a set of difference Fourier electron density maps from 
% time-resolved serial femtosecond crystallography (TR-SFX) experiments on
% bacteriorhodopsin. 
% - Difference electron density within a sphere around every atom in 
%   the resting state structure is extracted 
% - The average positive and negative density over the sphere is
%   calculated for each map, with or without a sigma cut-off
% - The resulting dual amplitude function may be plotted along the trace of
%   selected atoms or used for further analysis
%
% This script is an example for analysis of three maps (16 ns, 760 ns and 1.725 ms).
% Before analysis, execute "process_maps.sh" to convert the maps to 
% cartesian coordinates and .h5 format.
% -------------------------------------------------------------------------

% INPUT
% -------------------------------------------------------------------------
clear all;

% Sphere and sigma settings
radius = 1.7; % ??
distance = 0.2; % ??, how dense grid within sphere

mode = 1;
% Files
here = '/das/work/p18/p18594/cecilia-offline/NLSA/data_rho_2/results_NLSA/map_analysis';
pdbpath = [here '/bov_nlsa_refine_96_edited.pdb']; % resting state pdb
indir  = [here '/output_m_0_' num2str(mode) '/']; % where to find .h5 maps
outdir = [here '/results_m_0_' num2str(mode) '/'];

nrmaps = 2115;
mapnames = cell(nrmaps, 1);
for idx = 1:nrmaps
    i = 100*idx;
    nm = ['2.0_rho_light_mode_0_' num2str(mode) '_timestep_' num2str(i, '%0.6d') '_light--dark_rho_alldark_mode_0_avg'];
    mapnames(idx,1) = {nm};
end


% START CALCULATIONS
% ------------------------------------------------------------------------
time = tic;

% LOAD ATOMS

% In this case all HETATM are written as ATOM in pdb
%atomcoord = [[pdb.Model.Atom.X]' [pdb.Model.Atom.Y]' [pdb.Model.Atom.Z]'];
xs_positive_peak = [9.69, -21.22]
ys_positive_peak = [25.67, 31.38]
zs_positive_peak = [26.92, 47.82]
atomcoord = [xs_positive_peak' ys_positive_peak' zs_positive_peak'];
nratoms =  size(atomcoord,1); 

% CALCULATE SPHERE COORDINATES
calcdist = @(x,y,z) sqrt(x^2 + y^2 + z^2);
 
% radius 2 distance 0.5 -> spots = [-2 -1.5 -1 -0.5 0 0.5 1 1.5 2]
spots = distance:distance:ceil(radius/distance)*distance;
spots = [-fliplr(spots) 0 spots];
nrspots = length(spots);
 
% spherelist column 1 2 3 = coordinates
spherelist= zeros(nrspots^3,7);
count = 0;
for i = 1:nrspots
   for j = 1:nrspots
        for k = 1:nrspots
            dist = calcdist(spots(i),spots(j),spots(k));
            if dist <= radius
               count = count + 1;
               spherelist(count,:) = [spots(i) spots(j) spots(k) i j k dist];
            end
        end
   end
end
spherelist = spherelist(1:count,:); 
nrpoints = size(spherelist,1);


% PRECALCULATE ALL COORDINATES IN ALL SPHERES
% (example: 1834 rows = atoms, 2109 columns = points) 
X = repmat(atomcoord(:,1),1,nrpoints)+repmat(spherelist(:,1)',nratoms,1);
Y = repmat(atomcoord(:,2),1,nrpoints)+repmat(spherelist(:,2)',nratoms,1);
Z = repmat(atomcoord(:,3),1,nrpoints)+repmat(spherelist(:,3)',nratoms,1);

% Reshape to a single column starting with first point for each atom, then second etc. 
X=reshape(X, nratoms*nrpoints,1);
Y=reshape(Y, nratoms*nrpoints,1);
Z=reshape(Z, nratoms*nrpoints,1);


% LOAD GRID
% (using first experimental XYZinfo file - here all files have the same grid)
XYZinfo = dlmread([indir 'log/' mapnames{1} '_XYZinfo.dat']);

celldimensions = XYZinfo(1,1:3);
gridpoints = XYZinfo(2,1:3); 
axislimits = XYZinfo(3,:);  
axisorder = XYZinfo(4,1:3);  % Used to extract correct axislimits for each axis

% grid distance
dX = celldimensions(1)/gridpoints(1);
dY = celldimensions(2)/gridpoints(2);
dZ = celldimensions(3)/gridpoints(3);

tmp = [axisorder(1) axislimits(1:2); axisorder(2) axislimits(3:4); axisorder(3) axislimits(5:6)];
axes = sortrows(tmp); % By def., in order 1 2 3 (ie x, y, z)

% points at each side
sX = [axes(1,2):axes(1,3)]*dX;
sY = [axes(2,2):axes(2,3)]*dY;
sZ = [axes(3,2):axes(3,3)]*dZ;

[gY,gZ,gX] = meshgrid(sY,sZ,sX); % order so that gY, gZ, gX have same dim as map.


% LOAD MAPS AND EXTRACT DENSITIES
% ------------------------------------------------------------------------
%nrmaps = length(mapnames);
sigma = zeros(nrmaps, 2);
mapd0 = zeros(nrmaps, nratoms, nrpoints);

for m = 1:nrmaps
    fprintf(['Currently at map ' num2str(m) ' time ' num2str(toc(time)) ' s\n'])

    % Load sigma info
    XYZinfo = dlmread([indir 'log/' mapnames{m} '_XYZinfo.dat']);
    sigma(m,1:2) = XYZinfo(5:6,1);

    
    % LOAD MAP AND CALCULATE DENSITY AT POINTS BY INTERPOLATION
    % reshape back to rows = atoms, columns = points
    map = hdf5read([indir mapnames{m} '_cartesian.h5'],'map');
    tmp = interp3(gY,gZ,gX,map,Y,Z,X);

    mapdensities = reshape(tmp, nratoms, nrpoints);
    % divide with sigma of original map
    mapd0(m,:,:) = mapdensities/sigma(m,2); 
end
 
save([outdir 'mapd0_positivepeak.mat'], 'mapd0', '-v7.3');