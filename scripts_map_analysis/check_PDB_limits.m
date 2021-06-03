clear all;


%path = '/das/work/p17/p17491/Cecilia_Casadei/NLSA/data_bR/map_analysis/maptool_CMC';
%pdbpath = [path, '/6g7h_edited_nonH.pdb']
path = '/das/work/p18/p18594/cecilia-offline/NLSA/data_rho_2/results_NLSA/Phase_model';
pdbpath = [path, '/bov_nlsa_refine_96.pdb']
pdb = pdbread(pdbpath);

% PDB COOS ARE CARTESIAN, and in A
x_min = min([pdb.Model.Atom(:).X]);
x_max = max([pdb.Model.Atom(:).X]);
y_min = min([pdb.Model.Atom(:).Y]);
y_max = max([pdb.Model.Atom(:).Y]);
z_min = min([pdb.Model.Atom(:).Z]);
z_max = max([pdb.Model.Atom(:).Z]);
lim_pdb = [x_min x_max y_min y_max z_min z_max];

dx = x_max - x_min;
dy = y_max - y_min;
dz = z_max - z_min;

% CELL WORK for MAPROT
border = 5
A = round((x_max+border) - (x_min-border));
B = round((y_max+border) - (y_min-border));
C = round((z_max+border) - (z_min-border));
alpha = 90;
beta = 90;
gamma = 90;
sprintf('MAPROT: \nCELL WORK %d %d %d %d %d %d', A, B, C, alpha, beta, gamma)

% CELL WORK for MAPROT
factor = 8
grid_A = factor*A;
grid_B = factor*B;
grid_C = factor*C;
sprintf('MAPROT: \nGRID WORK %d %d %d', grid_A, grid_B, grid_C)

% XYZLIM for MAPROT
x_lim_min = round(factor*(x_min-border));
x_lim_max = grid_A + x_lim_min;
y_lim_min = round(factor*(y_min-border));
y_lim_max = grid_B + y_lim_min;
z_lim_min = round(factor*(z_min-border));
z_lim_max = grid_C + z_lim_min;
sprintf('MAPROT: \nXYZLIM %d %d %d %d %d %d', x_lim_min, x_lim_max, y_lim_min, y_lim_max, z_lim_min, z_lim_max)

