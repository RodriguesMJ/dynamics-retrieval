clear all;
pdbpath = './6g7h_edited_nonH.pdb';
pdb = pdbread(pdbpath);
   %lim_map = [sX(1) sX(end) sY(1) sY(end) sZ(1) sZ(end)];
lim_pdb = [min([pdb.Model.Atom(:).X]) max([pdb.Model.Atom(:).X]) min([pdb.Model.Atom(:).Y]) max([pdb.Model.Atom(:).Y]) min([pdb.Model.Atom(:).Z]) max([pdb.Model.Atom(:).Z])];
