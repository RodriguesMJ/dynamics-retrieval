#!/bin/sh
# PROCESS MAPS OF BACTERIORHOPDOPSIN/RHODOPSIN FOR ANALYSIS WITH MAPTOOL
#----------------------------------------------------------------------------------
# - Converts map to cartesian coordinates (with approx. 5 ?? border around protein)
# - Converts map to .h5 format
# - Reads map dimensions using mapdump
#----------------------------------------------------------------------------------


# INPUT
# Note! Also set cell work, grid work, xyzlim in maprot command.
#----------------------------------------------------------------------------------
here=/das/work/p18/p18594/cecilia-offline/NLSA/data_rho_2/results_NLSA/map_analysis
filetype='ccp4'    #'map' or 'ccp4'

indir=$here/input_m_0_1_step_10_range_50000_99990
outdir=$here/output_m_0_1_step_10_range_50000_99990
scriptdir=/das/work/p17/p17491/Cecilia_Casadei/NLSA/code/scripts_map_analysis
#----------------------------------------------------------------------------------



# PROCESS FILES
#----------------------------------------------------------------------------------
mkdir $outdir
mkdir $outdir/log
mkdir $outdir/tmp

cd $indir

for f in *.$filetype 
do

name=${f%.*}
echo Preparing file $name $f



# 1. EXTEND INITIAL MAP TO COVER FULL CELL (necessary for mapprot to run)
#----------------------------------------
mapmask mapin $f \
        mapout $outdir/tmp/$name'_fullcell.ccp4' << eof > $outdir/log/$name'_fullcell.log'
XYZLIM CELL
eof



# 2. TRANSLATE MAP TO CARTESIAN COORDINATES
#    Set CELL WORK / GRID WORK / XYZLIM to cover the volume of interest
#-----------------------------------------
# CELL WORK: A B C 90 90 90
#   ->  A B C sides of cell in Angstrom (= size of pdb + 5?? border)
#   angles 90 degree 
# GRID WORK: 4*A 4*B 4*C 
#   -> 0.25 ?? distance between grid points
# XYZLIM: 4*xmin 4*xmax 4*ymin 4*ymax 4*zmin 4*zmax
#-----------------------------------------

# Get maprot parameters from check_PDB_limits.m

# BACTERIORHODOPSIN (originally used)
# CELL WORK 42 54 77 90 90 90
# GRID WORK 336 432 616
# XYZLIM -40 296 60 492 0 616

# BACTERIORHODOPSIN (from check_PDB_limits.m)i
# CELL WORK 42 54 78 90 90 90'
# GRID WORK 336 432 624
# XYZLIM -40 296 77 509 7 631

# RHODOPSIN (from check_PDB_limits.m 
# CELL WORK 74 65 91 90 90 90
# GRID WORK 592 520 728
# XYZLIM -348 244 -32 488 -63 665

maprot mapin $outdir/tmp/$name'_fullcell.ccp4' \
       wrkout $outdir/$name'_cartesian.map' << eof >> $outdir/log/$name'_cartesian.log'
MODE FROM
CELL WORK 74 65 91 90 90 90
GRID WORK 592 520 728
XYZLIM -348 244 -32 488 -63 665
SYMM WORK 1
AVER
ROTA POLAR 0 0 0
TRANS 0 0 0
eof



# 3. CONVERT TO .h5 (for matlab)
#----------------------------------------------------------------------------
python $scriptdir/map_to_h5.py $outdir/$name'_cartesian.map'  $outdir/$name'_cartesian.h5'



# 4. EXTRACT INFORMATION ON MAP DIMESIONS
#    cell dimensions, grid sampling, start and stop, axis order
#    and sigma (=RMSD) from new and original map
#    (Sigma of the original map is used in the matlab calculations since the 
#    sigma value for the cartesian map is slightly changed in the conversion.)
#----------------------------------------------------------------------------
# Example of data in mapdump output:
# Cell dimensions .................................   42.0000    54.0000    77.0000    90.0000    90.0000    90.0000
# Grid sampling on x, y, z ........................  168  216  308
# Start and stop points on columns, rows, sections  -136  172 -144   24 -176 
# Fast, medium, slow axes  3  1  2
# Rms deviation from mean density .................     0.01195
#----------------------------------------------------------------------------

mapdump mapin $outdir/$name'_cartesian.map' << eof > $outdir/log/$name'_cartesian_header.txt'
eof

mapdump mapin $f << eof > $outdir/log/$name'_original_header.txt'
eof

grep 'Cell dimensions' $outdir/log/$name'_cartesian_header.txt' | awk '{ print $4,$5,$6,$7,$8,$9}' > $outdir/log/$name'_XYZinfo.dat'
grep 'Grid sampling on x, y, z' $outdir/log/$name'_cartesian_header.txt' | awk '{ print $8,$9,$10}' >> $outdir/log/$name'_XYZinfo.dat'
grep 'Start and stop points on columns, rows, sections' $outdir/log/$name'_cartesian_header.txt' | awk '{ print $9,$10,$11,$12,$13,$14}' >> $outdir/log/$name'_XYZinfo.dat'
grep '^ Fast, medium, slow axes' $outdir/log/$name'_cartesian_header.txt' | awk '{ print $5,$6,$7}' >> $outdir/log/$name'_XYZinfo.dat'

grep 'Rms deviation from mean density' $outdir/log/$name'_cartesian_header.txt' | awk '{ print $7}'  >> $outdir/log/$name'_XYZinfo.dat'
grep 'Rms deviation from mean density' $outdir/log/$name'_original_header.txt' | awk '{ print $7}'  >> $outdir/log/$name'_XYZinfo.dat'

done
