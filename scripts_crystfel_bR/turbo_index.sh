#!/bin/sh

# Split a large indexing job into many small tasks and submit using SLURM

# ./turbo-index-slurm my-files.lst label my.geom /location/for/streams

# Copyright © 2016-2017 Deutsches Elektronen-Synchrotron DESY,
#                       a research centre of the Helmholtz Association.
#
# Authors:
#   2016      Steve Aplin <steve.aplin@desy.de>
#   2016-2017 Thomas White <taw@physics.org>

SPLIT=1000  # Size of job chunks

INPUT=$1
RUN=$2
GEOM=$3
STREAMDIR=$4

# Set up environment here if necessary
module clear
source /etc/scripts/mx_fel.sh
module unload hdf5_serial/1.8.20
module load hdf5_serial/1.8.17

mkdir $STREAMDIR
cd $STREAMDIR

cp $INPUT events-${RUN}.lst

# Count total number of events
wc -l events-${RUN}.lst

# Split the events up, will create files with $SPLIT lines
split -a 3 -d -l $SPLIT events-${RUN}.lst split-events-${RUN}.lst

ln -s ../BR.cell .
# Clean up
#rm -f events-${RUN}.lst

# Loop over the event list files, and submit a batch job for each of them
for FILE in split-events-${RUN}.lst*; do

    # Stream file is the output of crystfel
    STREAM=`echo $FILE | sed -e "s/split-events-${RUN}.lst/${RUN}.stream/"`

    # Job name
    NAME=`echo $FILE | sed -e "s/split-events-${RUN}.lst/${RUN}-/"`

    echo "$NAME: $FILE  --->  $STREAM"

    SLURMFILE="$STREAMDIR/${NAME}.sh"

    echo "#!/bin/sh" > $SLURMFILE
    echo >> $SLURMFILE
    echo "#SBATCH --partition=hour" >> $SLURMFILE  
    echo "#SBATCH --time=00:59:00" >> $SLURMFILE
    echo "#SBATCH --nodes=1" >> $SLURMFILE
    echo >> $SLURMFILE
    echo "#SBATCH --chdir     $STREAMDIR" >> $SLURMFILE
    echo "#SBATCH --job-name  $NAME" >> $SLURMFILE
    echo "#SBATCH --output    $NAME-%N-%j.out" >> $SLURMFILE
    echo "#SBATCH --error     $NAME-%N-%j.err" >> $SLURMFILE
    echo >> $SLURMFILE
    echo "module clear" >> $SLURMFILE
    echo "source /etc/scripts/mx_fel.sh" >> $SLURMFILE
    echo "module unload hdf5_serial/1.8.20" >> $SLURMFILE
    echo "module load hdf5_serial/1.8.17" >> $SLURMFILE
    echo >> $SLURMFILE

    echo "nproc=\`grep proce /proc/cpuinfo | wc -l\`" >> $SLURMFILE
    
    command="indexamajig -i $FILE -o $STREAMDIR/$STREAM --indexing=xgandalf-latt-cell --geometry=$GEOM --pdb=$STREAMDIR/BR.cell --peaks=cxi --integration=rings --int-radius=3,4,7 --peak-radius=3,4,7 -j \${nproc} "
    
    echo $command >> $SLURMFILE

    sbatch $SLURMFILE
    
done
