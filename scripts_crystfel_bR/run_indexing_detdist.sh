#!/bin/bash 

RUN=$1 # E.G. 0057

GEO_START=/das/work/p17/p17491/Cecilia_Casadei/NLSA/data_bR_2/indexing/geometry/start.geom
OUT_DIR=./run_${RUN}
LISTFILE=/das/work/p17/p17491/Cecilia_Casadei/NLSA/data_bR_2/indexing/event_lists/lists_new_short/event_list_run_${RUN}_1000_events.txt
CELL=/das/work/p17/p17491/Cecilia_Casadei/NLSA/data_bR_2/indexing/cell/BR.cell

mkdir ${OUT_DIR}

DETDISTMINUS=82250
DETDISTPLUS=83750

for len in `seq $DETDISTMINUS 20 $DETDISTPLUS`
	do 
		sed 's/clen.*/clen = 0.0'$len'/' ${GEO_START} > ${OUT_DIR}/geo_$len.geom 
	done
		
DETDISTMINUS=81000
DETDISTPLUS=82200

for len in `seq $DETDISTMINUS 200 $DETDISTPLUS`
	do 
		sed 's/clen.*/clen = 0.0'$len'/' ${GEO_START} > ${OUT_DIR}/geo_$len.geom 
	done	
	
DETDISTMINUS=83800
DETDISTPLUS=85000

for len in `seq $DETDISTMINUS 200 $DETDISTPLUS`
	do 
		sed 's/clen.*/clen = 0.0'$len'/' ${GEO_START} > ${OUT_DIR}/geo_$len.geom 
	done

cd ${OUT_DIR}

for filename in geo_*.geom 
	do
    		sbatch -p hour <<EOF
#!/bin/bash
 
module clear
source /etc/scripts/mx_fel.sh
module unload hdf5_serial/1.8.20
module load hdf5_serial/1.8.17

nproc=\`grep proce /proc/cpuinfo | wc -l\`

indexamajig -i ${LISTFILE} -o run_${RUN}_${filename}.stream --indexing=xgandalf-latt-cell --geometry=${filename} --pdb=${CELL} --peaks=cxi --integration=rings --int-radius=3,4,7 --peak-radius=3,4,7 -j \${nproc} > run_${RUN}_${filename}.log 2>&1

EOF
	done

