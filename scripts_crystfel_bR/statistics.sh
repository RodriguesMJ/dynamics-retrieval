#!/bin/sh
echo 'syntax: statistics.sh Filename(without .hkl1/hkl2) ResCut(in A)'

HKLFILE=$1
RESO=$2

CELLFILE=BR.cell

check_hkl ${HKLFILE} -y 6/m -p ${CELLFILE} --highres=${RESO}
mv shells.dat check.dat
compare_hkl  ${HKLFILE}1 ${HKLFILE}2 -p ${CELLFILE} --highres=${RESO} --fom=rsplit
mv shells.dat rsplit.dat
compare_hkl ${HKLFILE}1 ${HKLFILE}2 -p ${CELLFILE} --highres=${RESO} --fom=ccstar
mv shells.dat ccstar.dat
compare_hkl ${HKLFILE}1 ${HKLFILE}2 -p ${CELLFILE} --highres=${RESO} --fom=cc
mv shells.dat cc.dat
cat rsplit.dat ccstar.dat cc.dat check.dat > statistics_${HKLFILE}.dat

#for i in *.hkl; do ./statistics.sh $i 1.50; done
