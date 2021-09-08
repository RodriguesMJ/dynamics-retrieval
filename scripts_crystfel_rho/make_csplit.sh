#!/bin/bash 

for file in *.stream; do
  grep filename $file | awk '{print $3}' > ${file}.lst
  grep Event $file | awk '{print $2}' > ${file}_events.lst
  paste ${file}.lst ${file}_events.lst > ${file}.dat
done


for timepoint in *.dat; do
 sed -e "s/$/\ $timepoint/" ${timepoint} >> csplit1.dat
done
