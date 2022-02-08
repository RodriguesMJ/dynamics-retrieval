#!/bin/sh
module unload crystfel
#module load crystfel/0.6.3	  
module load crystfel      
partialator -i all_detwinned.stream -o partialator.hkl -y 6/m --model=xsphere --min-measurements=1 --push-res=0 --iterations=1 --custom-split=csplit_partialitymodeling.dat --no-logs -j 24 > partialator.log 2>&1 	    


