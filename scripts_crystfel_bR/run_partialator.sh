#!/bin/sh
module unload crystfel
module load crystfel/0.6.3	        
partialator -i all_detwinned.stream -o all_detwinned.hkl -y 6/m --model=unity --push-res=1.2 --iterations=1 --custom-split=csplit.dat -j 24 > partialator.log 2>&1 	    


