#!/bin/sh
	        
partialator -i all_detwinned.stream -o all_detwinned.hkl -y 6/m --model=unity --push-res=1.2 --iterations=1 -j 24 > partialator.log 2>&1 	    


