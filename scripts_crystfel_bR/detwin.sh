#!/bin/bash

module unload crystfel
module load crystfel/0.6.3

ambigator all.stream -o all_detwinned.stream -y 6/m -w 6/mmm --lowres=20 --highres=3.0 --ncorr=2000 -j 24 > ambigator.log 2>&1 
