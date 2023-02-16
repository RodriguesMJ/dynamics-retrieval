#!/bin/bash
module load phenix
for i in bR_light_p_0_2_modes_timestep_*.mtz; do ./maps_phenix_bR.sh $i I_dark_avg.mtz 1.8; done
