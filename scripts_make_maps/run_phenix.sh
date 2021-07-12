#!/bin/bash
module load phenix
for i in rho_light_mode_0_2_timestep_*.mtz; do ./maps_phenix_rho.sh $i rho_alldark_mode_0_avg.mtz 2.0; done
