#!/bin/bash
module load phenix
for i in rho_light_avg_plus_mode_0_timestep_*.mtz; do ./maps_phenix_rho.sh $i rho_dark_avg.mtz 2.0; done
