#!/bin/bash
module load phenix
for i in extracted_*.mtz; do ./maps_phenix_rho.sh $i extracted_Is_p_0_1_modes_timestep_000000.mtz 1.8; done
