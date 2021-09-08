#!/bin/bash
srun -p week -t 8-00:00:00 partialator -i rho_nlsa_scans_light_alldark.stream -o rho.hkl -y mmm --model=unity --iterations=1 --custom-split=csplit.dat --output-every-cycle -j 24 > rho_nlsa.log 2>&1 &

