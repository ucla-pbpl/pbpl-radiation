#!/bin/sh
mpirun -np 101 --use-hwthread-cpus pbpl-radiation-calc-trajectories calc-trajectories.toml
mpirun -np 101 --use-hwthread-cpus pbpl-radiation-calc-farfield calc-farfield.toml
pbpl-radiation-plot-farfield plot-farfield.toml
