# Crystal
Molecular dynamics simulation using Python language

You can launch this program by typing: 'python crystal.py '{parameterfilenameNUMBER}.txt'' and output will include .txt files:
* energiesNUMBER
* pressuresNUMBER
* temperaturesNUMBER
.xyz file:
* trajectoryNUMBER
and .png files:
* energies
* pressures
* temperatures

Parameters file should contain 9 rows like:
* N = 729 (number of particles in the crystal)
* dim = 3 (number of dimensions of the crystal)
* epsilon = 1 (minimum potential of the van der Waals bonding)
* f = 1e4 (elasticity coefficient)
* R = 0.38 (interatomic distance)
* a = 0.38 (≈R)
* m = 40 (mass of one atom)
* T = 100 (initial temperature of the crystal)
* kB = 8.31e-3 (Boltzmann constant)

You can visualise simulation of the crystal e.g. in Jmol using .xyz file

Number of steps and length of time steps are embedded in the program, succesively: 10000 and 0.001
