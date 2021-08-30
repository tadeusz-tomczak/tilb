# Taichi Lattice Boltmann

Taichi Lattice Boltzmann (tilb) is an implementation of the lattice-Boltzmann method in [Taichi](https://github.com/taichi-dev/taichi). The code is loosely based on [LBM_Taichi](https://github.com/hietwll/LBM_Taichi) project, but with significantly redesigned the code to improve performance and handle a wider set of boundary conditions.

## Usage
tilb is designed as a main library (``tilb.py`` file) and separate script used for setting and running a simulation. Four example simulation are provided:
- lid-driven cavity (file ``cavity.py``),
- channel flow (file ``channel.py``),
- flow past cylinder (file ``cylinder.py``)
- flow through sparse geometry with given porosity (file ``sparse.py``)

The simulation can be run by simply running given script, for example:
```
ti cavity.py
```
launches lid-driven cavity simulation.

Simulations script contain commented sets of different settings, for details please consult the source code (I hope it is self-explaining). Each simulation has the main function ``simulate()``, which builds case and launches computational engine. At the end of simulation, file ``out.vts`` is generated, which can be further processed with [ParaView](https://www.paraview.org/).
