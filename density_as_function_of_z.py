from MDAnalysis import Universe
from MDAnalysis.analysis import lineardensity as lin
import re
from sys import argv
import numpy as np
import matplotlib.pyplot as plt

def get_universe_with_names(topo, traj, pdb_name):
    with open(pdb_name) as f:
        pdb_lines = f.readlines()[1:-1]
    rows = [re.split(r'\s+', row) for row in pdb_lines]
    atom_names = [row[2] for row in rows]

    u = Universe(topo, traj, dt=0.20)
    u.add_TopologyAttr('name', atom_names)

    return u


def density(topo, traj, pdb, output_file_name):
    u = get_universe_with_names(topo, traj, pdb)

    water_ag = u.select_atoms("name XOW or name XHW")
    CO2_ag = u.select_atoms("name XXC or name XXO")

    plt.figure(0)
    water_udensity = lin.LinearDensity(water_ag, grouping='atoms', binsize=2).run(start=-500)
    water_udens = water_udensity.results['z']['mass_density']
    plt.plot(np.linspace(0, 100, 100), water_udens)
    plt.xlabel("Density Bin (2 Ang Binsize)")
    plt.ylabel("Mass Density (kg/m^3)")
    plt.savefig('Water_Density_{}.png'.format(output_file_name))

    plt.figure(1)
    CO2_udensity = lin.LinearDensity(CO2_ag, grouping="atoms", binsize=2).run(start=-500)
    CO2_udens = CO2_udensity.results['z']['mass_density']
    plt.plot(np.linspace(0, 100, 100), CO2_udens)
    plt.xlabel("Density Bin (2 Ang Binsize)")
    plt.ylabel("Mass Density (kg/m^3)")
    plt.savefig('CO2_Density_{}.png'.format(output_file_name))


if __name__ == '__main__':
    density(argv[1], argv[2], argv[3], argv[4])
