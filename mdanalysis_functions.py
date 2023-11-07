from MDAnalysis import Universe
import re


def get_universe_with_names(topo, traj, pdb_name):
    with open(pdb_name) as f:
        pdb_lines = f.readlines()[1:-1]
    rows = [re.split(r'\s+', row) for row in pdb_lines]
    atom_names = [row[2] for row in rows]

    u = Universe(topo, traj, dt=0.20)
    u.add_TopologyAttr('name', atom_names)

    return u

