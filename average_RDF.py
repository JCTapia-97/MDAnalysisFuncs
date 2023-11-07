from MDAnalysis.analysis import rdf
import matplotlib.pyplot as plt
from MDAnalysis import Universe
import numpy as np
from sys import argv
import re
from MDAnalysis.analysis import lineardensity as lin
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis as HBA
from MDAnalysis.analysis.base import (AnalysisBase,
                                      AnalysisFromFunction,
                                      analysis_class)


def density(topo, traj, pdb, ag1, last_frame, output_file_name):
    u = get_universe_with_names(topo, traj, pdb)
    last_frame = int(last_frame)

    z = float(u.dimensions[2])
    bin_size = 2
    num_bins = np.ceil(z / bin_size).astype(int)
    x_graph = np.linspace(0, z, num_bins)

    water_ag = u.select_atoms(ag1)

    water_udensity = lin.LinearDensity(water_ag, grouping='atoms', binsize=bin_size).run(start=-500)
    water_udens = water_udensity.results['z']['mass_density']
    plt.plot(x_graph, water_udens)
    plt.xlabel("Density Bin (2 Ang Binsize)")
    plt.ylabel("Mass Density (kg/m^3)")
    plt.savefig('Water_Density.png')

    raw_data_file(x_graph, water_udens, "{}_water_density".format(output_file_name))


def raw_data_file(x_values, y_values, output_file_name):
    output_format = '%s %s \n'

    with open('{}_raw_data.txt'.format(output_file_name), "+w") as out_file:
        for count, _ in enumerate(x_values):
            out_file.write(output_format % (x_values[count], y_values[count]))


def radius_of_gyration(topo, traj, pdb_name, last_frames, output_file_name):
    last_frames = int(last_frames)
    u = get_universe_with_names(topo, traj, pdb_name)
    # u = Universe(topo, traj)

    ag1 = u.select_atoms("(name C1 or name CHR or name CH2 or name Cin) and index 864:1400")
    ag2 = u.select_atoms("(name C1 or name CHR or name CH2 or name Cin) and index 1401:1937")
    ag3 = u.select_atoms("(name C1 or name CHR or name CH2 or name Cin) and index 1938:2474")
    ag4 = u.select_atoms("(name C1 or name CHR or name CH2 or name Cin) and index 2475:3011")
    ag5 = u.select_atoms("(name C1 or name CHR or name CH2 or name Cin) and index 3012:3548")
    ag6 = u.select_atoms("(name C1 or name CHR or name CH2 or name Cin) and index 3549:4085")
    ag7 = u.select_atoms("(name C1 or name CHR or name CH2 or name Cin) and index 4086:4622")
    ag8 = u.select_atoms("(name C1 or name CHR or name CH2 or name Cin) and index 4623:5159")
    ag9 = u.select_atoms("(name C1 or name CHR or name CH2 or name Cin) and index 5160:5697")

    rog1 = AnalysisFromFunction(radgyr, u.trajectory,
                                ag1, ag1.masses,
                                total_mass=np.sum(ag1.masses))

    rog1.run(start=last_frames)

    rog2 = AnalysisFromFunction(radgyr, u.trajectory,
                                ag2, ag2.masses,
                                total_mass=np.sum(ag2.masses))

    rog2.run(start=last_frames)

    rog3 = AnalysisFromFunction(radgyr, u.trajectory,
                                ag3, ag3.masses,
                                total_mass=np.sum(ag3.masses))

    rog3.run(start=last_frames)

    rog4 = AnalysisFromFunction(radgyr, u.trajectory,
                                ag4, ag4.masses,
                                total_mass=np.sum(ag4.masses))

    rog4.run(start=last_frames)

    rog5 = AnalysisFromFunction(radgyr, u.trajectory,
                                ag5, ag5.masses,
                                total_mass=np.sum(ag5.masses))

    rog5.run(start=last_frames)

    rog6 = AnalysisFromFunction(radgyr, u.trajectory,
                                ag6, ag6.masses,
                                total_mass=np.sum(ag6.masses))

    rog6.run(start=last_frames)

    rog7 = AnalysisFromFunction(radgyr, u.trajectory,
                                ag7, ag7.masses,
                                total_mass=np.sum(ag7.masses))

    rog7.run(start=last_frames)

    rog8 = AnalysisFromFunction(radgyr, u.trajectory,
                                ag8, ag8.masses,
                                total_mass=np.sum(ag8.masses))

    rog8.run(start=last_frames)

    rog9 = AnalysisFromFunction(radgyr, u.trajectory,
                                ag9, ag9.masses,
                                total_mass=np.sum(ag9.masses))

    rog9.run(start=last_frames)

    concat_data = [rog1.results['timeseries'], rog2.results['timeseries'], rog3.results['timeseries'],
                                  rog4.results['timeseries'], rog5.results['timeseries'], rog6.results['timeseries'],
                                  rog7.results['timeseries'], rog8.results['timeseries'], rog9.results['timeseries']]
    concat_data = np.asarray(concat_data)
    print(concat_data)

    averaged_rog = np.average(concat_data, axis=0)
    std_rog = np.std(concat_data, axis=0)

    all_data = averaged_rog.T[0]
    x_data = averaged_rog.T[1]
    y_data = averaged_rog.T[2]
    z_data = averaged_rog.T[3]

    all_std = std_rog.T[0]
    x_std = std_rog.T[1]
    y_std = std_rog.T[2]
    z_std = std_rog.T[3]

    # labels = ['all', 'x-axis', 'y-axis', 'z-axis']
    # for col, label in zip(rog.results['timeseries'].T, labels):
    #     plt.plot(col, label=label)
    # plt.legend()
    # plt.ylabel('Radius of Gyration (Angstroms)')
    # plt.xlabel('Frame')
    # plt.savefig('{}_ROG.png'.format(output_file_name))

    timestep = np.linspace(1, np.absolute(last_frames), np.absolute(last_frames))
    condensed_data = [timestep, all_data, all_std, x_data, x_std, y_data, y_std, z_data, z_std]
    condensed_data = np.asarray(condensed_data)
    np.savetxt("{}_ROG_raw_data.txt".format(output_file_name), condensed_data.T, delimiter=' ')


def radgyr(atomgroup, masses, total_mass=None):
    # coordinates change for each frame
    coordinates = atomgroup.positions
    center_of_mass = atomgroup.center_of_mass()

    # get squared distance from center
    ri_sq = (coordinates - center_of_mass) ** 2
    # sum the unweighted positions
    sq = np.sum(ri_sq, axis=1)
    sq_x = np.sum(ri_sq[:, [1, 2]], axis=1)  # sum over y and z
    sq_y = np.sum(ri_sq[:, [0, 2]], axis=1)  # sum over x and z
    sq_z = np.sum(ri_sq[:, [0, 1]], axis=1)  # sum over x and y

    # make into array
    sq_rs = np.array([sq, sq_x, sq_y, sq_z])

    # weight positions
    rog_sq = np.sum(masses * sq_rs, axis=1) / total_mass
    # square root and return
    return np.sqrt(rog_sq)


def hydrogen_bonding(topo, traj, pdb_name, ag1, ag2, last_frames, output_file_name):
    last_frames = int(last_frames)
    u = get_universe_with_names(topo, traj, pdb_name)
    O_atom_count = []

    for ts in u.trajectory[last_frames:]:
        O_atom_count.append(len(u.select_atoms(ag2, updating=True)))

    hbonds = HBA(universe=u, hydrogens_sel=ag1, acceptors_sel=ag2)  # , d_a_cutoff=2.5, d_h_a_angle_cutoff=113.58)
    hbonds.run(start=last_frames)
    count_by_frame = hbonds.count_by_time()
    O_atom_count = np.asarray(O_atom_count).astype(int)
    norm_data = np.divide(count_by_frame, O_atom_count)
    norm_data = np.asarray(norm_data).astype(np.float32)

    average_num_hbonds = np.average(norm_data)
    std_of_hbonds = np.std(norm_data)
    stats = np.asarray([average_num_hbonds, std_of_hbonds]).astype(np.float32)

    print(count_by_frame, O_atom_count, norm_data, average_num_hbonds, std_of_hbonds)

    condensed_data = [count_by_frame, O_atom_count, norm_data]
    condensed_data = np.asarray(condensed_data).astype(np.float32)
    condensed_data = np.transpose(condensed_data)
    print(condensed_data)

    np.savetxt("{}_H_Bond_raw_data.txt".format(output_file_name), condensed_data, delimiter=' ')
    np.savetxt("{}_H_Bond_Average_STDEV_data.txt".format(output_file_name), stats, delimiter=' ')


def get_universe_with_names(topo, traj, pdb_name):
    with open(pdb_name) as f:
        pdb_lines = f.readlines()[1:-1]
    rows = [re.split(r'\s+', row) for row in pdb_lines]
    atom_names = [row[2] for row in rows]

    u = Universe(topo, traj, dt=0.20)
    u.add_TopologyAttr('name', atom_names)

    return u


def average_rdf(type, topo, traj, pdb_name, ag1, ag2, last_frames, output_file_name):
    last_frames = int(last_frames)
    u = get_universe_with_names(topo, traj, pdb_name)

    atom_group1 = u.select_atoms(ag1)
    atom_group2 = u.select_atoms(ag2)

    if "w" in type:
        ss_rdf = rdf.InterRDF(atom_group1, atom_group2, exclusion_block=(1, 1))
    else:
        ss_rdf = rdf.InterRDF(atom_group1, atom_group2)

    ss_rdf.run(start=last_frames)

    # print(np.cumsum(ss_rdf.results.count))
    # print(u.trajectory.n_frames)

    cum_rad_rdf = np.cumsum(ss_rdf.results.count)

    # vols = np.power(ss_rdf.results.edges, 3)
    # print(ss_rdf.results.edges)
    # print(np.diff(vols))
    # norm = u.trajectory.n_frames
    # norm = 4/3 * np.pi * np.diff(vols)
    num_density = cum_rad_rdf / len(atom_group1)
    # pseudo_count = ss_rdf.results.rdf * norm

    # print(pseudo_count, rdf_num_integral)

    if "prop" in ag1 or "prop" in ag2:
        z_cutoff = float(ag2.split()[-1])
        z_box = float(u.dimensions[2])
        z_frac = 1
        if z_cutoff > z_box:
            print("z cutoff is unphysical. z cutoff ({}) must be <= z box ({}),"
                  " using z_frac =1 ".format(z_cutoff, z_box))
        elif '>' in ag1 or '>' in ag2:
            z_frac = (z_box - z_cutoff) / z_box
        elif '<' in ag1 or '<' in ag2:
            z_frac = z_cutoff / z_box
        print(z_frac)
        ss_rdf.results.rdf *= z_frac
        num_density /= z_frac

    plt.figure(1)
    plt.plot(ss_rdf.results.bins, ss_rdf.results.rdf)
    plt.xlabel("Distance (A)")
    plt.ylabel("g(x)")
    plt.savefig('{}_RDF.png'.format(output_file_name))

    plt.figure(2)
    plt.plot(ss_rdf.results.bins, num_density)
    plt.xlabel("Distance (A)")
    plt.ylabel("Number of atoms")
    plt.savefig('{}_Number_Density.png'.format(output_file_name))

    raw_data_file(ss_rdf.results.bins, ss_rdf.results.rdf, "{}_RDF".format(output_file_name))
    raw_data_file(ss_rdf.results.bins, ss_rdf.results.count, "{}_Count".format(output_file_name))
    raw_data_file(ss_rdf.results.bins, num_density, "{}_Num_Integral".format(output_file_name))
    raw_data_file(ss_rdf.results.bins, ss_rdf.results.edges, "{}_Edges".format(output_file_name))


if __name__ == '__main__':
    if 'dens' in argv[1]:
        density(argv[2], argv[3], argv[4], argv[5])
    elif 'rdf' in argv[1]:
        average_rdf(argv[1], argv[2], argv[3], argv[4], argv[5], argv[6], argv[7], argv[8])
    elif 'hbond' in argv[1]:
        hydrogen_bonding(argv[2], argv[3], argv[4], argv[5], argv[6], argv[7], argv[8])
    elif 'rog' in argv[1]:
        # radius_of_gyration(PSF, DCD, "a", "protein", 0, "test")
        radius_of_gyration(argv[2], argv[3], argv[4], argv[5], argv[6])
    # average_rdf(TPR, XTC)
    # average_rdf('P2.data', 'movie.lammpsdump', 'P2_WaterOnly_1CO2_Polymer_3.5590_6.39_56.028.pdb',
    #             "name XOW and prop z <= 150", "name XOW and prop z <= 150", 'AAA')
