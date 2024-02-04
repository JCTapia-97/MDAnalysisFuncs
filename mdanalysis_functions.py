import matplotlib.pyplot as plt
import heapq
import numpy as np
import argparse, re, time, os, shutil, ast
from scipy.signal import correlate
from MDAnalysis import Universe
from MDAnalysis.analysis import rdf
from MDAnalysis.analysis import lineardensity as lin
from MDAnalysis.analysis.hydrogenbonds import HydrogenBondAnalysis as HBA
from MDAnalysis.analysis import distances
from MDAnalysis.lib.distances import calc_angles
from MDAnalysis.analysis import density
from MDAnalysis import Writer
from MDAnalysis import transformations as trans
from clustering import Clustering
from math import floor, ceil


class MDAFunctions:
    def __init__(self,
                 topology=None,
                 trajectory=None,
                 pdb_file=None,
                 atomgroup1=None,
                 atomgroup2=None,
                 start_from_frame=None,
                 end_on_frame=None,
                 segment=1,
                 topo_file_type=None,
                 traj_file_type=None,
                 outfile_name=None,
                 function=None,
                 rdf_exclusion=None,
                 timestep=None,
                 verbose=False,
                 very_verbose=False,
                 time=False,
                 xtc_convert=False,
                 add_names=False,
                 hbond_cutoff=None):

        self.topology = topology
        self.trajectory = trajectory
        self.pdb = pdb_file
        self.ag1 = atomgroup1
        self.ag2 = atomgroup2
        self.start_from_frame = start_from_frame
        self.end_on_frame = end_on_frame
        self.segment = segment
        self.topo_file_type = topo_file_type
        self.traj_file_type = traj_file_type
        self.outfile_name = outfile_name
        self.analysis_type = function
        self.exclusion = rdf_exclusion
        self.dt = timestep
        self.hbond_cutoff = hbond_cutoff

        self.xtc_convert = xtc_convert
        self.add_names = add_names

        self.verbose = verbose
        self.very_verbose = very_verbose
        self.time = time

        self.time_dependent = False
        self.transformed = False

        self._verbose_print("""Initializing MDAFunctions:
              topology file = {}
              topology file type = {}
              trajectory file = {}
              trajectory file type = {}
              PDB file = {}
              Analysis Preformed = {}
              Atom Group 1 = {}
              Atom Group 2 = {}
              Starting from frame {}
              Timestep = {} ps
              segmentd {} time(s)
              Output file name = {}
              RDF Exclusion = {}""".format(self.topology,
                                           self.topo_file_type,
                                           self.trajectory,
                                           self.traj_file_type,
                                           self.pdb,
                                           self.analysis_type,
                                           self.ag1,
                                           self.ag2,
                                           self.start_from_frame,
                                           self.dt,
                                           self.segment,
                                           self.outfile_name,
                                           self.exclusion))

    # Generates universe based on topology and trajectory.
    # If lammpsdump file is used a PDB must be supplied for names
    def get_universe(self):
        traj_list = MDAFunctions._str_to_list(self.trajectory)
        self._verbose_print("Trajectory list is {}".format(traj_list))
        if self.xtc_convert:
            traj_list = self._xtc_converter(traj_list)
        u = self._mda_universe_generator(traj_list)
        self._verbose_print("Universe has been created!")
        # adds names based on PDB if lammpsdump file
        self._verbose_print("Total number of frames in Universe: {}".format(
            u.trajectory.n_frames))
        return u

    # Runs analysis and generated output files
    def run_analysis(self):
        # considering adding implementation to run multiple jobs sequencially,
        # but may not be worth it
        start_time = self._get_time()
        u = self.get_universe()
        self._time_since(start_time, "Universe creation time")

        analysis_functions = {
            'dens': self.density_as_func_z,
            'rdf': self.average_rdf,
            'hbond': self.hydrogen_bonding,
            'rog': self.radius_of_gyration,
            'e2e': self.end_to_end,
            'oto': self.tetra_order
        }
        analysis_function = analysis_functions.get(self.analysis_type)

        if analysis_function is None:
            raise ValueError("Analysis function are limited to "
                             "dens, rdf, hbond, rog, e2e")

        segment_list = self._traj_segmenter()
        analysis_start_time = self._get_time()

        for seg_index, segment in enumerate(segment_list):
            starting_frame, ending_frame = \
                MDAFunctions._get_frame_limits(segment)
            analysis_data = analysis_function(u, starting_frame, ending_frame)
            output_file_name = self.output_file_name_maker(seg_index)
            self._make_output_files(output_file_name, analysis_data)
        self._file_mover()
        self._time_since(analysis_start_time, "Analysis time")

        if self.time_dependent:
            acf_start_time = self._get_time()
            acf_data = self.autocorrelation()
            acf_out_file_name = self.output_file_name_maker("ACF")
            self._make_output_files(acf_out_file_name, acf_data)
            self._file_mover()
            self._time_since(acf_start_time, "ACF time")

        self._time_since(start_time, "Total script time")

    def _make_output_files(self, output_file_name, data):
        MDAFunctions._save_data(output_file_name, data.T)
        self._very_verbose_print("Data saved for {}".format(output_file_name))
        self._plot_array(output_file_name)

    ### Analysis function sections 
    # Calculates density across z dimension. Only requires ag1
    def density_as_func_z(self, u, starting_frame, ending_frame):
        self._verbose_print(
            "Starting Density Analysis from frame {} to frame {}".format(
                starting_frame, ending_frame))

        bin_size = 3

        atom_group_1 = u.select_atoms(self.ag1)

        density_analysis = lin.LinearDensity(atom_group_1,
                                             grouping='atoms',
                                             binsize=bin_size)
        density_analysis.run(start=starting_frame,
                             stop=ending_frame)

        bin_edges = density_analysis.results.x.hist_bin_edges
        density_results = density_analysis.results['z']['mass_density']
        bin_centers = bin_edges[:-1] + 0.5
        self._verbose_print("Density results: {}".format(density_results))

        data_labels = self._data_label_maker("z-box length (A)",
                                             "Density (g/ml)")

        condensed_data = self._data_condenser(data_labels,
                                              bin_centers,
                                              density_results)

        return condensed_data

    # Calculates RDF requires only ag1 to be specified but ag2 can be specified as well
    def average_rdf(self, u, starting_frame, ending_frame):
        self._verbose_print(
            "Starting RDF Analysis from frame {} to frame {}".format(
                starting_frame, ending_frame))
        atom_group_1 = u.select_atoms(self.ag1, updating=True)
        if self.ag2 is None:
            atom_group_2 = u.select_atoms(self.ag1, updating=True)
        else:
            atom_group_2 = u.select_atoms(self.ag2, updating=True)

        z_frac = 1
        if "prop" in self.ag1 or (self.ag2 is not None and "prop" in self.ag2):
            z_box_length = float(u.dimensions[2])
            z_cutoff = self._limit_finder(self.ag1, self.ag2)
            z_frac = self._volume_fraction(z_cutoff,
                                           z_box_length,
                                           self.ag1,
                                           self.ag2)
            print(z_frac)

        rdf_analysis = rdf.InterRDF(atom_group_1,
                                    atom_group_2,
                                    exclusion_block=self.exclusion)
        rdf_analysis.run(start=starting_frame,
                         stop=ending_frame)

        rdf_bins = MDAFunctions._truncate(2, rdf_analysis.results.bins)
        crdf, num_dens = MDAFunctions._cum_num(rdf_analysis.results.count,
                                               len(atom_group_1))
        rdf_results = rdf_analysis.results.rdf

        num_dens = self._volume_adjustment(float(1/z_frac), num_dens)
        rdf_results = self._volume_adjustment(z_frac, rdf_results)

        data_labels = self._data_label_maker("r (A)",
                                             "RDF",
                                             "G(x)",
                                             "Number Density")
        condensed_data = self._data_condenser(data_labels,
                                              rdf_bins,
                                              rdf_results,
                                              crdf, num_dens)

        return condensed_data

    # Calculates average number of H-bonds in the simulation. Requires ag1 be hydrogen and ag2 be oxygen
    def hydrogen_bonding(self, u, starting_frame, ending_frame):
        if "OW" in self.ag1 or "HW" in self.ag2:
            raise ValueError("For H-bond analysis, ag1 must be Hydrogen and"
                             "ag2 must be Oxygen. Change atom selection or"
                             "atom naming to remove O from ag1 and H from ag2")
        self._verbose_print(
            "Starting H-bonding Analysis from frame {} to frame {}".format(
                starting_frame, ending_frame))

        u = self._make_whole(u,
                             self.ag1,
                             self.ag2)

        oxy_atoms = u.select_atoms(self.ag2)
        wat_atoms = u.select_atoms("{} or {}".format(self.ag1, self.ag2))
        n_oxys = len(oxy_atoms)

        hbonds = HBA(universe=u,
                     donors_sel=None,
                     hydrogens_sel=self.ag1,
                     acceptors_sel=self.ag2,
                     d_a_cutoff=3.3,
                     d_h_a_angle_cutoff=113.58,
                     update_selections=True)

        hbonds.run(start=starting_frame, stop=ending_frame)
        starting_frame_index = hbonds.results.hbonds[0][0]

        cleaned_data = {}

        for sublist in hbonds.results.hbonds:
            index = sublist[0]
            if index not in cleaned_data:
                cleaned_data[index] = []
            cleaned_data[index].append([sublist[1], sublist[3]])
        transformed_data = [value for _, value in sorted(cleaned_data.items())]
        del cleaned_data

        p_cluster_analysis = Clustering(len(u.atoms))

        z = float(u.dimensions[2])
        bin_size = 1
        num_bins = ceil(z / bin_size)
        z_edges = MDAFunctions._truncate(2, np.linspace(0, z, (num_bins + 1)))
        z_centers = z_edges[:-1] + 0.5

        donor_counts = np.full(z_centers.size, fill_value=0.0)
        acceptor_counts = np.full(z_centers.size, fill_value=0.0)
        oxy_counts = np.full(z_centers.size, fill_value=0.0)

        cluster_edges = np.linspace(0, n_oxys, (n_oxys + 1))
        cluster_centers = cluster_edges[:-1]
        p_cluster_counts = np.full(cluster_centers.size, fill_value=0.0)
        max_moments = np.full(cluster_centers.size, fill_value=0.0)
        mid_moments = np.full(cluster_centers.size, fill_value=0.0)
        min_moments = np.full(cluster_centers.size, fill_value=0.0)
        p_single_cluster_counter = 0

        for frame_index, frame in enumerate(transformed_data):
            current_frame = int(frame_index + starting_frame_index)
            u.trajectory[current_frame]
            atoms_not_in_cluster = len(u.atoms) - len(oxy_atoms.select_atoms("prop z < {}".format(self.hbond_cutoff)))
            p_single_cluster_counter += atoms_not_in_cluster

            for oxy_atom in oxy_atoms:
                oxy_zpos = oxy_atom.position[2]
                oxy_his, *_ = np.histogram(oxy_zpos, bins=z_edges)
                oxy_counts += oxy_his

            for hbond_pair in frame:
                donor_ix = int(hbond_pair[0])
                acceptor_ix = int(hbond_pair[1])

                donor_zpos = MDAFunctions._find_coord(u,
                                                      "z",
                                                      donor_ix)

                donor_counts = MDAFunctions._adjust_hist_counts(donor_counts,
                                                                z_edges,
                                                                donor_zpos)

                acceptor_zpos = MDAFunctions._find_coord(u,
                                                         "z",
                                                         acceptor_ix)

                acceptor_counts = MDAFunctions._adjust_hist_counts(acceptor_counts,
                                                                   z_edges,
                                                                   acceptor_zpos)

                if donor_zpos < self.hbond_cutoff:
                    p_cluster_analysis.merge(donor_ix, acceptor_ix)

            p_cluster_analysis.rebuild
            p_clusters = p_cluster_analysis.get_all_clusters()

            for p_cluster in p_clusters:
                p_cluster_size = len(p_cluster)
                p_cluster_counts = MDAFunctions._adjust_hist_counts(p_cluster_counts,
                                                                    cluster_edges,
                                                                    p_cluster_size)
                oxy_selection = ' or '.join([f"index {i}" for i in p_cluster])
                oxy_only_inertia_atomgroup = oxy_atoms.select_atoms("prop z < {} and {}".format(self.hbond_cutoff, oxy_selection))
                if oxy_only_inertia_atomgroup:
                    water_res_ids = oxy_only_inertia_atomgroup.atoms.resids
                    water_selection = ' or '.join([f"resid {resid}" for resid in water_res_ids])
                    inertia_atomgroup = wat_atoms.select_atoms("{}".format(water_selection))
                    inertia_tensor = inertia_atomgroup.moment_of_inertia()
                    eigen_val, _ = np.linalg.eig(inertia_tensor)
                    eigen_val = np.sort(eigen_val)[::-1]
                    print("This is the principal axes {}".format(eigen_val))
                    max_moments[p_cluster_size] += eigen_val[0]
                    mid_moments[p_cluster_size] += eigen_val[1]
                    min_moments[p_cluster_size] += eigen_val[2]

            p_cluster_counts[1] -= p_single_cluster_counter
            p_single_cluster_counter = 0
            p_cluster_analysis.reset()

        self._very_verbose_print(oxy_counts)
        total_frames = ending_frame - starting_frame
        donor_counts /= hbonds.n_frames
        acceptor_counts /= hbonds.n_frames
        oxy_counts /= total_frames
        donor_hbonds = donor_counts / oxy_counts
        acceptor_hbonds = acceptor_counts / oxy_counts
        max_moments /= p_cluster_counts
        mid_moments /= p_cluster_counts
        min_moments /= p_cluster_counts
        p_cluster_counts /= total_frames
        water_in_cluster_prob = p_cluster_counts * cluster_centers
        total_water_in_clusters = np.sum(water_in_cluster_prob)
        water_in_cluster_prob /= total_water_in_clusters
        total_clusters = np.sum(p_cluster_counts)
        cluster_size_prob = (p_cluster_counts / total_clusters)

        p_donors = np.nan_to_num(donor_hbonds[:self.hbond_cutoff])
        b_donors = np.nan_to_num(donor_hbonds[self.hbond_cutoff:])

        p_acceptors = np.nan_to_num(acceptor_hbonds[:self.hbond_cutoff])
        b_acceptors = np.nan_to_num(acceptor_hbonds[self.hbond_cutoff:])

        p_donor_average = np.average(p_donors)
        b_donor_average = np.average(b_donors)

        p_acceptor_average = np.average(p_acceptors)
        b_acceptor_average = np.average(b_acceptors)

        averaged_array = np.asarray([p_donor_average, b_donor_average, p_acceptor_average, b_acceptor_average])
        np.savetxt("{}_averaged_pbHbond_data_{}-{}".format(self.outfile_name, starting_frame, ending_frame), averaged_array)

        data_labels = self._data_label_maker("Cluster Number",
                                             "Polymer Clusters",
                                             "Water in Cluster Probability",
                                             "Cluster Size Probability",
                                             "max_inertia",
                                             "mid_inertia",
                                             "min_inertia")
        condensed_data = self._data_condenser(data_labels,
                                              cluster_centers,
                                              np.nan_to_num(p_cluster_counts),
                                              water_in_cluster_prob,
                                              cluster_size_prob,
                                              np.nan_to_num(max_moments),
                                              np.nan_to_num(mid_moments),
                                              np.nan_to_num(min_moments))

        return condensed_data

    def tetra_order(self, u, starting_frame, ending_frame):
        ow_selection = u.select_atoms(self.ag1, updating=True)
        box_dimension = u.dimensions
        n_closest_neighbors = 4
        q_edges = np.linspace(-3, 1, 401)
        q_centers = q_edges[:-1]
        q_counts = np.full(q_centers.size, fill_value=0.0)
        for ts in u.trajectory[starting_frame:ending_frame]:
            distance_array = distances.distance_array(ow_selection,
                                                      ow_selection,
                                                      box=box_dimension)

            distance_array = np.where(distance_array == 0.0, np.inf, distance_array)
            for central_atom_index, distance_list in enumerate(distance_array):
                neighbor_indices = np.argpartition(distance_list,
                                                   n_closest_neighbors)[1:n_closest_neighbors+1]
                central_atom_coord = ow_selection[central_atom_index].position
                double_sum = 0
                for j in range(0, 4):
                    j_atom_coord = ow_selection[neighbor_indices[j]].position
                    for k in range(j+1, 4):
                        k_atom_coord = ow_selection[neighbor_indices[k]].position
                        angle = calc_angles(j_atom_coord,
                                            central_atom_coord,
                                            k_atom_coord,
                                            box=box_dimension)
                        double_sum += (np.cos(angle) + 1/3) ** 2
                q = 1 - (3/8) * double_sum
                MDAFunctions._adjust_hist_counts(q_counts, q_edges, q)

        q_sum = np.sum(q_counts)
        q_prob = q_counts/q_sum

        data_labels = self._data_label_maker('q_value',
                                             'Counts',
                                             'Probability')

        condensed_data = self._data_condenser(data_labels,
                                              q_centers,
                                              q_counts,
                                              q_prob)

        return condensed_data

    @staticmethod
    def _get_atom_coords(selection, atom_index):
        return selection.atoms[int(atom_index)].position

    @staticmethod
    def _adjust_hist_counts(counts_array, edges, input_data):
        hist, *_ = np.histogram(input_data, bins=edges)
        counts_array += hist

        return counts_array

    @staticmethod
    def _find_coord(u, axis, atom_index):
        axis_to_int = {
            "x": 0,
            "y": 1,
            "z": 2
        }
        dim = axis_to_int[axis]
        atom = u.atoms[atom_index]
        coord = atom.position[dim]

        return coord

    # Calculates radius of gyration. Only uses ag1
    def radius_of_gyration(self, u, starting_frame, ending_frame):
        self.time_dependent = True
        self._verbose_print("Starting ROG Analysis from frame {} to frame {}".format(starting_frame, ending_frame))
        atom_group_1 = u.select_atoms(self.ag1)

        unique_resids = self._extract_unique_resids(atom_group_1)
        individual_rogs = []
        for unique_resid in unique_resids:
            individual_rog = []
            refined_atom_group = u.select_atoms("{} and resid {}".format(self.ag1, unique_resid))
            for ts in u.trajectory[starting_frame:(ending_frame)]:
                individual_rog.append(refined_atom_group.radius_of_gyration())
            individual_rogs.append(individual_rog)
        average_rog = self._averager(individual_rogs)
        rog_data = np.vstack((average_rog, individual_rogs))
        rog_data = MDAFunctions._matrix_to_many_row_vectors(rog_data)
        frame_number = MDAFunctions._get_frame_numbers(starting_frame, ending_frame)

        data_labels = self._data_label_maker("Frame Number", "Average", unique_resids)
        condensed_data = self._data_condenser(data_labels, frame_number, rog_data)

        return condensed_data

    # Calculates end-to-end distance. NOTE: terminal atoms must be unique or else it doesn't work
    def end_to_end(self, u, starting_frame, ending_frame):
        self.time_dependent = True
        self._verbose_print("Starting end-to-end Analysis from frame {} to frame {}".format(starting_frame, ending_frame))
        atom_group_1 = u.select_atoms(self.ag1)

        unique_resids = self._extract_unique_resids(atom_group_1)
        individual_e2es = []
        for unique_resid in unique_resids:
            individual_e2e = []
            refined_atom_group_1 = u.select_atoms("{} and resid {}".format(self.ag1, unique_resid))
            refined_atom_group_2 = u.select_atoms("{} and resid {}".format(self.ag2, unique_resid))
            for ts in u.trajectory[starting_frame:ending_frame]:
                output_array = distances.dist(refined_atom_group_1, refined_atom_group_2)
                distance = output_array[2]
                individual_e2e.append(distance)
            individual_e2es.append(individual_e2e)

        average_e2e = self._averager(individual_e2es)
        print(individual_e2es)
        e2e_data = np.vstack((average_e2e, individual_e2es))
        e2e_data = MDAFunctions._matrix_to_many_row_vectors(e2e_data)
        frame_number = np.arange(start=starting_frame, stop=ending_frame, step=1).astype(float)

        data_labels = self._data_label_maker("Frame Number", "Average", unique_resids)
        condensed_data = self._data_condenser(data_labels, frame_number, e2e_data)

        return condensed_data

    def density_xy(self, u, starting_frame, ending_frame):
        self._verbose_print("Starting 2-D averaging from frame {} to frame {}".format(starting_frame, ending_frame))
        solvent_group = u.select_atoms(self.ag1)
        solvated_group = u.select_atoms(self.ag2)
        u = self._preprocess_center_solvated_group(u, solvent_group, solvated_group)

        dens = density.DensityAnalysis(solvent_group, delta=1.0, padding=1)
        dens.run(start=starting_frame, stop=ending_frame)
        grid = dens.results.density.grid

        avg = grid.mean(axis=-1)
        fig, ax = plt.subplots()
        im = ax.imshow(avg)
        cbar = plt.colorbar(im)
        cbar.set_label('Density')
        plt.xlabel('X-axis (A)')
        plt.ylabel('Y-axis')
        plt.savefig('Test')

    def _preprocess_center_solvated_group(self, u, solvent_group, solvated_group):
        traj_transformations = [trans.unwrap(u.atoms),
                                trans.center_in_box(solvated_group, center="geometry"),
                                trans.wrap(solvent_group, compound='residues'),
                                trans.fit_rot_trans(solvated_group, solvated_group, weights='mass')]

        u.trajectory.add_transformations(*traj_transformations)
        return u

    # Calculates ACF based on a time dependent quantity
    def autocorrelation(self):
        self._verbose_print("Starting ACF Analysis")
        full_data_file = "{}_Full_Data".format(self.outfile_name)
        self._data_combiner(full_data_file)
        full_data = np.genfromtxt(full_data_file, names=True, delimiter='\t')
        x_label = full_data.dtype.names[0]
        y_labels = full_data.dtype.names[1:]
        acf_labels = []
        acfs = []
        lags = np.arange(0, len(full_data[x_label]))

        for y_label in y_labels:
            data = full_data[y_label]
            centered_data = data - np.average(data)
            self._very_verbose_print("Centered Data: {}".format(centered_data))

            acf = correlate(centered_data, centered_data, mode='full')
            acf /= np.max(np.abs(acf))
            acf = MDAFunctions._trim_list(acf)
            self._very_verbose_print("ACF results: {}".format(acf))
            acfs.append(acf)
            acf_labels.append("ACF_{}".format(y_label))

        data_labels = self._data_label_maker("Lag", acf_labels)
        condensed_data = self._data_condenser(data_labels, lags, acfs)

        return condensed_data

    @staticmethod
    def _trim_list(long_list):
        last_half = len(long_list) // 2
        long_list = long_list[last_half:]
        return long_list

    ### Helper Functions sections ###

    ### Functions that modify Universe parameters
    def _mda_universe_generator(self, traj_file):
        u = Universe(self.topology,
                     traj_file,
                     topology_format=self.topo_file_type,
                     format=self.traj_file_type,
                     dt=self.dt)
        if self.add_names:
            atom_names, _ = self._get_info_from_pdb()
            u = self._add_names_to_universe(u, atom_names)
        return u

    def _add_names_to_universe(self, u, atom_names):
        self._very_verbose_print("atom names are as follows:{}".format(atom_names))
        u.add_TopologyAttr('name', atom_names)
        self._verbose_print("Names added from PBD file!")
        return u

    def _get_info_from_pdb(self):
        if self.pdb is None:
            raise FileNotFoundError("PDB is required to add names")
        self._verbose_print("Getting info from pdb")
        atom_names = []
        res_id = []
        with open(self.pdb) as f:
            pdb_lines = f.readlines()
        stripped_pdb_lines = [line for line in pdb_lines if line.startswith('ATOM') or line.startswith('HETATM')]
        self._very_verbose_print("info from PBD: {}".format(stripped_pdb_lines))
        for stripped_line in stripped_pdb_lines:
            atom_names.append(stripped_line[12:16].strip())
            res_id.append(int(stripped_line[22:26].strip()))
        return atom_names, res_id

    def _extract_unique_resids(self, atom_group_1):
        unique_resids = sorted(set(atom_group_1.resids))
        self._very_verbose_print("Unique RESIDs: {}".format(unique_resids))
        return unique_resids

    @staticmethod
    def _get_frame_numbers(starting_frame, ending_frame):
        frame_list = np.arange(start=(starting_frame+1), stop=(ending_frame+1), step=1).astype(float)
        return frame_list

    def _traj_segmenter(self):
        if self.end_on_frame:
            self._verbose_print("Splicing frames from {} to {} into {} segments".format(self.start_from_frame, self.end_on_frame, self.segment))
            total_frames = self.end_on_frame - self.start_from_frame
            delta_step = total_frames / self.segment
        else:
            self._verbose_print("Splicing last {} frames into {} segments".format(self.start_from_frame, self.segment))
            delta_step = self.start_from_frame / self.segment
        i = 0
        segment_step_list = []
        while i < self.segment:
            starting_frame = floor(self.start_from_frame - i * delta_step - 1)
            ending_frame = ceil(self.start_from_frame - (i+1) * delta_step - 1)
            segment_step = [starting_frame, ending_frame]
            segment_step_list.append(segment_step)
            i += 1
        self._verbose_print("Segments analysized are the following: {}".format(segment_step_list))
        return segment_step_list

    def _limit_finder(self, ag1, ag2=None):
        if "prop" in ag1:
            z_cutoff = MDAFunctions._extract_limit_from_string(ag1)
        elif ag2 is not None and 'prop' in ag2:
            z_cutoff = MDAFunctions._extract_limit_from_string(ag2)
        else:
            raise ValueError("keyword 'prop' expected in ag1 or ag2 but was not found")
        self._verbose_print("z_cutoff(s) found to be {}".format(z_cutoff))
        return z_cutoff

    @staticmethod
    def _extract_limit_from_string(input_string):
        pattern = r'prop\s\w\s([><]=?)\s(\d+(?:\.\d+)?)'
        matches = re.findall(pattern, input_string)
        extracted_values = []
        if matches:
            for operator, value in matches:
                extracted_values.append(float(value))

        return extracted_values if extracted_values else None

    @staticmethod
    def _get_frame_limits(segment):
        starting_frame = segment[0]
        ending_frame = segment[1]
        return starting_frame, ending_frame

    def _make_whole(self, u, ag1, ag2=None):
        if self.transformed:
            return u

        if ag2 is not None:
            full_ag = "{} or {}".format(ag1, ag2)
        else:
            full_ag = ag1
        ag = u.select_atoms(full_ag)

        transformations = [trans.unwrap(ag),
                           trans.wrap(ag, compound='residues')]
        u.trajectory.add_transformations(*transformations)

        self.transformed = True
        return u

    ### File associated helper functions ###

    def _xtc_converter(self, traj_files):
        xtc_traj_file = []
        for traj_file in traj_files:
            xtc_traj_file.append(traj_file)
            u = self._mda_universe_generator(traj_file)
            with Writer("{}.xtc".format(traj_file), u.atoms.n_atoms) as f:
                for ts in u.trajectory:
                    f.write(u)
        return xtc_traj_file

    def output_file_name_maker(self, seg):
        output_file_name = "{}_{}".format(self.outfile_name, seg)
        return output_file_name

    def _file_mover(self):
        current_dir = os.getcwd()
        destination_path = self._dir_maker(current_dir)
        data_files = os.listdir(current_dir)

        for data_file in data_files:
            if self.outfile_name in data_file and data_file != destination_path:
                source_file_path = os.path.join(current_dir, data_file)
                destination_file_path = os.path.join(destination_path, data_file)
                if os.path.exists(destination_file_path):
                    os.remove(destination_file_path)
                shutil.move(source_file_path, destination_path)

    def _data_combiner(self, out_file):
        data_path, file_list = self._get_file_list()
        data_list = []
        header_list = None
        for i, filename in enumerate(file_list):
            file_path = os.path.join(data_path, filename)
            self._verbose_print("file to open: {}".format(file_path))
            if header_list is None:
                header_list = np.genfromtxt(file_path, delimiter='\t', max_rows=1, dtype=str).T
                self._very_verbose_print("Headers: {}".format(header_list))
            data = np.loadtxt(file_path, skiprows=1, delimiter='\t')
            data_list.append(data)
        concat_data = np.concatenate(data_list, axis=0)
        concat_data = np.vstack((header_list, concat_data))
        self._save_data(out_file, concat_data)

    def _get_file_list(self):
        current_dir = os.getcwd()
        data_path = os.path.join(current_dir, self.outfile_name)
        file_list = os.listdir(data_path)
        bad_words = ["ACF", "graph", "Full"]
        file_list = [data_file for data_file in file_list
                     if data_file.startswith(self.outfile_name)
                     and all(bad_word not in data_file for bad_word in bad_words)]
        self._very_verbose_print("Data files to combine: {}".format(file_list))
        file_list.sort()
        return data_path, file_list

    def _dir_maker(self, current_dir):
        output_folder = self.outfile_name
        output_folder_path = os.path.join(current_dir, output_folder)
        os.makedirs(output_folder_path, exist_ok=True)
        return output_folder_path

    ### Data management associated functions ###
    @staticmethod
    def _save_data(output_file, data):
        np.savetxt(output_file, data, fmt="%s", delimiter='\t')

    def _data_condenser(self, data_labels, *data):
        flattened_data = [data_value if type(data_value) is np.ndarray else sub_data for data_value in data for sub_data in (data_value if isinstance(data_value, list) else [data_value])]
        self._very_verbose_print("Flattened_data:{}".format(flattened_data))
        condensed_data = np.asarray(flattened_data)
        condensed_data = np.hstack((data_labels, flattened_data))
        self._very_verbose_print("Labelled data:{}".format(condensed_data))
        return condensed_data

    def _data_label_maker(self, *labels):
        self._verbose_print(labels)
        flattened_labels = [label if type(label) is str else str(inner_label) for label in labels for inner_label in (label if type(label) is list else [label])]
        # flattened_labels = [item for sublist in flattened_labels for item in sublist]
        self._verbose_print("Flattened Array: {}".format(flattened_labels))
        label_array = np.asanyarray(flattened_labels)
        label_array = label_array[:, np.newaxis]
        self._verbose_print("Data Labels Generated!: {}".format(label_array))
        return label_array

    @staticmethod
    def _truncate(kept_decimals, array):
        array = np.floor(array * 10**kept_decimals) / 10**kept_decimals
        return array

    @staticmethod
    def _matrix_to_many_row_vectors(array):
        row_vectors = [np.array(row) for row in array]
        return row_vectors

    def _averager(self, data):
        average_data = np.average(data, axis=0)
        self._very_verbose_print("Averaged data:{}".format(average_data))
        return average_data

    def _plot_array(self, data_file):
        self._verbose_print("I'm Graphin here")
        data = np.genfromtxt(data_file, delimiter='\t', names=True)
        x_label = data.dtype.names[0]
        y_labels = data.dtype.names[1:]
        self._very_verbose_print("All labels to be graphed {}".format(y_labels))
        for y_label in y_labels:
            self._very_verbose_print("graphing {}".format(y_label))
            output_file_name = "{}_graph_{}_v_{}".format(data_file, y_label, x_label)
            MDAFunctions.grapher(data[x_label], data[y_label], x_label, y_label, output_file_name)

    @staticmethod
    def grapher(x_data, y_data, x_label, y_label, output_file_name):
        plt.figure(num=y_label)
        plt.plot(x_data, y_data)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.savefig(output_file_name)
        plt.clf()

    ### RDF associated helper functions ###
    @staticmethod
    def _volume_adjustment(z_frac, data):
        data *= z_frac
        return data

    def _volume_fraction(self, z_cutoff, total_z_length, ag1, ag2):
        self._verbose_print("fixing void fraction")
        if len(z_cutoff) == 1:
            z_total = z_cutoff[0]
        else:
            z_total = max(z_cutoff) - min(z_cutoff)

        if z_total >= total_z_length:
            raise ValueError("z cutoff(s) are unphysical. z cutoff ({}) must be <= z box ({}),"
                  " please change bounds ".format(z_total, total_z_length))
        elif '>' in ag1 and '<' in ag1 or (ag2 is not None and '>' in ag2 and '<' in ag2):
            z_frac = (max(z_cutoff) - min(z_cutoff)) / total_z_length
        elif '>' in ag1 or (ag2 is not None and '>' in ag2):
            z_frac = (total_z_length - z_cutoff[0]) / total_z_length
        elif '<' in ag1 or (ag2 is not None and '<' in ag2):
            z_frac = z_cutoff[0] / total_z_length
        self._verbose_print("Fraction considered:{}".format(z_frac))
        return z_frac

    @staticmethod
    def _cum_num(rdf_count, natoms):
        cumulative_rdf = np.cumsum(rdf_count)
        number_density = cumulative_rdf / natoms
        return cumulative_rdf, number_density

    ### H-bonding associated helper functions ###
    def _count_oxygen_in_selection(self, u, start, finish):
        O_atom_count = []
        for ts in u.trajectory[start:finish]:
            O_atom_count.append(len(u.select_atoms(self.ag2, updating=True)))
        O_atom_count = np.asarray(O_atom_count).astype(int)
        return O_atom_count

    ### Misc helper functions ###

    # Converts input string into a list
    @staticmethod
    def _str_to_list(input_str):
        return input_str.split()

    ### Debug functions ###
    def _very_verbose_print(self, message):
        if self.very_verbose:
            print(message)

    def _verbose_print(self, message):
        if self.verbose or self.very_verbose:
            print(message)

    def _get_time(self):
        if self.time:
            time_now = time.perf_counter()
        else:
            time_now = None
        return time_now

    def _time_since(self, start, message):
        if self.time:
            end = time.perf_counter()
            total_time = end - start
            print("{}:{} s".format(message, total_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Runs various MD analysis scripts for analysizing MD systems")
    parser.add_argument('--topology', '-topo', type=str, help='Topology input file, for LAMMPS foo.data', default=None)
    parser.add_argument('--trajectory', '-traj', type=str, help='Trajectory input file, for LAMMPS foo.lammpsdump', default=None)
    parser.add_argument('--atomgroup1', '-ag1', type=str, help='Atom group 1 for MD Analysis', default=None)
    parser.add_argument('--atomgroup2', '-ag2', type=str, help='Atom group 2 for MD analysis (used with hbond and e2e. optional with rdf)', default=None)
    parser.add_argument('--start_from_frame', '-sf', type=int, help='Start analysis from frame X. should be a negative Int if you want to start last X frames', default=-1)
    parser.add_argument('--end_on_frame', '-ef', type=int, help='Ends analysis on frame X.', default=None)
    parser.add_argument('--segment', '-seg', type=int, help='If doing block averaging, splits the run into X many groups', default=1)
    parser.add_argument('--topo_file_type', '-toft', type=str, help='Specify the topology file type', default=None)
    parser.add_argument('--traj_file_type', '-trft', type=str, help='Specify the trajectory file type', default=None)
    parser.add_argument('--pdb_file', '-pdb', type=str, help='Specify PDB file. Used to extract atomnames when using lammpsdump files', default=None)
    parser.add_argument('--outfile_name', '-o', type=str, help='Output file name', default=None)
    parser.add_argument('--function', '-f', type=str, help='Chooses which analysis function to do', choices=['rdf', 'e2e', 'hbond', 'rog', 'dens', 'oto'], default=None)
    parser.add_argument('--rdf_exclusion', '-exc', type=lambda x: ast.literal_eval(x), help='Exclusion block used for RDF function written as tuple written "(X,Y)"', default=None)
    parser.add_argument('--timestep', '-dt', type=float, help='time step in ps (1fs = 0.001ps)', default=None)
    parser.add_argument('--hbond_cutoff', '-hcut', type=int, help='Cutoff value for HBond Analysis', default=None)

    parser.add_argument('--convertXTC', '-xtc', action='store_true', help="Converts trajectory files into XTC file format before running simulations", default=False)
    parser.add_argument('--addnames', '-add', action='store_true', help="Adds atom names to the topology file from a given pdb", default=False)

    parser.add_argument('--verbose', '-v', action='store_true', help='Helpful flag for debuging code', default=False)
    parser.add_argument('--very_verbose', '-vv', action='store_true', help='displays even more information', default=False)
    parser.add_argument('--time', '-t', action='store_true', help='times length of functions', default=False)

    args = parser.parse_args()

    clf = MDAFunctions(topology=args.topology,
                       trajectory=args.trajectory,
                       atomgroup1=args.atomgroup1,
                       atomgroup2=args.atomgroup2,
                       start_from_frame=args.start_from_frame,
                       end_on_frame=args.end_on_frame,
                       segment=args.segment,
                       topo_file_type=args.topo_file_type,
                       traj_file_type=args.traj_file_type,
                       pdb_file=args.pdb_file,
                       outfile_name=args.outfile_name,
                       function=args.function,
                       rdf_exclusion=args.rdf_exclusion,
                       timestep=args.timestep,
                       verbose=args.verbose,
                       very_verbose=args.very_verbose,
                       time=args.time,
                       xtc_convert=args.convertXTC,
                       add_names=args.addnames,
                       hbond_cutoff=args.hbond_cutoff)

    clf.run_analysis()
