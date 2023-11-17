import matplotlib.pyplot as plt
import numpy as np
import argparse, re, time, os, shutil, ast
from scipy.signal import correlate
from MDAnalysis import Universe
from MDAnalysis.analysis import rdf
from MDAnalysis.analysis import lineardensity as lin
from MDAnalysis.analysis.hydrogenbonds.hbond_analysis import HydrogenBondAnalysis as HBA
from MDAnalysis.analysis import distances
from MDAnalysis import Writer
from math import floor, ceil


class MDAFunctions:
    def __init__(self,
                 topology=None,
                 trajectory=None,
                 pdb_file=None,
                 atomgroup1=None,
                 atomgroup2=None,
                 start_from_frame=None,
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
                 add_names=False):

        self.topology = topology
        self.trajectory = trajectory
        self.pdb = pdb_file
        self.ag1 = atomgroup1
        self.ag2 = atomgroup2
        self.start_from_frame = start_from_frame
        self.segment = segment
        self.topo_file_type = topo_file_type
        self.traj_file_type = traj_file_type
        self.outfile_name = outfile_name
        self.analysis_type = function
        self.exclusion = rdf_exclusion
        self.dt = timestep

        self.xtc_convert = xtc_convert
        self.add_names = add_names

        self.verbose = verbose
        self.very_verbose = very_verbose
        self.time = time

        self.time_dependent = False

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

    # Generates universe based on topology and trajectory. If lammpsdump file is used a PDB must be supplied for names
    def get_universe(self):
        traj_list = MDAFunctions._str_to_list(self.trajectory)
        self._verbose_print("Trajectory list is {}".format(traj_list))
        if self.xtc_convert:
            traj_list = self._xtc_converter(traj_list)
        u = self._mda_universe_generator(traj_list)
        self._verbose_print("Universe has been created!")
        # adds names based on PDB if lammpsdump file
        self._verbose_print("Total number of frames in Universe: {}".format(u.trajectory.n_frames))
        return u

    # Runs analysis and generated output files
    def run_analysis(self):
        # considering adding implementation to run multiple jobs sequencially, but may not be worth it
        start_time = self._get_time()
        u = self.get_universe()
        self._time_since(start_time, "Universe creation time")

        analysis_functions = {
            'dens': self.density_as_func_z,
            'rdf': self.average_rdf,
            'hbond': self.hydrogen_bonding,
            'rog': self.radius_of_gyration,
            'e2e': self.end_to_end
        }
        analysis_function = analysis_functions.get(self.analysis_type)

        if analysis_function is None:
            raise ValueError("Analysis function are limited to dens, rdf, hbond, rog, e2e")

        segment_list = self._traj_segmenter()
        analysis_start_time = self._get_time()
        for seg_index, segment in enumerate(segment_list):
            analysis_data = analysis_function(u, segment)
            output_file_name = self.output_file_name_maker(seg_index)
            MDAFunctions._save_data(output_file_name, analysis_data.T)
            self._very_verbose_print("Data saved for segment {}".format(segment))
        self._file_mover()
        self._time_since(analysis_start_time, "Analysis time")

        acf_start_time = self._get_time()
        if self.time_dependent:
            acf_data = self.autocorrelation()
            acf_out_file_name = self.output_file_name_maker("ACF")
            MDAFunctions._save_data(acf_out_file_name, acf_data.T)
            self._very_verbose_print("Data saved for ACF")
        self._file_mover()
        self._time_since(acf_start_time, "ACF time")

        self._time_since(start_time, "Total script time")

    ### Analysis function sections 
    # Calculates density across z dimension. Only requires ag1
    def density_as_func_z(self, u, segment):
        starting_frame, ending_frame = MDAFunctions._get_frame_limits(segment)
        self._verbose_print("Starting Density Analysis from frame {} to frame {}".format(starting_frame, ending_frame))

        z = float(u.dimensions[2])
        bin_size = 1
        num_bins = np.floor(z/bin_size).astype(int)
        z_span = MDAFunctions._truncate(2, np.linspace(0, z, num_bins))

        atom_group_1 = u.select_atoms(self.ag1)

        density_analysis = lin.LinearDensity(atom_group_1, grouping='atoms', binsize=bin_size).run(start=starting_frame, stop=ending_frame)
        density_results = density_analysis.results['z']['mass_density']
        data_labels = self._data_label_maker("z-box length (A)", "Density (g/ml)")
        condensed_data = self._data_condenser(data_labels, z_span, density_results)

        return condensed_data

    # Calculates RDF requires only ag1 to be specified but ag2 can be specified as well
    def average_rdf(self, u, segment):
        starting_frame, ending_frame = MDAFunctions._get_frame_limits(segment)
        self._verbose_print("Starting RDF Analysis from frame {} to frame {}".format(starting_frame, ending_frame))
        
        atom_group_1 = u.select_atoms(self.ag1)
        if self.ag2 is None:
            atom_group_2 = u.select_atoms(self.ag1)
        else:
            atom_group_2 = u.select_atoms(self.ag2)

        z_frac = 1
        if "prop" in self.ag1 or (self.ag2 is not None and "prop" in self.ag2):
            z_box_length = float(u.dimensions[2])
            z_cutoff = MDAFunctions._limit_finder(self.ag1, self.ag2)
            z_frac = self._volume_fraction(z_cutoff, z_box_length, self.ag1, self.ag2)

        rdf_analysis = rdf.InterRDF(atom_group_1, atom_group_2, exclusion_block=self.exclusion)
        rdf_analysis.run(start=starting_frame, stop=ending_frame)

        rdf_bins = MDAFunctions._truncate(2, rdf_analysis.results.bins)
        crdf, num_dens = MDAFunctions._cum_num(rdf_analysis.results.count, len(atom_group_1))
        rdf_results = rdf_analysis.results.rdf

        num_dens = self._volume_adjustment(float(1/z_frac), num_dens)
        rdf_results = self._volume_adjustment(z_frac, rdf_results)

        data_labels = self._data_label_maker("r (A)", "g(x)", "G(x)", "Number Density")
        condensed_data = self._data_condenser(data_labels, rdf_bins, rdf_results, crdf, num_dens)

        return condensed_data

    # Calculates average number of H-bonds in the simulation. Requires ag1 be hydrogen and ag2 be oxygen
    def hydrogen_bonding(self, u, segment):
        self.time_dependent = True
        if "OW" in self.ag1 or "HW" in self.ag2:
            raise ValueError("For H-bond analysis, ag1 must be Hydrogen and ag2 must be Oxygen. Change atom selection or atom naming to remove O from ag1 and H from ag2")
        starting_frame, ending_frame = MDAFunctions._get_frame_limits(segment)
        self._verbose_print("Starting H-bonding Analysis from frame {} to frame {}".format(starting_frame, ending_frame))

        O_atom_count = self._count_oxygen_in_selection(u, starting_frame, ending_frame)

        hbond_analysis = HBA(universe=u, hydrogens_sel=self.ag1, acceptors_sel=self.ag2)
        hbond_analysis.run(start=starting_frame, stop=ending_frame)
        hbonds_per_frame = hbond_analysis.count_by_time()

        frame_number = MDAFunctions._get_frame_numbers(starting_frame, ending_frame)
        normalized_hbonds_per_frame = np.divide(hbonds_per_frame, O_atom_count)
        data_labels = self._data_label_maker("frame number", "H-bonds in frame", "Normalized H-bonds in frame")
        condensed_data = self._data_condenser(data_labels, frame_number, hbonds_per_frame, normalized_hbonds_per_frame)

        return condensed_data

    # Calculates radius of gyration. Only uses ag1
    def radius_of_gyration(self, u, segment):
        self.time_dependent = True
        starting_frame, ending_frame = MDAFunctions._get_frame_limits(segment)
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
    def end_to_end(self, u, segment):
        self.time_dependent = True
        self._verbose_print("Starting end-to-end Analysis from frame {} to frame {}".format(starting_frame, ending_frame))
        starting_frame, ending_frame = MDAFunctions._get_frame_limits(segment)
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
    
    # Calculates ACF based on a time dependent quantity
    def autocorrelation(self):
        self._verbose_print("Starting ACF Analysis")
        full_data = self._get_data()
        data = full_data[1][:]
        centered_data = data - np.average(data)
        self._very_verbose_print("Centered Data: {}".format(centered_data))

        acf = correlate(centered_data, centered_data, mode='full')

        acf /= np.max(np.abs(acf))

        lags = np.arange(-len(data)+1, len(data))
        self._very_verbose_print("ACF results: {} \n Lags: {}".format(acf, lags))

        data_labels = self._data_label_maker("Lag", "ACF")
        condensed_data = self._data_condenser(data_labels, lags, acf)

        return condensed_data

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

    @staticmethod
    def _limit_finder(ag1, ag2):
        if "prop" in ag1:
            z_cutoff = float(ag1.split()[-1])
        else:
            z_cutoff = float(ag2.split()[-1])
        return z_cutoff

    @staticmethod
    def _get_frame_limits(segment):
        starting_frame = segment[0]
        ending_frame = segment[1]
        return starting_frame, ending_frame

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

    def _get_data(self):
        data_path, file_list = self._get_file_list()
        data_list = []
        for _, filename in enumerate(file_list):
            file_path = os.path.join(data_path, filename)
            data = np.loadtxt(file_path, skiprows=1, delimiter='\t')
            data_list.append(data)
        concat_data = np.concatenate(data_list, axis=0)
        self._very_verbose_print("Total Data:{}".format(concat_data))
        return concat_data.T

    def _get_file_list(self):
        current_dir = os.getcwd()
        data_path = os.path.join(current_dir, self.outfile_name)
        file_list = os.listdir(data_path)
        file_list = [data_file for data_file in file_list if data_file.startswith(self.outfile_name) and "ACF" not in data_file]
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

    @staticmethod
    def plotter(x_data, y_data, x_label, y_label, output_file_name):
        plt.plot(x_data, y_data)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.savefig(output_file_name)

    ### RDF associated helper functions ###
    @staticmethod
    def _volume_adjustment(z_frac, data):
        data *= z_frac
        return data

    def _volume_fraction(self, z_cutoff, total_z_length, ag1, ag2):
        self._verbose_print("fixing void fraction")
        if z_cutoff >= total_z_length:
            raise ValueError("z cutoff is unphysical. z cutoff ({}) must be <= z box ({}),"
                  " using z_frac =2 ".format(z_cutoff, total_z_length))
        elif '>' in ag1 or '>' in ag2:
            z_frac = (total_z_length - z_cutoff) / total_z_length
        elif '<' in ag1 or '<' in ag2:
            z_frac = z_cutoff / total_z_length
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
            total_time = start - end
            print("{}:{} s".format(message, total_time))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Runs various MD analysis scripts for analysizing MD systems")
    parser.add_argument('--topology', '-topo', type=str, help='Topology input file, for LAMMPS foo.data', default=None)
    parser.add_argument('--trajectory', '-traj', type=str, help='Trajectory input file, for LAMMPS foo.lammpsdump', default=None)
    parser.add_argument('--atomgroup1', '-ag1', type=str, help='Atom group 1 for MD Analysis', default=None)
    parser.add_argument('--atomgroup2', '-ag2', type=str, help='Atom group 2 for MD analysis (used with hbond and e2e. optional with rdf)', default=None)
    parser.add_argument('--start_from_frame', '-sf', type=int, help='Start analysis from frame X. should be a negative Int if you want to start last X frames', default=-1)
    parser.add_argument('--segment', '-seg', type=int, help='If doing block averaging, splits the run into X many groups', default=1)
    parser.add_argument('--topo_file_type', '-toft', type=str, help='Specify the topology file type', default=None)
    parser.add_argument('--traj_file_type', '-trft', type=str, help='Specify the trajectory file type', default=None)
    parser.add_argument('--pdb_file', '-pdb', type=str, help='Specify PDB file. Used to extract atomnames when using lammpsdump files', default=None)
    parser.add_argument('--outfile_name', '-o', type=str, help='Output file name', default=None)
    parser.add_argument('--function', '-f', type=str, help='Chooses which analysis function to do', choices=['rdf', 'e2e', 'hbond', 'rog', 'dens'], default=None)
    parser.add_argument('--rdf_exclusion', '-exc', type=lambda x: ast.literal_eval(x), help='Exclusion block used for RDF function written as tuple written "(X,Y)"', default=None)
    parser.add_argument('--timestep', '-dt', type=float, help='time step in ps (1fs = 0.001ps)', default=None)

    parser.add_argument('--convertXTC', '-xtc', action='store_true', help="Converts trajectory files into XTC file format before running simulations", default=False)
    parser.add_argument('-addnames', '-add', action='store_true', help="Adds atom names to the topology file from a given pdb", default=False)

    parser.add_argument('--verbose', '-v', action='store_true', help='Helpful flag for debuging code', default=False)
    parser.add_argument('--very_verbose', '-vv', action='store_true', help='displays even more information', default=False)
    parser.add_argument('--time', '-t', action='store_true', help='times length of functions', default=False)

    args = parser.parse_args()

    clf = MDAFunctions(topology=args.topology,
                       trajectory=args.trajectory,
                       atomgroup1=args.atomgroup1,
                       atomgroup2=args.atomgroup2,
                       start_from_frame=args.start_from_frame,
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
                       xtc_convert=args.convertXTC)

    clf.run_analysis()
