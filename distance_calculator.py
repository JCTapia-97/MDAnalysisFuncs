import numpy as np
from sys import  argv

def distance_calculator(input_file):
    with open(input_file, 'r') as f:
        lines = f.readlines()
    lines = [line.split() for line in lines]

    distace_matrix = []
    for i in range(0, len(lines), 3):
        print(i)
        index = [i, i+1, i+2]
        for j in range(0, 3):
            if lines[i+j][2] == '23':
                C_atom_xyz = np.array(list(map(float,lines[i+j][3:])))
                index.pop(j)
            else:
                continue
        O1_atom_xyz = np.array(list(map(float, lines[index[0]][3:])))
        O2_atom_xyz = np.array(list(map(float, lines[index[1]][3:])))

        print(C_atom_xyz, O1_atom_xyz, O2_atom_xyz)

        distace_matrix.append(np.linalg.norm((C_atom_xyz - O1_atom_xyz)))
        distace_matrix.append(np.linalg.norm((C_atom_xyz - O2_atom_xyz)))

        index = []
    average = np.average(distace_matrix)
    stdev = np.std(distace_matrix)

    print(average, stdev)

if __name__ == '__main__':
    distance_calculator(argv[1])
