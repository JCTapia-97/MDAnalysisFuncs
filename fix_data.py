from sys import argv


def normalize_data(data_file, total_frames):
    format = "%s %s \n"
    with open(data_file, "r") as f:
        lines = f.readlines()
    lines = [line.split() for line in lines]
    new_lines = []
    for line in lines:
        line[1] = float(line[1]) / float(total_frames)
        new_lines.append(line)
    data_file_name = data_file.strip('.txt')

    with open("{}_fixed.txt".format(data_file_name), "w+") as ff:
        for line in new_lines:
            ff.write(format % (line[0], line[1]))

if __name__ == "__main__":
    normalize_data(argv[1], argv[2])