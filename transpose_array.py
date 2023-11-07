from sys import argv


def transpose(input_file, output_file):
    output_format = ('%s %s \n')

    with open(input_file, 'r') as f:
        lines = f.readlines()
    x_values = lines[0]
    y_values = lines[1]
    x_values = x_values.split(',')
    y_values = y_values.split(',')


    with open(output_file, 'w') as ff:
        for count, value in enumerate(x_values):
            ff.write(output_format % (x_values[count].strip(), y_values[count].strip()))


if __name__ == '__main__':
    transpose(argv[1],argv[2])
