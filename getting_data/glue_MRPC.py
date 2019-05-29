from commons import data_io


def get_MRPC_data(file):
    def parse_line(line):
        label, id1, id2, texta, textb = line.split('\t')
        return {
            'text': texta,
            'textb': textb,
            'labels': label}

    lines_g = data_io.read_lines(file)
    next(lines_g)
    data = [parse_line(line) for line in lines_g]
    return data


