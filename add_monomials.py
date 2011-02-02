import csv
import numpy

def get_cols(in_file):
    rows = []
    for line in csv.reader(in_file):
        rows.append(line)

    header = rows[0]
    rows = rows[1:]
    cols = zip(*rows)
    table = {}
    for i, (name, col) in enumerate(zip(header, cols)):
        if i > 0:
            col = [float(x) for x in col]
        table[name] = col
    return table, header

def xy_from_cols(cols, header):
    y = numpy.asarray(cols[header[-1]])
    n_row = len(y)
    n_vars = len(header) - 2
    x = numpy.zeros((n_vars, n_row), dtype = numpy.float)
    for i, name in enumerate(header[1:-1]):
        x[i, :] = numpy.asarray(cols[name])
    return x, y

def gen_multi_indices(dim, degree):
    if dim == 1:
        yield (degree, )
    else:
        for n in xrange(degree + 1):
            for tail in gen_multi_indices(dim - 1, degree - n):
                yield (n, ) + tail

def gen_all_multi_indices(dim, max_degree):
    for degree in xrange(1, max_degree + 1):
        for alpha in gen_multi_indices(dim, degree):
            yield alpha

def make_monomial(x, alpha):
    y = numpy.ones(numpy.shape(x[0]))
    for (x_i, alpha_i) in zip(x, alpha):
        if alpha_i > 0:
            y *= (x_i ** alpha_i)
    return y

def fancy_multi_index_name(alpha, names):
    return ' '.join('%s^%d' % (name, exponent) for (name, exponent) in zip(names, alpha) if exponent > 0)


def make_extended_cols(header, cols, monomials):
    d = {}
    for name in monomials:
        d[name] = monomials[name]
    d[header[0]] = cols[header[0]]
    d[header[-1]] = cols[header[-1]]
    d_header = [header[0]] + sorted(monomials.keys()) + [header[-1]]
    return d, d_header

def save_as_csv(cols, header, out_file):
    rows = zip(*[cols[name] for name in header])
    writer = csv.writer(out_file, quoting = csv.QUOTE_NONNUMERIC)
    writer.writerow(header)
    writer.writerows(rows)

def add_monomials(in_file, out_file, max_degree, display_progress):

    def cautiously_print(s):
        if display_progress:
            print(s)

    numpy.seterr(invalid = 'raise')

    cautiously_print('parsing input')
    cols, header = get_cols(in_file)
    x, y = xy_from_cols(cols, header)

    # how to normalise things? (THIS CHANGES STUFF A LOT!)
    # x = x - numpy.mean(x, axis = 1)[:, numpy.newaxis]
    x_var = numpy.var(x, axis = 1)
    mask = x_var > 0.0
    x = x[mask, :]
    x_var = x_var[mask]
    x_header = [name for name, flag in zip(header[1:-1], mask) if flag]
    x = x / x_var[:, numpy.newaxis]

    y = y - numpy.mean(y)
    y = y / numpy.linalg.norm(y)

    cautiously_print('generating polynomials of 0 < degree <= %d' % max_degree)

    all_indices = list(gen_all_multi_indices(x.shape[0], max_degree))
    n_indices = len(all_indices)

    monomials = {}
    pc_complete = 0
    for i, alpha in enumerate(all_indices):
        name = fancy_multi_index_name(alpha, x_header)
        monomials[name] = make_monomial(x, alpha)
        current_pc_complete = int(10.0 * i / n_indices) * 10
        if current_pc_complete > pc_complete:
            pc_complete = current_pc_complete
            cautiously_print('%d %% complete (%d of %d)' % (pc_complete, i, n_indices))

    cautiously_print('saving output')
    extended_cols, extended_header = make_extended_cols(header, cols, monomials)
    save_as_csv(extended_cols, extended_header, out_file)

def main():
    import sys
    import argparse

    # wow argparse. it's so bright, it's so clean, ....

    parser = argparse.ArgumentParser(
        description = 'add monomials! yeah!'
    )
    parser.add_argument(
        '--source',
        default = sys.stdin,
        type = argparse.FileType('r'),
        action = 'store',
        dest = 'source',
        help = 'input csv file',
    )
    parser.add_argument(
        '--dest',
        default = sys.stdout,
        type = argparse.FileType('w'),
        action = 'store',
        dest = 'dest',
        help = 'output csv file'
    )
    parser.add_argument(
        '--max_degree',
        default = 2,
        type = int,
        action = 'store',
        dest = 'max_degree',
        help = 'max degree of monomials to add',
    )
    parsed = parser.parse_args()

    add_monomials(
        parsed.source,
        parsed.dest,
        parsed.max_degree,
        display_progress = (parsed.dest != sys.stdout),
    )

if __name__ == '__main__':
    main()
