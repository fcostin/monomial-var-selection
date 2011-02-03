"""
late-night implementation of variable selection approach as described in

    MLPs (Mono-Layer Polynomials and Multi-Layer Perceptrons) for Nonlinear Modeling
    Isabelle Rivals, L\'eon Personnaz
    http://jmlr.csail.mit.edu/papers/volume3/rivals03a/rivals03a.pdf
"""

import csv
import numpy
import random

def get_cols(file_name):
    rows = []
    for line in csv.reader(open(file_name, 'r')):
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
    for degree in xrange(0, max_degree + 1):
        for alpha in gen_multi_indices(dim, degree):
            yield alpha

def make_monomial(x, alpha):
    y = numpy.ones(numpy.shape(x[0]))
    for (x_i, alpha_i) in zip(x, alpha):
        if alpha_i > 0:
            y *= (x_i ** alpha_i)
    return y

def cosine_similarity(x, y):
    """
    fancy name for <x, y> / ||x||||y||
    """
    # deal with divide-by-zero case vaguely reasonably
    if numpy.linalg.norm(x) * numpy.linalg.norm(y) == 0.0:
        return 0.0
    else:
        return numpy.dot(x, y) / (numpy.linalg.norm(x) * numpy.linalg.norm(y))

def fancy_multi_index_name(alpha, names):
    return '*'.join('%s^%d' % (name, exponent) for (name, exponent) in zip(names, alpha) if exponent > 0)

def extract_alpha_from_fancy_name(name, var_names):
    """
    blimey!
    """
    exponents = {}
    for token in set(name.split('*')):
        if token == '':
            continue
        var_name, exponent = token.split('^')
        exponent = int(exponent)
        assert var_name in var_names
        exponents[var_name] = exponent
    return tuple([exponents.get(var_name, 0) for var_name in var_names])

def make_rows_from_distance_matrix(distance, row_names, col_names):
    rows = []
    rows.append([''] + col_names)
    for name, row in zip(row_names, distance):
        rows.append([name] + list(row))
    return rows

def make_rows_from_vectors(vectors, row_names):
    rows = []
    n_cols = len(vectors[0])
    rows.append([''] * (1 + n_cols))
    for name, row in zip(row_names, vectors):
        rows.append([name] + list(row))
    return rows

def save_rows_as_csv(rows, out_file):
    writer = csv.writer(out_file, quoting = csv.QUOTE_NONNUMERIC)
    writer.writerows(rows)

def mangle_name(s):
    s = s.replace('wt.%', '')
    s = s.replace('ppm', '')
    s = s.replace('^', '')
    s = s.replace('*', '.')
    s = ''.join(c for c in s if (c.isalnum() or c == '.'))
    if s == '':
        s = '(intercept)'
    return s

def main():
    numpy.seterr(invalid = 'raise')

    file_name = 'tensile-strength-177c.csv' # real file name here

    cols, header = get_cols(file_name)
    x, y = xy_from_cols(cols, header)
    x_names = header[1:-1]

    test_fraction = 1.0 / 3.0
    test_size = int(len(y) * test_fraction)
    
    n_iters = 15

    ranks = {}
    
    for iter in xrange(n_iters):
        test_indices = random.sample(xrange(len(y)), test_size)
        test_mask = numpy.zeros(len(y), dtype = numpy.bool)
        test_mask[test_indices] = True
        training_mask = numpy.logical_not(test_mask)
        x_test = x[:, test_mask]
        y_test = y[test_mask]
        x_training = x[:, training_mask]
        y_training = y[training_mask]
        shortlist, x_header = shortlist_variables(
            x_training,
            y_training,
            header,
            max_degree = 3,
            shortlist_length = 25,
        )
        for i, alpha in enumerate(shortlist):
            fancy_name = fancy_multi_index_name(alpha, x_header)
            global_alpha = extract_alpha_from_fancy_name(fancy_name, x_names)
            if global_alpha not in ranks:
                ranks[global_alpha] = [i]
            else:
                ranks[global_alpha].append(i)

    shortlist = list(sorted(ranks.keys()))
    shortlist_length = len(shortlist)
    proximity = numpy.zeros((shortlist_length, ) * 2, dtype = numpy.float)
    vectors = []
    for i, alpha in enumerate(shortlist):
        m_alpha = make_monomial(x, alpha)
        vectors.append(m_alpha)
        for j, beta in enumerate(shortlist):
            m_beta = make_monomial(x, beta)
            proximity[i, j] = numpy.abs(cosine_similarity(m_alpha, m_beta))
    distance = 1.0 - proximity

    
    def check_uniqueness(a):
        if not len(set(a)) == len(a):
            for x in sorted(a):
                print str(x)
            raise ValueError('uniqueness failure')

    check_uniqueness(x_names)
    check_uniqueness(shortlist)
    shortlist_names = [fancy_multi_index_name(alpha, x_names) for alpha in shortlist]
    check_uniqueness(shortlist_names)
    shortlist_names = [mangle_name(name) for name in shortlist_names]
    check_uniqueness(shortlist_names)

    for name in sorted(shortlist_names):
        print name
    
    save_rows_as_csv(
        make_rows_from_distance_matrix(
            distance,
            shortlist_names,
            shortlist_names,
        ),
        open('distance.csv', 'w'),
    )

    save_rows_as_csv(
        make_rows_from_vectors(
            vectors,
            shortlist_names,
        ),
        open('vectors.csv', 'w'),
    )


    plot_distance_matrix(distance)


def plot_distance_matrix(distance):
    import pylab
    pylab.imshow(
        distance,
        interpolation = 'nearest',
    )
    pylab.colorbar()
    pylab.show()


def shortlist_variables(x, y, header, max_degree, shortlist_length):
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

    print x_header
    print x.shape
    print y.shape
    all_indices = list(gen_all_multi_indices(x.shape[0], max_degree))
    n_indices = len(all_indices)
    print 'we have quite a few multi indices to play with: %d' % n_indices

    monomials = {}
    pc_complete = 0
    for i, alpha in enumerate(all_indices):
        monomials[alpha] = make_monomial(x, alpha)
        current_pc_complete = int(10.0 * i / n_indices) * 10
        if current_pc_complete > pc_complete:
            pc_complete = current_pc_complete
            print '%d %% complete' % pc_complete

    def compute_scores(monomials, y):
        scores = {}
        for alpha in monomials:
            scores[alpha] = cosine_similarity(monomials[alpha], y)
        return scores

    # rank scores in decreasing order
    basis = []
    basis_size = min(shortlist_length, len(y) - 1)
    y_norms = [numpy.linalg.norm(y)]
    while len(basis) < basis_size:
        scores = compute_scores(monomials, y)
        if not scores:
            break
        alpha, score = max(scores.iteritems(), key = lambda x : numpy.abs(x[1]))
        basis.append(alpha)
        m_alpha = monomials[alpha]
        del monomials[alpha]
        m_alpha_norm = numpy.linalg.norm(m_alpha)
        if m_alpha_norm > 0.0:
            m_alpha_hat = m_alpha / (m_alpha_norm ** 2)
        else:
            m_alpha_hat = 0.0
        def project(z):
            return z - numpy.dot(m_alpha, z) * m_alpha_hat
        # orthogonalise y and remaining monomials wrt monomial alpha
        y = project(y)
        y_norm = numpy.linalg.norm(y)
        y_norms.append(y_norm)
        for beta in monomials:
            monomials[beta] = project(monomials[beta])

        print '[%3d] chose %s with score %.3f; norm of residual is %.3f' % (len(basis), fancy_multi_index_name(alpha, x_header), score, y_norm)

    return basis, x_header

def plot_ynorms(ynorms, max_degree):
    import pylab
    pylab.plot(y_norms, 'o-')
    pylab.xlabel('basis size')
    pylab.ylabel('residual norm')
    pylab.title('approximation of y by monomials of degree at most %d' % max_degree)
    pylab.show()

if __name__ == '__main__':
    main()
