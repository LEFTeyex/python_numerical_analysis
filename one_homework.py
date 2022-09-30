import numpy as np
import matplotlib.pyplot as plt


def deal_xy(x, y):
    # to numpy
    x = np.asarray(x)
    y = np.asarray(y)
    # sort x and y
    index = np.argsort(x, kind='mergesort')
    x = x[index]
    y = y[index]
    return x, y


def check_np_inter(x, y):
    # check inputs which the dimension must be one and the shape must be equal
    assert x.shape == y.shape, f'The shape of {x} and {y} is not equal'
    assert x.ndim == 1, f'The dimension of {x} and {y} must be 1'


def check_np_boundary(*args):
    for x in args:
        assert x.shape[0] == 2, f'The length of {x} must be 2'
        assert x.ndim == 1, f'The dimension of {x} must be 1'


def fn_list_computer(x, fn_list: list, fn_intervals: list):
    """
    x: the interpolation point of x

    fn_list: the list contains interpolation function for fn_intervals

    fn_intervals: every two adjacent elements in fn_intervals corresponding to a function in fn_list

    return: the data y that computed by x
    """
    assert len(fn_list) + 1 == len(fn_intervals), f'The fn_list do not match fn_intervals'
    x = np.asarray(x)
    y = np.empty_like(x)
    assert x.ndim == 1, f'The dimension of {x} must be 1'

    if not isinstance(fn_list, list):
        fn_list = list(fn_list)
    if not isinstance(fn_intervals, list):
        fn_intervals = list(fn_intervals)

    # sort x
    index = np.argsort(x, kind='mergesort')
    x = x[index]

    left_y, right_y = np.empty(0), np.empty(0)
    left_index = []
    right_index = []

    # take out the extrapolated value
    for j, v in enumerate(x):
        if v < fn_intervals[0]:
            left_index.append(j)
        if v > fn_intervals[-1]:
            right_index.append(j)
    if len(left_index):
        left_x = x[left_index]
        left_y = fn_list[0](left_x)
    if len(right_index):
        right_x = x[right_index]
        right_y = fn_list[-1](right_x)

    # get the values of x for interpolating
    if len(left_index) or len(right_index):
        x = np.delete(x, left_index + right_index)

    # compute x to y
    inter_y = np.empty_like(x)
    for j, value in enumerate(x):
        while True:
            if fn_intervals[0] <= value <= fn_intervals[1]:
                inter_y[j] = fn_list[0](value)
                break
            else:
                fn_list.pop(0)
                fn_intervals.pop(0)

    # turn to the original order for corresponding to x that do not sort
    y[index] = np.concatenate((left_y, inter_y, right_y))
    return y


def lagrange_fn(x, y):
    """
    x: the interpolation node of x

    y: the interpolation node of y

    return: the function of interpolation
    """
    # to numpy array
    x, y = deal_xy(x, y)
    check_np_inter(x, y)
    # init polynomial
    # p = np.poly1d(0.0)
    node = []
    div = []
    for j, x_j in enumerate(x):
        # p_j = np.poly1d(y[j])  # get y
        node_j = np.delete(x, obj=j, axis=0)  # remove the x_j
        div_j = np.prod(x_j - node_j)  # compute (x_j - x_0)(x_j - x_1)...(x_j - x_n) that without x_j as divisor
        # compute (x - x_0)(x - x_1)...(x - x_n) that without x_j
        # for value in node_j:
        #     p_j *= np.poly1d([1.0, -value])
        # p += p_j / div_j  # compute the j basis function with y_j
        node.append(node_j)
        div.append(div_j)
    node = np.asarray(node)
    div = np.asarray(div)

    def function(inter_x):
        inter_x = np.asarray(inter_x).reshape((-1, 1))
        out = np.zeros_like(inter_x)
        inter_x = inter_x.repeat(node.shape[1], axis=1)
        for k, (n, d) in enumerate(zip(node, div)):
            out += (y[k] * np.prod(inter_x - n, axis=1, keepdims=True)) / d
        return np.reshape(out, (-1))

    return function


def piecewise_inter(x, y, *, kind: str = 'lagrange', times: int = 1):
    """
    x: the interpolation node of x

    y: the interpolation node of y

    kind: the kind of piecewise interpolation

    times: the times of piecewise interpolation

    return: the function of piecewise interpolation
    """
    # to numpy array and check
    x, y = deal_xy(x, y)
    x_intervals = list(x)
    check_np_inter(x, y)
    # to list
    x = list(x)
    y = list(y)
    kind = kind.lower().replace(' ', '')
    assert (len(x) - 1) % times == 0, f'The input do not match the interpolation_old times--{times}'
    assert kind in ['lagrange'], f'The kind--{kind} must be one of [lagrange]'

    # get function of interpolation_old
    if kind == 'lagrange':
        fn_piece = lagrange_fn
    else:
        raise f'The kind--{kind} do not match something'

    fn_list = []
    # get values and compute piecewise interpolation_old function
    for j in range(int((len(x) - 1) / times)):
        x_piece = [x.pop(0) for _ in range(times)] + [x[0]]
        y_piece = [y.pop(0) for _ in range(times)] + [y[0]]
        p = fn_piece(x_piece, y_piece)
        fn_list.append(p)

    # get function of piecewise interpolation_old
    x_intervals = [x_intervals[i] for i in range(0, len(x_intervals), times)]

    def function(inter_x):
        return fn_list_computer(inter_x, fn_list, fn_intervals=x_intervals)

    return function


def cubic_spline_inter(x, y, condition: str, *, df_on: list = None, ddf_on: list = None):
    """
    x: the interpolation node of x

    y: the interpolation node of y

    condition: boundary condition
        'one': is given the first derivative of the boundary

        'two': is given the second derivative of the boundary

        'three': periodic function

    df_on: first boundary condition for boundary first derivative

    ddf_on: second boundary condition for boundary second derivative

    return: the function of cubic spline interpolation
    """
    x, y = deal_xy(x, y)
    check_np_inter(x, y)
    condition = condition.lower().replace(' ', '')
    assert condition in ['one', 'two', 'three'], f'The condition--{condition} must be one of [one, two, three]'

    # deal the data of boundary condition
    if df_on is not None:
        df_on = np.asarray(df_on)
        check_np_boundary(df_on)
    if ddf_on is not None:
        ddf_on = np.asarray(ddf_on)
        check_np_boundary(ddf_on)

    # compute the coefficient
    h = x[1:] - x[:-1]  # compute h_k
    n_fn = h.shape[0]  # numbers of spline function
    a = h[:-1] / (h[:-1] + h[1:])
    b = h[1:] / (h[:-1] + h[1:])
    d = (y[1:] - y[:-1]) / h
    d = 6 * (d[1:] - d[:-1]) / (h[:-1] + h[1:])

    if condition == 'one':
        if df_on is None:
            raise f'The condition--{condition} must contain df_on that is boundary first derivative'
        d = np.insert(d, 0, 6 * ((y[1] - y[0]) / h[0] - df_on[0]) / h[0])
        d = np.append(d, 6 * (df_on[1] - (y[-1] - y[-2]) / h[-1]) / h[-1])

        # build the matrix for solving equations
        n = d.shape[0]
        matrix = np.zeros((n, n))
        matrix[0, 0], matrix[0, 1] = 2, 1
        matrix[-1, -1], matrix[-1, -2] = 2, 1

        for j in range(1, n - 1):
            matrix[j, j - 1], matrix[j, j], matrix[j, j + 1] = a[j - 1], 2, b[j - 1]
        m = np.linalg.solve(matrix, d)

    elif condition == 'two':
        if ddf_on is None:
            raise f'The condition--{condition} must contain df_on that is boundary second derivative'
        d[0] -= a[0] * ddf_on[0]
        d[-1] -= b[-1] * ddf_on[1]

        # build the matrix for solving equations
        n = d.shape[0]
        matrix = np.zeros((n, n))
        matrix[0, 0], matrix[0, 1] = 2, b[0]
        matrix[-1, -1], matrix[-1, -2] = 2, a[-1]

        for j in range(1, n - 1):
            matrix[j, j - 1], matrix[j, j], matrix[j, j + 1] = a[j], 2, b[j]
        m = np.linalg.solve(matrix, d)
        m = np.insert(m, 0, ddf_on[0])
        m = np.append(m, ddf_on[1])

    elif condition == 'three':
        d = np.append(d, 6 * ((y[1] - y[0]) / h[0] - (y[-1] - y[-2]) / h[-1]) / (h[0] + h[-1]))

        # build the matrix for solving equations
        n = d.shape[0]
        matrix = np.zeros((n, n))
        matrix[0, 0], matrix[0, 1], matrix[0, -1] = 2, b[0], a[0]
        matrix[-1, -1], matrix[-1, -2], matrix[-1, 0] = 2, h[-1] / (h[0] + h[-1]), h[0] / (h[0] + h[-1])

        for j in range(1, n - 1):
            matrix[j, j - 1], matrix[j, j], matrix[j, j + 1] = a[j], 2, b[j]
        m = np.linalg.solve(matrix, d)
        m = np.insert(m, 0, m[-1])

    else:
        raise f'The condition--{condition} do not match something'

    fn_list = []
    coe1 = (m[1:] - m[:-1]) / (6 * h)
    coe2 = m[:-1] / 2
    coe3 = (y[1:] - y[:-1]) / h - h * (m[1:] + 2 * m[:-1]) / 6

    # compute all function in intervals
    for j in range(n_fn):
        base = np.poly1d([1.0, -x[j]])
        p = np.poly1d(coe1[j]) * base ** 3 + np.poly1d(coe2[j]) * base ** 2 + np.poly1d(coe3[j]) * base + \
            np.poly1d(y[j])
        fn_list.append(p)

    # get function of total intervals
    def function(inter_x):
        return fn_list_computer(inter_x, fn_list, fn_intervals=x)

    return function


def plot_vs(interval: tuple, fn1, fn2, inter_x, *,
            sub: int = 111, title: str = None, lin1: str = 'original', lin2: str = 'interpolation',
            point_num: int = 101):
    """
    interval: the interval of plot x

    fn1: the function of original for plot

    fn2: the function of interpolation for plot

    inter_x: the interpolation point of x

    sub: the subplot of plot

    title: the title of plot

    lin1: the name of line1 for fn1

    lin2: the name of line2 for fn2

    point_num: the numbers of point to plot
    """
    assert len(interval) == 2, f'the length of interval--{interval} must be 2'
    x = np.linspace(*interval, num=point_num)
    x = np.sort(np.concatenate((x, inter_x)))
    inter_y = fn1(inter_x)
    y1 = fn1(x)
    y2 = fn2(x)
    plt.subplot(sub)
    plt.title(title)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.plot(inter_x, inter_y, 'go')
    plt.plot(x, y1, 'g-', label=lin1)
    plt.plot(x, y2, 'r--', label=lin2)
    plt.legend(loc='best')


def fn(x):
    """
    The given function of subject

    x: the interpolation node of x

    return: the interpolation node of y
    """
    x = map(lambda v: 1 / (1 + 25 * v ** 2), x)
    return list(x)


def chebyshev_zero(interval: tuple, times: int):
    """
    interval: the interval of interpolation node

    times: the times of interpolation

    return: list of the chebyshev zero for interpolating
    """
    x = np.array(range(times + 1))
    x = map(lambda v: 0.5 * (interval[1] - interval[0]) * np.cos(np.pi * (2 * v + 1) / (2 * times + 2)) + 0.5 * (
            interval[1] + interval[0]), x)
    return list(x)


def little_one():
    # 11 points for 10 times lagrange
    x1 = np.linspace(-1, 1, num=11)
    y1 = fn(x1)
    f1 = lagrange_fn(x1, y1)
    plot_vs((-1, 1), fn, f1, inter_x=x1, sub=121, title='10 times lagrange', lin1='f(x)', lin2='L(x)')

    # 21 points for 20 times lagrange
    x2 = np.linspace(-1, 1, num=21)
    y2 = fn(x2)
    f2 = lagrange_fn(x2, y2)
    plot_vs((-1, 1), fn, f2, inter_x=x2, sub=122, title='20 times lagrange', lin1='f(x)', lin2='L(x)')
    plt.show()


def little_two():
    # get 19 rand real number and end points
    x = np.random.uniform(-1, 1, 19)
    x = np.append(x, [-1, 1])
    y = fn(x)
    f = lagrange_fn(x, y)
    plot_vs((-1, 1), fn, f, inter_x=x, sub=111, title='20 times lagrange', lin1='f(x)', lin2='L(x)')
    plt.show()


def little_three():
    x = np.linspace(-1, 1, num=11)
    y = fn(x)
    f = piecewise_inter(x, y, times=2)
    plot_vs((-1, 1), fn, f, inter_x=x, sub=111, title='2 times piecewise interpolation',
            lin1='f(x)', lin2='piecewise interpolation')
    plt.show()


def little_four():
    x1 = chebyshev_zero((-1, 1), 10)
    y1 = fn(x1)
    f1 = lagrange_fn(x1, y1)
    plot_vs((-1, 1), fn, f1, inter_x=x1, sub=131, title='10 times chebyshev lagrange', lin1='f(x)', lin2='L(x)')

    x2 = chebyshev_zero((-1, 1), 20)
    y2 = fn(x2)
    f2 = lagrange_fn(x2, y2)
    plot_vs((-1, 1), fn, f2, inter_x=x2, sub=132, title='20 times chebyshev lagrange', lin1='f(x)', lin2='L(x)')

    x3 = chebyshev_zero((-1, 1), 40)
    y3 = fn(x3)
    f3 = lagrange_fn(x3, y3)
    plot_vs((-1, 1), fn, f3, inter_x=x3, sub=133, title='40 times chebyshev lagrange', lin1='f(x)', lin2='L(x)')
    plt.show()


def little_five():
    x = np.linspace(-1, 1, num=11)
    y = fn(x)
    f = cubic_spline_inter(x, y, condition='three')
    plot_vs((-1, 1), fn, f, inter_x=x, sub=111, title='cubic spline interpolation',
            lin1='f(x)', lin2='cubic spline interpolation')
    plt.show()


if __name__ == '__main__':
    little_one()
    little_two()
    little_three()
    little_four()
    little_five()
