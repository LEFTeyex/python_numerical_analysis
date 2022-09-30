import sympy
import numpy as np

from sympy.solvers.solveset import solveset_real
from scipy import integrate

# init the global symbol x
x = sympy.symbols('x')


def check_all(interval, kind, error, n):
    assert kind in ['trapezoid', 'simpson']
    assert len(interval) == 2, f'the length of interval--{interval} must be 2'
    assert interval[0] < interval[1], f'the interval--{interval} is not right'
    assert error or n, 'abs_error and n can not be None at the same time'
    if n is not None:
        if kind == 'trapezoid':
            assert n > 1, f'n--{n} must be greater than 1'
        else:
            assert n > 2, f'n--{n} must be greater than 2'
            assert n & 1 == 0, f'n--{n} do not match the integration kind--{kind}'


def comp_integrate(fn, interval: tuple, kind: str = 'trapezoid', abs_error=None, n=None):
    """
    If both abs_error and n is not None at the same time, the n will be ignored

    fn: the function for integrating

    interval: the interval of integration

    kind: the kind of integration

    abs_error: the absolute error limit

    n: the number of equal divisions

    return: the result of integration and the number of equal divisions
    """
    kind = kind.lower().replace(' ', '')
    check_all(interval, kind, abs_error, n)
    d_max = None
    # compute the max error and get n
    if abs_error is not None:
        if kind == 'trapezoid':
            n = 1
            result = []
            d_3 = sympy.diff(fn, x, 3)
            _result = solveset_real(d_3, x)
            b_num = abs(interval[0]) if abs(interval[0]) > abs(interval[1]) else abs(interval[1])
            b_num = int(b_num * 100)
            for j, value in enumerate(_result):
                if interval[0] <= value <= interval[1]:
                    result.append(float(value))
                if j > b_num and abs(value) > b_num / 100:
                    break
            result += [*interval]
            d_2 = sympy.diff(fn, x, 2)
            for v in result:
                _max = abs(d_2.subs(x, v))
                if d_max is not None:
                    if _max > d_max:
                        d_max = _max
                else:
                    d_max = abs(d_2.subs(x, v))
            while True:
                error = abs(d_max * (interval[1] - interval[0]) ** 3 / (12 * n ** 2))
                if error <= abs_error:
                    print(f'n--{n} error--{error}')
                    break
                else:
                    print(f'n--{n} error--{error}')
                    n += 1
        else:
            n = 2
            result = []
            d_5 = sympy.diff(fn, x, 5)
            _result = solveset_real(d_5, x)
            b_num = abs(interval[0]) if abs(interval[0]) > abs(interval[1]) else abs(interval[1])
            b_num = int(b_num * 100)
            for j, value in enumerate(_result):
                if interval[0] <= value <= interval[1]:
                    result.append(float(value))
                if j > b_num and abs(value) > b_num / 100:
                    break
            result += [*interval]
            d_4 = sympy.diff(fn, x, 4)
            for v in result:
                _max = abs(d_4.subs(x, v))
                if d_max is not None:
                    if _max > d_max:
                        d_max = _max
                else:
                    d_max = abs(d_4.subs(x, v))
            while True:
                error = abs(d_max * (interval[1] - interval[0]) ** 5 / (2880 * n ** 4))
                if error <= abs_error:
                    print(f'n--{n} error--{error}')
                    break
                else:
                    print(f'n--{n} error--{error}')
                    n += 2

    # compute the cotes coefficient
    if kind == 'trapezoid':
        value = []
        coe = np.array([[1.0, 1.0]])
        node = list(np.linspace(*interval, num=n + 1))
        for _ in range(n):
            value.append([fn.subs(x, node.pop(0))] + [fn.subs(x, node[0])])
        value = np.array(value)
        out = ((value * coe) / n * 0.5 * (interval[1] - interval[0])).sum()
    else:
        value = []
        coe = np.array([[1.0, 4.0, 1.0]])
        node = list(np.linspace(*interval, num=n + 1, dtype=np.float64))
        for _ in range(int(n / 2)):
            value.append([fn.subs(x, node.pop(0)) for _ in range(2)] + [fn.subs(x, node[0])])
        value = np.array(value)
        out = (value * coe).sum() * (1 / 3) * (interval[1] - interval[0]) / n
    return out, n


def comp_integrate_rbg(fn, interval: tuple, abs_error: float):
    """
    fn: the function for integrating

    interval: the interval of integration

    abs_error: the absolute error limit

    return: the result of integration and the list of Romberg
    """
    assert len(interval) == 2, f'the length of interval--{interval} must be 2'
    assert interval[0] < interval[1], f'the interval--{interval} is not right'
    interval = np.asarray(interval)
    # get T1
    h = (interval[1] - interval[0])
    t = h / 2 * fn(interval).sum()
    out = [[t]]
    # recursion by line
    while True:
        h = h / 2
        new_node = 0.5 * (np.delete(interval, 0) + np.delete(interval, -1))
        interval = np.sort(np.concatenate((interval, new_node), axis=None), kind='mergesort')
        t = 0.5 * t + h * fn(new_node).sum()
        last_row = [t]
        for m, v in enumerate(out[-1]):
            v_last = last_row[m]
            coe = 4 ** (m + 1)
            last_row.append((coe * v_last - v) / (coe - 1))

        out.append(last_row.copy())
        if abs(last_row[-1] - last_row[-2]) < abs_error:
            break
    return out[-1][-1], out


def comp_gauss_legendre3(fn, interval: tuple, abs_error=None, n: int = 2):
    """
    fn: the function for integrating

    interval: the interval of integration

    abs_error: the absolute error limit

    n: the number of equal divisions

    return: the result of integration and the number of equal divisions
    """
    assert abs_error and n, 'abs_error and n can not be None all the time'
    assert len(interval) == 2, f'the length of interval--{interval} must be 2'
    assert interval[0] < interval[1], f'the interval--{interval} is not right'
    abs_integration = integrate.quadrature(fn, *interval)[0]

    while True:
        integration = 0.0
        _interval = list(np.linspace(*interval, num=n + 1))
        for j in range(n):
            little_interval = [_interval.pop(0)] + [_interval[0]]
            integration += gauss_legendre3(fn, little_interval)
        error = abs(integration - abs_integration)
        print(f'n--{n} error--{error}')
        if abs(integration - abs_integration) <= abs_error:
            break
        else:
            n += 1
    return integration, n


def gauss_legendre3(fn, interval):
    """
    fn: the function for integrating

    interval: the interval of integration
    """
    assert len(interval) == 2, f'the length of interval--{interval} must be 2'
    assert interval[0] < interval[1], f'the interval--{interval} is not right'
    interval = np.asarray(interval)
    coe1 = (interval[1] - interval[0]) / 2
    coe2 = (interval[0] + interval[1]) / 2
    coe = np.array([5 / 9, 8 / 9, 5 / 9])
    node = np.array([-np.sqrt(15) / 5, 0.0, np.sqrt(15) / 5])
    node = node * coe1 + coe2
    integration = (coe * fn(node)).sum() * coe1
    return integration


def func(v):
    v = map(lambda w: np.exp(w) * np.cos(w), v)
    v = np.array(list(v))
    return v


def little_one():
    f = sympy.exp(x) * sympy.cos(x)
    integration, nums = comp_integrate(f, (0.0, np.pi), kind='trapezoid', abs_error=1e-6)
    print('----------------------------------------')
    print('The compound trapezoid')
    print(f'The integration is {np.trunc(integration * 1e5) / 1e5:.5f}')
    print(f'The number of equal divisions is {nums}')
    print('----------------------------------------')


def little_two():
    f = sympy.exp(x) * sympy.cos(x)
    integration, nums = comp_integrate(f, (0.0, np.pi), kind='simpson', abs_error=1e-6)
    print('----------------------------------------')
    print('The compound simpson')
    print(f'The integration is {np.trunc(integration * 1e5) / 1e5:.5f}')
    print(f'The number of equal divisions is {nums}')
    print('----------------------------------------')


def little_three():
    integration, matrix = comp_integrate_rbg(func, (0.0, np.pi), abs_error=1e-6)
    print('----------------------------------------')
    print('The Romberg integration')
    print(f'The integration is {np.trunc(integration * 1e5) / 1e5:.5f}')
    print('\t')
    print('The list:')
    for row in matrix:
        print(np.round(row, 10))
    print('----------------------------------------')


def little_four():
    integration, nums = comp_gauss_legendre3(func, (0.0, np.pi), abs_error=1e-6)
    print('----------------------------------------')
    print('The compound Gauss-Legendre of three points')
    print(f'The integration is {np.trunc(integration * 1e5) / 1e5:.5f}')
    print(f'The number of equal divisions is {nums}')
    print('----------------------------------------')


if __name__ == '__main__':
    little_one()
    little_two()
    little_three()
    little_four()
