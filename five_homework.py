import matplotlib.pyplot as plt
import numpy as np
import sympy

# init the global symbol x
x = sympy.symbols('x')


def func_np(v):
    return np.sin(10 * v) - v * np.exp(-v) - v


def get_func_sympy():
    return sympy.sin(10 * x) - x * sympy.exp(-x) - x


def bisection_root(fn, interval: tuple, abs_error):
    """
    fn: the function of root

    interval: the interval of root

    abs_error: the absolute error

    return: the bisection times, the root, the fn(root)
    """
    assert np.prod([fn(v) for v in interval]) < 0, f'interval--{interval} do not meet the condition'
    a, b = interval
    n = 0
    while True:
        n += 1
        m = 0.5 * (a + b)
        fn_m = fn(m)
        if abs(fn_m) <= abs_error:
            return n, m, fn_m
        if fn(a) * fn_m < 0:
            b = m
        else:
            a = m


def newton_root(fn, x_0, abs_error):
    """
    fn: the function of root

    x_0: the initial value

    abs_error: the absolute error

    return: the initial value, the iteration times, the root, the fn(root)
    """
    d1_fn = sympy.diff(fn, x, 1)
    n = 0
    v = x_0
    while True:
        n += 1
        v = v - fn.subs(x, v) / d1_fn.subs(x, v)
        fn_v = fn.subs(x, v)
        if abs(fn_v) <= abs_error:
            return x_0, n, v, fn_v


def secant_root(fn, init_x: tuple, abs_error):
    """
     fn: the function of root

     x_0: the initial value

     abs_error: the absolute error

     return: the initial value, the iteration times, the root, the fn(root)
     """
    n = 0
    v0, v1 = init_x
    while True:
        n += 1
        v0, v1 = v1, v1 - (v1 - v0) * fn(v1) / (fn(v1) - fn(v0))
        fn_v = fn(v1)
        if abs(fn_v) <= abs_error:
            return init_x, n, v1, fn_v


def convergence_order(fn, kind: str, root, k: int = 3):
    """
    fn: the function for computing convergence order

    kind: the kind of method for the numerical convergence order

    root: one of the roots of function

    k: the times of iteration

    return: the numerical convergence order of the function on root
    """
    kind = kind.lower().replace(' ', '')
    assert kind in ['bisection', 'newton', 'secant'], f'kind do not match'
    assert k >= 3, f'the times of iteration k--{k} must be larger than 2'
    v_p = [0., 0., 0.]
    if kind == 'bisection':
        a, b = root - 0.1, root + 0.2
        for j in range(k):
            m = 0.5 * (a + b)
            v_p.pop(0)
            v_p.append(m)
            if fn.subs(x, a) * fn.subs(x, m) < 0:
                b = m
            else:
                a = m

    elif kind == 'newton':
        v = root + 0.1
        d1_fn = sympy.diff(fn, x, 1)
        for j in range(k):
            v = v - fn.subs(x, v) / d1_fn.subs(x, v)
            v_p.pop(0)
            v_p.append(v)

    else:
        v0, v1 = root - 0.1, root + 0.1
        for j in range(k):
            v0, v1 = v1, v1 - (v1 - v0) * fn.subs(x, v1) / (fn.subs(x, v1) - fn.subs(x, v0))
            v_p.pop(0)
            v_p.append(v1)

    e0, e1, e2 = [float(w) for w in np.absolute(np.array(v_p))]
    p = (np.log(e2) - np.log(e1)) / (np.log(e1) - np.log(e0))
    return p


def little_one():
    node = np.linspace(-0.4, 0.4, num=1000)
    y = func_np(node)
    plt.plot((-0.4, 0.4), (0, 0))
    plt.plot(node, y)
    plt.title('The interval of roots is (-0.3,-0.2), (-0.1,0.1), (0.2,0.3)')
    plt.show()


def little_two():
    intervals = ((-0.3, -0.2), (-0.1, 0.1), (0.2, 0.3))
    for v in intervals:
        times, root, fn_root = bisection_root(func_np, v, 1e-8)
        print(f'the interval of root is {v}, the bisection times is {times}, the root is {root}, fn_root is {fn_root}')


def little_three():
    initial_value = (-0.2, 0.1, 0.3)
    fn = get_func_sympy()
    for v in initial_value:
        x0, times, root, fn_root = newton_root(fn, v, 1e-8)
        print(f'the initial value is {x0}, the iteration times is {times}, the root is {root}, fn_root is {fn_root}')


def little_four():
    initial_value = ((-0.3, -0.2), (-0.1, 0.1), (0.2, 0.3))
    for v in initial_value:
        init_x, times, root, fn_root = secant_root(func_np, v, 1e-8)
        print(
            f'the initial value is {init_x}, the iteration times is {times}, the root is {root}, fn_root is {fn_root}')


def little_five():
    print('choosing the root of 0.0 for numerical convergence order')
    root = 0.0
    fn = get_func_sympy()
    all_kind = ['bisection', 'newton', 'secant']
    all_k = [6, 6, 6]
    for kind, k in zip(all_kind, all_k):
        p = convergence_order(fn, kind, root, k=k)
        print(f'the numerical convergence order of {kind} method on root--{root} is p--{p:.3f}')


if __name__ == '__main__':
    little_one()
    # little_two()
    # little_three()
    # little_four()
    # little_five()
