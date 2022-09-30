import numpy as np
import matplotlib.pyplot as plt


def plot_real_fn(interval: tuple, nums=1000):
    x = np.linspace(*interval, num=nums)
    y = np.exp(-20 * x)
    plt.plot(x, y, 'r', label='the real function')


def dfn(v_x, v_y):
    return -20 * v_y


def euler(d_fn, init_point: tuple, h, right_interval):
    """
    d_fn: the first derivative

    init_point: the point of initial value

    h: the interval of x

    right_interval: the value of the max of interval for the differential equation

    return: the list of the value xy for the differential equation
    """
    x, y = init_point
    x_list = [x]
    y_list = [y]
    while True:
        x += h
        if x >= right_interval:
            return x_list, y_list
        y += h * d_fn(x, y)
        x_list.append(x)
        y_list.append(y)


def improve_euler(d_fn, init_point: tuple, h, right_interval):
    """
    d_fn: the first derivative

    init_point: the point of initial value

    h: the interval of x

    right_interval: the value of the max of interval for the differential equation

    return: the list of the value xy for the differential equation
    """
    x, y = init_point
    x_list = [x]
    y_list = [y]
    while True:
        x += h
        if x >= right_interval:
            return x_list, y_list
        yp = y + h * d_fn(x, y)
        yc = y + h * d_fn(x + h, yp)
        y = 0.5 * (yp + yc)
        x_list.append(x)
        y_list.append(y)


def runge_kutta4(d_fn, init_point: tuple, h, right_interval):
    """
    d_fn: the first derivative

    init_point: the point of initial value

    h: the interval of x

    right_interval: the value of the max of interval for the differential equation

    return: the list of the value xy for the differential equation
    """
    x, y = init_point
    x_list = [x]
    y_list = [y]
    while True:
        x += h
        if x >= right_interval:
            return x_list, y_list
        k1 = d_fn(x, y)
        k2 = d_fn(x + 0.5 * h, y + 0.5 * h * k1)
        k3 = d_fn(x + 0.5 * h, y + 0.5 * h * k2)
        k4 = d_fn(x + h, y + h * k2)
        y += (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        x_list.append(x)
        y_list.append(y)


def little_one():
    initial_point = (0, 1.)
    x1, y1 = euler(dfn, initial_point, 0.1, 1.)
    x2, y2 = euler(dfn, initial_point, 0.05, 1.)
    plt.title('The Euler method')
    plt.plot(x1, y1, 'b:', label='h = 0.1')
    plt.plot(x2, y2, 'g--', label='h = 0.05')
    plot_real_fn((0, 1))
    plt.legend(loc='best')
    plt.show()


def little_two():
    initial_point = (0, 1.)
    x1, y1 = improve_euler(dfn, initial_point, 0.1, 1.)
    x2, y2 = improve_euler(dfn, initial_point, 0.05, 1.)
    plt.title('The improvement Euler method')
    plt.plot(x1, y1, 'b:', label='h = 0.1')
    plt.plot(x2, y2, 'g--', label='h = 0.05')
    plot_real_fn((0, 1))
    plt.legend(loc='best')
    plt.show()


def little_three():
    initial_point = (0, 1.)
    x1, y1 = runge_kutta4(dfn, initial_point, 0.1, 1.)
    x2, y2 = runge_kutta4(dfn, initial_point, 0.05, 1.)
    plt.title('The Runge-Kutta4 method')
    plt.plot(x1, y1, 'b:', label='h = 0.1')
    plt.plot(x2, y2, 'g--', label='h = 0.05')
    plot_real_fn((0, 1))
    plt.legend(loc='best')
    plt.show()


if __name__ == '__main__':
    little_one()
    little_two()
    little_three()
