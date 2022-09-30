import numpy as np
import matplotlib.pyplot as plt


def check_np_inter(x, y):
    # check inputs which the dimension must be one and the shape must be equal
    assert x.shape == y.shape, f'The shape of {x} and {y} is not equal'
    assert x.ndim == 1, f'The dimension of {x} and {y} must be 1'


def ls_poly_fit(x, y, times: int, w=None):
    """
    This function use the orthogonal polynomial to fit data

    x: the data of x

    y: the data of y

    times: times of the polynomial

    w: the weights of x, y

    return: the function of polynomial fitting
    """
    x = np.asarray(x)
    y = np.asarray(y)
    check_np_inter(x, y)
    assert times > 0, f'The times--{times} of polynomial must > 0'

    if w is None:
        w = np.ones_like(x)
    else:
        w = np.asarray(w)

    # get p_0 p_1
    p = [np.poly1d(1.0)]
    alpha = np.sum(np.poly1d([1.0, 0.0])(x) * w) / np.sum(w)
    p.append(np.poly1d([1.0, -alpha]))

    # get p_2...p_n
    if times > 1:
        for j in range(1, times):
            coe1 = np.sum(p[j](x) * p[j](x) * w)
            alpha = np.sum((np.poly1d([1.0, 0.0]) * p[j])(x) * p[j](x) * w) / coe1
            beta = coe1 / np.sum(p[j - 1](x) * p[j - 1](x) * w)
            p.append(np.poly1d([1.0, -alpha]) * p[j] - np.poly1d(beta * p[j - 1]))

    # get the fitting polynomial s
    s = np.poly1d(0.0)  # init the fitting polynomial s
    for p_k in p:
        a_k = np.poly1d(np.sum(p_k(x) * y * w) / np.sum(p_k(x) * p_k(x) * w))
        s += a_k * p_k
    return s


def plot(interval: tuple, fn, inter_x, inter_y, *,
         sub: int = 111, title: str = None, lin1: str = 'fitting curve', point_num: int = 101):
    """
    interval: the interval of plot x

    fn: the function of plot

    inter_x: the scatter of x

    inter_y: the scatter of y

    sub: the subplot of plot

    title: the title of plot

    lin1: the name of line1

    point_num: the numbers of point to plot
    """
    assert len(interval) == 2, f'the length of interval--{interval} must be 2'
    x = np.linspace(*interval, num=point_num)
    y = fn(x)
    plt.subplot(sub)
    plt.title(title)
    plt.xlabel('precipitation')
    plt.ylabel('current Speed')
    plt.plot(inter_x, inter_y, 'go')
    plt.plot(x, y, 'r', label=lin1)
    plt.legend(loc='best')


def read_data():  # get the data of homework
    x = [88.9, 108.5, 104.1, 139.7, 127, 94, 116.8, 99.1]
    y = [14.6, 16.7, 15.3, 23.2, 19.5, 16.1, 18.1, 16.6]
    x = np.array(x)
    y = np.array(y)
    return x, y


def little_one():
    x, y = read_data()
    plt.subplot(111)
    plt.title('precipitation--current Speed scatter diagram')
    plt.xlabel('precipitation')
    plt.ylabel('current Speed')
    plt.plot(x, y, 'go')
    plt.show()


def little_two():
    x, y = read_data()
    f = ls_poly_fit(x, y, times=1)
    plot((80, 150), f, x, y, title='1 times fitting curve')
    plt.show()


def little_three():
    x, y = read_data()
    f = ls_poly_fit(x, y, times=2)
    plot((80, 150), f, x, y, title='2 times fitting curve')
    plt.show()


def little_four():
    x, y = read_data()
    f = ls_poly_fit(x, y, times=1)
    plot((80, 150), f, x, y, title='1 times fitting curve')
    x_year = 120
    y_year = f(x_year)
    plt.plot(x_year, y_year, 'bo', label=f'precipitation-{x_year:.2f}\ncurrent Speed-{y_year:.2f}')
    plt.legend(loc='best')
    plt.show()


def little_five():
    x, y = read_data()
    time_year = 365 * 24 * 3600
    area = 1100_0000
    x_rest = y * time_year / area
    x_loss = x - x_rest
    loss_ratio = x_loss / x
    rest_ratio = x_rest / x
    mean_loss = loss_ratio.mean()
    year = np.array([n for n in range(1, 1 + len(x))])
    size = 12
    plt.title('precipitation loss ratio', fontsize=size)
    plt.ylim((0, 1.5))
    plt.xlim((0, 12))
    plt.xlabel('index of year', fontsize=size)
    plt.ylabel('ratio', fontsize=size)
    plt.bar(year, loss_ratio, width=0.6, color='#8B0000', label='loss ratio')
    plt.bar(year, rest_ratio, width=0.6, color='#006400', label='rest ratio', bottom=loss_ratio)
    for a, b, c in zip(year, loss_ratio, rest_ratio):
        plt.text(a, b, s=f'{b:.2f}', ha='center', va='bottom', fontsize=size)
        plt.text(a, 1.0, s=f'{c:.2f}', ha='center', va='bottom', fontsize=size)
    plt.plot([0.5, 9.5], [mean_loss, mean_loss], 'r')
    plt.text(10, mean_loss, s=f'estimated mean\nloss ratio-{mean_loss:.2f}', ha='center', va='bottom', fontsize=size)
    plt.legend(loc='best', fontsize=size)
    plt.show()


if __name__ == '__main__':
    little_one()
    little_two()
    little_three()
    little_four()
    little_five()
