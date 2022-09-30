import numpy as np


def get_matrix_and_b1(n: int):
    # get matrix and b for problem 1
    assert n >= 4, f'n--{n} must be bigger than 4'
    b = np.ones((n, 1), np.float)
    matrix = np.zeros((n, n), np.float)
    middle_row = (1., -4., 6., -4.), 1.
    matrix[0, :3] = (9., -4., 1.)
    matrix[1, :4] = (-4., 6., -4., 1.)
    matrix[- 2, -4:] = (1., -4., 5., -2.)
    matrix[- 1, -3:] = (1., -2., 1.)
    for j in range(2, n - 2):
        matrix[j, j - 2:j + 2], matrix[j, j + 2] = middle_row
    return matrix, b


def get_matrix_and_b2(n: int):
    # get matrix and b for problem 2
    assert n >= 3, f'n--{n} must be bigger than 3'
    b = np.zeros((n, 1), np.float)
    b[0], b[-1] = 1., 1.
    matrix = np.zeros((n, n), np.float)
    middle_row = (-1., 2.), -1.
    matrix[0, :2] = (2., -1.)
    matrix[- 1, -2:] = (-1., 2.)
    for j in range(1, n - 1):
        matrix[j, j - 1:j + 1], matrix[j, j + 1] = middle_row
    return matrix, b


def lu_decompose_solve(matrix, b):
    """
    matrix: the matrix to decompose

    b: the coefficient y for the equation

    return: the solution, L matrix, U matrix
    """
    matrix = np.array(matrix, np.float)
    b = np.asarray(b, np.float)
    n = matrix.shape[0]
    # lu decompose
    for j in range(n):
        matrix[j, j:] = matrix[j, j:] - (matrix[:j, j:] * matrix[j, :j].reshape(j, 1)).sum(0)
        matrix[j + 1:, j] = (1. / matrix[j, j]) * \
                            (matrix[j + 1:, j] - (matrix[j + 1:, :j] * matrix[:j, j].reshape(1, j)).sum(1))

    # get l and u in matrix
    l_matrix = np.zeros_like(matrix, np.float)
    u_matrix = np.zeros_like(matrix, np.float)
    for j, row in enumerate(matrix):
        l_matrix[j, :j], l_matrix[j, j], u_matrix[j, j:] = row[:j], 1., row[j:]

    del matrix  # to save memory

    # solve
    y = np.zeros_like(b, np.float)
    for j, v in enumerate(b):
        y[j, 0] = v - (l_matrix[j, :j] * y[:j].reshape(-1)).sum()
    x = np.zeros_like(b, np.float)
    for j, v in enumerate(y[::-1]):
        x[n - j - 1, 0] = (1 / u_matrix[n - j - 1, n - j - 1]) * \
                          (v - (u_matrix[n - j - 1, n - j - 1:] * x[n - j - 1:].reshape(-1)).sum())
    return x, l_matrix, u_matrix


def cholesky_decompose_solve(matrix, b):
    """
    matrix: the matrix to decompose

    b: the coefficient y for the equation

    return: the solution, L matrix, L.T matrix
    """
    matrix = np.array(matrix, np.float)
    b = np.asarray(b, np.float)
    n = matrix.shape[0]
    # cholesky decompose
    matrix = np.linalg.cholesky(matrix)

    # solve
    y = np.zeros_like(b, np.float)
    for j, v in enumerate(b):
        y[j] = (1 / matrix[j, j]) * (v - (matrix[j, :j] * y[:j].reshape(-1)).sum())
    x = np.zeros_like(b, np.float)
    matrix_t = matrix.transpose((1, 0))
    for j, v in enumerate(y[::-1]):
        x[n - j - 1] = (1 / matrix_t[n - j - 1, n - j - 1]) * \
                       (v - (matrix_t[n - j - 1, n - j - 1:] * x[n - j - 1:].reshape(-1)).sum())
    return x, matrix, matrix_t


def decompose_ldu(matrix):
    """
    matrix: decompose the matrix to l_matrix, d_matrix, u_matrix

    return: l_matrix, d_matrix, u_matrix
    """
    matrix = np.asarray(matrix, np.float)
    l_matrix = np.zeros_like(matrix, np.float)
    d_matrix = np.zeros_like(matrix, np.float)
    u_matrix = np.zeros_like(matrix, np.float)
    for j, row in enumerate(matrix):
        l_matrix[j, :j], d_matrix[j, j], u_matrix[j, j + 1:] = row[:j], row[j], row[j + 1:]
    return l_matrix, d_matrix, u_matrix


def jacobi_iter(matrix, b, abs_error):
    """
    matrix: the coefficient matrix of equation

    b: the coefficient y for the equation

    abs_error: the absolute error limit

    return: the number of iterations, error, the iterative matrix
    """
    matrix = np.asarray(matrix, np.float)
    b = np.asarray(b, np.float)
    x = np.zeros_like(b, np.float)
    l_matrix, d_matrix, u_matrix = decompose_ldu(matrix)
    # get the inverse of d_matrix
    inv_d_matrix = np.linalg.inv(d_matrix)
    # get the iterative matrix
    iter_matrix = np.matmul(inv_d_matrix, l_matrix + u_matrix)
    n = 1
    while True:
        old_x = x.copy()
        x = np.matmul(iter_matrix, x) + np.matmul(inv_d_matrix, b)
        error = np.linalg.norm(x - old_x, np.inf)
        if error < abs_error:
            break
        n += 1
    return n, error, iter_matrix


def gauss_seidel_iter(matrix, b, abs_error):
    """
    matrix: the coefficient matrix of equation

    b: the coefficient y for the equation

    abs_error: the absolute error limit

    return: the number of iterations, error, the iterative matrix
    """
    matrix = np.asarray(matrix, np.float)
    b = np.asarray(b, np.float)
    x = np.zeros_like(b, np.float)
    l_matrix, d_matrix, u_matrix = decompose_ldu(matrix)
    # get the inverse of (d_matrix - l_matrix)
    inv_dml_matrix = np.linalg.inv(d_matrix - l_matrix)
    # get the iterative matrix
    iter_matrix = np.matmul(inv_dml_matrix, u_matrix)
    n = 1
    while True:
        old_x = x.copy()
        x = np.matmul(iter_matrix, x) + np.matmul(inv_dml_matrix, b)
        error = np.linalg.norm(x - old_x, np.inf)
        if error < abs_error:
            break
        n += 1
    return n, error, iter_matrix


def little_one_1():
    for n in (100, 1000):
        a_matrix, a_to_b = get_matrix_and_b1(n)
        x, l_matrix, u_matrix = lu_decompose_solve(a_matrix, a_to_b)
        max_res = np.linalg.norm(a_to_b - np.matmul(a_matrix, x), np.inf)
        cond = np.linalg.cond(a_matrix, np.inf)
        print('----------------------------------------')
        print(f'the n--{n}')
        # print('the L_matrix')
        # print(l_matrix)
        # print('the U_matrix')
        # print(u_matrix)
        # print('the solution x')
        # print(x)
        print(f'the maximum residual--{max_res}')
        print(f'the condition number--{cond}')
        print('----------------------------------------')


def little_one_2():
    for n in (5, 10):
        a_matrix, a_to_b = get_matrix_and_b1(n)
        x, matrix, matrix_t = cholesky_decompose_solve(a_matrix, a_to_b)
        max_res = np.linalg.norm(a_to_b - np.matmul(a_matrix, x), np.inf)
        print('----------------------------------------')
        print(f'the n--{n}')
        # print('the L_matrix')
        # print(matrix)
        # print('the U_matrix'
        # print(matrix_t)
        # print('the solution x')
        # print(x)
        print(f'the maximum residual--{max_res}')
        print('----------------------------------------')


def little_two_1():
    for n in (10, 20, 50, 100, 200):
        a_matrix, a_to_b = get_matrix_and_b2(n)
        num, error, iter_matrix = jacobi_iter(a_matrix, a_to_b, 1e-8)
        r = np.max(np.linalg.eigvals(iter_matrix)).real
        print('----------------------------------------')
        print(f'the n--{n}')
        print(f'the number of iterations--{num}')
        print(f'the error--{error}')
        print(f'the spectral radius of iterative matrix--{r}')
        print('----------------------------------------')


def little_two_2():
    for n in (10, 20, 50, 100, 200):
        a_matrix, a_to_b = get_matrix_and_b2(n)
        num, error, iter_matrix = gauss_seidel_iter(a_matrix, a_to_b, 1e-8)
        r = np.max(np.linalg.eigvals(iter_matrix)).real
        print('----------------------------------------')
        print(f'the n--{n}')
        print(f'the number of iterations--{num}')
        print(f'the error--{error}')
        print(f'the spectral radius of iterative matrix--{r}')
        print('----------------------------------------')


if __name__ == '__main__':
    little_one_1()
    little_one_2()
    little_two_1()
    little_two_2()
