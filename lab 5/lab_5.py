import math
import numpy as np


def get_norm(matrix):
    return np.linalg.norm(matrix)


def get_lam(x0, A, eps):
    norm = get_norm(A)
    x = np.array(x0[:])
    lam = 0
    n_iter = 0

    while norm >= eps:
        n_iter += 1
        y = np.dot(A, x)
        lam = np.dot(np.transpose(y), x)
        y = y / get_norm(y)
        norm = get_norm(y - x)
        x = y[:]

        if n_iter > 100:
            break

    return lam


def get_lam_vector(lam_1, A, eps):
    x_0 = np.ones(len(A))
    E = np.eye(len(A))
    A_1 = A - lam_1 * E
    norm = get_norm(np.abs(x_0))
    n_iter = 0

    while norm > eps:
        n_iter += 1
        y = np.linalg.solve(A_1, x_0)
        norm = get_norm(np.abs(y))
        y = y / norm
        x_0 = np.abs(y) - np.abs(x_0)
        norm = math.sqrt(get_norm(np.abs(x_0)))
        x_0 = y[:]

    return x_0

A = [[1.2, 0.5, 2.0, 1.0],
     [0.5, 1.0, 0.6, 2.0],
     [2.0, 0.6, 1.0, 1.0],
     [1.0, 2.0, 1.0, 2.0]]

# A = [[1.0, 1.5, 0.4, 2.0],
#      [1.5, -1.2, 1.0, -0.5],
#      [0.4, 1.0, 2.0, 1.2],
#      [2.0, -0.5, 1.2, 2.5]]

# из методички
A_1 = [[1.0, 1.0, 2.0, 3.0],
       [0.0, 2.0, 2.0, 4.0],
       [0.0, 0.0, 1.0, -2.0],
       [0.0, 0.0, 0.0, 2.0]]

# x0 = [[1], [1], [1], [1]]
x0 = [1, 1, 1, 1]
eps = math.pow(10, -4)
lam_1 = get_lam(x0, A, eps)
print("A: {}".format(np.array(A)))
print('lam_1: ', lam_1)
# A1 = A - lam_1 * np.eye(4)
lam_2 = get_lam(x0, A - lam_1 * np.eye(4), eps) + lam_1
print('lam_2: ', lam_2)
lam_vec = get_lam_vector(lam_1, A, eps)
print('lam_vec:', lam_vec)
print('A * lam_vec', np.dot(np.array(A), lam_vec))
print('lam_vec * lam_1', np.dot(lam_vec, np.array(lam_1)))
print('------------------------------------------------------------------')
print('Пример из методички:')
print("A: {}".format(np.array(A_1)))
lam_1 = get_lam(x0, A_1, eps)
print('lam_1: ', lam_1)
lam_2 = get_lam(x0, A_1 - lam_1 * np.eye(4), eps) + lam_1
print('lam_2: ', lam_2)
lam_vec = get_lam_vector(lam_1, A_1, eps)
print(lam_vec)
print('A * lam_vec', np.dot(np.array(A_1), lam_vec))
print('lam_vec * lam_1', np.dot(lam_vec, np.array(lam_1)))