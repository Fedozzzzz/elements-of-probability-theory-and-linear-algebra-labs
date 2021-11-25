import math
import numpy as np


def get_norm(matrix):
    return np.linalg.norm(matrix, ord=np.inf)


def get_c(A, b):
    return [b[i] / A[i][i] for i in range(len(b))]


def get_B(A):
    rows = len(A)
    columns = len(A[0])
    B = np.zeros((rows, columns))
    for i in range(len(A)):
        for j in range(len(A[0])):
            if i != j:
                B[i][j] = - A[i][j] / A[i][i]
    return B


def jacobi_method(B, c, eps):
    x1 = c.copy()
    x = np.zeros(len(c)).tolist()

    norm = get_norm(b)
    iterations = 0
    while norm > eps:
        iterations += 1

        x = [c[i] + np.dot(B[i], x1) for i in range(len(c))]

        norm = math.sqrt(np.dot(x1, x1))

        x1 = [x[k] - x1[k] for k in range(len(x1))]

        norm = math.sqrt(np.dot(x1, x1)) / norm

        x1 = x[:]

    print("Amount of iterations: {}".format(iterations))
    return x


def seidel(A, b, eps):
    n = len(A)
    x = np.zeros(n)  # zero vector

    converge = False
    while not converge:
        x_new = np.copy(x)
        for i in range(n):
            s1 = sum(A[i][j] * x_new[j] for j in range(i))
            s2 = sum(A[i][j] * x[j] for j in range(i + 1, n))
            x_new[i] = (b[i] - s1 - s2) / A[i][i]

        converge = np.sqrt(sum((x_new[i] - x[i]) ** 2 for i in range(n))) <= eps
        x = x_new

    return x


def jacobi(A, b, tolerance=1e-10, max_iterations=10000):
    x = np.zeros_like(b, dtype=np.double)

    T = A - np.diag(np.diagonal(A))

    for k in range(max_iterations):

        x_old = x.copy()

        x[:] = (b - np.dot(T, x)) / np.diagonal(A)

        if get_norm(x - x_old) / get_norm(x) < tolerance:
            break

    return x


A = [[0.58, 0.32, -0.03, 0.00],
     [-0.11, 1.26, 0.36, 0.00],
     [-0.12, -0.08, 1.14, 0.24],
     [-0.15, 0.35, 0.18, 1.00]]

b = [0.4400, 1.4200, -0.8300, -1.4200]

# print(get_c(A, b))

A_2 = [[0.4000, 0.0003, 0.0008, 0.0014],
       [-0.0029, -0.5000, -0.0018, -0.0012],
       [-0.0055, -0.0050, -1.4000, -0.003],
       [-0.0082, -0.0076, -0.0070, -2.3000]]

b_2 = [0.1220, -0.2532, -0.9876, -2.0812]

# print(get_c(A_2, b_2))

# print(get_B(A))
print(get_norm(get_B(A_2)))
print(get_norm(A_2))
B_2 = get_B(A_2)
c_2 = get_c(A_2, b_2)
# print(c_2)
eps = math.pow(10, -7)
# print(jacobi_method(B_2, c_2, eps))

B = get_B(A)
c = get_c(A, b)
print(jacobi_method(B, c, eps))
print(seidel(A, b, eps))
print(np.linalg.solve(A, b))
# print(jacobi(A, b, eps))
# print(get_norm(b))

A_3 = [[8.20, 0.23, 0.18, 0.14],
       [0.37, 7.30, 0.26, 0.21],
       [0.45, 0.39, 6.40, 0.29],
       [0.53, 0.48, 0.42, 5.50]]

b_3 = [7.5591, 8.1741, 8.4281, 8.3210]

B_3 = get_B(A_3)
c_3 = get_c(A_3, b_3)
# print(jacobi_method(B_3, c_3, eps))

