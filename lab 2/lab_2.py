import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import stats
import statistics as s


def plot_histogram(dist_arr, mu, sigma):
    count, bins, ignored = plt.hist(dist_arr, 15, density=True)
    plt.plot(bins, 1 / (sigma * np.sqrt(2 * np.pi)) *
             np.exp(- (bins - mu) ** 2 / (2 * sigma ** 2)),
             linewidth=2, color='r')
    plt.grid()
    plt.show()


def check_H1(dist_arr, mu, sigma, alpha):
    z_right = stats.norm.ppf(1 - alpha / 2)
    z_left = stats.norm.ppf(alpha / 2)

    sample_mean = s.mean(dist_arr)

    z = (sample_mean - mu) / (sigma / math.sqrt(len(dist_arr)))

    print(z_left, z_right)
    print(z)

    return (z < z_right) * (z > z_left)


def check_H2(dist_arr, mu, alpha):
    N = len(dist_arr)
    z_right = stats.t.ppf(1 - alpha / 2, N - 1)
    z_left = stats.t.ppf(alpha / 2, N - 1)

    sample_mean = s.mean(dist_arr)
    sample_variance = s.variance(dist_arr, sample_mean)

    z = (sample_mean - mu) / math.sqrt(sample_variance / N)

    print(z_left, z_right)
    print(z)

    return (z < z_right) * (z > z_left)


def check_H3(dist_arr, sigma, alpha):
    N = len(dist_arr)
    z_right = stats.chi2.ppf(1 - alpha / 2, N - 1)
    z_left = stats.chi2.ppf(alpha / 2, N - 1)

    sample_mean = s.mean(dist_arr)
    sample_variance = s.variance(dist_arr, sample_mean)

    z = ((N - 1) * sample_variance) / sigma ** 2

    print(z_left, z_right)
    print(z)

    return (z < z_right) * (z > z_left)


def check_H4(dist_arr_x, dist_arr_y, dx, dy, alpha):
    n_x = len(dist_arr_x)
    n_y = len(dist_arr_y)

    z_right = stats.norm.ppf(1 - alpha / 2)
    z_left = stats.norm.ppf(alpha / 2)

    sample_mean_x = s.mean(dist_arr_x)
    sample_mean_y = s.mean(dist_arr_y)

    z = (sample_mean_x - sample_mean_y) / math.sqrt((dx / n_x) + (dy / n_y))

    print(z_left, z_right)
    print(z)

    return (z < z_right) * (z > z_left)


def check_H5(dist_arr_x, dist_arr_y, alpha):
    n_x = len(dist_arr_x)
    n_y = len(dist_arr_y)

    z_right = stats.t.ppf(1 - alpha / 2, n_x + n_y - 2)
    z_left = stats.t.ppf(alpha / 2, n_x + n_y - 2)

    sample_mean_x = s.mean(dist_arr_x)
    sample_mean_y = s.mean(dist_arr_y)

    sample_variance_x = s.variance(dist_arr_x, sample_mean_x)
    sample_variance_y = s.variance(dist_arr_y, sample_mean_y)

    nominator = sample_mean_x - sample_mean_y
    denominator = math.sqrt(
        ((1 / n_x) + (1 / n_y)) * ((sample_variance_x * (n_x - 1) + sample_variance_y * (n_y - 1)) / (n_x + n_y - 2)))
    z = nominator / denominator

    print(z_left, z_right)
    print(z)

    return (z < z_right) * (z > z_left)


def check_H6(dist_arr_x, dist_arr_y, alpha):
    n_x = len(dist_arr_x)
    n_y = len(dist_arr_y)

    z_right = stats.f.ppf(1 - alpha / 2, n_x - 1, n_y - 1)
    z_left = stats.f.ppf(alpha / 2, n_x - 1, n_y - 1)

    sample_mean_x = s.mean(dist_arr_x)
    sample_mean_y = s.mean(dist_arr_y)

    sample_variance_x = s.variance(dist_arr_x, sample_mean_x)
    sample_variance_y = s.variance(dist_arr_y, sample_mean_y)

    z = sample_variance_x / sample_variance_y

    print(z_left, z_right)
    print(z)

    return (z < z_right) * (z > z_left)


N_X = 50
N_Y = 100
MX = 3
MY = MX + 1.5
DX = 11
DY = DX + 3
alpha = 0.1

sigma_x = math.sqrt(DX)
sigma_y = math.sqrt(DY)

norm_dist_x = np.random.normal(MX, sigma_x, N_X)
norm_dist_y = np.random.normal(MY, sigma_y, N_Y)

plot_histogram(norm_dist_x, MX, sigma_x)
plot_histogram(norm_dist_y, MY, sigma_y)

print('------------------------------------------------------------------')
print('1)')
print("Гипотеза верна" if check_H1(norm_dist_x, MX, sigma_x, alpha) else "Гипотеза неверна")
print('------------------------------------------------------------------')
print('2)')
print("Гипотеза верна" if check_H2(norm_dist_x, MX, alpha) else "Гипотеза неверна")
print('------------------------------------------------------------------')
print('3)')
print("Гипотеза верна" if check_H3(norm_dist_x, sigma_x, alpha) else "Гипотеза неверна")
print('------------------------------------------------------------------')
print('4)')
print("Гипотеза верна" if check_H4(norm_dist_x, norm_dist_y, DX, DY, alpha) else "Гипотеза неверна")
print('------------------------------------------------------------------')
print('5)')
print("Гипотеза верна" if check_H5(norm_dist_x, norm_dist_y, alpha) else "Гипотеза неверна")
print('------------------------------------------------------------------')
print('6)')
print("Гипотеза верна" if check_H6(norm_dist_x, norm_dist_y, alpha) else "Гипотеза неверна")
