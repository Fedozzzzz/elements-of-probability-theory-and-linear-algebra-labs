import math
import matplotlib.pyplot as plt
import numpy as np
import random as rand
from scipy import integrate
from scipy import stats
from scipy.stats import gamma


def generate_random_num(alpha, lam, n):
    if alpha <= 0:
        return None

    def algorithm_1():
        X = []
        for i in range(n):
            r1 = rand.random()
            r2 = rand.random()
            r3 = rand.random()
            S1 = math.pow(r1, 1 / alpha)
            S2 = math.pow(r2, 1 / (1 - alpha))
            while S1 + S2 > 1:
                r1 = rand.random()
                r2 = rand.random()
                r3 = rand.random()
                S1 = math.pow(r1, 1 / alpha)
                S2 = math.pow(r2, 1 / (1 - alpha))
            X.append(-(S1 * math.log(r3) / (lam * (S1 + S2))))
        X.sort()
        return X

    def algorithm_2_1():
        X = []
        for i in range(n):
            b = math.exp(1) / (math.exp(1) + alpha)
            res = None
            while res is None:
                r1 = rand.random()
                r2 = rand.random()
                if r1 < b:
                    v = math.pow((r1 / b), (1 / alpha))
                    if r2 <= math.exp(-v):
                        res = v / lam
                else:
                    v = 1 - math.log((1 - r1) / (1 - b))
                    if r2 <= math.pow(v, alpha - 1):
                        res = v / lam
            X.append(res)
        X.sort()
        return X

    def algorithm_2_2():

        def get_exponential_distribution_random(l):
            return -math.log(1 - rand.random()) / l

        X = []
        for i in range(n):
            e1 = get_exponential_distribution_random(1)
            e2 = get_exponential_distribution_random(1)
            while e2 < ((alpha - 1) * (e1 - math.log(e1) - 1)):
                e1 = get_exponential_distribution_random(1)
                e2 = get_exponential_distribution_random(1)
            X.append(alpha * e1 / lam)

        X.sort()
        return X

    # return algorithm_1() if 0 < alpha < 1 else algorithm_2_1()
    return algorithm_1() if 0 < alpha < 1 else algorithm_2_2()


def get_probability_density(X, alpha, lam):
    def get_fx(x):
        return (math.pow(lam, alpha) / math.gamma(alpha)) * math.pow(x, alpha - 1) * math.exp(-x * lam)

    return [get_fx(i) for i in X]


def get_probability_distribution(X, alpha, lam):
    def get_Fx(x):
        return (1 / math.gamma(alpha)) * integrate.quad(lambda t: math.pow(t, alpha - 1) * math.exp(-t), 0, lam * x)[0]

    return [get_Fx(i) for i in X]


def plot_percentiles(X, alpha, lam):
    percentile = np.arange(0, 101, 1)
    score = np.percentile(X, percentile)
    fig, ax = plt.subplots()
    plt.bar(percentile, score, label='alpha={},lam={}'.format(alpha, lam))
    ax.set_xlabel('Percent Rank')
    ax.set_ylabel('Percentile values')
    ax.grid()
    ax.legend()
    plt.show()


def plot_probability_density(X_arr, params):
    fig, ax = plt.subplots()
    for i in range(len(X_arr)):
        x = X_arr[i]
        alpha = params[i]['alpha']
        lam = params[i]['lam']
        prob_density = get_probability_density(x, alpha, lam)
        ax.plot(x, prob_density, label='alpha={},lam={}'.format(alpha, lam))
    ax.set_xlabel('x')
    ax.set_ylabel('f(x)')
    plt.ylim(0.0, 0.5)
    plt.xlim(0, 10)
    ax.grid()
    ax.legend()
    plt.show()


def plot_probability_distribution(X_arr, params):
    fig, ax = plt.subplots()
    for i in range(len(X_arr)):
        x = X_arr[i]
        alpha = params[i]['alpha']
        lam = params[i]['lam']
        prob_distr = get_probability_distribution(x, alpha, lam)
        ax.plot(x, prob_distr, label='alpha={},lam={}'.format(alpha, lam))
    ax.set_xlabel('x')
    ax.set_ylabel('F(x)')
    plt.ylim(0.0, 1.0)
    plt.xlim(0, 10)
    ax.grid()
    ax.legend()
    plt.show()


def plot_histogram(X):
    fig, ax = plt.subplots()
    plt.hist(X, density=True, alpha=0.5, facecolor='blue', edgecolor='black', linewidth=1.2)
    ax.set_xlabel('x')
    ax.set_ylabel('Frequency')
    plt.show()


# THEORETICAL CHARACTERISTICS
# MX
def get_expected_value(alpha, lam):
    return alpha / lam


# Mode
def get_mode(alpha, lam):
    return (alpha - 1) / lam if alpha >= 1 else None


# DX
def get_variance(alpha, lam):
    return alpha / lam ** 2


# Asymmetry
def get_skewness(alpha):
    return 2 / math.sqrt(alpha)


# Excess
def get_kurtosis(alpha):
    return 6 / alpha


# STATISTICAL CHARACTERISTICS
# Mode statistical
def get_mode_statistical(X, alpha, lam):
    fx_arr = get_probability_density(X, alpha, lam)
    index_of_max = np.argmax(fx_arr)
    return X[index_of_max]


def get_median(X):
    return np.median(X)


def get_range(X):
    return max(X) - min(X)


# выборочная несмещенная дисперсия
def get_s2(arr, mean):
    sum = 0
    for x in arr:
        sum += (x - mean) ** 2
    return sum / (len(arr) - 1), sum / len(arr)


def get_mean(arr):
    sum = 0
    for x in arr:
        sum += x
    return sum / len(arr)


def get_mx(X, alpha, lam):
    min_val = min(X)
    max_val = max(X)
    return (math.pow(lam, alpha) / math.gamma(alpha)) * \
           integrate.quad(lambda x: math.pow(x, alpha - 1) * math.exp(-lam * x) * x, min_val, max_val)[0]


def get_dx(X, alpha, lam):
    min_val = min(X)
    max_val = max(X)
    mx = (math.pow(lam, alpha) / math.gamma(alpha)) * \
         integrate.quad(lambda x: math.pow(x, alpha - 1) * math.exp(-lam * x) * x, min_val, max_val)[0]
    mx2 = (math.pow(lam, alpha) / math.gamma(alpha)) * \
          integrate.quad(lambda x: math.pow(x, alpha - 1) * math.exp(-lam * x) * x ** 2, min_val, max_val)[0]
    return mx2 - mx ** 2


# INTERVAL CHARACTERISTICS
def get_interval_expected_value(mean, b, n, Dx):
    eps_b = math.sqrt(Dx / n) / (stats.norm.cdf((1 + b) / 2))
    return mean - eps_b, mean + eps_b


def get_interval_variance(Dx, b, n):
    eps_b = math.sqrt(2 / (n - 1)) * Dx / (stats.norm.cdf((1 + b) / 2))
    return Dx - eps_b, Dx + eps_b


# точная интервальная оценка
def get_interval_expected_value_2(mean, b, n, s2):
    t_val = stats.t.interval(b, n - 1)[1]
    return mean - t_val * math.sqrt(s2 / n), mean + t_val * math.sqrt(s2 / n)


def get_interval_variance_2(b, n, s2):
    chi2 = stats.chi2.interval(b, n - 1)
    return (n - 1) * s2 / chi2[1], (n - 1) * s2 / chi2[0]


def get_gamma(N, alpha, lam):
    res = gamma.rvs(alpha, size=N, scale=1 / lam)
    res.sort()
    return res


if __name__ == '__main__':

    # params = [
    #     {
    #         "alpha": 4.0,
    #         "lam": 3.0
    #     },
    #     {
    #         "alpha": 4.0,
    #         "lam": 1.5
    #     },
    #     {
    #         "alpha": 4.0,
    #         "lam": 1.0
    #     },
    # ]
    # params = [
    #     {
    #         "alpha": 0.3,
    #         "lam": 1.2
    #     },
    #     {
    #         "alpha": 0.5,
    #         "lam": 1.2
    #     },
    #     {
    #         "alpha": 0.9,
    #         "lam": 1.2
    #     },
    # ]

    params = [
        {
            "alpha": 0.5,
            "lam": 1.0
        },
        {
            "alpha": 2.0,
            "lam": 1.0
        },
        {
            "alpha": 4.0,
            "lam": 1.0
        },
    ]

    # params = [
    #     {
    #         "alpha": 1.0,
    #         "lam": 1.0
    #     },
    #     {
    #         "alpha": 1.5,
    #         "lam": 1.0
    #     },
    #     {
    #         "alpha": 2.5,
    #         "lam": 1.0
    #     },
    #     {
    #         "alpha": 3.5,
    #         "lam": 1.0
    #     },
    #     {
    #         "alpha": 4.5,
    #         "lam": 1.0
    #     },
    # ]
    N = 100
    # N = 10000
    # N = 100000

    rand_num_arr = []
    b = 0.95
    i = 1

    for p in params:
        alpha = p['alpha']
        lam = p['lam']
        rand_nums = generate_random_num(alpha, lam, N)
        rand_num_arr.append(rand_nums)
        plot_percentiles(rand_nums, alpha, lam)
        plot_histogram(rand_nums)
        print('-----------------------------------------------------------------------')
        print('Набор данных #{}: alpha={}, lambda={}'.format(i, alpha, lam))
        print('-----------------------------------------------------------------------')
        print('ТЕОРЕТИЧЕСКИЕ ХАРАКТЕРИСТИКИ:')
        mx = get_expected_value(alpha, lam)
        dx = get_variance(alpha, lam)
        print('Математическое ожидание: {}'.format(mx))
        # print('Математическое ожидание (по общей формуле): {}'.format(get_mx(rand_nums, alpha, lam)))
        print('Дисперсия: {}'.format(dx))
        print('Стандартное отклонение: {}'.format(math.sqrt(dx)))
        # print('Дисперсия (по общей формуле): {}'.format(get_dx(rand_nums, alpha, lam)))
        if alpha >= 1:
            print('Мода: {}'.format(get_mode(alpha, lam)))
        print('Асимметрия: {}'.format(get_skewness(alpha)))
        print('Эксцесс: {}'.format(get_kurtosis(alpha)))
        print('-----------------------------------------------------------------------')
        print('СТАТИСТИЧЕСКИЕ ХАРАКТЕРИСТИКИ:')
        mean = np.mean(rand_nums)
        s2, s_default = get_s2(rand_nums, mean)
        print('Выборочное среднее: {}'.format(mean))
        print('Выборочная дисперсия: {}'.format(s_default))
        print('Выборочное стандартное отклонение: {}'.format(math.sqrt(s_default)))
        print('Несмещенная выборочная дисперсия: {}'.format(s2))
        print('Несмещенное выборочное стандартное отклонение: {}'.format(math.sqrt(s2)))
        print('Размах вариации: {}'.format(get_range(rand_nums)))
        if alpha >= 1:
            print('Мода: {}'.format(get_mode_statistical(rand_nums, alpha, lam)))
        print('Медиана: {}'.format(get_median(rand_nums)))
        print('-----------------------------------------------------------------------')
        print('ИНТЕРВАЛЬНЫЕ ОЦЕНКИ:')
        mx_interval = get_interval_expected_value_2(mean, b, len(rand_nums), s2)
        dx_interval = get_interval_variance_2(b, len(rand_nums), s2)
        print('Интервальная оценка мат. ожидания: \n{} <= MX <= {}'.format(mx_interval[0], mx_interval[1]))
        print('Интервальная оценка дисперсии: \n{} <= DX <= {}'.format(dx_interval[0], dx_interval[1]))
        print('-----------------------------------------------------------------------')
        i += 1

    plot_probability_density(rand_num_arr, params)
    plot_probability_distribution(rand_num_arr, params)
