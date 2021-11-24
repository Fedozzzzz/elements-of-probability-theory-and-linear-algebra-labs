import math
from scipy import stats
from sklearn.linear_model import LinearRegression
import numpy as np
import statistics as s
from prettytable import PrettyTable
import matplotlib.pyplot as plt

x = [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55]
y = [2.43, 2.67, 2.71, 3.15, 3.47, 3.76, 3.91, 4.46, 4.76, 5.15, 5.54, 5.61]
# x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])
# y = np.array([0.0, 0.23, 0.32, 0.24, 0.35, 0.77, 0.68, 0.92, 0.97, 1.08, 1.15, 1.37])

x_sm = s.mean(x)
# x_sm = np.mean(x)

xi_xsm = [x[i] - x_sm for i in range(len(x))]
xi_xsm2 = [xi_xsm[i] ** 2 for i in range(len(xi_xsm))]

# y_sm = np.mean(y)
y_sm = s.mean(y)

yi_ysm = [y[i] - y_sm for i in range(len(y))]
yi_ysm2 = [el ** 2 for el in yi_ysm]
xi_xsm_yi_ysm2 = [xi_xsm[i] * yi_ysm[i] for i in range(len(yi_ysm))]

# print('x_sm={}'.format(x_sm)))

pt = PrettyTable()
pt.add_column('x', x)
pt.add_column('y', y)
pt.add_column('xi-x_sm', xi_xsm)
pt.add_column('yi-y_sm', yi_ysm)
pt.add_column('(xi-x_sm)^2', xi_xsm2)
pt.add_column('(xi-x_sm)(yi-y_sm)', xi_xsm_yi_ysm2)

print(pt.get_string())

lin_reg = LinearRegression()
lin_reg.fit(np.array(x).reshape(-1, 1), y)

alpha = y_sm
beta = sum(xi_xsm_yi_ysm2) / sum(xi_xsm2)

# print('beta ', beta)
print('Уравнение регрессии: y_observed={}-{}*(x-{})'.format(alpha, beta, x_sm))
y_observed = [alpha + beta * xi_xsm[i] for i in range(len(xi_xsm))]
yi_yo = [y[i] - y_observed[i] for i in range(len(y))]
yi_yo2 = [yi_yo[i] ** 2 for i in range(len(yi_yo))]

pt = PrettyTable()
pt.add_column('x', x)
pt.add_column('y', y)
pt.add_column('y_obs_i', y_observed)
pt.add_column('yi-y_obs_i', yi_yo)
pt.add_column('(yi-y_obs_i)**2', yi_yo2)

print(pt.get_string())

fig, ax = plt.subplots()
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.plot(x, y, 'o', color='blue', label='y')
# plt.title('Полученные данные')
# plt.ylim(0.0, 1.0)
# plt.xlim(0, 10)
# ax.grid()
# ax.legend()
# plt.show()

# fig, ax = plt.subplots()
# ax.set_xlabel('x')
# ax.set_ylabel('y_observed')
ax.plot(x, y_observed, 'o', color='orange', label='y_observed')
# plt.title('Полученные данные')
# plt.ylim(0.0, 1.0)
# plt.xlim(0, 10)
ax.grid()
ax.legend()
plt.show()

# D_o = s.mean(yi_yo2)
D_o = np.mean(yi_ysm2)
sigma_o = math.sqrt(D_o)
Da = D_o / len(yi_ysm2)
Db = D_o / sum(xi_xsm2)

# print('D(y)={}*(x-{})'.format(alpha, beta, x_sm))

beta_level = 0.9
alpha_level = 1 - beta_level
n = len(x)
t = 2.228
# t = stats.t.ppf(alpha_level, n - 2)

print("t={}".format(t))
# print("Do={}".format(t))
# print("t={}".format(stats.t.ppf(beta, n - 2)))
e_a = t * math.sqrt(D_o / (n - 2))
# e_a = stats.t.ppf(beta_level, n - 2) * math.sqrt(D_o / (n - 2))
e_b = t * math.sqrt((len(x) / (n - 2)) * (D_o / sum(xi_xsm2)))

print('eps_a={}, eps_b={}'.format(e_a, e_b))


def get_eps_y_i(xi_xsm2_i):
    return t * (sigma_o / math.sqrt(n - 2) * math.sqrt(1 + (n * xi_xsm2_i / sum(xi_xsm2))))


x_copy = x.copy()
x_copy[5] = x_sm
# e_y = [get_eps_y_i(xi_xsm2_i) for xi_xsm2_i in xi_xsm2]
s2 = [(el - x_sm) ** 2 for el in x_copy]
# s2 = [(el - x_sm) ** 2 for el in x]
# e_y = [t * (sigma_o / math.sqrt(n - 2) * math.sqrt(1 + (n * ((el - x_sm) ** 2) / sum(s2)))) for el in x]
e_y = [t * (sigma_o / math.sqrt(n - 2) * math.sqrt(1 + (n * ((el - x_sm) ** 2) / sum(s2)))) for el in x_copy]
# e_y = [t * sigma_o / math.sqrt(len(x)-2) * math.sqrt(1 + len(x) * (xx - x_sm) ** 2 / sum(s2)) for xx in x_copy]

pt = PrettyTable()
pt.add_column('x', x)
pt.add_column('eps_y', e_y)

print(pt.get_string())

inter = lin_reg.intercept_

Ia_left = alpha - e_a
Ia_right = alpha + e_a

c = lin_reg.coef_[0]

Ib_left = beta - e_a
Ib_right = beta + e_a

print('Доверительный интервал Ia: ({}; {})'.format(Ia_left, Ia_right))
print('Доверительный интервал Ib: ({}; {})'.format(Ib_left, Ib_right))

# graph_up = [lin_reg.predict(x[i].reshape(-1, 1))[0] + e_y[i] for i in range(len(x))]
# graph_down = [lin_reg.predict(x[i].reshape(-1, 1))[0] - e_y[i] for i in range(len(x))]

graph_up = [y_observed[i] + e_y[i] for i in range(len(y_observed))]
graph_down = [y_observed[i] - e_y[i] for i in range(len(y_observed))]

# print(graph)
# graph = lin_reg.predict(x.reshape(-1, 1))
fig, ax = plt.subplots()
# ax.set_xlabel('x')
# ax.set_ylabel('y')
# ax.plot(x, y, 'o', color='blue', label='y')
ax.plot(x, y, color='blue', label='y')
ax.plot(x, y_observed, color='red')
ax.plot(x, graph_up, color='black')
ax.plot(x, graph_down, color='black')
# ax.plot(x, graph + e_y, color='red')
# ax.plot(x, graph - e_y, color='red')
plt.title('Наблюдаемая и регрессионная зависимости с доверительной полосой')
plt.grid()
plt.show()

# критерий Пирса
row_mean = []
x_pirs = []
y_pirs = []
for i in range(len(x)):
    row_mean.append(x[i]+y[i])
s = sum(row_mean) + sum(x) + sum(y)
for i in range(len(x)):
    x_pirs.append(row_mean[i]*sum(x)/s)
    y_pirs.append(row_mean[i] * sum(y) / s)
chi_observe = 0
for i in range(len(x)):
    chi_observe = chi_observe + (x[i]-x_pirs[i])**2/x_pirs[i] + (y[i]-y_pirs[i])**2/y_pirs[i]
print("Полученное значение критерия Пирсона: ", chi_observe)
print("Критическое значение критерия Пирсона: ", stats.chi2.ppf(alpha_level, (len(x)-1)*(len(y)-1)))