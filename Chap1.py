# Chap1 Python代码
# 自动从chap1.html同步生成

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

x = np.arange(0, 10)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 概率分布列
for p, color, ls in [(0.1, 'C0', '-'), (0.2, 'C1', '--'),
                      (0.3, 'C2', '-.'), (0.5, 'C3', ':')]:
    axes[0].plot(x, stats.binom.pmf(x, 9, p), color=color, ls=ls,
                 label=f'p={p}')
axes[0].set_xlabel('k'); axes[0].set_ylabel('p(k)')
axes[0].set_title('二项分布概率质量函数'); axes[0].legend()

# 累积分布函数
for p, color, ls in [(0.1, 'C0', '-'), (0.2, 'C1', '--'),
                      (0.3, 'C2', '-.'), (0.5, 'C3', ':')]:
    axes[1].plot(x, stats.binom.cdf(x, 9, p), color=color, ls=ls,
                 label=f'p={p}')
axes[1].set_xlabel('k'); axes[1].set_ylabel('F(k)')
axes[1].set_title('二项分布累积分布函数'); axes[1].legend()
plt.tight_layout(); plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

x = np.arange(0, 11)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for lam, color, ls in [(1, 'C0', '-'), (2, 'C1', '--'),
                        (3, 'C2', '-.'), (5, 'C3', ':')]:
    axes[0].plot(x, stats.poisson.pmf(x, lam), color=color, ls=ls,
                 label=f'λ={lam}')
axes[0].set_xlabel('k'); axes[0].set_ylabel('p(k)')
axes[0].set_title('泊松分布概率质量函数'); axes[0].legend()

for lam, color, ls in [(1, 'C0', '-'), (2, 'C1', '--'),
                        (3, 'C2', '-.'), (5, 'C3', ':')]:
    axes[1].plot(x, stats.poisson.cdf(x, lam), color=color, ls=ls,
                 label=f'λ={lam}')
axes[1].set_xlabel('k'); axes[1].set_ylabel('F(k)')
axes[1].set_title('泊松分布累积分布函数'); axes[1].legend()
plt.tight_layout(); plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

x = np.arange(0, 11)
fig, ax = plt.subplots(figsize=(8, 5))

for beta, color, ls in [(0.1, 'C0', '-'), (0.2, 'C1', '--'),
                         (0.3, 'C2', '-.'), (0.5, 'C3', ':')]:
    p = 1 / (1 + beta)
    ax.plot(x, stats.nbinom.pmf(x, 2, p), color=color, ls=ls,
            label=f'β={beta}')
ax.set_xlabel('k'); ax.set_ylabel('p(k)')
ax.set_title('负二项分布概率质量函数'); ax.legend()
plt.tight_layout(); plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

x = np.arange(0, 11)
p_nb = stats.nbinom.pmf(x, 4, 0.7)
p0 = p_nb[0]

# 零截断
pt = np.zeros_like(p_nb)
pt[1:] = p_nb[1:] / (1 - p0)

# 零调整
pm0 = 0.3
pm = np.zeros_like(p_nb)
pm[0] = pm0
pm[1:] = (1 - pm0) * p_nb[1:] / (1 - p0)

fig, ax = plt.subplots(figsize=(10, 5))
width = 0.35
ax.bar(x - width/2, p_nb, width, label='负二项', alpha=0.7)
ax.bar(x + width/2, pt, width, label='零截断负二项', alpha=0.7)
ax.set_xlabel('k'); ax.set_ylabel('概率')
ax.set_title('负二项分布与零截断负二项分布')
ax.set_xticks(x); ax.legend()
plt.tight_layout(); plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

x = np.linspace(0, 5, 500)
fig, ax = plt.subplots(figsize=(8, 5))
for rate, color, ls in [(0.5, 'C0', '-'), (1, 'C1', '--'),
                         (2, 'C2', '-.'), (5, 'C3', ':')]:
    ax.plot(x, stats.expon.pdf(x, scale=1/rate), color=color, ls=ls,
            label=f'rate={rate}')
ax.set_xlabel('x'); ax.set_ylabel('f(x)')
ax.set_title('指数分布概率密度函数'); ax.legend()
plt.tight_layout(); plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

x = np.linspace(0, 4, 1000)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for shape, color, ls in [(1, 'C0', '-'), (2, 'C1', '--'),
                          (3, 'C2', '-.'), (0.5, 'C3', ':')]:
    axes[0].plot(x, stats.gamma.pdf(x, shape, scale=1), color=color, ls=ls,
                 label=f'shape={shape}')
axes[0].set_xlabel('x'); axes[0].set_ylabel('f(x)')
axes[0].set_title('伽马分布（改变形状参数）'); axes[0].legend()

for scale, color, ls in [(1, 'C0', '-'), (2, 'C1', '--'),
                          (3, 'C2', '-.'), (0.5, 'C3', ':')]:
    axes[1].plot(x, stats.gamma.pdf(x, 1, scale=scale), color=color, ls=ls,
                 label=f'scale={scale}')
axes[1].set_xlabel('x'); axes[1].set_ylabel('f(x)')
axes[1].set_title('伽马分布（改变尺度参数）'); axes[1].legend()
plt.tight_layout(); plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

x = np.linspace(0.01, 5, 500)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 改变均值参数
for mu, color, ls in [(1, 'C0', '-'), (2, 'C1', '--'),
                       (5, 'C2', '-.'), (0.5, 'C3', ':')]:
    lam = 1
    y = np.sqrt(lam/(2*np.pi*x**3)) * np.exp(-lam*(x-mu)**2/(2*mu**2*x))
    axes[0].plot(x, y, color=color, ls=ls, label=f'mu={mu}')
axes[0].set_xlabel('x'); axes[0].set_ylabel('f(x)')
axes[0].set_title('逆高斯分布（改变均值）'); axes[0].legend()

# 改变形状参数
for lam, color, ls in [(1, 'C0', '-'), (2, 'C1', '--'),
                        (5, 'C2', '-.'), (0.5, 'C3', ':')]:
    mu = 1
    y = np.sqrt(lam/(2*np.pi*x**3)) * np.exp(-lam*(x-mu)**2/(2*mu**2*x))
    axes[1].plot(x, y, color=color, ls=ls, label=f'shape={lam}')
axes[1].set_xlabel('x'); axes[1].set_ylabel('f(x)')
axes[1].set_title('逆高斯分布（改变形状）'); axes[1].legend()
plt.tight_layout(); plt.show()

import numpy as np
import matplotlib.pyplot as plt

def pareto_pdf(x, alpha, lam):
    return alpha * lam**alpha / (lam + x)**(alpha + 1)

x = np.linspace(0, 3, 300)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for alpha, color, ls in [(1, 'C0', '-'), (2, 'C1', '--'),
                          (5, 'C2', '-.'), (0.5, 'C3', ':')]:
    axes[0].plot(x, pareto_pdf(x, alpha, 3), color=color, ls=ls,
                 label=f'α={alpha}')
axes[0].set_xlabel('x'); axes[0].set_ylabel('f(x)')
axes[0].set_title('帕累托分布（改变形状参数）'); axes[0].legend()

for lam, color, ls in [(1, 'C0', '-'), (2, 'C1', '--'),
                        (5, 'C2', '-.'), (0.5, 'C3', ':')]:
    axes[1].plot(x, pareto_pdf(x, 2, lam), color=color, ls=ls,
                 label=f'λ={lam}')
axes[1].set_xlabel('x'); axes[1].set_ylabel('f(x)')
axes[1].set_title('帕累托分布（改变尺度参数）'); axes[1].legend()
plt.tight_layout(); plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

x = np.linspace(0, 7, 500)
fig, ax = plt.subplots(figsize=(8, 5))
for sdlog, color, ls in [(0.5, 'C0', '-'), (1, 'C1', '--'),
                          (3, 'C2', '-.'), (10, 'C3', ':')]:
    ax.plot(x, stats.lognorm.pdf(x, s=sdlog, scale=np.exp(1)),
            color=color, ls=ls, label=f'sdlog={sdlog}')
ax.set_xlabel('x'); ax.set_ylabel('f(x)')
ax.set_title('对数正态分布'); ax.legend()
plt.tight_layout(); plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

x = np.linspace(0, 3, 500)
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for shape, color, ls in [(1, 'C0', '-'), (2, 'C1', '--'),
                          (3, 'C2', '-.'), (0.5, 'C3', ':')]:
    axes[0].plot(x, stats.weibull_min.pdf(x, shape, scale=1),
                 color=color, ls=ls, label=f'shape={shape}')
axes[0].set_xlabel('x'); axes[0].set_ylabel('f(x)')
axes[0].set_title('威布尔分布（改变形状参数）'); axes[0].legend()

for scale, color, ls in [(1, 'C0', '-'), (2, 'C1', '--'),
                          (3, 'C2', '-.'), (0.5, 'C3', ':')]:
    axes[1].plot(x, stats.weibull_min.pdf(x, 1, scale=scale),
                 color=color, ls=ls, label=f'scale={scale}')
axes[1].set_xlabel('x'); axes[1].set_ylabel('f(x)')
axes[1].set_title('威布尔分布（改变尺度参数）'); axes[1].legend()
plt.tight_layout(); plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

x = np.linspace(0.001, 1, 1000)
f_mix = 0.3 * stats.lognorm.pdf(x, s=2, scale=np.exp(1)) + \
        0.7 * stats.lognorm.pdf(x, s=4, scale=np.exp(3))
f1 = stats.lognorm.pdf(x, s=2, scale=np.exp(1))
f2 = stats.lognorm.pdf(x, s=4, scale=np.exp(3))

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, f_mix, 'r-', lw=2, label='混合分布')
ax.plot(x, f1, 'b--', label='LN(1,2)')
ax.plot(x, f2, 'g-.', label='LN(3,4)')
ax.set_xlabel('x'); ax.set_ylabel('f(x)')
ax.set_title('混合分布'); ax.legend()
ax.set_ylim(0, 2)
plt.tight_layout(); plt.show()

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(123)
x = np.random.gamma(shape=2, scale=1/2.5, size=100)

# MLE
shape_mle, loc_mle, scale_mle = stats.gamma.fit(x, floc=0)
print(f"伽马分布MLE: shape={shape_mle:.4f}, scale={scale_mle:.4f}, rate={1/scale_mle:.4f}")

# 矩估计
xbar = np.mean(x)
s2 = np.var(x, ddof=1)
shape_mme = xbar**2 / s2
scale_mme = s2 / xbar
print(f"伽马分布MME: shape={shape_mme:.4f}, scale={scale_mme:.4f}")

# 绘图
fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(x, bins=20, density=True, alpha=0.5, label='数据')
xx = np.linspace(0, max(x), 200)
ax.plot(xx, stats.gamma.pdf(xx, shape_mle, scale=scale_mle),
        'r-', lw=2, label='MLE拟合')
ax.plot(xx, stats.gamma.pdf(xx, shape_mme, scale=scale_mme),
        'b--', lw=2, label='MME拟合')
ax.set_xlabel('x'); ax.set_ylabel('f(x)')
ax.set_title('伽马分布拟合'); ax.legend()
plt.tight_layout(); plt.show()

import numpy as np
from scipy import stats

# 卡方拟合优度检验：维修费用数据
lower = np.array([0, 1000, 2000, 3000, 4000, 5000])
upper = np.array([1000, 2000, 3000, 4000, 5000, np.inf])
freq = np.array([250, 300, 250, 150, 100, 0])

# 指数分布MLE
midpoints = np.array([500, 1500, 2500, 3500, 4500, 5500])
n_total = freq.sum()
xbar = np.sum(freq * midpoints) / n_total
lambda_hat = 1 / xbar

# 期望频数
p_exp = stats.expon.cdf(upper, scale=1/lambda_hat) - \
        stats.expon.cdf(lower, scale=1/lambda_hat)
exp_freq = n_total * p_exp

# 卡方统计量
chi_sq = np.sum((freq - exp_freq)**2 / exp_freq)
df = len(freq) - 1 - 1
p_value = 1 - stats.chi2.cdf(chi_sq, df)
print(f"卡方拟合优度检验（指数分布）:")
print(f"  lambda_hat = {lambda_hat:.6f}")
print(f"  chi_sq = {chi_sq:.4f}, df = {df}, p-value = {p_value:.4f}")

# 泊松分布检验
X = np.arange(0, 7)
Y = np.array([7, 10, 12, 8, 3, 2, 0])
lambda_hat_pois = np.sum(X * Y) / np.sum(Y)

p_pois = np.zeros(len(Y))
p_pois[0] = stats.poisson.cdf(0, lambda_hat_pois)
p_pois[-1] = 1 - stats.poisson.cdf(len(Y)-2, lambda_hat_pois)
for i in range(1, len(Y)-1):
    p_pois[i] = stats.poisson.cdf(X[i], lambda_hat_pois) - \
                stats.poisson.cdf(X[i]-1, lambda_hat_pois)

exp_freq_pois = np.sum(Y) * p_pois
chi_sq_pois = np.sum((Y - exp_freq_pois)**2 / exp_freq_pois)
df_pois = len(Y) - 1 - 1
p_value_pois = 1 - stats.chi2.cdf(chi_sq_pois, df_pois)
print(f"\n卡方拟合优度检验（泊松分布）:")
print(f"  lambda_hat = {lambda_hat_pois:.4f}")
print(f"  chi_sq = {chi_sq_pois:.4f}, df = {df_pois}, p-value = {p_value_pois:.4f}")