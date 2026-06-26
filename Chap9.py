# Chap9 Python代码
# 自动从chap9.html同步生成

# 导入所需的Python模块
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

print("正态分布的风险度量:")
for p in p_values:
    VaR = stats.norm.ppf(p, mu, sigma)
    z = stats.norm.ppf(p)
    TVaR = mu + sigma * stats.norm.pdf(z) / (1 - p)
    print(f"  p={p}: VaR={VaR:.4f}, TVaR={TVaR:.4f}")

print("\n对数正态分布的风险度量:")
for p in p_values:
    VaR = stats.lognorm.ppf(p, s=1, scale=np.exp(0))
    print(f"  p={p}: VaR={VaR:.4f}")


##############################################################################
# 9.2 广义极值分布（GEV）
##############################################################################
def gev_pdf(x, mu, sigma, xi):
    """GEV概率密度函数"""
    z = (x - mu) / sigma
    if abs(xi) < 1e-8:
        t = np.exp(-z)
        return (1/sigma) * t * np.exp(-t)
    else:
        if np.any(1 + xi * z <= 0):
            return 0
        t = (1 + xi * z) ** (-1/xi)
        return (1/sigma) * t**(xi + 1) * np.exp(-t)

x = np.linspace(-4, 4, 100)
fig, ax = plt.subplots(figsize=(8, 5))
for xi, label, color in [(0.5, 'ξ=0.5 (Frechet)', 'C0'),
                          (0, 'ξ=0 (Gumbel)', 'C1'),

##############################################################################
# 第9章 极值理论
# 对应教材：section9.tex
# 内容：风险度量、广义极值分布（GEV）、广义帕累托分布（GPD）、
#       阈值选择、参数估计、应用
##############################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


##############################################################################
# 9.1 风险度量：VaR与TVaR
##############################################################################
mu, sigma = 0, 1
p_values = [0.95, 0.99, 0.999]

print("正态分布的风险度量:")
for p in p_values:
    VaR = stats.norm.ppf(p, mu, sigma)
    z = stats.norm.ppf(p)
    TVaR = mu + sigma * stats.norm.pdf(z) / (1 - p)
    print(f"  p={p}: VaR={VaR:.4f}, TVaR={TVaR:.4f}")

print("\n对数正态分布的风险度量:")
for p in p_values:
    VaR = stats.lognorm.ppf(p, s=1, scale=np.exp(0))
    print(f"  p={p}: VaR={VaR:.4f}")


##############################################################################
# 9.2 广义极值分布（GEV）
##############################################################################
def gev_pdf(x, mu, sigma, xi):
    """GEV概率密度函数"""
    z = (x - mu) / sigma
    if abs(xi) < 1e-8:
        t = np.exp(-z)
        return (1/sigma) * t * np.exp(-t)
    else:
        if np.any(1 + xi * z <= 0):
            return 0
        t = (1 + xi * z) ** (-1/xi)
        return (1/sigma) * t**(xi + 1) * np.exp(-t)

x = np.linspace(-4, 4, 100)
fig, ax = plt.subplots(figsize=(8, 5))
for xi, label, color in [(0.5, 'ξ=0.5 (Frechet)', 'C0'),
                          (0, 'ξ=0 (Gumbel)', 'C1'),
                          (-0.5, 'ξ=-0.5 (Weibull)', 'C2')]:
    f = np.array([gev_pdf(xi_, 0, 1, xi) for xi_ in x])
    ax.plot(x, f, color=color, label=label)
ax.set_xlabel('x'); ax.set_ylabel('f(x)')
ax.set_title('GEV分布密度函数'); ax.legend()
plt.tight_layout(); plt.show()


##############################################################################
# 9.3 区块最大值法（Block Maxima）
##############################################################################
np.random.seed(123)
n_blocks = 100
block_size = 50

maxima = np.zeros(n_blocks)
for b in range(n_blocks):
    x_block = np.random.exponential(1, block_size)
    maxima[b] = x_block.max()

print(f"\n区块最大值法:")
print(f"  均值={maxima.mean():.4f}, 标准差={maxima.std():.4f}, 最大值={maxima.max():.4f}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(maxima, bins=20, density=True, alpha=0.5)
kde_x = np.linspace(maxima.min(), maxima.max(), 200)
from scipy.stats import gaussian_kde
kde = gaussian_kde(maxima)
ax.plot(kde_x, kde(kde_x), 'r-')
ax.set_title('区块最大值直方图')
plt.tight_layout(); plt.show()


##############################################################################
# 9.4 GEV参数估计
##############################################################################
def gev_nll(parms, x):
    mu = parms[0]
    sigma = np.exp(parms[1])
    xi = parms[2]
    z = (x - mu) / sigma
    if np.any(1 + xi * z <= 0):
        return 1e10
    if abs(xi) < 1e-8:
        t = np.exp(-z)
    else:
        t = (1 + xi * z) ** (-1/xi)
    return np.sum(np.log(sigma) + (1 + 1/xi) * np.log(t) + t)

result_gev = minimize(gev_nll, x0=[3, 0, 0.1], args=(maxima,),
                       method='Nelder-Mead')
mu_hat = result_gev.x[0]
sigma_hat = np.exp(result_gev.x[1])
xi_hat = result_gev.x[2]
print(f"\nGEV参数MLE估计:")
print(f"  μ_hat={mu_hat:.4f}, σ_hat={sigma_hat:.4f}, ξ_hat={xi_hat:.4f}")


##############################################################################
# 9.5 广义帕累托分布（GPD）
##############################################################################
def gpd_pdf(x, sigma, xi):
    """GPD概率密度函数"""
    if abs(xi) < 1e-8:
        return (1/sigma) * np.exp(-x/sigma)
    else:
        if np.any(1 + xi * x / sigma <= 0):
            return 0
        return (1/sigma) * (1 + xi * x / sigma) ** (-1/xi - 1)

x = np.linspace(0, 5, 100)
fig, ax = plt.subplots(figsize=(8, 5))
for xi, label, color in [(0.5, 'ξ=0.5', 'C0'),
                          (0, 'ξ=0', 'C1'),
                          (-0.5, 'ξ=-0.5', 'C2')]:
    f = np.array([gpd_pdf(xi_, 1, xi) for xi_ in x])
    ax.plot(x, f, color=color, label=label)
ax.set_xlabel('x'); ax.set_ylabel('f(x)')
ax.set_title('GPD分布密度函数'); ax.legend()
plt.tight_layout(); plt.show()


##############################################################################
# 9.6 阈值选择（POT方法）
##############################################################################
np.random.seed(456)
n = 1000
x_pot = np.random.exponential(1, n)

# 平均超出量图
u_grid = np.linspace(0, 3, 50)
me = np.zeros_like(u_grid)
for i, u in enumerate(u_grid):
    excess = x_pot[x_pot > u] - u
    if len(excess) > 0:
        me[i] = excess.mean()

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(u_grid, me)
ax.set_xlabel('阈值 u'); ax.set_ylabel('平均超出量')
ax.set_title('平均超出量图')
plt.tight_layout(); plt.show()

# 不同阈值下的超出量
print(f"\n不同阈值下的超出量:")
for u in [0.5, 1.0, 1.5, 2.0, 2.5]:
    excess = x_pot[x_pot > u] - u
    print(f"  u={u}: 超出数={len(excess)}, 平均超出量={excess.mean():.4f}")


##############################################################################
# 9.7 GPD参数估计
##############################################################################
np.random.seed(789)
n = 1000
sigma_true = 1
xi_true = 0.3

u_gpd = np.random.uniform(0, 1, n)
if abs(xi_true) < 1e-8:
    x_gpd = -sigma_true * np.log(1 - u_gpd)
else:
    x_gpd = sigma_true / xi_true * ((1 - u_gpd)**(-xi_true) - 1)

def gpd_nll(parms, x):
    sigma = np.exp(parms[0])
    xi = parms[1]
    if np.any(1 + xi * x / sigma <= 0):
        return 1e10
    if abs(xi) < 1e-8:
        return np.sum(np.log(sigma) + x / sigma)
    return np.sum(np.log(sigma) + (1 + 1/xi) * np.log(1 + xi * x / sigma))

result_gpd = minimize(gpd_nll, x0=[0, 0.3], args=(x_gpd,),
                       method='Nelder-Mead')
sigma_hat = np.exp(result_gpd.x[0])
xi_hat = result_gpd.x[1]
print(f"\nGPD参数MLE估计:")
print(f"  σ_true={sigma_true}, σ_hat={sigma_hat:.4f}")
print(f"  ξ_true={xi_true}, ξ_hat={xi_hat:.4f}")


##############################################################################
# 9.8 极值理论应用：VaR估计
##############################################################################
np.random.seed(101)
n = 1000
x_var = np.random.exponential(1, n)

# 选择阈值
u = 2
excess = x_var[x_var > u] - u
n_excess = len(excess)
n_total = len(x_var)

# GPD参数估计（矩估计）
xbar = excess.mean()
s2 = excess.var()
xi_hat_var = 0.5 * (xbar**2 / s2 - 1)
sigma_hat_var = 0.5 * xbar * (xbar**2 / s2 + 1)

print(f"\nVaR估计:")
print(f"  阈值u={u}, 超出数={n_excess}")
print(f"  ξ_hat={xi_hat_var:.4f}, σ_hat={sigma_hat_var:.4f}")

for p in [0.95, 0.99, 0.999]:
    VaR_gpd = u + sigma_hat_var / xi_hat_var * \
              ((n_total / n_excess * (1 - p))**(-xi_hat_var) - 1)
    VaR_emp = np.percentile(x_var, p * 100)
    print(f"  p={p}: VaR_GPD={VaR_gpd:.4f}, VaR_emp={VaR_emp:.4f}")


##############################################################################
# 9.9 Danish火灾损失数据应用
##############################################################################
np.random.seed(42)
# 模拟Danish数据（对数正态）
danish = np.random.lognormal(0.672, 0.732, 2167)
danish = danish[danish > 1]

# 使用POT方法
u_danish = np.percentile(danish, 95)
excess_danish = danish[danish > u_danish] - u_danish
n_excess_d = len(excess_danish)
n_total_d = len(danish)

# GPD参数估计
xbar_d = excess_danish.mean()
s2_d = excess_danish.var()
xi_hat_d = 0.5 * (xbar_d**2 / s2_d - 1)
sigma_hat_d = 0.5 * xbar_d * (xbar_d**2 / s2_d + 1)

print(f"\nDanish火灾损失POT分析:")
print(f"  阈值u={u_danish:.2f}, 超出数={n_excess_d}")
print(f"  ξ_hat={xi_hat_d:.4f}, σ_hat={sigma_hat_d:.4f}")

for p in [0.95, 0.99, 0.999]:
    VaR_gpd = u_danish + sigma_hat_d / xi_hat_d * \
              ((n_total_d / n_excess_d * (1 - p))**(-xi_hat_d) - 1)
    VaR_emp = np.percentile(danish, p * 100)
    print(f"  p={p}: VaR_GPD={VaR_gpd:.2f}, VaR_emp={VaR_emp:.2f}")

##############################################################################
# 第9章 极值理论
# 对应教材：section9.tex
# 内容：风险度量、广义极值分布（GEV）、广义帕累托分布（GPD）、
#       阈值选择、参数估计、应用
##############################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


##############################################################################
# 9.1 风险度量：VaR与TVaR
##############################################################################
mu, sigma = 0, 1
p_values = [0.95, 0.99, 0.999]

print("正态分布的风险度量:")
for p in p_values:
    VaR = stats.norm.ppf(p, mu, sigma)
    z = stats.norm.ppf(p)
    TVaR = mu + sigma * stats.norm.pdf(z) / (1 - p)
    print(f"  p={p}: VaR={VaR:.4f}, TVaR={TVaR:.4f}")

print("\n对数正态分布的风险度量:")
for p in p_values:
    VaR = stats.lognorm.ppf(p, s=1, scale=np.exp(0))
    print(f"  p={p}: VaR={VaR:.4f}")


##############################################################################
# 9.2 广义极值分布（GEV）
##############################################################################
def gev_pdf(x, mu, sigma, xi):
    """GEV概率密度函数"""
    z = (x - mu) / sigma
    if abs(xi) < 1e-8:
        t = np.exp(-z)
        return (1/sigma) * t * np.exp(-t)
    else:
        if np.any(1 + xi * z <= 0):
            return 0
        t = (1 + xi * z) ** (-1/xi)
        return (1/sigma) * t**(xi + 1) * np.exp(-t)

x = np.linspace(-4, 4, 100)
fig, ax = plt.subplots(figsize=(8, 5))
for xi, label, color in [(0.5, 'ξ=0.5 (Frechet)', 'C0'),
                          (0, 'ξ=0 (Gumbel)', 'C1'),
                          (-0.5, 'ξ=-0.5 (Weibull)', 'C2')]:
    f = np.array([gev_pdf(xi_, 0, 1, xi) for xi_ in x])
    ax.plot(x, f, color=color, label=label)
ax.set_xlabel('x'); ax.set_ylabel('f(x)')
ax.set_title('GEV分布密度函数'); ax.legend()
plt.tight_layout(); plt.show()


##############################################################################
# 9.3 区块最大值法（Block Maxima）
##############################################################################
np.random.seed(123)
n_blocks = 100
block_size = 50

maxima = np.zeros(n_blocks)
for b in range(n_blocks):
    x_block = np.random.exponential(1, block_size)
    maxima[b] = x_block.max()

print(f"\n区块最大值法:")
print(f"  均值={maxima.mean():.4f}, 标准差={maxima.std():.4f}, 最大值={maxima.max():.4f}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(maxima, bins=20, density=True, alpha=0.5)
kde_x = np.linspace(maxima.min(), maxima.max(), 200)
from scipy.stats import gaussian_kde
kde = gaussian_kde(maxima)
ax.plot(kde_x, kde(kde_x), 'r-')
ax.set_title('区块最大值直方图')
plt.tight_layout(); plt.show()


##############################################################################
# 9.4 GEV参数估计
##############################################################################
def gev_nll(parms, x):
    mu = parms[0]
    sigma = np.exp(parms[1])
    xi = parms[2]
    z = (x - mu) / sigma
    if np.any(1 + xi * z <= 0):
        return 1e10
    if abs(xi) < 1e-8:
        t = np.exp(-z)
    else:
        t = (1 + xi * z) ** (-1/xi)
    return np.sum(np.log(sigma) + (1 + 1/xi) * np.log(t) + t)

result_gev = minimize(gev_nll, x0=[3, 0, 0.1], args=(maxima,),
                       method='Nelder-Mead')
mu_hat = result_gev.x[0]
sigma_hat = np.exp(result_gev.x[1])
xi_hat = result_gev.x[2]
print(f"\nGEV参数MLE估计:")
print(f"  μ_hat={mu_hat:.4f}, σ_hat={sigma_hat:.4f}, ξ_hat={xi_hat:.4f}")


##############################################################################
# 9.5 广义帕累托分布（GPD）
##############################################################################
def gpd_pdf(x, sigma, xi):
    """GPD概率密度函数"""
    if abs(xi) < 1e-8:
        return (1/sigma) * np.exp(-x/sigma)
    else:
        if np.any(1 + xi * x / sigma <= 0):
            return 0
        return (1/sigma) * (1 + xi * x / sigma) ** (-1/xi - 1)

x = np.linspace(0, 5, 100)
fig, ax = plt.subplots(figsize=(8, 5))
for xi, label, color in [(0.5, 'ξ=0.5', 'C0'),
                          (0, 'ξ=0', 'C1'),
                          (-0.5, 'ξ=-0.5', 'C2')]:
    f = np.array([gpd_pdf(xi_, 1, xi) for xi_ in x])
    ax.plot(x, f, color=color, label=label)
ax.set_xlabel('x'); ax.set_ylabel('f(x)')
ax.set_title('GPD分布密度函数'); ax.legend()
plt.tight_layout(); plt.show()


##############################################################################
# 9.6 阈值选择（POT方法）
##############################################################################
np.random.seed(456)
n = 1000
x_pot = np.random.exponential(1, n)

# 平均超出量图
u_grid = np.linspace(0, 3, 50)
me = np.zeros_like(u_grid)
for i, u in enumerate(u_grid):
    excess = x_pot[x_pot > u] - u
    if len(excess) > 0:
        me[i] = excess.mean()

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(u_grid, me)
ax.set_xlabel('阈值 u'); ax.set_ylabel('平均超出量')
ax.set_title('平均超出量图')
plt.tight_layout(); plt.show()

# 不同阈值下的超出量
print(f"\n不同阈值下的超出量:")
for u in [0.5, 1.0, 1.5, 2.0, 2.5]:
    excess = x_pot[x_pot > u] - u
    print(f"  u={u}: 超出数={len(excess)}, 平均超出量={excess.mean():.4f}")


##############################################################################
# 9.7 GPD参数估计
##############################################################################
np.random.seed(789)
n = 1000
sigma_true = 1
xi_true = 0.3

u_gpd = np.random.uniform(0, 1, n)
if abs(xi_true) < 1e-8:
    x_gpd = -sigma_true * np.log(1 - u_gpd)
else:
    x_gpd = sigma_true / xi_true * ((1 - u_gpd)**(-xi_true) - 1)

def gpd_nll(parms, x):
    sigma = np.exp(parms[0])
    xi = parms[1]
    if np.any(1 + xi * x / sigma <= 0):
        return 1e10
    if abs(xi) < 1e-8:
        return np.sum(np.log(sigma) + x / sigma)
    return np.sum(np.log(sigma) + (1 + 1/xi) * np.log(1 + xi * x / sigma))

result_gpd = minimize(gpd_nll, x0=[0, 0.3], args=(x_gpd,),
                       method='Nelder-Mead')
sigma_hat = np.exp(result_gpd.x[0])
xi_hat = result_gpd.x[1]
print(f"\nGPD参数MLE估计:")
print(f"  σ_true={sigma_true}, σ_hat={sigma_hat:.4f}")
print(f"  ξ_true={xi_true}, ξ_hat={xi_hat:.4f}")


##############################################################################
# 9.8 极值理论应用：VaR估计
##############################################################################
np.random.seed(101)
n = 1000
x_var = np.random.exponential(1, n)

# 选择阈值
u = 2
excess = x_var[x_var > u] - u
n_excess = len(excess)
n_total = len(x_var)

# GPD参数估计（矩估计）
xbar = excess.mean()
s2 = excess.var()
xi_hat_var = 0.5 * (xbar**2 / s2 - 1)
sigma_hat_var = 0.5 * xbar * (xbar**2 / s2 + 1)

print(f"\nVaR估计:")
print(f"  阈值u={u}, 超出数={n_excess}")
print(f"  ξ_hat={xi_hat_var:.4f}, σ_hat={sigma_hat_var:.4f}")

for p in [0.95, 0.99, 0.999]:
    VaR_gpd = u + sigma_hat_var / xi_hat_var * \
              ((n_total / n_excess * (1 - p))**(-xi_hat_var) - 1)
    VaR_emp = np.percentile(x_var, p * 100)
    print(f"  p={p}: VaR_GPD={VaR_gpd:.4f}, VaR_emp={VaR_emp:.4f}")


##############################################################################
# 9.9 Danish火灾损失数据应用
##############################################################################
np.random.seed(42)
# 模拟Danish数据（对数正态）
danish = np.random.lognormal(0.672, 0.732, 2167)
danish = danish[danish > 1]

# 使用POT方法
u_danish = np.percentile(danish, 95)
excess_danish = danish[danish > u_danish] - u_danish
n_excess_d = len(excess_danish)
n_total_d = len(danish)

# GPD参数估计
xbar_d = excess_danish.mean()
s2_d = excess_danish.var()
xi_hat_d = 0.5 * (xbar_d**2 / s2_d - 1)
sigma_hat_d = 0.5 * xbar_d * (xbar_d**2 / s2_d + 1)

print(f"\nDanish火灾损失POT分析:")
print(f"  阈值u={u_danish:.2f}, 超出数={n_excess_d}")
print(f"  ξ_hat={xi_hat_d:.4f}, σ_hat={sigma_hat_d:.4f}")

for p in [0.95, 0.99, 0.999]:
    VaR_gpd = u_danish + sigma_hat_d / xi_hat_d * \
              ((n_total_d / n_excess_d * (1 - p))**(-xi_hat_d) - 1)
    VaR_emp = np.percentile(danish, p * 100)
    print(f"  p={p}: VaR_GPD={VaR_gpd:.2f}, VaR_emp={VaR_emp:.2f}")

# 导入所需的Python模块
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

print("正态分布的风险度量:")
for p in p_values:
    VaR = stats.norm.ppf(p, mu, sigma)
    z = stats.norm.ppf(p)
    TVaR = mu + sigma * stats.norm.pdf(z) / (1 - p)
    print(f"  p={p}: VaR={VaR:.4f}, TVaR={TVaR:.4f}")

print("\n对数正态分布的风险度量:")
for p in p_values:
    VaR = stats.lognorm.ppf(p, s=1, scale=np.exp(0))
    print(f"  p={p}: VaR={VaR:.4f}")


##############################################################################
# 9.2 广义极值分布（GEV）
##############################################################################
def gev_pdf(x, mu, sigma, xi):
    """GEV概率密度函数"""
    z = (x - mu) / sigma
    if abs(xi) < 1e-8:
        t = np.exp(-z)
        return (1/sigma) * t * np.exp(-t)
    else:
        if np.any(1 + xi * z <= 0):
            return 0
        t = (1 + xi * z) ** (-1/xi)
        return (1/sigma) * t**(xi + 1) * np.exp(-t)

x = np.linspace(-4, 4, 100)
fig, ax = plt.subplots(figsize=(8, 5))
for xi, label, color in [(0.5, 'ξ=0.5 (Frechet)', 'C0'),
                          (0, 'ξ=0 (Gumbel)', 'C1'),

##############################################################################
# 第9章 极值理论
# 对应教材：section9.tex
# 内容：风险度量、广义极值分布（GEV）、广义帕累托分布（GPD）、
#       阈值选择、参数估计、应用
##############################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


##############################################################################
# 9.1 风险度量：VaR与TVaR
##############################################################################
mu, sigma = 0, 1
p_values = [0.95, 0.99, 0.999]

print("正态分布的风险度量:")
for p in p_values:
    VaR = stats.norm.ppf(p, mu, sigma)
    z = stats.norm.ppf(p)
    TVaR = mu + sigma * stats.norm.pdf(z) / (1 - p)
    print(f"  p={p}: VaR={VaR:.4f}, TVaR={TVaR:.4f}")

print("\n对数正态分布的风险度量:")
for p in p_values:
    VaR = stats.lognorm.ppf(p, s=1, scale=np.exp(0))
    print(f"  p={p}: VaR={VaR:.4f}")


##############################################################################
# 9.2 广义极值分布（GEV）
##############################################################################
def gev_pdf(x, mu, sigma, xi):
    """GEV概率密度函数"""
    z = (x - mu) / sigma
    if abs(xi) < 1e-8:
        t = np.exp(-z)
        return (1/sigma) * t * np.exp(-t)
    else:
        if np.any(1 + xi * z <= 0):
            return 0
        t = (1 + xi * z) ** (-1/xi)
        return (1/sigma) * t**(xi + 1) * np.exp(-t)

x = np.linspace(-4, 4, 100)
fig, ax = plt.subplots(figsize=(8, 5))
for xi, label, color in [(0.5, 'ξ=0.5 (Frechet)', 'C0'),
                          (0, 'ξ=0 (Gumbel)', 'C1'),
                          (-0.5, 'ξ=-0.5 (Weibull)', 'C2')]:
    f = np.array([gev_pdf(xi_, 0, 1, xi) for xi_ in x])
    ax.plot(x, f, color=color, label=label)
ax.set_xlabel('x'); ax.set_ylabel('f(x)')
ax.set_title('GEV分布密度函数'); ax.legend()
plt.tight_layout(); plt.show()


##############################################################################
# 9.3 区块最大值法（Block Maxima）
##############################################################################
np.random.seed(123)
n_blocks = 100
block_size = 50

maxima = np.zeros(n_blocks)
for b in range(n_blocks):
    x_block = np.random.exponential(1, block_size)
    maxima[b] = x_block.max()

print(f"\n区块最大值法:")
print(f"  均值={maxima.mean():.4f}, 标准差={maxima.std():.4f}, 最大值={maxima.max():.4f}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(maxima, bins=20, density=True, alpha=0.5)
kde_x = np.linspace(maxima.min(), maxima.max(), 200)
from scipy.stats import gaussian_kde
kde = gaussian_kde(maxima)
ax.plot(kde_x, kde(kde_x), 'r-')
ax.set_title('区块最大值直方图')
plt.tight_layout(); plt.show()


##############################################################################
# 9.4 GEV参数估计
##############################################################################
def gev_nll(parms, x):
    mu = parms[0]
    sigma = np.exp(parms[1])
    xi = parms[2]
    z = (x - mu) / sigma
    if np.any(1 + xi * z <= 0):
        return 1e10
    if abs(xi) < 1e-8:
        t = np.exp(-z)
    else:
        t = (1 + xi * z) ** (-1/xi)
    return np.sum(np.log(sigma) + (1 + 1/xi) * np.log(t) + t)

result_gev = minimize(gev_nll, x0=[3, 0, 0.1], args=(maxima,),
                       method='Nelder-Mead')
mu_hat = result_gev.x[0]
sigma_hat = np.exp(result_gev.x[1])
xi_hat = result_gev.x[2]
print(f"\nGEV参数MLE估计:")
print(f"  μ_hat={mu_hat:.4f}, σ_hat={sigma_hat:.4f}, ξ_hat={xi_hat:.4f}")


##############################################################################
# 9.5 广义帕累托分布（GPD）
##############################################################################
def gpd_pdf(x, sigma, xi):
    """GPD概率密度函数"""
    if abs(xi) < 1e-8:
        return (1/sigma) * np.exp(-x/sigma)
    else:
        if np.any(1 + xi * x / sigma <= 0):
            return 0
        return (1/sigma) * (1 + xi * x / sigma) ** (-1/xi - 1)

x = np.linspace(0, 5, 100)
fig, ax = plt.subplots(figsize=(8, 5))
for xi, label, color in [(0.5, 'ξ=0.5', 'C0'),
                          (0, 'ξ=0', 'C1'),
                          (-0.5, 'ξ=-0.5', 'C2')]:
    f = np.array([gpd_pdf(xi_, 1, xi) for xi_ in x])
    ax.plot(x, f, color=color, label=label)
ax.set_xlabel('x'); ax.set_ylabel('f(x)')
ax.set_title('GPD分布密度函数'); ax.legend()
plt.tight_layout(); plt.show()


##############################################################################
# 9.6 阈值选择（POT方法）
##############################################################################
np.random.seed(456)
n = 1000
x_pot = np.random.exponential(1, n)

# 平均超出量图
u_grid = np.linspace(0, 3, 50)
me = np.zeros_like(u_grid)
for i, u in enumerate(u_grid):
    excess = x_pot[x_pot > u] - u
    if len(excess) > 0:
        me[i] = excess.mean()

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(u_grid, me)
ax.set_xlabel('阈值 u'); ax.set_ylabel('平均超出量')
ax.set_title('平均超出量图')
plt.tight_layout(); plt.show()

# 不同阈值下的超出量
print(f"\n不同阈值下的超出量:")
for u in [0.5, 1.0, 1.5, 2.0, 2.5]:
    excess = x_pot[x_pot > u] - u
    print(f"  u={u}: 超出数={len(excess)}, 平均超出量={excess.mean():.4f}")


##############################################################################
# 9.7 GPD参数估计
##############################################################################
np.random.seed(789)
n = 1000
sigma_true = 1
xi_true = 0.3

u_gpd = np.random.uniform(0, 1, n)
if abs(xi_true) < 1e-8:
    x_gpd = -sigma_true * np.log(1 - u_gpd)
else:
    x_gpd = sigma_true / xi_true * ((1 - u_gpd)**(-xi_true) - 1)

def gpd_nll(parms, x):
    sigma = np.exp(parms[0])
    xi = parms[1]
    if np.any(1 + xi * x / sigma <= 0):
        return 1e10
    if abs(xi) < 1e-8:
        return np.sum(np.log(sigma) + x / sigma)
    return np.sum(np.log(sigma) + (1 + 1/xi) * np.log(1 + xi * x / sigma))

result_gpd = minimize(gpd_nll, x0=[0, 0.3], args=(x_gpd,),
                       method='Nelder-Mead')
sigma_hat = np.exp(result_gpd.x[0])
xi_hat = result_gpd.x[1]
print(f"\nGPD参数MLE估计:")
print(f"  σ_true={sigma_true}, σ_hat={sigma_hat:.4f}")
print(f"  ξ_true={xi_true}, ξ_hat={xi_hat:.4f}")


##############################################################################
# 9.8 极值理论应用：VaR估计
##############################################################################
np.random.seed(101)
n = 1000
x_var = np.random.exponential(1, n)

# 选择阈值
u = 2
excess = x_var[x_var > u] - u
n_excess = len(excess)
n_total = len(x_var)

# GPD参数估计（矩估计）
xbar = excess.mean()
s2 = excess.var()
xi_hat_var = 0.5 * (xbar**2 / s2 - 1)
sigma_hat_var = 0.5 * xbar * (xbar**2 / s2 + 1)

print(f"\nVaR估计:")
print(f"  阈值u={u}, 超出数={n_excess}")
print(f"  ξ_hat={xi_hat_var:.4f}, σ_hat={sigma_hat_var:.4f}")

for p in [0.95, 0.99, 0.999]:
    VaR_gpd = u + sigma_hat_var / xi_hat_var * \
              ((n_total / n_excess * (1 - p))**(-xi_hat_var) - 1)
    VaR_emp = np.percentile(x_var, p * 100)
    print(f"  p={p}: VaR_GPD={VaR_gpd:.4f}, VaR_emp={VaR_emp:.4f}")


##############################################################################
# 9.9 Danish火灾损失数据应用
##############################################################################
np.random.seed(42)
# 模拟Danish数据（对数正态）
danish = np.random.lognormal(0.672, 0.732, 2167)
danish = danish[danish > 1]

# 使用POT方法
u_danish = np.percentile(danish, 95)
excess_danish = danish[danish > u_danish] - u_danish
n_excess_d = len(excess_danish)
n_total_d = len(danish)

# GPD参数估计
xbar_d = excess_danish.mean()
s2_d = excess_danish.var()
xi_hat_d = 0.5 * (xbar_d**2 / s2_d - 1)
sigma_hat_d = 0.5 * xbar_d * (xbar_d**2 / s2_d + 1)

print(f"\nDanish火灾损失POT分析:")
print(f"  阈值u={u_danish:.2f}, 超出数={n_excess_d}")
print(f"  ξ_hat={xi_hat_d:.4f}, σ_hat={sigma_hat_d:.4f}")

for p in [0.95, 0.99, 0.999]:
    VaR_gpd = u_danish + sigma_hat_d / xi_hat_d * \
              ((n_total_d / n_excess_d * (1 - p))**(-xi_hat_d) - 1)
    VaR_emp = np.percentile(danish, p * 100)
    print(f"  p={p}: VaR_GPD={VaR_gpd:.2f}, VaR_emp={VaR_emp:.2f}")

##############################################################################
# 第9章 极值理论
# 对应教材：section9.tex
# 内容：风险度量、广义极值分布（GEV）、广义帕累托分布（GPD）、
#       阈值选择、参数估计、应用
##############################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


##############################################################################
# 9.1 风险度量：VaR与TVaR
##############################################################################
mu, sigma = 0, 1
p_values = [0.95, 0.99, 0.999]

print("正态分布的风险度量:")
for p in p_values:
    VaR = stats.norm.ppf(p, mu, sigma)
    z = stats.norm.ppf(p)
    TVaR = mu + sigma * stats.norm.pdf(z) / (1 - p)
    print(f"  p={p}: VaR={VaR:.4f}, TVaR={TVaR:.4f}")

print("\n对数正态分布的风险度量:")
for p in p_values:
    VaR = stats.lognorm.ppf(p, s=1, scale=np.exp(0))
    print(f"  p={p}: VaR={VaR:.4f}")


##############################################################################
# 9.2 广义极值分布（GEV）
##############################################################################
def gev_pdf(x, mu, sigma, xi):
    """GEV概率密度函数"""
    z = (x - mu) / sigma
    if abs(xi) < 1e-8:
        t = np.exp(-z)
        return (1/sigma) * t * np.exp(-t)
    else:
        if np.any(1 + xi * z <= 0):
            return 0
        t = (1 + xi * z) ** (-1/xi)
        return (1/sigma) * t**(xi + 1) * np.exp(-t)

x = np.linspace(-4, 4, 100)
fig, ax = plt.subplots(figsize=(8, 5))
for xi, label, color in [(0.5, 'ξ=0.5 (Frechet)', 'C0'),
                          (0, 'ξ=0 (Gumbel)', 'C1'),
                          (-0.5, 'ξ=-0.5 (Weibull)', 'C2')]:
    f = np.array([gev_pdf(xi_, 0, 1, xi) for xi_ in x])
    ax.plot(x, f, color=color, label=label)
ax.set_xlabel('x'); ax.set_ylabel('f(x)')
ax.set_title('GEV分布密度函数'); ax.legend()
plt.tight_layout(); plt.show()


##############################################################################
# 9.3 区块最大值法（Block Maxima）
##############################################################################
np.random.seed(123)
n_blocks = 100
block_size = 50

maxima = np.zeros(n_blocks)
for b in range(n_blocks):
    x_block = np.random.exponential(1, block_size)
    maxima[b] = x_block.max()

print(f"\n区块最大值法:")
print(f"  均值={maxima.mean():.4f}, 标准差={maxima.std():.4f}, 最大值={maxima.max():.4f}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(maxima, bins=20, density=True, alpha=0.5)
kde_x = np.linspace(maxima.min(), maxima.max(), 200)
from scipy.stats import gaussian_kde
kde = gaussian_kde(maxima)
ax.plot(kde_x, kde(kde_x), 'r-')
ax.set_title('区块最大值直方图')
plt.tight_layout(); plt.show()


##############################################################################
# 9.4 GEV参数估计
##############################################################################
def gev_nll(parms, x):
    mu = parms[0]
    sigma = np.exp(parms[1])
    xi = parms[2]
    z = (x - mu) / sigma
    if np.any(1 + xi * z <= 0):
        return 1e10
    if abs(xi) < 1e-8:
        t = np.exp(-z)
    else:
        t = (1 + xi * z) ** (-1/xi)
    return np.sum(np.log(sigma) + (1 + 1/xi) * np.log(t) + t)

result_gev = minimize(gev_nll, x0=[3, 0, 0.1], args=(maxima,),
                       method='Nelder-Mead')
mu_hat = result_gev.x[0]
sigma_hat = np.exp(result_gev.x[1])
xi_hat = result_gev.x[2]
print(f"\nGEV参数MLE估计:")
print(f"  μ_hat={mu_hat:.4f}, σ_hat={sigma_hat:.4f}, ξ_hat={xi_hat:.4f}")


##############################################################################
# 9.5 广义帕累托分布（GPD）
##############################################################################
def gpd_pdf(x, sigma, xi):
    """GPD概率密度函数"""
    if abs(xi) < 1e-8:
        return (1/sigma) * np.exp(-x/sigma)
    else:
        if np.any(1 + xi * x / sigma <= 0):
            return 0
        return (1/sigma) * (1 + xi * x / sigma) ** (-1/xi - 1)

x = np.linspace(0, 5, 100)
fig, ax = plt.subplots(figsize=(8, 5))
for xi, label, color in [(0.5, 'ξ=0.5', 'C0'),
                          (0, 'ξ=0', 'C1'),
                          (-0.5, 'ξ=-0.5', 'C2')]:
    f = np.array([gpd_pdf(xi_, 1, xi) for xi_ in x])
    ax.plot(x, f, color=color, label=label)
ax.set_xlabel('x'); ax.set_ylabel('f(x)')
ax.set_title('GPD分布密度函数'); ax.legend()
plt.tight_layout(); plt.show()


##############################################################################
# 9.6 阈值选择（POT方法）
##############################################################################
np.random.seed(456)
n = 1000
x_pot = np.random.exponential(1, n)

# 平均超出量图
u_grid = np.linspace(0, 3, 50)
me = np.zeros_like(u_grid)
for i, u in enumerate(u_grid):
    excess = x_pot[x_pot > u] - u
    if len(excess) > 0:
        me[i] = excess.mean()

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(u_grid, me)
ax.set_xlabel('阈值 u'); ax.set_ylabel('平均超出量')
ax.set_title('平均超出量图')
plt.tight_layout(); plt.show()

# 不同阈值下的超出量
print(f"\n不同阈值下的超出量:")
for u in [0.5, 1.0, 1.5, 2.0, 2.5]:
    excess = x_pot[x_pot > u] - u
    print(f"  u={u}: 超出数={len(excess)}, 平均超出量={excess.mean():.4f}")


##############################################################################
# 9.7 GPD参数估计
##############################################################################
np.random.seed(789)
n = 1000
sigma_true = 1
xi_true = 0.3

u_gpd = np.random.uniform(0, 1, n)
if abs(xi_true) < 1e-8:
    x_gpd = -sigma_true * np.log(1 - u_gpd)
else:
    x_gpd = sigma_true / xi_true * ((1 - u_gpd)**(-xi_true) - 1)

def gpd_nll(parms, x):
    sigma = np.exp(parms[0])
    xi = parms[1]
    if np.any(1 + xi * x / sigma <= 0):
        return 1e10
    if abs(xi) < 1e-8:
        return np.sum(np.log(sigma) + x / sigma)
    return np.sum(np.log(sigma) + (1 + 1/xi) * np.log(1 + xi * x / sigma))

result_gpd = minimize(gpd_nll, x0=[0, 0.3], args=(x_gpd,),
                       method='Nelder-Mead')
sigma_hat = np.exp(result_gpd.x[0])
xi_hat = result_gpd.x[1]
print(f"\nGPD参数MLE估计:")
print(f"  σ_true={sigma_true}, σ_hat={sigma_hat:.4f}")
print(f"  ξ_true={xi_true}, ξ_hat={xi_hat:.4f}")


##############################################################################
# 9.8 极值理论应用：VaR估计
##############################################################################
np.random.seed(101)
n = 1000
x_var = np.random.exponential(1, n)

# 选择阈值
u = 2
excess = x_var[x_var > u] - u
n_excess = len(excess)
n_total = len(x_var)

# GPD参数估计（矩估计）
xbar = excess.mean()
s2 = excess.var()
xi_hat_var = 0.5 * (xbar**2 / s2 - 1)
sigma_hat_var = 0.5 * xbar * (xbar**2 / s2 + 1)

print(f"\nVaR估计:")
print(f"  阈值u={u}, 超出数={n_excess}")
print(f"  ξ_hat={xi_hat_var:.4f}, σ_hat={sigma_hat_var:.4f}")

for p in [0.95, 0.99, 0.999]:
    VaR_gpd = u + sigma_hat_var / xi_hat_var * \
              ((n_total / n_excess * (1 - p))**(-xi_hat_var) - 1)
    VaR_emp = np.percentile(x_var, p * 100)
    print(f"  p={p}: VaR_GPD={VaR_gpd:.4f}, VaR_emp={VaR_emp:.4f}")


##############################################################################
# 9.9 Danish火灾损失数据应用
##############################################################################
np.random.seed(42)
# 模拟Danish数据（对数正态）
danish = np.random.lognormal(0.672, 0.732, 2167)
danish = danish[danish > 1]

# 使用POT方法
u_danish = np.percentile(danish, 95)
excess_danish = danish[danish > u_danish] - u_danish
n_excess_d = len(excess_danish)
n_total_d = len(danish)

# GPD参数估计
xbar_d = excess_danish.mean()
s2_d = excess_danish.var()
xi_hat_d = 0.5 * (xbar_d**2 / s2_d - 1)
sigma_hat_d = 0.5 * xbar_d * (xbar_d**2 / s2_d + 1)

print(f"\nDanish火灾损失POT分析:")
print(f"  阈值u={u_danish:.2f}, 超出数={n_excess_d}")
print(f"  ξ_hat={xi_hat_d:.4f}, σ_hat={sigma_hat_d:.4f}")

for p in [0.95, 0.99, 0.999]:
    VaR_gpd = u_danish + sigma_hat_d / xi_hat_d * \
              ((n_total_d / n_excess_d * (1 - p))**(-xi_hat_d) - 1)
    VaR_emp = np.percentile(danish, p * 100)
    print(f"  p={p}: VaR_GPD={VaR_gpd:.2f}, VaR_emp={VaR_emp:.2f}")

# 导入所需的Python模块
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

print("正态分布的风险度量:")
for p in p_values:
    VaR = stats.norm.ppf(p, mu, sigma)
    z = stats.norm.ppf(p)
    TVaR = mu + sigma * stats.norm.pdf(z) / (1 - p)
    print(f"  p={p}: VaR={VaR:.4f}, TVaR={TVaR:.4f}")

print("\n对数正态分布的风险度量:")
for p in p_values:
    VaR = stats.lognorm.ppf(p, s=1, scale=np.exp(0))
    print(f"  p={p}: VaR={VaR:.4f}")


##############################################################################
# 9.2 广义极值分布（GEV）
##############################################################################
def gev_pdf(x, mu, sigma, xi):
    """GEV概率密度函数"""
    z = (x - mu) / sigma
    if abs(xi) < 1e-8:
        t = np.exp(-z)
        return (1/sigma) * t * np.exp(-t)
    else:
        if np.any(1 + xi * z <= 0):
            return 0
        t = (1 + xi * z) ** (-1/xi)
        return (1/sigma) * t**(xi + 1) * np.exp(-t)

x = np.linspace(-4, 4, 100)
fig, ax = plt.subplots(figsize=(8, 5))
for xi, label, color in [(0.5, 'ξ=0.5 (Frechet)', 'C0'),
                          (0, 'ξ=0 (Gumbel)', 'C1'),

##############################################################################
# 第9章 极值理论
# 对应教材：section9.tex
# 内容：风险度量、广义极值分布（GEV）、广义帕累托分布（GPD）、
#       阈值选择、参数估计、应用
##############################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import minimize
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


##############################################################################
# 9.1 风险度量：VaR与TVaR
##############################################################################
mu, sigma = 0, 1
p_values = [0.95, 0.99, 0.999]

print("正态分布的风险度量:")
for p in p_values:
    VaR = stats.norm.ppf(p, mu, sigma)
    z = stats.norm.ppf(p)
    TVaR = mu + sigma * stats.norm.pdf(z) / (1 - p)
    print(f"  p={p}: VaR={VaR:.4f}, TVaR={TVaR:.4f}")

print("\n对数正态分布的风险度量:")
for p in p_values:
    VaR = stats.lognorm.ppf(p, s=1, scale=np.exp(0))
    print(f"  p={p}: VaR={VaR:.4f}")


##############################################################################
# 9.2 广义极值分布（GEV）
##############################################################################
def gev_pdf(x, mu, sigma, xi):
    """GEV概率密度函数"""
    z = (x - mu) / sigma
    if abs(xi) < 1e-8:
        t = np.exp(-z)
        return (1/sigma) * t * np.exp(-t)
    else:
        if np.any(1 + xi * z <= 0):
            return 0
        t = (1 + xi * z) ** (-1/xi)
        return (1/sigma) * t**(xi + 1) * np.exp(-t)

x = np.linspace(-4, 4, 100)
fig, ax = plt.subplots(figsize=(8, 5))
for xi, label, color in [(0.5, 'ξ=0.5 (Frechet)', 'C0'),
                          (0, 'ξ=0 (Gumbel)', 'C1'),
                          (-0.5, 'ξ=-0.5 (Weibull)', 'C2')]:
    f = np.array([gev_pdf(xi_, 0, 1, xi) for xi_ in x])
    ax.plot(x, f, color=color, label=label)
ax.set_xlabel('x'); ax.set_ylabel('f(x)')
ax.set_title('GEV分布密度函数'); ax.legend()
plt.tight_layout(); plt.show()


##############################################################################
# 9.3 区块最大值法（Block Maxima）
##############################################################################
np.random.seed(123)
n_blocks = 100
block_size = 50

maxima = np.zeros(n_blocks)
for b in range(n_blocks):
    x_block = np.random.exponential(1, block_size)
    maxima[b] = x_block.max()

print(f"\n区块最大值法:")
print(f"  均值={maxima.mean():.4f}, 标准差={maxima.std():.4f}, 最大值={maxima.max():.4f}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(maxima, bins=20, density=True, alpha=0.5)
kde_x = np.linspace(maxima.min(), maxima.max(), 200)
from scipy.stats import gaussian_kde
kde = gaussian_kde(maxima)
ax.plot(kde_x, kde(kde_x), 'r-')
ax.set_title('区块最大值直方图')
plt.tight_layout(); plt.show()


##############################################################################
# 9.4 GEV参数估计
##############################################################################
def gev_nll(parms, x):
    mu = parms[0]
    sigma = np.exp(parms[1])
    xi = parms[2]
    z = (x - mu) / sigma
    if np.any(1 + xi * z <= 0):
        return 1e10
    if abs(xi) < 1e-8:
        t = np.exp(-z)
    else:
        t = (1 + xi * z) ** (-1/xi)
    return np.sum(np.log(sigma) + (1 + 1/xi) * np.log(t) + t)

result_gev = minimize(gev_nll, x0=[3, 0, 0.1], args=(maxima,),
                       method='Nelder-Mead')
mu_hat = result_gev.x[0]
sigma_hat = np.exp(result_gev.x[1])
xi_hat = result_gev.x[2]
print(f"\nGEV参数MLE估计:")
print(f"  μ_hat={mu_hat:.4f}, σ_hat={sigma_hat:.4f}, ξ_hat={xi_hat:.4f}")


##############################################################################
# 9.5 广义帕累托分布（GPD）
##############################################################################
def gpd_pdf(x, sigma, xi):
    """GPD概率密度函数"""
    if abs(xi) < 1e-8:
        return (1/sigma) * np.exp(-x/sigma)
    else:
        if np.any(1 + xi * x / sigma <= 0):
            return 0
        return (1/sigma) * (1 + xi * x / sigma) ** (-1/xi - 1)

x = np.linspace(0, 5, 100)
fig, ax = plt.subplots(figsize=(8, 5))
for xi, label, color in [(0.5, 'ξ=0.5', 'C0'),
                          (0, 'ξ=0', 'C1'),
                          (-0.5, 'ξ=-0.5', 'C2')]:
    f = np.array([gpd_pdf(xi_, 1, xi) for xi_ in x])
    ax.plot(x, f, color=color, label=label)
ax.set_xlabel('x'); ax.set_ylabel('f(x)')
ax.set_title('GPD分布密度函数'); ax.legend()
plt.tight_layout(); plt.show()


##############################################################################
# 9.6 阈值选择（POT方法）
##############################################################################
np.random.seed(456)
n = 1000
x_pot = np.random.exponential(1, n)

# 平均超出量图
u_grid = np.linspace(0, 3, 50)
me = np.zeros_like(u_grid)
for i, u in enumerate(u_grid):
    excess = x_pot[x_pot > u] - u
    if len(excess) > 0:
        me[i] = excess.mean()

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(u_grid, me)
ax.set_xlabel('阈值 u'); ax.set_ylabel('平均超出量')
ax.set_title('平均超出量图')
plt.tight_layout(); plt.show()

# 不同阈值下的超出量
print(f"\n不同阈值下的超出量:")
for u in [0.5, 1.0, 1.5, 2.0, 2.5]:
    excess = x_pot[x_pot > u] - u
    print(f"  u={u}: 超出数={len(excess)}, 平均超出量={excess.mean():.4f}")


##############################################################################
# 9.7 GPD参数估计
##############################################################################
np.random.seed(789)
n = 1000
sigma_true = 1
xi_true = 0.3

u_gpd = np.random.uniform(0, 1, n)
if abs(xi_true) < 1e-8:
    x_gpd = -sigma_true * np.log(1 - u_gpd)
else:
    x_gpd = sigma_true / xi_true * ((1 - u_gpd)**(-xi_true) - 1)

def gpd_nll(parms, x):
    sigma = np.exp(parms[0])
    xi = parms[1]
    if np.any(1 + xi * x / sigma <= 0):
        return 1e10
    if abs(xi) < 1e-8:
        return np.sum(np.log(sigma) + x / sigma)
    return np.sum(np.log(sigma) + (1 + 1/xi) * np.log(1 + xi * x / sigma))

result_gpd = minimize(gpd_nll, x0=[0, 0.3], args=(x_gpd,),
                       method='Nelder-Mead')
sigma_hat = np.exp(result_gpd.x[0])
xi_hat = result_gpd.x[1]
print(f"\nGPD参数MLE估计:")
print(f"  σ_true={sigma_true}, σ_hat={sigma_hat:.4f}")
print(f"  ξ_true={xi_true}, ξ_hat={xi_hat:.4f}")


##############################################################################
# 9.8 极值理论应用：VaR估计
##############################################################################
np.random.seed(101)
n = 1000
x_var = np.random.exponential(1, n)

# 选择阈值
u = 2
excess = x_var[x_var > u] - u
n_excess = len(excess)
n_total = len(x_var)

# GPD参数估计（矩估计）
xbar = excess.mean()
s2 = excess.var()
xi_hat_var = 0.5 * (xbar**2 / s2 - 1)
sigma_hat_var = 0.5 * xbar * (xbar**2 / s2 + 1)

print(f"\nVaR估计:")
print(f"  阈值u={u}, 超出数={n_excess}")
print(f"  ξ_hat={xi_hat_var:.4f}, σ_hat={sigma_hat_var:.4f}")

for p in [0.95, 0.99, 0.999]:
    VaR_gpd = u + sigma_hat_var / xi_hat_var * \
              ((n_total / n_excess * (1 - p))**(-xi_hat_var) - 1)
    VaR_emp = np.percentile(x_var, p * 100)
    print(f"  p={p}: VaR_GPD={VaR_gpd:.4f}, VaR_emp={VaR_emp:.4f}")


##############################################################################
# 9.9 Danish火灾损失数据应用
##############################################################################
np.random.seed(42)
# 模拟Danish数据（对数正态）
danish = np.random.lognormal(0.672, 0.732, 2167)
danish = danish[danish > 1]

# 使用POT方法
u_danish = np.percentile(danish, 95)
excess_danish = danish[danish > u_danish] - u_danish
n_excess_d = len(excess_danish)
n_total_d = len(danish)

# GPD参数估计
xbar_d = excess_danish.mean()
s2_d = excess_danish.var()
xi_hat_d = 0.5 * (xbar_d**2 / s2_d - 1)
sigma_hat_d = 0.5 * xbar_d * (xbar_d**2 / s2_d + 1)

print(f"\nDanish火灾损失POT分析:")
print(f"  阈值u={u_danish:.2f}, 超出数={n_excess_d}")
print(f"  ξ_hat={xi_hat_d:.4f}, σ_hat={sigma_hat_d:.4f}")

for p in [0.95, 0.99, 0.999]:
    VaR_gpd = u_danish + sigma_hat_d / xi_hat_d * \
              ((n_total_d / n_excess_d * (1 - p))**(-xi_hat_d) - 1)
    VaR_emp = np.percentile(danish, p * 100)
    print(f"  p={p}: VaR_GPD={VaR_gpd:.2f}, VaR_emp={VaR_emp:.2f}")