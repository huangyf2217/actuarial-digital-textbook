# Chap8 Python代码
# 自动从chap8.html同步生成

##############################################################################
# 第8章 相依风险与Copula
# 对应教材：section8.tex
# 内容：Copula概念、常见Copula函数、参数估计、尾部相关性、
#       多元Copula、vine Copula
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
# 8.1 Copula基本概念
##############################################################################
u = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
v = np.array([0.2, 0.4, 0.6, 0.8, 1.0])

# 独立Copula
C_indep = u * v
print(f"独立Copula: {C_indep}")


##############################################################################
# 8.2 高斯Copula
##############################################################################
def gaussian_copula(u, v, rho):
    """高斯Copula"""
    z1 = stats.norm.ppf(u)
    z2 = stats.norm.ppf(v)
    return stats.multivariate_normal.cdf(np.column_stack([z1, z2]),
                                          cov=[[1, rho], [rho, 1]])

rho = 0.5
C_gaussian = [gaussian_copula(ui, vi, rho) for ui, vi in zip(u, v)]
print(f"\n高斯Copula (ρ=0.5): {C_gaussian}")


## 8.2.1 高斯Copula模拟
np.random.seed(123)
n = 1000
rho = 0.7

z1 = np.random.normal(0, 1, n)
z2 = rho * z1 + np.sqrt(1 - rho**2) * np.random.normal(0, 1, n)

u1 = stats.norm.cdf(z1)
u2 = stats.norm.cdf(z2)

# 转换为指数分布
x1_g = stats.expon.ppf(u1)
x2_g = stats.expon.ppf(u2)

print(f"\n高斯Copula模拟: 经验相关系数={np.corrcoef(x1_g, x2_g)[0,1]:.4f}")

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(x1_g, x2_g, alpha=0.3)
ax.set_xlabel('x1'); ax.set_ylabel('x2')
ax.set_title('高斯Copula模拟散点图')
plt.tight_layout(); plt.show()


##############################################################################
# 8.3 T Copula
##############################################################################
np.random.seed(456)
rho_t = 0.5
df_t = 4

z1_t = np.random.normal(0, 1, n)
z2_t = rho_t * z1_t + np.sqrt(1 - rho_t**2) * np.random.normal(0, 1, n)
w = np.random.chisquare(df_t) / df_t
z1_t = z1_t / np.sqrt(w)
z2_t = z2_t / np.sqrt(w)

u1_t = stats.t.cdf(z1_t, df_t)
u2_t = stats.t.cdf(z2_t, df_t)
x1_t = stats.expon.ppf(u1_t)
x2_t = stats.expon.ppf(u2_t)

print(f"\nT Copula模拟 (df=4): 经验相关系数={np.corrcoef(x1_t, x2_t)[0,1]:.4f}")


##############################################################################
# 8.4 阿基米德Copula
##############################################################################

## 8.4.1 Gumbel Copula
def gumbel_copula(u, v, theta):
    return np.exp(-((-np.log(u))**theta + (-np.log(v))**theta)**(1/theta))

C_gumbel = gumbel_copula(u, v, 2)
print(f"\nGumbel Copula (θ=2): {C_gumbel}")

## 8.4.2 Clayton Copula
def clayton_copula(u, v, theta):
    return (u**(-theta) + v**(-theta) - 1)**(-1/theta)

C_clayton = clayton_copula(u, v, 2)
print(f"Clayton Copula (θ=2): {C_clayton}")

## 8.4.3 Frank Copula
def frank_copula(u, v, theta):
    return -1/theta * np.log(1 + (np.exp(-theta * u) - 1) *
                              (np.exp(-theta * v) - 1) / (np.exp(-theta) - 1))

C_frank = frank_copula(u, v, 5)
print(f"Frank Copula (θ=5): {C_frank}")


## 8.4.4 Clayton Copula模拟
np.random.seed(789)
theta_c = 2

u1_c = np.random.uniform(0, 1, n)
t_c = np.random.uniform(0, 1, n)
u2_c = (1 - u1_c**(-theta_c) +
        u1_c**(-theta_c) * t_c**(-theta_c/(1+theta_c)))**(-1/theta_c)

x1_c = stats.expon.ppf(u1_c)
x2_c = stats.expon.ppf(u2_c)

print(f"\nClayton Copula模拟 (θ=2): 经验相关系数={np.corrcoef(x1_c, x2_c)[0,1]:.4f}")

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(x1_c, x2_c, alpha=0.3, color='red')
ax.set_xlabel('x1'); ax.set_ylabel('x2')
ax.set_title('Clayton Copula模拟散点图')
plt.tight_layout(); plt.show()


##############################################################################
# 8.5 尾部相关性
##############################################################################
print(f"\n尾部相关性:")

# 高斯Copula: 尾部相关性为0
print("高斯Copula: λ_upper=0, λ_lower=0 (ρ<1)")

# T Copula
print("T Copula (df=4):")
for rho in [0.0, 0.3, 0.5, 0.7, 0.9]:
    df = 4
    lam = 2 * stats.t.ppf(1 - 0.5, df + 1) * (1 - rho) / \
          np.sqrt((df + 1) * (1 - rho**2))
    print(f"  ρ={rho}: λ={lam:.4f}")

# Clayton Copula下尾
print("Clayton Copula下尾相关性:")
for theta in [1, 2, 5, 10]:
    lam_lower = 2**(-1/theta)
    print(f"  θ={theta}: λ_lower={lam_lower:.4f}")

# Gumbel Copula上尾
print("Gumbel Copula上尾相关性:")
for theta in [1, 2, 5, 10]:
    lam_upper = 2 - 2**(1/theta)
    print(f"  θ={theta}: λ_upper={lam_upper:.4f}")


##############################################################################
# 8.6 Copula参数估计
##############################################################################
np.random.seed(123)
theta_true = 2

u1_est = np.random.uniform(0, 1, n)
t_est = np.random.uniform(0, 1, n)
u2_est = (1 - u1_est**(-theta_true) +
          u1_est**(-theta_true) * t_est**(-theta_true/(1+theta_true)))**(-1/theta_true)
x1_est = stats.expon.ppf(u1_est)
x2_est = stats.expon.ppf(u2_est)

# 方法1：基于Kendall tau的矩估计
from scipy.stats import kendalltau
tau, _ = kendalltau(x1_est, x2_est)
theta_hat = 2 * tau / (1 - tau)
print(f"\nClayton Copula参数估计:")
print(f"  Kendall tau = {tau:.4f}")
print(f"  θ_hat（矩估计）= {theta_hat:.4f}")

# 方法2：极大似然估计
def clayton_nll(theta, u1, u2):
    if theta <= 0:
        return 1e10
    c = (1 + theta) * (u1 * u2)**(-theta - 1) * \
        (u1**(-theta) + u2**(-theta) - 1)**(-2 - 1/theta)
    return -np.sum(np.log(c))

result = minimize(clayton_nll, x0=[1.0], args=(u1_est, u2_est),
                  method='Nelder-Mead')
print(f"  θ_hat（MLE）= {result.x[0]:.4f}")


##############################################################################
# 8.7 多元Copula应用
##############################################################################
np.random.seed(202)
n_multi = 1000

Sigma = np.array([[1.0, 0.5, 0.3],
                   [0.5, 1.0, 0.4],
                   [0.3, 0.4, 1.0]])

L = np.linalg.cholesky(Sigma)
Z = np.random.normal(0, 1, (n_multi, 3)) @ L.T
U = stats.norm.cdf(Z)

# 不同边缘分布
x1_multi = stats.expon.ppf(U[:, 0])
x2_multi = stats.gamma.ppf(U[:, 1], 2)
x3_multi = stats.lognorm.ppf(U[:, 2], s=1)

corr_emp = np.corrcoef(np.column_stack([x1_multi, x2_multi, x3_multi]).T)
print(f"\n三元高斯Copula:")
print(f"  经验相关矩阵:")
print(f"  {corr_emp}")


##############################################################################
# 8.8 Copula比较
##############################################################################
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 高斯Copula
axes[0,0].scatter(x1_g[:500], x2_g[:500], alpha=0.3)
axes[0,0].set_title('高斯Copula')

# T Copula
axes[0,1].scatter(x1_t[:500], x2_t[:500], alpha=0.3, color='orange')
axes[0,1].set_title('T Copula')

# Clayton Copula
axes[1,0].scatter(x1_c[:500], x2_c[:500], alpha=0.3, color='red')
axes[1,0].set_title('Clayton Copula')

# Gumbel Copula模拟
np.random.seed(111)
theta_gu = 2
u1_gu = np.random.uniform(0, 1, n)
t_gu = np.random.uniform(0, 1, n)
# Gumbel Copula条件分布法（简化）
u2_gu = np.exp(-((-np.log(u1_gu))**theta_gu +
                  (-np.log(t_gu))**theta_gu)**(1/theta_gu))
x1_gu = stats.expon.ppf(u1_gu)
x2_gu = stats.expon.ppf(u2_gu)
axes[1,1].scatter(x1_gu[:500], x2_gu[:500], alpha=0.3, color='green')
axes[1,1].set_title('Gumbel Copula')

plt.tight_layout(); plt.show()

##############################################################################
# 第8章 相依风险与Copula
# 对应教材：section8.tex
# 内容：Copula概念、常见Copula函数、参数估计、尾部相关性、
#       多元Copula、vine Copula
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
# 8.1 Copula基本概念
##############################################################################
u = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
v = np.array([0.2, 0.4, 0.6, 0.8, 1.0])

# 独立Copula
C_indep = u * v
print(f"独立Copula: {C_indep}")


##############################################################################
# 8.2 高斯Copula
##############################################################################
def gaussian_copula(u, v, rho):
    """高斯Copula"""
    z1 = stats.norm.ppf(u)
    z2 = stats.norm.ppf(v)
    return stats.multivariate_normal.cdf(np.column_stack([z1, z2]),
                                          cov=[[1, rho], [rho, 1]])

rho = 0.5
C_gaussian = [gaussian_copula(ui, vi, rho) for ui, vi in zip(u, v)]
print(f"\n高斯Copula (ρ=0.5): {C_gaussian}")


## 8.2.1 高斯Copula模拟
np.random.seed(123)
n = 1000
rho = 0.7

z1 = np.random.normal(0, 1, n)
z2 = rho * z1 + np.sqrt(1 - rho**2) * np.random.normal(0, 1, n)

u1 = stats.norm.cdf(z1)
u2 = stats.norm.cdf(z2)

# 转换为指数分布
x1_g = stats.expon.ppf(u1)
x2_g = stats.expon.ppf(u2)

print(f"\n高斯Copula模拟: 经验相关系数={np.corrcoef(x1_g, x2_g)[0,1]:.4f}")

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(x1_g, x2_g, alpha=0.3)
ax.set_xlabel('x1'); ax.set_ylabel('x2')
ax.set_title('高斯Copula模拟散点图')
plt.tight_layout(); plt.show()


##############################################################################
# 8.3 T Copula
##############################################################################
np.random.seed(456)
rho_t = 0.5
df_t = 4

z1_t = np.random.normal(0, 1, n)
z2_t = rho_t * z1_t + np.sqrt(1 - rho_t**2) * np.random.normal(0, 1, n)
w = np.random.chisquare(df_t) / df_t
z1_t = z1_t / np.sqrt(w)
z2_t = z2_t / np.sqrt(w)

u1_t = stats.t.cdf(z1_t, df_t)
u2_t = stats.t.cdf(z2_t, df_t)
x1_t = stats.expon.ppf(u1_t)
x2_t = stats.expon.ppf(u2_t)

print(f"\nT Copula模拟 (df=4): 经验相关系数={np.corrcoef(x1_t, x2_t)[0,1]:.4f}")


##############################################################################
# 8.4 阿基米德Copula
##############################################################################

## 8.4.1 Gumbel Copula
def gumbel_copula(u, v, theta):
    return np.exp(-((-np.log(u))**theta + (-np.log(v))**theta)**(1/theta))

C_gumbel = gumbel_copula(u, v, 2)
print(f"\nGumbel Copula (θ=2): {C_gumbel}")

## 8.4.2 Clayton Copula
def clayton_copula(u, v, theta):
    return (u**(-theta) + v**(-theta) - 1)**(-1/theta)

C_clayton = clayton_copula(u, v, 2)
print(f"Clayton Copula (θ=2): {C_clayton}")

## 8.4.3 Frank Copula
def frank_copula(u, v, theta):
    return -1/theta * np.log(1 + (np.exp(-theta * u) - 1) *
                              (np.exp(-theta * v) - 1) / (np.exp(-theta) - 1))

C_frank = frank_copula(u, v, 5)
print(f"Frank Copula (θ=5): {C_frank}")


## 8.4.4 Clayton Copula模拟
np.random.seed(789)
theta_c = 2

u1_c = np.random.uniform(0, 1, n)
t_c = np.random.uniform(0, 1, n)
u2_c = (1 - u1_c**(-theta_c) +
        u1_c**(-theta_c) * t_c**(-theta_c/(1+theta_c)))**(-1/theta_c)

x1_c = stats.expon.ppf(u1_c)
x2_c = stats.expon.ppf(u2_c)

print(f"\nClayton Copula模拟 (θ=2): 经验相关系数={np.corrcoef(x1_c, x2_c)[0,1]:.4f}")

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(x1_c, x2_c, alpha=0.3, color='red')
ax.set_xlabel('x1'); ax.set_ylabel('x2')
ax.set_title('Clayton Copula模拟散点图')
plt.tight_layout(); plt.show()


##############################################################################
# 8.5 尾部相关性
##############################################################################
print(f"\n尾部相关性:")

# 高斯Copula: 尾部相关性为0
print("高斯Copula: λ_upper=0, λ_lower=0 (ρ<1)")

# T Copula
print("T Copula (df=4):")
for rho in [0.0, 0.3, 0.5, 0.7, 0.9]:
    df = 4
    lam = 2 * stats.t.ppf(1 - 0.5, df + 1) * (1 - rho) / \
          np.sqrt((df + 1) * (1 - rho**2))
    print(f"  ρ={rho}: λ={lam:.4f}")

# Clayton Copula下尾
print("Clayton Copula下尾相关性:")
for theta in [1, 2, 5, 10]:
    lam_lower = 2**(-1/theta)
    print(f"  θ={theta}: λ_lower={lam_lower:.4f}")

# Gumbel Copula上尾
print("Gumbel Copula上尾相关性:")
for theta in [1, 2, 5, 10]:
    lam_upper = 2 - 2**(1/theta)
    print(f"  θ={theta}: λ_upper={lam_upper:.4f}")


##############################################################################
# 8.6 Copula参数估计
##############################################################################
np.random.seed(123)
theta_true = 2

u1_est = np.random.uniform(0, 1, n)
t_est = np.random.uniform(0, 1, n)
u2_est = (1 - u1_est**(-theta_true) +
          u1_est**(-theta_true) * t_est**(-theta_true/(1+theta_true)))**(-1/theta_true)
x1_est = stats.expon.ppf(u1_est)
x2_est = stats.expon.ppf(u2_est)

# 方法1：基于Kendall tau的矩估计
from scipy.stats import kendalltau
tau, _ = kendalltau(x1_est, x2_est)
theta_hat = 2 * tau / (1 - tau)
print(f"\nClayton Copula参数估计:")
print(f"  Kendall tau = {tau:.4f}")
print(f"  θ_hat（矩估计）= {theta_hat:.4f}")

# 方法2：极大似然估计
def clayton_nll(theta, u1, u2):
    if theta <= 0:
        return 1e10
    c = (1 + theta) * (u1 * u2)**(-theta - 1) * \
        (u1**(-theta) + u2**(-theta) - 1)**(-2 - 1/theta)
    return -np.sum(np.log(c))

result = minimize(clayton_nll, x0=[1.0], args=(u1_est, u2_est),
                  method='Nelder-Mead')
print(f"  θ_hat（MLE）= {result.x[0]:.4f}")


##############################################################################
# 8.7 多元Copula应用
##############################################################################
np.random.seed(202)
n_multi = 1000

Sigma = np.array([[1.0, 0.5, 0.3],
                   [0.5, 1.0, 0.4],
                   [0.3, 0.4, 1.0]])

L = np.linalg.cholesky(Sigma)
Z = np.random.normal(0, 1, (n_multi, 3)) @ L.T
U = stats.norm.cdf(Z)

# 不同边缘分布
x1_multi = stats.expon.ppf(U[:, 0])
x2_multi = stats.gamma.ppf(U[:, 1], 2)
x3_multi = stats.lognorm.ppf(U[:, 2], s=1)

corr_emp = np.corrcoef(np.column_stack([x1_multi, x2_multi, x3_multi]).T)
print(f"\n三元高斯Copula:")
print(f"  经验相关矩阵:")
print(f"  {corr_emp}")


##############################################################################
# 8.8 Copula比较
##############################################################################
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 高斯Copula
axes[0,0].scatter(x1_g[:500], x2_g[:500], alpha=0.3)
axes[0,0].set_title('高斯Copula')

# T Copula
axes[0,1].scatter(x1_t[:500], x2_t[:500], alpha=0.3, color='orange')
axes[0,1].set_title('T Copula')

# Clayton Copula
axes[1,0].scatter(x1_c[:500], x2_c[:500], alpha=0.3, color='red')
axes[1,0].set_title('Clayton Copula')

# Gumbel Copula模拟
np.random.seed(111)
theta_gu = 2
u1_gu = np.random.uniform(0, 1, n)
t_gu = np.random.uniform(0, 1, n)
# Gumbel Copula条件分布法（简化）
u2_gu = np.exp(-((-np.log(u1_gu))**theta_gu +
                  (-np.log(t_gu))**theta_gu)**(1/theta_gu))
x1_gu = stats.expon.ppf(u1_gu)
x2_gu = stats.expon.ppf(u2_gu)
axes[1,1].scatter(x1_gu[:500], x2_gu[:500], alpha=0.3, color='green')
axes[1,1].set_title('Gumbel Copula')

plt.tight_layout(); plt.show()

##############################################################################
# 第8章 相依风险与Copula
# 对应教材：section8.tex
# 内容：Copula概念、常见Copula函数、参数估计、尾部相关性、
#       多元Copula、vine Copula
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
# 8.1 Copula基本概念
##############################################################################
u = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
v = np.array([0.2, 0.4, 0.6, 0.8, 1.0])

# 独立Copula
C_indep = u * v
print(f"独立Copula: {C_indep}")


##############################################################################
# 8.2 高斯Copula
##############################################################################
def gaussian_copula(u, v, rho):
    """高斯Copula"""
    z1 = stats.norm.ppf(u)
    z2 = stats.norm.ppf(v)
    return stats.multivariate_normal.cdf(np.column_stack([z1, z2]),
                                          cov=[[1, rho], [rho, 1]])

rho = 0.5
C_gaussian = [gaussian_copula(ui, vi, rho) for ui, vi in zip(u, v)]
print(f"\n高斯Copula (ρ=0.5): {C_gaussian}")


## 8.2.1 高斯Copula模拟
np.random.seed(123)
n = 1000
rho = 0.7

z1 = np.random.normal(0, 1, n)
z2 = rho * z1 + np.sqrt(1 - rho**2) * np.random.normal(0, 1, n)

u1 = stats.norm.cdf(z1)
u2 = stats.norm.cdf(z2)

# 转换为指数分布
x1_g = stats.expon.ppf(u1)
x2_g = stats.expon.ppf(u2)

print(f"\n高斯Copula模拟: 经验相关系数={np.corrcoef(x1_g, x2_g)[0,1]:.4f}")

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(x1_g, x2_g, alpha=0.3)
ax.set_xlabel('x1'); ax.set_ylabel('x2')
ax.set_title('高斯Copula模拟散点图')
plt.tight_layout(); plt.show()


##############################################################################
# 8.3 T Copula
##############################################################################
np.random.seed(456)
rho_t = 0.5
df_t = 4

z1_t = np.random.normal(0, 1, n)
z2_t = rho_t * z1_t + np.sqrt(1 - rho_t**2) * np.random.normal(0, 1, n)
w = np.random.chisquare(df_t) / df_t
z1_t = z1_t / np.sqrt(w)
z2_t = z2_t / np.sqrt(w)

u1_t = stats.t.cdf(z1_t, df_t)
u2_t = stats.t.cdf(z2_t, df_t)
x1_t = stats.expon.ppf(u1_t)
x2_t = stats.expon.ppf(u2_t)

print(f"\nT Copula模拟 (df=4): 经验相关系数={np.corrcoef(x1_t, x2_t)[0,1]:.4f}")


##############################################################################
# 8.4 阿基米德Copula
##############################################################################

## 8.4.1 Gumbel Copula
def gumbel_copula(u, v, theta):
    return np.exp(-((-np.log(u))**theta + (-np.log(v))**theta)**(1/theta))

C_gumbel = gumbel_copula(u, v, 2)
print(f"\nGumbel Copula (θ=2): {C_gumbel}")

## 8.4.2 Clayton Copula
def clayton_copula(u, v, theta):
    return (u**(-theta) + v**(-theta) - 1)**(-1/theta)

C_clayton = clayton_copula(u, v, 2)
print(f"Clayton Copula (θ=2): {C_clayton}")

## 8.4.3 Frank Copula
def frank_copula(u, v, theta):
    return -1/theta * np.log(1 + (np.exp(-theta * u) - 1) *
                              (np.exp(-theta * v) - 1) / (np.exp(-theta) - 1))

C_frank = frank_copula(u, v, 5)
print(f"Frank Copula (θ=5): {C_frank}")


## 8.4.4 Clayton Copula模拟
np.random.seed(789)
theta_c = 2

u1_c = np.random.uniform(0, 1, n)
t_c = np.random.uniform(0, 1, n)
u2_c = (1 - u1_c**(-theta_c) +
        u1_c**(-theta_c) * t_c**(-theta_c/(1+theta_c)))**(-1/theta_c)

x1_c = stats.expon.ppf(u1_c)
x2_c = stats.expon.ppf(u2_c)

print(f"\nClayton Copula模拟 (θ=2): 经验相关系数={np.corrcoef(x1_c, x2_c)[0,1]:.4f}")

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(x1_c, x2_c, alpha=0.3, color='red')
ax.set_xlabel('x1'); ax.set_ylabel('x2')
ax.set_title('Clayton Copula模拟散点图')
plt.tight_layout(); plt.show()


##############################################################################
# 8.5 尾部相关性
##############################################################################
print(f"\n尾部相关性:")

# 高斯Copula: 尾部相关性为0
print("高斯Copula: λ_upper=0, λ_lower=0 (ρ<1)")

# T Copula
print("T Copula (df=4):")
for rho in [0.0, 0.3, 0.5, 0.7, 0.9]:
    df = 4
    lam = 2 * stats.t.ppf(1 - 0.5, df + 1) * (1 - rho) / \
          np.sqrt((df + 1) * (1 - rho**2))
    print(f"  ρ={rho}: λ={lam:.4f}")

# Clayton Copula下尾
print("Clayton Copula下尾相关性:")
for theta in [1, 2, 5, 10]:
    lam_lower = 2**(-1/theta)
    print(f"  θ={theta}: λ_lower={lam_lower:.4f}")

# Gumbel Copula上尾
print("Gumbel Copula上尾相关性:")
for theta in [1, 2, 5, 10]:
    lam_upper = 2 - 2**(1/theta)
    print(f"  θ={theta}: λ_upper={lam_upper:.4f}")


##############################################################################
# 8.6 Copula参数估计
##############################################################################
np.random.seed(123)
theta_true = 2

u1_est = np.random.uniform(0, 1, n)
t_est = np.random.uniform(0, 1, n)
u2_est = (1 - u1_est**(-theta_true) +
          u1_est**(-theta_true) * t_est**(-theta_true/(1+theta_true)))**(-1/theta_true)
x1_est = stats.expon.ppf(u1_est)
x2_est = stats.expon.ppf(u2_est)

# 方法1：基于Kendall tau的矩估计
from scipy.stats import kendalltau
tau, _ = kendalltau(x1_est, x2_est)
theta_hat = 2 * tau / (1 - tau)
print(f"\nClayton Copula参数估计:")
print(f"  Kendall tau = {tau:.4f}")
print(f"  θ_hat（矩估计）= {theta_hat:.4f}")

# 方法2：极大似然估计
def clayton_nll(theta, u1, u2):
    if theta <= 0:
        return 1e10
    c = (1 + theta) * (u1 * u2)**(-theta - 1) * \
        (u1**(-theta) + u2**(-theta) - 1)**(-2 - 1/theta)
    return -np.sum(np.log(c))

result = minimize(clayton_nll, x0=[1.0], args=(u1_est, u2_est),
                  method='Nelder-Mead')
print(f"  θ_hat（MLE）= {result.x[0]:.4f}")


##############################################################################
# 8.7 多元Copula应用
##############################################################################
np.random.seed(202)
n_multi = 1000

Sigma = np.array([[1.0, 0.5, 0.3],
                   [0.5, 1.0, 0.4],
                   [0.3, 0.4, 1.0]])

L = np.linalg.cholesky(Sigma)
Z = np.random.normal(0, 1, (n_multi, 3)) @ L.T
U = stats.norm.cdf(Z)

# 不同边缘分布
x1_multi = stats.expon.ppf(U[:, 0])
x2_multi = stats.gamma.ppf(U[:, 1], 2)
x3_multi = stats.lognorm.ppf(U[:, 2], s=1)

corr_emp = np.corrcoef(np.column_stack([x1_multi, x2_multi, x3_multi]).T)
print(f"\n三元高斯Copula:")
print(f"  经验相关矩阵:")
print(f"  {corr_emp}")


##############################################################################
# 8.8 Copula比较
##############################################################################
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 高斯Copula
axes[0,0].scatter(x1_g[:500], x2_g[:500], alpha=0.3)
axes[0,0].set_title('高斯Copula')

# T Copula
axes[0,1].scatter(x1_t[:500], x2_t[:500], alpha=0.3, color='orange')
axes[0,1].set_title('T Copula')

# Clayton Copula
axes[1,0].scatter(x1_c[:500], x2_c[:500], alpha=0.3, color='red')
axes[1,0].set_title('Clayton Copula')

# Gumbel Copula模拟
np.random.seed(111)
theta_gu = 2
u1_gu = np.random.uniform(0, 1, n)
t_gu = np.random.uniform(0, 1, n)
# Gumbel Copula条件分布法（简化）
u2_gu = np.exp(-((-np.log(u1_gu))**theta_gu +
                  (-np.log(t_gu))**theta_gu)**(1/theta_gu))
x1_gu = stats.expon.ppf(u1_gu)
x2_gu = stats.expon.ppf(u2_gu)
axes[1,1].scatter(x1_gu[:500], x2_gu[:500], alpha=0.3, color='green')
axes[1,1].set_title('Gumbel Copula')

plt.tight_layout(); plt.show()

##############################################################################
# 第8章 相依风险与Copula
# 对应教材：section8.tex
# 内容：Copula概念、常见Copula函数、参数估计、尾部相关性、
#       多元Copula、vine Copula
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
# 8.1 Copula基本概念
##############################################################################
u = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
v = np.array([0.2, 0.4, 0.6, 0.8, 1.0])

# 独立Copula
C_indep = u * v
print(f"独立Copula: {C_indep}")


##############################################################################
# 8.2 高斯Copula
##############################################################################
def gaussian_copula(u, v, rho):
    """高斯Copula"""
    z1 = stats.norm.ppf(u)
    z2 = stats.norm.ppf(v)
    return stats.multivariate_normal.cdf(np.column_stack([z1, z2]),
                                          cov=[[1, rho], [rho, 1]])

rho = 0.5
C_gaussian = [gaussian_copula(ui, vi, rho) for ui, vi in zip(u, v)]
print(f"\n高斯Copula (ρ=0.5): {C_gaussian}")


## 8.2.1 高斯Copula模拟
np.random.seed(123)
n = 1000
rho = 0.7

z1 = np.random.normal(0, 1, n)
z2 = rho * z1 + np.sqrt(1 - rho**2) * np.random.normal(0, 1, n)

u1 = stats.norm.cdf(z1)
u2 = stats.norm.cdf(z2)

# 转换为指数分布
x1_g = stats.expon.ppf(u1)
x2_g = stats.expon.ppf(u2)

print(f"\n高斯Copula模拟: 经验相关系数={np.corrcoef(x1_g, x2_g)[0,1]:.4f}")

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(x1_g, x2_g, alpha=0.3)
ax.set_xlabel('x1'); ax.set_ylabel('x2')
ax.set_title('高斯Copula模拟散点图')
plt.tight_layout(); plt.show()


##############################################################################
# 8.3 T Copula
##############################################################################
np.random.seed(456)
rho_t = 0.5
df_t = 4

z1_t = np.random.normal(0, 1, n)
z2_t = rho_t * z1_t + np.sqrt(1 - rho_t**2) * np.random.normal(0, 1, n)
w = np.random.chisquare(df_t) / df_t
z1_t = z1_t / np.sqrt(w)
z2_t = z2_t / np.sqrt(w)

u1_t = stats.t.cdf(z1_t, df_t)
u2_t = stats.t.cdf(z2_t, df_t)
x1_t = stats.expon.ppf(u1_t)
x2_t = stats.expon.ppf(u2_t)

print(f"\nT Copula模拟 (df=4): 经验相关系数={np.corrcoef(x1_t, x2_t)[0,1]:.4f}")


##############################################################################
# 8.4 阿基米德Copula
##############################################################################

## 8.4.1 Gumbel Copula
def gumbel_copula(u, v, theta):
    return np.exp(-((-np.log(u))**theta + (-np.log(v))**theta)**(1/theta))

C_gumbel = gumbel_copula(u, v, 2)
print(f"\nGumbel Copula (θ=2): {C_gumbel}")

## 8.4.2 Clayton Copula
def clayton_copula(u, v, theta):
    return (u**(-theta) + v**(-theta) - 1)**(-1/theta)

C_clayton = clayton_copula(u, v, 2)
print(f"Clayton Copula (θ=2): {C_clayton}")

## 8.4.3 Frank Copula
def frank_copula(u, v, theta):
    return -1/theta * np.log(1 + (np.exp(-theta * u) - 1) *
                              (np.exp(-theta * v) - 1) / (np.exp(-theta) - 1))

C_frank = frank_copula(u, v, 5)
print(f"Frank Copula (θ=5): {C_frank}")


## 8.4.4 Clayton Copula模拟
np.random.seed(789)
theta_c = 2

u1_c = np.random.uniform(0, 1, n)
t_c = np.random.uniform(0, 1, n)
u2_c = (1 - u1_c**(-theta_c) +
        u1_c**(-theta_c) * t_c**(-theta_c/(1+theta_c)))**(-1/theta_c)

x1_c = stats.expon.ppf(u1_c)
x2_c = stats.expon.ppf(u2_c)

print(f"\nClayton Copula模拟 (θ=2): 经验相关系数={np.corrcoef(x1_c, x2_c)[0,1]:.4f}")

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(x1_c, x2_c, alpha=0.3, color='red')
ax.set_xlabel('x1'); ax.set_ylabel('x2')
ax.set_title('Clayton Copula模拟散点图')
plt.tight_layout(); plt.show()


##############################################################################
# 8.5 尾部相关性
##############################################################################
print(f"\n尾部相关性:")

# 高斯Copula: 尾部相关性为0
print("高斯Copula: λ_upper=0, λ_lower=0 (ρ<1)")

# T Copula
print("T Copula (df=4):")
for rho in [0.0, 0.3, 0.5, 0.7, 0.9]:
    df = 4
    lam = 2 * stats.t.ppf(1 - 0.5, df + 1) * (1 - rho) / \
          np.sqrt((df + 1) * (1 - rho**2))
    print(f"  ρ={rho}: λ={lam:.4f}")

# Clayton Copula下尾
print("Clayton Copula下尾相关性:")
for theta in [1, 2, 5, 10]:
    lam_lower = 2**(-1/theta)
    print(f"  θ={theta}: λ_lower={lam_lower:.4f}")

# Gumbel Copula上尾
print("Gumbel Copula上尾相关性:")
for theta in [1, 2, 5, 10]:
    lam_upper = 2 - 2**(1/theta)
    print(f"  θ={theta}: λ_upper={lam_upper:.4f}")


##############################################################################
# 8.6 Copula参数估计
##############################################################################
np.random.seed(123)
theta_true = 2

u1_est = np.random.uniform(0, 1, n)
t_est = np.random.uniform(0, 1, n)
u2_est = (1 - u1_est**(-theta_true) +
          u1_est**(-theta_true) * t_est**(-theta_true/(1+theta_true)))**(-1/theta_true)
x1_est = stats.expon.ppf(u1_est)
x2_est = stats.expon.ppf(u2_est)

# 方法1：基于Kendall tau的矩估计
from scipy.stats import kendalltau
tau, _ = kendalltau(x1_est, x2_est)
theta_hat = 2 * tau / (1 - tau)
print(f"\nClayton Copula参数估计:")
print(f"  Kendall tau = {tau:.4f}")
print(f"  θ_hat（矩估计）= {theta_hat:.4f}")

# 方法2：极大似然估计
def clayton_nll(theta, u1, u2):
    if theta <= 0:
        return 1e10
    c = (1 + theta) * (u1 * u2)**(-theta - 1) * \
        (u1**(-theta) + u2**(-theta) - 1)**(-2 - 1/theta)
    return -np.sum(np.log(c))

result = minimize(clayton_nll, x0=[1.0], args=(u1_est, u2_est),
                  method='Nelder-Mead')
print(f"  θ_hat（MLE）= {result.x[0]:.4f}")


##############################################################################
# 8.7 多元Copula应用
##############################################################################
np.random.seed(202)
n_multi = 1000

Sigma = np.array([[1.0, 0.5, 0.3],
                   [0.5, 1.0, 0.4],
                   [0.3, 0.4, 1.0]])

L = np.linalg.cholesky(Sigma)
Z = np.random.normal(0, 1, (n_multi, 3)) @ L.T
U = stats.norm.cdf(Z)

# 不同边缘分布
x1_multi = stats.expon.ppf(U[:, 0])
x2_multi = stats.gamma.ppf(U[:, 1], 2)
x3_multi = stats.lognorm.ppf(U[:, 2], s=1)

corr_emp = np.corrcoef(np.column_stack([x1_multi, x2_multi, x3_multi]).T)
print(f"\n三元高斯Copula:")
print(f"  经验相关矩阵:")
print(f"  {corr_emp}")


##############################################################################
# 8.8 Copula比较
##############################################################################
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 高斯Copula
axes[0,0].scatter(x1_g[:500], x2_g[:500], alpha=0.3)
axes[0,0].set_title('高斯Copula')

# T Copula
axes[0,1].scatter(x1_t[:500], x2_t[:500], alpha=0.3, color='orange')
axes[0,1].set_title('T Copula')

# Clayton Copula
axes[1,0].scatter(x1_c[:500], x2_c[:500], alpha=0.3, color='red')
axes[1,0].set_title('Clayton Copula')

# Gumbel Copula模拟
np.random.seed(111)
theta_gu = 2
u1_gu = np.random.uniform(0, 1, n)
t_gu = np.random.uniform(0, 1, n)
# Gumbel Copula条件分布法（简化）
u2_gu = np.exp(-((-np.log(u1_gu))**theta_gu +
                  (-np.log(t_gu))**theta_gu)**(1/theta_gu))
x1_gu = stats.expon.ppf(u1_gu)
x2_gu = stats.expon.ppf(u2_gu)
axes[1,1].scatter(x1_gu[:500], x2_gu[:500], alpha=0.3, color='green')
axes[1,1].set_title('Gumbel Copula')

plt.tight_layout(); plt.show()

##############################################################################
# 第8章 相依风险与Copula
# 对应教材：section8.tex
# 内容：Copula概念、常见Copula函数、参数估计、尾部相关性、
#       多元Copula、vine Copula
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
# 8.1 Copula基本概念
##############################################################################
u = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
v = np.array([0.2, 0.4, 0.6, 0.8, 1.0])

# 独立Copula
C_indep = u * v
print(f"独立Copula: {C_indep}")


##############################################################################
# 8.2 高斯Copula
##############################################################################
def gaussian_copula(u, v, rho):
    """高斯Copula"""
    z1 = stats.norm.ppf(u)
    z2 = stats.norm.ppf(v)
    return stats.multivariate_normal.cdf(np.column_stack([z1, z2]),
                                          cov=[[1, rho], [rho, 1]])

rho = 0.5
C_gaussian = [gaussian_copula(ui, vi, rho) for ui, vi in zip(u, v)]
print(f"\n高斯Copula (ρ=0.5): {C_gaussian}")


## 8.2.1 高斯Copula模拟
np.random.seed(123)
n = 1000
rho = 0.7

z1 = np.random.normal(0, 1, n)
z2 = rho * z1 + np.sqrt(1 - rho**2) * np.random.normal(0, 1, n)

u1 = stats.norm.cdf(z1)
u2 = stats.norm.cdf(z2)

# 转换为指数分布
x1_g = stats.expon.ppf(u1)
x2_g = stats.expon.ppf(u2)

print(f"\n高斯Copula模拟: 经验相关系数={np.corrcoef(x1_g, x2_g)[0,1]:.4f}")

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(x1_g, x2_g, alpha=0.3)
ax.set_xlabel('x1'); ax.set_ylabel('x2')
ax.set_title('高斯Copula模拟散点图')
plt.tight_layout(); plt.show()


##############################################################################
# 8.3 T Copula
##############################################################################
np.random.seed(456)
rho_t = 0.5
df_t = 4

z1_t = np.random.normal(0, 1, n)
z2_t = rho_t * z1_t + np.sqrt(1 - rho_t**2) * np.random.normal(0, 1, n)
w = np.random.chisquare(df_t) / df_t
z1_t = z1_t / np.sqrt(w)
z2_t = z2_t / np.sqrt(w)

u1_t = stats.t.cdf(z1_t, df_t)
u2_t = stats.t.cdf(z2_t, df_t)
x1_t = stats.expon.ppf(u1_t)
x2_t = stats.expon.ppf(u2_t)

print(f"\nT Copula模拟 (df=4): 经验相关系数={np.corrcoef(x1_t, x2_t)[0,1]:.4f}")


##############################################################################
# 8.4 阿基米德Copula
##############################################################################

## 8.4.1 Gumbel Copula
def gumbel_copula(u, v, theta):
    return np.exp(-((-np.log(u))**theta + (-np.log(v))**theta)**(1/theta))

C_gumbel = gumbel_copula(u, v, 2)
print(f"\nGumbel Copula (θ=2): {C_gumbel}")

## 8.4.2 Clayton Copula
def clayton_copula(u, v, theta):
    return (u**(-theta) + v**(-theta) - 1)**(-1/theta)

C_clayton = clayton_copula(u, v, 2)
print(f"Clayton Copula (θ=2): {C_clayton}")

## 8.4.3 Frank Copula
def frank_copula(u, v, theta):
    return -1/theta * np.log(1 + (np.exp(-theta * u) - 1) *
                              (np.exp(-theta * v) - 1) / (np.exp(-theta) - 1))

C_frank = frank_copula(u, v, 5)
print(f"Frank Copula (θ=5): {C_frank}")


## 8.4.4 Clayton Copula模拟
np.random.seed(789)
theta_c = 2

u1_c = np.random.uniform(0, 1, n)
t_c = np.random.uniform(0, 1, n)
u2_c = (1 - u1_c**(-theta_c) +
        u1_c**(-theta_c) * t_c**(-theta_c/(1+theta_c)))**(-1/theta_c)

x1_c = stats.expon.ppf(u1_c)
x2_c = stats.expon.ppf(u2_c)

print(f"\nClayton Copula模拟 (θ=2): 经验相关系数={np.corrcoef(x1_c, x2_c)[0,1]:.4f}")

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(x1_c, x2_c, alpha=0.3, color='red')
ax.set_xlabel('x1'); ax.set_ylabel('x2')
ax.set_title('Clayton Copula模拟散点图')
plt.tight_layout(); plt.show()


##############################################################################
# 8.5 尾部相关性
##############################################################################
print(f"\n尾部相关性:")

# 高斯Copula: 尾部相关性为0
print("高斯Copula: λ_upper=0, λ_lower=0 (ρ<1)")

# T Copula
print("T Copula (df=4):")
for rho in [0.0, 0.3, 0.5, 0.7, 0.9]:
    df = 4
    lam = 2 * stats.t.ppf(1 - 0.5, df + 1) * (1 - rho) / \
          np.sqrt((df + 1) * (1 - rho**2))
    print(f"  ρ={rho}: λ={lam:.4f}")

# Clayton Copula下尾
print("Clayton Copula下尾相关性:")
for theta in [1, 2, 5, 10]:
    lam_lower = 2**(-1/theta)
    print(f"  θ={theta}: λ_lower={lam_lower:.4f}")

# Gumbel Copula上尾
print("Gumbel Copula上尾相关性:")
for theta in [1, 2, 5, 10]:
    lam_upper = 2 - 2**(1/theta)
    print(f"  θ={theta}: λ_upper={lam_upper:.4f}")


##############################################################################
# 8.6 Copula参数估计
##############################################################################
np.random.seed(123)
theta_true = 2

u1_est = np.random.uniform(0, 1, n)
t_est = np.random.uniform(0, 1, n)
u2_est = (1 - u1_est**(-theta_true) +
          u1_est**(-theta_true) * t_est**(-theta_true/(1+theta_true)))**(-1/theta_true)
x1_est = stats.expon.ppf(u1_est)
x2_est = stats.expon.ppf(u2_est)

# 方法1：基于Kendall tau的矩估计
from scipy.stats import kendalltau
tau, _ = kendalltau(x1_est, x2_est)
theta_hat = 2 * tau / (1 - tau)
print(f"\nClayton Copula参数估计:")
print(f"  Kendall tau = {tau:.4f}")
print(f"  θ_hat（矩估计）= {theta_hat:.4f}")

# 方法2：极大似然估计
def clayton_nll(theta, u1, u2):
    if theta <= 0:
        return 1e10
    c = (1 + theta) * (u1 * u2)**(-theta - 1) * \
        (u1**(-theta) + u2**(-theta) - 1)**(-2 - 1/theta)
    return -np.sum(np.log(c))

result = minimize(clayton_nll, x0=[1.0], args=(u1_est, u2_est),
                  method='Nelder-Mead')
print(f"  θ_hat（MLE）= {result.x[0]:.4f}")


##############################################################################
# 8.7 多元Copula应用
##############################################################################
np.random.seed(202)
n_multi = 1000

Sigma = np.array([[1.0, 0.5, 0.3],
                   [0.5, 1.0, 0.4],
                   [0.3, 0.4, 1.0]])

L = np.linalg.cholesky(Sigma)
Z = np.random.normal(0, 1, (n_multi, 3)) @ L.T
U = stats.norm.cdf(Z)

# 不同边缘分布
x1_multi = stats.expon.ppf(U[:, 0])
x2_multi = stats.gamma.ppf(U[:, 1], 2)
x3_multi = stats.lognorm.ppf(U[:, 2], s=1)

corr_emp = np.corrcoef(np.column_stack([x1_multi, x2_multi, x3_multi]).T)
print(f"\n三元高斯Copula:")
print(f"  经验相关矩阵:")
print(f"  {corr_emp}")


##############################################################################
# 8.8 Copula比较
##############################################################################
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 高斯Copula
axes[0,0].scatter(x1_g[:500], x2_g[:500], alpha=0.3)
axes[0,0].set_title('高斯Copula')

# T Copula
axes[0,1].scatter(x1_t[:500], x2_t[:500], alpha=0.3, color='orange')
axes[0,1].set_title('T Copula')

# Clayton Copula
axes[1,0].scatter(x1_c[:500], x2_c[:500], alpha=0.3, color='red')
axes[1,0].set_title('Clayton Copula')

# Gumbel Copula模拟
np.random.seed(111)
theta_gu = 2
u1_gu = np.random.uniform(0, 1, n)
t_gu = np.random.uniform(0, 1, n)
# Gumbel Copula条件分布法（简化）
u2_gu = np.exp(-((-np.log(u1_gu))**theta_gu +
                  (-np.log(t_gu))**theta_gu)**(1/theta_gu))
x1_gu = stats.expon.ppf(u1_gu)
x2_gu = stats.expon.ppf(u2_gu)
axes[1,1].scatter(x1_gu[:500], x2_gu[:500], alpha=0.3, color='green')
axes[1,1].set_title('Gumbel Copula')

plt.tight_layout(); plt.show()

# 8.4.1 基于相依性测度的Copula参数估计
import numpy as np
from scipy import stats
from scipy.stats import kendalltau
from scipy.optimize import minimize

# 使用模拟数据演示
np.random.seed(101)
n_obs = 100

# 使用Clayton Copula生成相依数据
theta_true = 3.0
u1 = np.random.uniform(0, 1, n_obs)
t = np.random.uniform(0, 1, n_obs)
u2 = (1 + u1**(-theta_true) * (t**(-theta_true/(1+theta_true)) - 1))**(-1/theta_true)

# 转换为伽马分布的边际
loss = stats.gamma.ppf(u1, a=2, scale=1/0.001)
alae = stats.gamma.ppf(u2, a=3, scale=1/0.0005)
loss_ALAE = np.column_stack([loss, alae])

# 计算Kendall秩相关系数
tau_kendall, _ = kendalltau(loss_ALAE[:, 0], loss_ALAE[:, 1])
print(f"Kendall秩相关系数: {tau_kendall:.4f}")

# 计算各Copula参数的估计值
# Clayton: theta = 2*tau/(1-tau)
theta_clayton = 2 * tau_kendall / (1 - tau_kendall)
print(f"Clayton Copula参数: {theta_clayton:.4f}")

# Frank: 通过数值求解
def frank_tau_eq(theta, tau):
    if abs(theta) < 1e-10:
        return tau
    return 1 - 4/theta * (1 - 4/(theta*(1 - np.exp(-theta)))) - tau

from scipy.optimize import brentq
theta_frank = brentq(frank_tau_eq, -50, 50, args=(tau_kendall,))
print(f"Frank Copula参数: {theta_frank:.4f}")

# Gumbel: theta = 1/(1-tau)
theta_gumbel = 1 / (1 - tau_kendall)
print(f"Gumbel Copula参数: {theta_gumbel:.4f}")

##############################################################################
# 第8章 相依风险与Copula
# 对应教材：section8.tex
# 内容：Copula概念、常见Copula函数、参数估计、尾部相关性、
#       多元Copula、vine Copula
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
# 8.1 Copula基本概念
##############################################################################
u = np.array([0.1, 0.3, 0.5, 0.7, 0.9])
v = np.array([0.2, 0.4, 0.6, 0.8, 1.0])

# 独立Copula
C_indep = u * v
print(f"独立Copula: {C_indep}")


##############################################################################
# 8.2 高斯Copula
##############################################################################
def gaussian_copula(u, v, rho):
    """高斯Copula"""
    z1 = stats.norm.ppf(u)
    z2 = stats.norm.ppf(v)
    return stats.multivariate_normal.cdf(np.column_stack([z1, z2]),
                                          cov=[[1, rho], [rho, 1]])

rho = 0.5
C_gaussian = [gaussian_copula(ui, vi, rho) for ui, vi in zip(u, v)]
print(f"\n高斯Copula (ρ=0.5): {C_gaussian}")


## 8.2.1 高斯Copula模拟
np.random.seed(123)
n = 1000
rho = 0.7

z1 = np.random.normal(0, 1, n)
z2 = rho * z1 + np.sqrt(1 - rho**2) * np.random.normal(0, 1, n)

u1 = stats.norm.cdf(z1)
u2 = stats.norm.cdf(z2)

# 转换为指数分布
x1_g = stats.expon.ppf(u1)
x2_g = stats.expon.ppf(u2)

print(f"\n高斯Copula模拟: 经验相关系数={np.corrcoef(x1_g, x2_g)[0,1]:.4f}")

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(x1_g, x2_g, alpha=0.3)
ax.set_xlabel('x1'); ax.set_ylabel('x2')
ax.set_title('高斯Copula模拟散点图')
plt.tight_layout(); plt.show()


##############################################################################
# 8.3 T Copula
##############################################################################
np.random.seed(456)
rho_t = 0.5
df_t = 4

z1_t = np.random.normal(0, 1, n)
z2_t = rho_t * z1_t + np.sqrt(1 - rho_t**2) * np.random.normal(0, 1, n)
w = np.random.chisquare(df_t) / df_t
z1_t = z1_t / np.sqrt(w)
z2_t = z2_t / np.sqrt(w)

u1_t = stats.t.cdf(z1_t, df_t)
u2_t = stats.t.cdf(z2_t, df_t)
x1_t = stats.expon.ppf(u1_t)
x2_t = stats.expon.ppf(u2_t)

print(f"\nT Copula模拟 (df=4): 经验相关系数={np.corrcoef(x1_t, x2_t)[0,1]:.4f}")


##############################################################################
# 8.4 阿基米德Copula
##############################################################################

## 8.4.1 Gumbel Copula
def gumbel_copula(u, v, theta):
    return np.exp(-((-np.log(u))**theta + (-np.log(v))**theta)**(1/theta))

C_gumbel = gumbel_copula(u, v, 2)
print(f"\nGumbel Copula (θ=2): {C_gumbel}")

## 8.4.2 Clayton Copula
def clayton_copula(u, v, theta):
    return (u**(-theta) + v**(-theta) - 1)**(-1/theta)

C_clayton = clayton_copula(u, v, 2)
print(f"Clayton Copula (θ=2): {C_clayton}")

## 8.4.3 Frank Copula
def frank_copula(u, v, theta):
    return -1/theta * np.log(1 + (np.exp(-theta * u) - 1) *
                              (np.exp(-theta * v) - 1) / (np.exp(-theta) - 1))

C_frank = frank_copula(u, v, 5)
print(f"Frank Copula (θ=5): {C_frank}")


## 8.4.4 Clayton Copula模拟
np.random.seed(789)
theta_c = 2

u1_c = np.random.uniform(0, 1, n)
t_c = np.random.uniform(0, 1, n)
u2_c = (1 - u1_c**(-theta_c) +
        u1_c**(-theta_c) * t_c**(-theta_c/(1+theta_c)))**(-1/theta_c)

x1_c = stats.expon.ppf(u1_c)
x2_c = stats.expon.ppf(u2_c)

print(f"\nClayton Copula模拟 (θ=2): 经验相关系数={np.corrcoef(x1_c, x2_c)[0,1]:.4f}")

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(x1_c, x2_c, alpha=0.3, color='red')
ax.set_xlabel('x1'); ax.set_ylabel('x2')
ax.set_title('Clayton Copula模拟散点图')
plt.tight_layout(); plt.show()


##############################################################################
# 8.5 尾部相关性
##############################################################################
print(f"\n尾部相关性:")

# 高斯Copula: 尾部相关性为0
print("高斯Copula: λ_upper=0, λ_lower=0 (ρ<1)")

# T Copula
print("T Copula (df=4):")
for rho in [0.0, 0.3, 0.5, 0.7, 0.9]:
    df = 4
    lam = 2 * stats.t.ppf(1 - 0.5, df + 1) * (1 - rho) / \
          np.sqrt((df + 1) * (1 - rho**2))
    print(f"  ρ={rho}: λ={lam:.4f}")

# Clayton Copula下尾
print("Clayton Copula下尾相关性:")
for theta in [1, 2, 5, 10]:
    lam_lower = 2**(-1/theta)
    print(f"  θ={theta}: λ_lower={lam_lower:.4f}")

# Gumbel Copula上尾
print("Gumbel Copula上尾相关性:")
for theta in [1, 2, 5, 10]:
    lam_upper = 2 - 2**(1/theta)
    print(f"  θ={theta}: λ_upper={lam_upper:.4f}")


##############################################################################
# 8.6 Copula参数估计
##############################################################################
np.random.seed(123)
theta_true = 2

u1_est = np.random.uniform(0, 1, n)
t_est = np.random.uniform(0, 1, n)
u2_est = (1 - u1_est**(-theta_true) +
          u1_est**(-theta_true) * t_est**(-theta_true/(1+theta_true)))**(-1/theta_true)
x1_est = stats.expon.ppf(u1_est)
x2_est = stats.expon.ppf(u2_est)

# 方法1：基于Kendall tau的矩估计
from scipy.stats import kendalltau
tau, _ = kendalltau(x1_est, x2_est)
theta_hat = 2 * tau / (1 - tau)
print(f"\nClayton Copula参数估计:")
print(f"  Kendall tau = {tau:.4f}")
print(f"  θ_hat（矩估计）= {theta_hat:.4f}")

# 方法2：极大似然估计
def clayton_nll(theta, u1, u2):
    if theta <= 0:
        return 1e10
    c = (1 + theta) * (u1 * u2)**(-theta - 1) * \
        (u1**(-theta) + u2**(-theta) - 1)**(-2 - 1/theta)
    return -np.sum(np.log(c))

result = minimize(clayton_nll, x0=[1.0], args=(u1_est, u2_est),
                  method='Nelder-Mead')
print(f"  θ_hat（MLE）= {result.x[0]:.4f}")


##############################################################################
# 8.7 多元Copula应用
##############################################################################
np.random.seed(202)
n_multi = 1000

Sigma = np.array([[1.0, 0.5, 0.3],
                   [0.5, 1.0, 0.4],
                   [0.3, 0.4, 1.0]])

L = np.linalg.cholesky(Sigma)
Z = np.random.normal(0, 1, (n_multi, 3)) @ L.T
U = stats.norm.cdf(Z)

# 不同边缘分布
x1_multi = stats.expon.ppf(U[:, 0])
x2_multi = stats.gamma.ppf(U[:, 1], 2)
x3_multi = stats.lognorm.ppf(U[:, 2], s=1)

corr_emp = np.corrcoef(np.column_stack([x1_multi, x2_multi, x3_multi]).T)
print(f"\n三元高斯Copula:")
print(f"  经验相关矩阵:")
print(f"  {corr_emp}")


##############################################################################
# 8.8 Copula比较
##############################################################################
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 高斯Copula
axes[0,0].scatter(x1_g[:500], x2_g[:500], alpha=0.3)
axes[0,0].set_title('高斯Copula')

# T Copula
axes[0,1].scatter(x1_t[:500], x2_t[:500], alpha=0.3, color='orange')
axes[0,1].set_title('T Copula')

# Clayton Copula
axes[1,0].scatter(x1_c[:500], x2_c[:500], alpha=0.3, color='red')
axes[1,0].set_title('Clayton Copula')

# Gumbel Copula模拟
np.random.seed(111)
theta_gu = 2
u1_gu = np.random.uniform(0, 1, n)
t_gu = np.random.uniform(0, 1, n)
# Gumbel Copula条件分布法（简化）
u2_gu = np.exp(-((-np.log(u1_gu))**theta_gu +
                  (-np.log(t_gu))**theta_gu)**(1/theta_gu))
x1_gu = stats.expon.ppf(u1_gu)
x2_gu = stats.expon.ppf(u2_gu)
axes[1,1].scatter(x1_gu[:500], x2_gu[:500], alpha=0.3, color='green')
axes[1,1].set_title('Gumbel Copula')

plt.tight_layout(); plt.show()

##############################################################################
# 8.2.5 应用复合函数生成Copula
##############################################################################
alpha = 2

def tau_inv(t, alpha):
    """Clayton Copula的拉普拉斯逆变换"""
    return t**(-alpha) - 1

def psi_clayton(t, alpha):
    """Clayton生成函数"""
    return (t**(-alpha) - 1) / alpha

# 验证：tau^{-1}(t) = alpha * psi(t)
t_vals = np.arange(0.01, 1.0, 0.1)
print("验证 tau^{-1}(t) = alpha * psi(t):")
for t in t_vals:
    print(f"  t={t:.2f}: tau_inv={tau_inv(t, alpha):.4f}, "
          f"alpha*psi={alpha * psi_clayton(t, alpha):.4f}")

# 基于脆弱性模型的Copula模拟
np.random.seed(2024)
alpha = 2
n = 1000

gamma = np.random.gamma(shape=1/alpha, scale=1, size=n)
U1 = np.random.uniform(0, 1, n)
U2 = np.random.uniform(0, 1, n)

def tau_laplace(s, alpha):
    """Clayton对应的拉普拉斯变换"""
    return (1 + s)**(-1/alpha)

u1 = tau_laplace(-np.log(U1) / gamma, alpha)
u2 = tau_laplace(-np.log(U2) / gamma, alpha)

x1 = stats.expon.ppf(u1)
x2 = stats.expon.ppf(u2)

from scipy.stats import kendalltau
tau_emp, _ = kendalltau(x1, x2)
print(f"\n脆弱性模型模拟 (alpha={alpha}):")
print(f"  Kendall tau (经验) = {tau_emp:.4f}")
print(f"  Kendall tau (理论) = {alpha/(alpha+2):.4f}")


##############################################################################
# 8.3.1 高斯Copula的模拟
##############################################################################
np.random.seed(123)

rho = 0.6
n = 2000

# Cholesky分解法
Sigma = np.array([[1, rho], [rho, 1]])
L = np.linalg.cholesky(Sigma)
Y = np.random.normal(0, 1, (n, 2))
Z = Y @ L.T
U = stats.norm.cdf(Z)

x1 = stats.expon.ppf(U[:, 0])
x2 = stats.expon.ppf(U[:, 1])

tau_emp, _ = kendalltau(x1, x2)
tau_theory = 2 / np.pi * np.arcsin(rho / 2)

print(f"\n二元高斯Copula模拟 (rho=0.6):")
print(f"  Kendall tau (经验) = {tau_emp:.4f}")
print(f"  Kendall tau (理论) = {tau_theory:.4f}")

# 多元高斯Copula模拟 (d=3)
Sigma3 = np.array([[1.0, 0.5, 0.3],
                    [0.5, 1.0, 0.4],
                    [0.3, 0.4, 1.0]])
L3 = np.linalg.cholesky(Sigma3)
Y3 = np.random.normal(0, 1, (n, 3))
Z3 = Y3 @ L3.T
U3 = stats.norm.cdf(Z3)

x1_m = stats.expon.ppf(U3[:, 0])
x2_m = stats.gamma.ppf(U3[:, 1], 2)
x3_m = stats.lognorm.ppf(U3[:, 2], s=1)

corr_emp = np.corrcoef(np.column_stack([x1_m, x2_m, x3_m]).T)
print(f"\n三元高斯Copula模拟:")
print(f"  经验相关矩阵:")
print(np.round(corr_emp, 4))

# 从相依性测度反推rho
tau_target = 0.4
rho_from_tau = np.sin(np.pi * tau_target / 2) * 2
print(f"\n从Kendall tau={tau_target:.2f}反推rho={rho_from_tau:.4f}")

rho_target = 0.5
rho_from_spearman = 2 * np.sin(np.pi * rho_target / 6)
print(f"从Spearman rho={rho_target:.2f}反推rho={rho_from_spearman:.4f}")


##############################################################################
# 8.3.2 阿基米德Copula的模拟
##############################################################################
np.random.seed(2024)
n = 2000

# 方法1：条件分布法（Clayton Copula）
alpha = 2
u1 = np.random.uniform(0, 1, n)
t = np.random.uniform(0, 1, n)
u2 = (1 - u1**(-alpha) + u1**(-alpha) * t**(-alpha/(1+alpha)))**(-1/alpha)

x1_clayton = stats.expon.ppf(u1)
x2_clayton = stats.expon.ppf(u2)

tau_emp, _ = kendalltau(x1_clayton, x2_clayton)
print(f"\nClayton Copula模拟 (条件分布法, alpha={alpha}):")
print(f"  Kendall tau (经验) = {tau_emp:.4f}")
print(f"  Kendall tau (理论) = {alpha/(alpha+2):.4f}")

# 方法1b：Frank Copula条件分布法
alpha_f = 5
u1_f = np.random.uniform(0, 1, n)
t_f = np.random.uniform(0, 1, n)
u2_f = -1/alpha_f * np.log(1 + (1 - np.exp(-alpha_f)) * t_f /
         (np.exp(-alpha_f * u1_f) - t_f * (np.exp(-alpha_f * u1_f) - 1) -
          np.exp(-alpha_f) + t_f * (np.exp(-alpha_f) - 1)))

x1_frank = stats.expon.ppf(u1_f)
x2_frank = stats.expon.ppf(u2_f)
tau_frank, _ = kendalltau(x1_frank, x2_frank)
print(f"\nFrank Copula模拟 (条件分布法, alpha={alpha_f}):")
print(f"  Kendall tau (经验) = {tau_frank:.4f}")

# 方法2：脆弱性模型法
gamma = np.random.gamma(shape=1/alpha, scale=1, size=n)
U1 = np.random.uniform(0, 1, n)
U2 = np.random.uniform(0, 1, n)

u1_frail = tau_laplace(-np.log(U1) / gamma, alpha)
u2_frail = tau_laplace(-np.log(U2) / gamma, alpha)

x1_frail = stats.expon.ppf(u1_frail)
x2_frail = stats.expon.ppf(u2_frail)
tau_frail, _ = kendalltau(x1_frail, x2_frail)
print(f"\nClayton Copula模拟 (脆弱性模型法, alpha={alpha}):")
print(f"  Kendall tau (经验) = {tau_frail:.4f}")

# 比较三种方法
print(f"\n三种模拟方法比较 (Clayton, alpha={alpha}):")
print(f"  条件分布法: tau = {tau_emp:.4f}")
print(f"  脆弱性模型法: tau = {tau_frail:.4f}")
