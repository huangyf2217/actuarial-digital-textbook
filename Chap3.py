# Chap3 Python代码
# 自动从chap3.html同步生成

##############################################################################
# 第3章 贝叶斯与信度
# 对应教材：section3.tex
# 内容：贝叶斯统计基础、贝叶斯估计、信度理论、贝叶斯信度、
#       经验贝叶斯信度（EBCT模型1、EBCT模型2）
##############################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


##############################################################################
# 3.1 贝叶斯公式与后验分布
##############################################################################

## 3.1.1 三家工厂次品问题（例3.1）
P_B = np.array([0.6, 0.3, 0.1])
P_A_given_B = np.array([0.10, 0.05, 0.15])
P_A = np.sum(P_A_given_B * P_B)
P_B3_given_A = P_A_given_B[2] * P_B[2] / P_A
print(f"例3.1: P(A)={P_A:.4f}, P(B3|A)={P_B3_given_A:.4f}")

##############################################################################
# 第3章 贝叶斯与信度
# 对应教材：section3.tex
# 内容：贝叶斯统计基础、贝叶斯估计、信度理论、贝叶斯信度、
#       经验贝叶斯信度（EBCT模型1、EBCT模型2）
##############################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


##############################################################################
# 3.1 贝叶斯公式与后验分布
##############################################################################

## 3.1.2 S/M/L三类索赔（例3.2）
theta = np.array([100, 1000, 2500])
P_prior = np.array([0.80, 0.15, 0.05])
x = 5000
P_A_given_B = 2 * theta**2 / x**3
P_A = np.sum(P_A_given_B * P_prior)
P_post = P_A_given_B * P_prior / P_A
print(f"例3.2: 后验概率 S={P_post[0]:.4f}, M={P_post[1]:.4f}, L={P_post[2]:.4f}")

##############################################################################
# 第3章 贝叶斯与信度
# 对应教材：section3.tex
# 内容：贝叶斯统计基础、贝叶斯估计、信度理论、贝叶斯信度、
#       经验贝叶斯信度（EBCT模型1、EBCT模型2）
##############################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


##############################################################################
# 3.1 贝叶斯公式与后验分布
##############################################################################

## 3.1.3 二项分布/贝塔共轭先验（例3.3）
alpha = 2; beta = 3
x = np.array([1, 0, 1, 1, 0, 1, 1, 1, 0, 1])
n = len(x)
alpha_post = x.sum() + alpha
beta_post = n - x.sum() + beta
E_theta_post = alpha_post / (alpha_post + beta_post)
E_prior = alpha / (alpha + beta)
print(f"例3.3: 先验期望={E_prior:.4f}, 后验期望={E_theta_post:.4f}, 样本比例={x.mean():.4f}")


##############################################################################
# 3.2 共轭先验分布
##############################################################################

##############################################################################
# 第3章 贝叶斯与信度
# 对应教材：section3.tex
# 内容：贝叶斯统计基础、贝叶斯估计、信度理论、贝叶斯信度、
#       经验贝叶斯信度（EBCT模型1、EBCT模型2）
##############################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


##############################################################################
# 3.1 贝叶斯公式与后验分布
##############################################################################

## 3.2.1 泊松/伽马共轭（例3.4）
alpha = 100; lam = 1
x = np.array([144, 144, 174, 148, 151, 156, 168, 147, 140, 161])
n = len(x)
alpha_post = x.sum() + alpha
lambda_post = n + lam
E_mu_post = alpha_post / lambda_post
print(f"\n例3.4: 泊松/伽马共轭后验期望 = {E_mu_post:.4f}")

##############################################################################
# 第3章 贝叶斯与信度
# 对应教材：section3.tex
# 内容：贝叶斯统计基础、贝叶斯估计、信度理论、贝叶斯信度、
#       经验贝叶斯信度（EBCT模型1、EBCT模型2）
##############################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


##############################################################################
# 3.1 贝叶斯公式与后验分布
##############################################################################

## 3.2.2 正态/正态共轭（例3.5）
sigma1 = 50; mu0 = 300; sigma2 = 20
n = 10; xbar = 270
mu_post = (sigma1**2 * mu0 + n * sigma2**2 * xbar) / (sigma1**2 + n * sigma2**2)
sigma2_post = sigma1**2 * sigma2**2 / (sigma1**2 + n * sigma2**2)
print(f"例3.5: 正态/正态共轭 后验均值={mu_post:.2f}, 标准差={np.sqrt(sigma2_post):.2f}")

##############################################################################
# 第3章 贝叶斯与信度
# 对应教材：section3.tex
# 内容：贝叶斯统计基础、贝叶斯估计、信度理论、贝叶斯信度、
#       经验贝叶斯信度（EBCT模型1、EBCT模型2）
##############################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


##############################################################################
# 3.1 贝叶斯公式与后验分布
##############################################################################

## 3.2.3 贝塔后验的二次损失贝叶斯估计（例3.6）
alpha = 2; beta = 3
sum_x = 7; n = 10
theta_hat = (sum_x + alpha) / (n + alpha + beta)
print(f"例3.6: 二次损失贝叶斯估计 theta_hat = {theta_hat:.4f}")


##############################################################################
# 3.3 贝叶斯估计（不同损失函数）
##############################################################################

##############################################################################
# 第3章 贝叶斯与信度
# 对应教材：section3.tex
# 内容：贝叶斯统计基础、贝叶斯估计、信度理论、贝叶斯信度、
#       经验贝叶斯信度（EBCT模型1、EBCT模型2）
##############################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


##############################################################################
# 3.1 贝叶斯公式与后验分布
##############################################################################

## 3.3.1 三种损失函数下的贝叶斯估计（例3.7）
mu_hat1 = 7 / 13                                    # 二次损失
mu_hat2 = stats.gamma.ppf(0.5, 7, scale=1/13)      # 绝对损失
mu_hat3 = (7 - 1) / 13                              # 0/1损失
print(f"\n例3.7: 二次损失={mu_hat1:.4f}, 绝对损失={mu_hat2:.4f}, 0/1损失={mu_hat3:.4f}")


##############################################################################
# 3.4 信度保费与信度因子
##############################################################################

##############################################################################
# 第3章 贝叶斯与信度
# 对应教材：section3.tex
# 内容：贝叶斯统计基础、贝叶斯估计、信度理论、贝叶斯信度、
#       经验贝叶斯信度（EBCT模型1、EBCT模型2）
##############################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


##############################################################################
# 3.1 贝叶斯公式与后验分布
##############################################################################

## 3.4.2 逆伽马先验下的信度估计（例3.8）
theta_ig = 40; alpha_ig = 1.5; sum_x = 9826; n = 100
mu_hat = (sum_x + theta_ig) / (n + alpha_ig - 1)
Z = n / (n + alpha_ig - 1)
print(f"例3.8: mu_hat={mu_hat:.4f}, Z={Z:.4f}")


##############################################################################
# 3.5 贝叶斯信度模型
##############################################################################

##############################################################################
# 第3章 贝叶斯与信度
# 对应教材：section3.tex
# 内容：贝叶斯统计基础、贝叶斯估计、信度理论、贝叶斯信度、
#       经验贝叶斯信度（EBCT模型1、EBCT模型2）
##############################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


##############################################################################
# 3.1 贝叶斯公式与后验分布
##############################################################################

## 3.5.1 泊松/伽马模型：信度因子随时间变化（例3.9）
x = np.array([144, 144, 174, 148, 151, 156, 168, 147, 140, 161])
n = len(x)

# 先验Ga(100, 1)
alpha1, beta1 = 100, 1
Z1 = np.zeros(n); E_mu1 = np.zeros(n)
for k in range(n):
    Z1[k] = (k+1) / (k+1 + beta1)
    E_mu1[k] = (x[:k+1].sum() + alpha1) / (k+1 + beta1)

# 先验Ga(500, 5)
alpha2, beta2 = 500, 5
Z2 = np.zeros(n); E_mu2 = np.zeros(n)
for k in range(n):
    Z2[k] = (k+1) / (k+1 + beta2)
    E_mu2[k] = (x[:k+1].sum() + alpha2) / (k+1 + beta2)

print(f"\n例3.9: Ga(100,1)最终Z={Z1[-1]:.4f}, 最终E={E_mu1[-1]:.2f}")
print(f"       Ga(500,5)最终Z={Z2[-1]:.4f}, 最终E={E_mu2[-1]:.2f}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes[0,0].plot(range(1, n+1), Z1, 'ro-', label='Ga(100,1)')
axes[0,0].set_title('信度因子 Z'); axes[0,0].legend()
axes[0,1].plot(range(1, n+1), Z2, 'bo-', label='Ga(500,5)')
axes[0,1].set_title('信度因子 Z'); axes[0,1].legend()
axes[1,0].plot(range(1, n+1), E_mu1, 'ro-')
axes[1,0].axhline(y=x.mean(), ls='--', color='black')
axes[1,0].set_title('E(λ|x) - Ga(100,1)')
axes[1,1].plot(range(1, n+1), E_mu2, 'bo-')
axes[1,1].axhline(y=x.mean(), ls='--', color='black')
axes[1,1].set_title('E(λ|x) - Ga(500,5)')
plt.tight_layout(); plt.show()

##############################################################################
# 第3章 贝叶斯与信度
# 对应教材：section3.tex
# 内容：贝叶斯统计基础、贝叶斯估计、信度理论、贝叶斯信度、
#       经验贝叶斯信度（EBCT模型1、EBCT模型2）
##############################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


##############################################################################
# 3.1 贝叶斯公式与后验分布
##############################################################################

## 3.5.2 伯努利/贝塔模型（例3.10）
a = 2; b = 3
x = np.array([1, 0, 1, 1, 0, 1, 1, 1, 0, 1])
n = len(x)
a_post = x.sum() + a
b_post = n - x.sum() + b
E_p_post = a_post / (a_post + b_post)
Z = n / (n + a + b)
print(f"\n例3.10: 后验期望={E_p_post:.4f}, 信度因子Z={Z:.4f}")

##############################################################################
# 第3章 贝叶斯与信度
# 对应教材：section3.tex
# 内容：贝叶斯统计基础、贝叶斯估计、信度理论、贝叶斯信度、
#       经验贝叶斯信度（EBCT模型1、EBCT模型2）
##############################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


##############################################################################
# 3.1 贝叶斯公式与后验分布
##############################################################################

## 3.5.3 正态/正态模型（例3.11）
sigma1 = 50; mu0 = 300; sigma2 = 20
n = 10; xbar = 270

# (1) 先验概率P(mu < 270)
P_prior = stats.norm.cdf(270, mu0, sigma2)
print(f"\n例3.11: 先验概率 P(mu<270) = {P_prior:.5f}")

# (2) 后验分布
mu_post = (sigma1**2 * mu0 + n * sigma2**2 * xbar) / (sigma1**2 + n * sigma2**2)
sigma2_post = sigma1**2 * sigma2**2 / (sigma1**2 + n * sigma2**2)
sigma_post = np.sqrt(sigma2_post)

# 后验概率P(mu < 270 | x)
P_post = stats.norm.cdf(270, mu_post, sigma_post)
print(f"       后验概率 P(mu<270|x) = {P_post:.4f}")

# 信度因子
Z = n * sigma2**2 / (sigma1**2 + n * sigma2**2)
print(f"       信度因子 Z = {Z:.4f}")

##############################################################################
# 第3章 贝叶斯与信度
# 对应教材：section3.tex
# 内容：贝叶斯统计基础、贝叶斯估计、信度理论、贝叶斯信度、
#       经验贝叶斯信度（EBCT模型1、EBCT模型2）
##############################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


##############################################################################
# 3.1 贝叶斯公式与后验分布
##############################################################################

## 3.5.4 EBCT模型1示例：四个国家火灾保单（例3.12）
Y = np.array([
    [48, 53, 42, 50, 59],
    [64, 71, 64, 73, 70],
    [85, 54, 76, 65, 90],
    [44, 52, 69, 55, 71]
])
N, n = Y.shape

X_bar_i = Y.mean(axis=1)
X_bar = Y.mean()

# 补充缺失项
s2_i = np.var(Y, axis=1, ddof=1)

E_m = X_bar
E_s2 = np.mean(s2_i)
Var_m = np.var(X_bar_i, ddof=1) - E_s2 / n

Z = n / (n + E_s2 / Var_m)
credibility_premium = Z * X_bar_i + (1 - Z) * E_m

print(f"\n例3.12: E[m]={E_m:.2f}, E[s²]={E_s2:.2f}, Var[m]={Var_m:.2f}, Z={Z:.4f}")
print(f"        信度保费: {credibility_premium}")


##############################################################################
# 3.6 EBCT模型1（Bühlmann信度）
##############################################################################

##############################################################################
# 第3章 贝叶斯与信度
# 对应教材：section3.tex
# 内容：贝叶斯统计基础、贝叶斯估计、信度理论、贝叶斯信度、
#       经验贝叶斯信度（EBCT模型1、EBCT模型2）
##############################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


##############################################################################
# 3.1 贝叶斯公式与后验分布
##############################################################################

## 3.6.1 EBCT模型1示例：五个车队（例3.13）
Y = np.array([
    [1250, 980, 1800, 2040, 1000, 1180],
    [1700, 3080, 1700, 2820, 5760, 3480],
    [2050, 3560, 2800, 1600, 4200, 2650],
    [4690, 4370, 4800, 9070, 3770, 5250],
    [7150, 3480, 5010, 4810, 8740, 7260]
])
N, n = Y.shape

X_bar_i = Y.mean(axis=1)
X_bar = Y.mean()
E_m = X_bar
E_s2 = np.mean(np.var(Y, axis=1, ddof=1))
Var_m = np.var(X_bar_i, ddof=1) - E_s2 / n
Z = n / (n + E_s2 / Var_m)
credibility_premium = Z * X_bar_i + (1 - Z) * E_m

print(f"\n例3.13: EBCT模型1")
print(f"  E[m(θ)] = {E_m:.2f}")
print(f"  E[s²(θ)] = {E_s2:.2f}")
print(f"  Var[m(θ)] = {Var_m:.2f}")
print(f"  Z = {Z:.4f}")
print(f"  信度保费: {credibility_premium}")


##############################################################################
# 3.7 EBCT模型2（Bühlmann-Straub信度）
##############################################################################

##############################################################################
# 第3章 贝叶斯与信度
# 对应教材：section3.tex
# 内容：贝叶斯统计基础、贝叶斯估计、信度理论、贝叶斯信度、
#       经验贝叶斯信度（EBCT模型1、EBCT模型2）
##############################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


##############################################################################
# 3.1 贝叶斯公式与后验分布
##############################################################################

## 3.7.1 EBCT模型2示例：五个车队（例3.14）
P = np.array([
    [5, 5, 4, 6, 5, 5],
    [11, 13, 10, 12, 15, 14],
    [3, 4, 4, 3, 3, 2],
    [9, 9, 8, 8, 9, 10],
    [7, 7, 8, 8, 9, 10]
])

X = Y / P
P_bar_i = P.sum(axis=1)
P_bar = P.sum()
X_bar_i = (P * X).sum(axis=1) / P_bar_i
X_bar = (P * X).sum() / P_bar

E_m = X_bar
E_s2 = np.sum(P * (X - X_bar_i[:, None])**2) / (N * (n - 1))
P_star = np.sum(P_bar_i * (1 - P_bar_i / P_bar)) / (N * n - 1)
Var_m = (np.sum(P * (X - X_bar)**2) / (N * n - 1) - E_s2) / P_star

Z_i = P_bar_i / (P_bar_i + E_s2 / Var_m)
premium_unit = Z_i * X_bar_i + (1 - Z_i) * E_m
P_2021 = np.array([5, 14, 2, 10, 10])
credibility_premium = premium_unit * P_2021

print(f"\n例3.14: EBCT模型2")
print(f"  E[m(θ)] = {E_m:.4f}")
print(f"  E[s²(θ)] = {E_s2:.4f}")
print(f"  Var[m(θ)] = {Var_m:.4f}")
print(f"  Z_i = {Z_i}")
print(f"  2021年信度保费: {credibility_premium}")

##############################################################################
# 第3章 贝叶斯与信度
# 对应教材：section3.tex
# 内容：贝叶斯统计基础、贝叶斯估计、信度理论、贝叶斯信度、
#       经验贝叶斯信度（EBCT模型1、EBCT模型2）
##############################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


##############################################################################
# 3.1 贝叶斯公式与后验分布
##############################################################################

## 3.7.2 EBCT模型2示例：三个保险公司（例3.15）
Y2 = np.array([
    [14.2, 15.8, 22.7, 19.0],
    [58.6, 63.1, 81.0, 64.2],
    [123, 132, 161, 133]
])
P2 = np.array([
    [163, 189, 252, 199],
    [4435, 4761, 5576, 4581],
    [16184, 17443, 20102, 18000]
])

N2, n2 = Y2.shape
X2 = Y2 / P2
P_bar_i2 = P2.sum(axis=1)
P_bar2 = P2.sum()
X_bar_i2 = (P2 * X2).sum(axis=1) / P_bar_i2
X_bar2 = (P2 * X2).sum() / P_bar2

E_m2 = X_bar2
E_s2_2 = np.sum(P2 * (X2 - X_bar_i2[:, None])**2) / (N2 * (n2 - 1))
P_star2 = np.sum(P_bar_i2 * (1 - P_bar_i2 / P_bar2)) / (N2 * n2 - 1)
Var_m2 = (np.sum(P2 * (X2 - X_bar2)**2) / (N2 * n2 - 1) - E_s2_2) / P_star2

Z_B = P_bar_i2[1] / (P_bar_i2[1] + E_s2_2 / Var_m2)
cred_B_unit = Z_B * X_bar_i2[1] + (1 - Z_B) * E_m2
cred_B = cred_B_unit * 4800

print(f"\n例3.15: 保险公司B")
print(f"  Z_B = {Z_B:.4f}")
print(f"  单位风险信度保费 = {cred_B_unit:.6f}")
print(f"  下年保费(4800) = {cred_B:.3f}")

##############################################################################
# 第3章 贝叶斯与信度
# 对应教材：section3.tex
# 内容：贝叶斯统计基础、贝叶斯估计、信度理论、贝叶斯信度、
#       经验贝叶斯信度（EBCT模型1、EBCT模型2）
##############################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


##############################################################################
# 3.1 贝叶斯公式与后验分布
##############################################################################

## 3.7.3 EBCT模型2假设条件示例（例3.16）
np.random.seed(123)
P = np.array([100, 150, 120])
m_theta = 5
s2_theta = 25

# 模拟各年索赔
Y = np.array([np.sum(np.random.normal(m_theta, np.sqrt(s2_theta), p)) for p in P])
X = Y / P

# 验证：E(X_j) ≈ m(theta)
print(f"\n例3.16: E(X_j)均值 = {X.mean():.4f} (理论值={m_theta})")

# 验证：P_j * Var(X_j) ≈ s^2(theta)
n_sim = 1000
X_sim = np.zeros((n_sim, len(P)))
for s in range(n_sim):
    Y_sim = np.array([np.sum(np.random.normal(m_theta, np.sqrt(s2_theta), p)) for p in P])
    X_sim[s, :] = Y_sim / P
P_var_X = np.var(X_sim, axis=0) * P
print(f"        P_j*Var(X_j)均值 = {P_var_X.mean():.4f} (理论值={s2_theta})")