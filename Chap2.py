# Chap2 Python代码
# 自动从chap2.html同步生成

# 导入所需的Python模块
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import gaussian_kde
import warnings
warnings.filterwarnings('ignore')

## 2.1.1 卷积法计算累积索赔金额的分布（例2.1）
pn = np.array([0.3, 0.5, 0.2])           # 索赔次数概率
fx = np.array([0.2, 0.4, 0.2, 0.1, 0.1]) # 索赔强度概率

# 卷积法计算S的分布
max_s = 4 * 4
FS = np.zeros(max_s + 1)

# N=0
FS[0] = pn[0]
# N=1
for k in range(1, 5):
    FS[k] += pn[1] * fx[k-1]
# N=2
for k1 in range(1, 5):
    for k2 in range(1, 5):
        s = k1 + k2
        FS[s] += pn[2] * fx[k1-1] * fx[k2-1]

print("卷积法计算累积索赔金额分布:")
print(f"  P(S=0) = {FS[0]:.4f}")
for s in range(1, min(9, max_s+1)):
    if FS[s] > 0:
        print(f"  P(S={s}) = {FS[s]:.4f}")

# 绘图
fig, ax = plt.subplots(figsize=(8, 5))
ax.step(range(max_s+1), np.cumsum(FS), where='post')
ax.set_xlabel('s'); ax.set_ylabel('F(s)')
ax.set_title('累积索赔金额分布函数')
plt.tight_layout(); plt.show()

##############################################################################
# 第2章 风险模型
# 对应教材：section2.tex
# 内容：短期聚合风险模型、复合分布、短期个体风险模型、
#       参数不确定性的影响、近似计算方法
##############################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


##############################################################################
# 2.1 短期聚合风险模型：卷积法
##############################################################################

## 2.1.1 卷积法计算累积索赔金额的分布
pn = np.array([0.3, 0.5, 0.2])           # 索赔次数概率
fx = np.array([0.2, 0.4, 0.2, 0.1, 0.1]) # 索赔强度概率

# 卷积法计算S的分布
max_s = 4 * 4
FS = np.zeros(max_s + 1)

# N=0
FS[0] = pn[0]
# N=1
for k in range(1, 5):
    FS[k] += pn[1] * fx[k-1]
# N=2
for k1 in range(1, 5):
    for k2 in range(1, 5):
        s = k1 + k2
        FS[s] += pn[2] * fx[k1-1] * fx[k2-1]

print("卷积法计算累积索赔金额分布:")
print(f"  P(S=0) = {FS[0]:.4f}")
for s in range(1, min(9, max_s+1)):
    if FS[s] > 0:
        print(f"  P(S={s}) = {FS[s]:.4f}")

# 绘图
fig, ax = plt.subplots(figsize=(8, 5))
ax.step(range(max_s+1), np.cumsum(FS), where='post')
ax.set_xlabel('s'); ax.set_ylabel('F(s)')
ax.set_title('累积索赔金额分布函数')
plt.tight_layout(); plt.show()


##############################################################################
# 2.2 复合分布的随机模拟
##############################################################################

## 2.2.1 复合泊松分布
np.random.seed(123)
n_sim = 10000
S_poisson = np.zeros(n_sim)
for i in range(n_sim):
    N = np.random.poisson(2.5)
    X = np.random.gamma(shape=2, scale=500, size=N)
    S_poisson[i] = X.sum()

print(f"\n复合泊松分布: E(S)={S_poisson.mean():.1f}, Var(S)={S_poisson.var():.1f}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(S_poisson, bins=range(0, 15001, 500), density=True, alpha=0.5, color='grey')
from scipy.stats import gaussian_kde
kde = gaussian_kde(S_poisson)
x_range = np.linspace(0, 15000, 200)
ax.plot(x_range, kde(x_range), 'r-')
ax.set_xlabel('S'); ax.set_ylabel('密度')
ax.set_title('复合泊松分布模拟')
plt.tight_layout(); plt.show()


## 2.2.2 复合二项分布
np.random.seed(123)
S_binomial = np.zeros(n_sim)
for i in range(n_sim):
    N = np.random.binomial(100, 0.01)
    X = np.random.gamma(shape=10, scale=5, size=N)  # rate=0.2 => scale=5
    S_binomial[i] = X.sum()

print(f"复合二项分布: E(S)={S_binomial.mean():.2f}, Var(S)={S_binomial.var():.1f}")


## 2.2.3 复合负二项分布
np.random.seed(123)
S_negbinom = np.zeros(n_sim)
for i in range(n_sim):
    N = np.random.negative_binomial(2.5, 0.5)
    X = np.random.gamma(shape=2, scale=500, size=N)
    S_negbinom[i] = X.sum()

print(f"复合负二项分布: E(S)={S_negbinom.mean():.1f}, Var(S)={S_negbinom.var():.1f}")


## 2.2.4 复合分布的数字特征（正态索赔强度）
for dist_name, N_dist in [('泊松', lambda: np.random.poisson(5, n_sim)),
                           ('二项', lambda: np.random.binomial(10, 0.5, n_sim)),
                           ('负二项', lambda: np.random.negative_binomial(2, 0.5, n_sim))]:
    N = N_dist()
    S = np.array([np.random.normal(1000, 10, n).sum() for n in N])
    ES = S.mean()
    varS = S.var()
    skewS = np.mean(S**3) - 3 * S.mean() * np.mean(S**2) + 2 * S.mean()**3
    print(f"\n复合{dist_name}分布（正态强度）: E(S)={ES:.1f}, Var(S)={varS:.1f}, skew={skewS:.1f}")


##############################################################################
# 2.3 短期个体风险模型
##############################################################################
np.random.seed(123)
S_individual = np.zeros(n_sim)
for k in range(n_sim):
    Ni = np.concatenate([np.random.binomial(1, 0.1, 50),
                          np.random.binomial(1, 0.2, 50)])
    Xi = np.concatenate([np.random.gamma(10, scale=50, size=50),    # rate=0.02
                          np.random.lognormal(5, 1, size=50)])
    S_individual[k] = np.sum(Ni * Xi)

print(f"\n个体风险模型: E(S)={S_individual.mean():.1f}, Var(S)={S_individual.var():.0f}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(S_individual, bins=50, density=True, alpha=0.5, color='grey')
kde = gaussian_kde(S_individual)
x_range = np.linspace(0, S_individual.max(), 200)
ax.plot(x_range, kde(x_range), 'r-')
ax.set_xlabel('S'); ax.set_ylabel('密度')
ax.set_title('个体风险模型模拟')
plt.tight_layout(); plt.show()


##############################################################################
# 2.4 聚合风险模型计算方法
##############################################################################

## 2.4.1 Panjer递推
def panjer_poisson(p, lam):
    """Panjer递推（泊松分布）"""
    cumul = np.exp(-lam * np.sum(p))
    f = [cumul]
    s = 0
    while cumul <= 0.99999999:
        s += 1
        m = min(s, len(p))
        last = lam / s * np.sum(np.arange(1, m+1) * p[:m] * f[s-m:s][::-1])
        f.append(last)
        cumul += last
    return np.array(f)

f_panjer = panjer_poisson(np.array([0.25, 0.5, 0.25]), 4) * np.exp(4)
print(f"\nPanjer递推结果: {f_panjer}")


## 2.4.2 FFT法
x_fft = np.array([0, 0.5, 0.4, 0.1] + [0]*40)
phi_x = np.fft.fft(x_fft)
phi_s = np.exp(3 * (phi_x - 1))
fs = np.real(np.fft.ifft(phi_s))
Fs = np.cumsum(fs)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(range(min(31, len(fs))), fs[:31], color='red')
axes[0].set_xlabel('s'); axes[0].set_ylabel('f(s)')
axes[0].set_title('FFT法：概率函数')
axes[1].step(range(min(31, len(Fs))), Fs[:31], where='post', color='red')
axes[1].set_xlabel('s'); axes[1].set_ylabel('F(s)')
axes[1].set_title('FFT法：分布函数')
plt.tight_layout(); plt.show()


## 2.4.3 随机模拟法
np.random.seed(42)
lam = 3; mu = 6; sigma = 1.5; u = 1000
s_sim = np.zeros(n_sim)
for i in range(n_sim):
    n = np.random.poisson(lam)
    s_sim[i] = np.sum(np.minimum(np.random.lognormal(mu, sigma, n), u))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(s_sim, bins=100, density=True, color='red', alpha=0.5)
axes[0].set_xlabel('s'); axes[0].set_title('随机模拟：频率直方图')
s_sorted = np.sort(s_sim)
axes[1].plot(s_sorted, np.cumsum(s_sorted)/s_sorted.sum(), 'r-')
axes[1].set_xlabel('s'); axes[1].set_title('Lorenz曲线')
plt.tight_layout(); plt.show()


## 2.4.4 随机模拟求累积损失的分布（例2.7）
np.random.seed(321)
d = 250; u = 1000
r_nb = 3; beta_nb = 2
alpha_g = 100; theta_g = 0.2
P = np.zeros(n_sim)

for i in range(n_sim):
    n = np.random.negative_binomial(r_nb, 1/(1+beta_nb))
    x = np.random.gamma(alpha_g, 1/theta_g, n)
    w = np.minimum(x, d)
    v = min(w.sum(), u)
    S = x.sum()
    P[i] = S - v

print(f"\n例2.7: E(P)={P.mean():.2f}, P95={np.percentile(P, 95):.2f}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(P, bins=50, density=True, color='grey', alpha=0.5)
ax.set_xlabel('累积赔款'); ax.set_ylabel('频率')
ax.set_title('保险人年度累积赔款')
plt.tight_layout(); plt.show()


##############################################################################
# 2.5 近似计算方法
##############################################################################

## 2.5.1 Tweedie分布模拟
np.random.seed(11)
lam_tw = 1; alpha_tw = 10; beta_tw = 2
Y = np.zeros(n_sim)
for i in range(n_sim):
    N = np.random.poisson(lam_tw)
    Y[i] = np.random.gamma(alpha_tw, 1/beta_tw, N).sum()

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(Y, bins=50, color='grey', alpha=0.5)
ax.set_title('Tweedie分布模拟')
plt.tight_layout(); plt.show()


## 2.5.2 近似计算比较
np.random.seed(123)
S_approx = np.zeros(n_sim)
for i in range(n_sim):
    N = np.random.binomial(1000, 0.001)
    S_approx[i] = N  # 每次索赔金额为1

ES = S_approx.mean()
varS = S_approx.var()

# 精确分布
p0 = 1 - stats.binom.cdf(3.5, 1000, 0.001)
p1 = 1 - stats.poisson.cdf(3.5, 1)

# 正态近似
p2 = 1 - stats.norm.cdf(3.5, ES, np.sqrt(varS))

# 平移伽马近似: S+1 ~ Gamma(4, 2)
p4 = 1 - stats.gamma.cdf(4.5, 4, scale=0.5)

# NP近似
mu_np = sigma_np = gamma_np = 1
a_np = -3/gamma_np + np.sqrt(9/gamma_np**2 + 1 + 6/gamma_np * (3.5 - mu_np) / sigma_np)
p5 = 1 - stats.norm.cdf(a_np)

print(f"\n近似计算比较 P(S>3.5):")
print(f"  二项分布(精确): {p0:.6f}")
print(f"  泊松分布(精确): {p1:.6f}")
print(f"  正态近似:       {p2:.6f}")
print(f"  平移伽马近似:   {p4:.6f}")
print(f"  NP近似:         {p5:.6f}")

##############################################################################
# 第2章 风险模型
# 对应教材：section2.tex
# 内容：短期聚合风险模型、复合分布、短期个体风险模型、
#       参数不确定性的影响、近似计算方法
##############################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


##############################################################################
# 2.1 短期聚合风险模型：卷积法
##############################################################################

## 2.1.1 卷积法计算累积索赔金额的分布
pn = np.array([0.3, 0.5, 0.2])           # 索赔次数概率
fx = np.array([0.2, 0.4, 0.2, 0.1, 0.1]) # 索赔强度概率

# 卷积法计算S的分布
max_s = 4 * 4
FS = np.zeros(max_s + 1)

# N=0
FS[0] = pn[0]
# N=1
for k in range(1, 5):
    FS[k] += pn[1] * fx[k-1]
# N=2
for k1 in range(1, 5):
    for k2 in range(1, 5):
        s = k1 + k2
        FS[s] += pn[2] * fx[k1-1] * fx[k2-1]

print("卷积法计算累积索赔金额分布:")
print(f"  P(S=0) = {FS[0]:.4f}")
for s in range(1, min(9, max_s+1)):
    if FS[s] > 0:
        print(f"  P(S={s}) = {FS[s]:.4f}")

# 绘图
fig, ax = plt.subplots(figsize=(8, 5))
ax.step(range(max_s+1), np.cumsum(FS), where='post')
ax.set_xlabel('s'); ax.set_ylabel('F(s)')
ax.set_title('累积索赔金额分布函数')
plt.tight_layout(); plt.show()


##############################################################################
# 2.2 复合分布的随机模拟
##############################################################################

## 2.2.1 复合泊松分布
np.random.seed(123)
n_sim = 10000
S_poisson = np.zeros(n_sim)
for i in range(n_sim):
    N = np.random.poisson(2.5)
    X = np.random.gamma(shape=2, scale=500, size=N)
    S_poisson[i] = X.sum()

print(f"\n复合泊松分布: E(S)={S_poisson.mean():.1f}, Var(S)={S_poisson.var():.1f}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(S_poisson, bins=range(0, 15001, 500), density=True, alpha=0.5, color='grey')
from scipy.stats import gaussian_kde
kde = gaussian_kde(S_poisson)
x_range = np.linspace(0, 15000, 200)
ax.plot(x_range, kde(x_range), 'r-')
ax.set_xlabel('S'); ax.set_ylabel('密度')
ax.set_title('复合泊松分布模拟')
plt.tight_layout(); plt.show()


## 2.2.2 复合二项分布
np.random.seed(123)
S_binomial = np.zeros(n_sim)
for i in range(n_sim):
    N = np.random.binomial(100, 0.01)
    X = np.random.gamma(shape=10, scale=5, size=N)  # rate=0.2 => scale=5
    S_binomial[i] = X.sum()

print(f"复合二项分布: E(S)={S_binomial.mean():.2f}, Var(S)={S_binomial.var():.1f}")


## 2.2.3 复合负二项分布
np.random.seed(123)
S_negbinom = np.zeros(n_sim)
for i in range(n_sim):
    N = np.random.negative_binomial(2.5, 0.5)
    X = np.random.gamma(shape=2, scale=500, size=N)
    S_negbinom[i] = X.sum()

print(f"复合负二项分布: E(S)={S_negbinom.mean():.1f}, Var(S)={S_negbinom.var():.1f}")


## 2.2.4 复合分布的数字特征（正态索赔强度）
for dist_name, N_dist in [('泊松', lambda: np.random.poisson(5, n_sim)),
                           ('二项', lambda: np.random.binomial(10, 0.5, n_sim)),
                           ('负二项', lambda: np.random.negative_binomial(2, 0.5, n_sim))]:
    N = N_dist()
    S = np.array([np.random.normal(1000, 10, n).sum() for n in N])
    ES = S.mean()
    varS = S.var()
    skewS = np.mean(S**3) - 3 * S.mean() * np.mean(S**2) + 2 * S.mean()**3
    print(f"\n复合{dist_name}分布（正态强度）: E(S)={ES:.1f}, Var(S)={varS:.1f}, skew={skewS:.1f}")


##############################################################################
# 2.3 短期个体风险模型
##############################################################################
np.random.seed(123)
S_individual = np.zeros(n_sim)
for k in range(n_sim):
    Ni = np.concatenate([np.random.binomial(1, 0.1, 50),
                          np.random.binomial(1, 0.2, 50)])
    Xi = np.concatenate([np.random.gamma(10, scale=50, size=50),    # rate=0.02
                          np.random.lognormal(5, 1, size=50)])
    S_individual[k] = np.sum(Ni * Xi)

print(f"\n个体风险模型: E(S)={S_individual.mean():.1f}, Var(S)={S_individual.var():.0f}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(S_individual, bins=50, density=True, alpha=0.5, color='grey')
kde = gaussian_kde(S_individual)
x_range = np.linspace(0, S_individual.max(), 200)
ax.plot(x_range, kde(x_range), 'r-')
ax.set_xlabel('S'); ax.set_ylabel('密度')
ax.set_title('个体风险模型模拟')
plt.tight_layout(); plt.show()


##############################################################################
# 2.4 聚合风险模型计算方法
##############################################################################

## 2.4.1 Panjer递推
def panjer_poisson(p, lam):
    """Panjer递推（泊松分布）"""
    cumul = np.exp(-lam * np.sum(p))
    f = [cumul]
    s = 0
    while cumul <= 0.99999999:
        s += 1
        m = min(s, len(p))
        last = lam / s * np.sum(np.arange(1, m+1) * p[:m] * f[s-m:s][::-1])
        f.append(last)
        cumul += last
    return np.array(f)

f_panjer = panjer_poisson(np.array([0.25, 0.5, 0.25]), 4) * np.exp(4)
print(f"\nPanjer递推结果: {f_panjer}")


## 2.4.2 FFT法
x_fft = np.array([0, 0.5, 0.4, 0.1] + [0]*40)
phi_x = np.fft.fft(x_fft)
phi_s = np.exp(3 * (phi_x - 1))
fs = np.real(np.fft.ifft(phi_s))
Fs = np.cumsum(fs)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(range(min(31, len(fs))), fs[:31], color='red')
axes[0].set_xlabel('s'); axes[0].set_ylabel('f(s)')
axes[0].set_title('FFT法：概率函数')
axes[1].step(range(min(31, len(Fs))), Fs[:31], where='post', color='red')
axes[1].set_xlabel('s'); axes[1].set_ylabel('F(s)')
axes[1].set_title('FFT法：分布函数')
plt.tight_layout(); plt.show()


## 2.4.3 随机模拟法
np.random.seed(42)
lam = 3; mu = 6; sigma = 1.5; u = 1000
s_sim = np.zeros(n_sim)
for i in range(n_sim):
    n = np.random.poisson(lam)
    s_sim[i] = np.sum(np.minimum(np.random.lognormal(mu, sigma, n), u))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(s_sim, bins=100, density=True, color='red', alpha=0.5)
axes[0].set_xlabel('s'); axes[0].set_title('随机模拟：频率直方图')
s_sorted = np.sort(s_sim)
axes[1].plot(s_sorted, np.cumsum(s_sorted)/s_sorted.sum(), 'r-')
axes[1].set_xlabel('s'); axes[1].set_title('Lorenz曲线')
plt.tight_layout(); plt.show()


## 2.4.4 随机模拟求累积损失的分布（例2.7）
np.random.seed(321)
d = 250; u = 1000
r_nb = 3; beta_nb = 2
alpha_g = 100; theta_g = 0.2
P = np.zeros(n_sim)

for i in range(n_sim):
    n = np.random.negative_binomial(r_nb, 1/(1+beta_nb))
    x = np.random.gamma(alpha_g, 1/theta_g, n)
    w = np.minimum(x, d)
    v = min(w.sum(), u)
    S = x.sum()
    P[i] = S - v

print(f"\n例2.7: E(P)={P.mean():.2f}, P95={np.percentile(P, 95):.2f}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(P, bins=50, density=True, color='grey', alpha=0.5)
ax.set_xlabel('累积赔款'); ax.set_ylabel('频率')
ax.set_title('保险人年度累积赔款')
plt.tight_layout(); plt.show()


##############################################################################
# 2.5 近似计算方法
##############################################################################

## 2.5.1 Tweedie分布模拟
np.random.seed(11)
lam_tw = 1; alpha_tw = 10; beta_tw = 2
Y = np.zeros(n_sim)
for i in range(n_sim):
    N = np.random.poisson(lam_tw)
    Y[i] = np.random.gamma(alpha_tw, 1/beta_tw, N).sum()

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(Y, bins=50, color='grey', alpha=0.5)
ax.set_title('Tweedie分布模拟')
plt.tight_layout(); plt.show()


## 2.5.2 近似计算比较
np.random.seed(123)
S_approx = np.zeros(n_sim)
for i in range(n_sim):
    N = np.random.binomial(1000, 0.001)
    S_approx[i] = N  # 每次索赔金额为1

ES = S_approx.mean()
varS = S_approx.var()

# 精确分布
p0 = 1 - stats.binom.cdf(3.5, 1000, 0.001)
p1 = 1 - stats.poisson.cdf(3.5, 1)

# 正态近似
p2 = 1 - stats.norm.cdf(3.5, ES, np.sqrt(varS))

# 平移伽马近似: S+1 ~ Gamma(4, 2)
p4 = 1 - stats.gamma.cdf(4.5, 4, scale=0.5)

# NP近似
mu_np = sigma_np = gamma_np = 1
a_np = -3/gamma_np + np.sqrt(9/gamma_np**2 + 1 + 6/gamma_np * (3.5 - mu_np) / sigma_np)
p5 = 1 - stats.norm.cdf(a_np)

print(f"\n近似计算比较 P(S>3.5):")
print(f"  二项分布(精确): {p0:.6f}")
print(f"  泊松分布(精确): {p1:.6f}")
print(f"  正态近似:       {p2:.6f}")
print(f"  平移伽马近似:   {p4:.6f}")
print(f"  NP近似:         {p5:.6f}")

##############################################################################
# 第2章 风险模型
# 对应教材：section2.tex
# 内容：短期聚合风险模型、复合分布、短期个体风险模型、
#       参数不确定性的影响、近似计算方法
##############################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


##############################################################################
# 2.1 短期聚合风险模型：卷积法
##############################################################################

## 2.1.1 卷积法计算累积索赔金额的分布
pn = np.array([0.3, 0.5, 0.2])           # 索赔次数概率
fx = np.array([0.2, 0.4, 0.2, 0.1, 0.1]) # 索赔强度概率

# 卷积法计算S的分布
max_s = 4 * 4
FS = np.zeros(max_s + 1)

# N=0
FS[0] = pn[0]
# N=1
for k in range(1, 5):
    FS[k] += pn[1] * fx[k-1]
# N=2
for k1 in range(1, 5):
    for k2 in range(1, 5):
        s = k1 + k2
        FS[s] += pn[2] * fx[k1-1] * fx[k2-1]

print("卷积法计算累积索赔金额分布:")
print(f"  P(S=0) = {FS[0]:.4f}")
for s in range(1, min(9, max_s+1)):
    if FS[s] > 0:
        print(f"  P(S={s}) = {FS[s]:.4f}")

# 绘图
fig, ax = plt.subplots(figsize=(8, 5))
ax.step(range(max_s+1), np.cumsum(FS), where='post')
ax.set_xlabel('s'); ax.set_ylabel('F(s)')
ax.set_title('累积索赔金额分布函数')
plt.tight_layout(); plt.show()


##############################################################################
# 2.2 复合分布的随机模拟
##############################################################################

## 2.2.1 复合泊松分布
np.random.seed(123)
n_sim = 10000
S_poisson = np.zeros(n_sim)
for i in range(n_sim):
    N = np.random.poisson(2.5)
    X = np.random.gamma(shape=2, scale=500, size=N)
    S_poisson[i] = X.sum()

print(f"\n复合泊松分布: E(S)={S_poisson.mean():.1f}, Var(S)={S_poisson.var():.1f}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(S_poisson, bins=range(0, 15001, 500), density=True, alpha=0.5, color='grey')
from scipy.stats import gaussian_kde
kde = gaussian_kde(S_poisson)
x_range = np.linspace(0, 15000, 200)
ax.plot(x_range, kde(x_range), 'r-')
ax.set_xlabel('S'); ax.set_ylabel('密度')
ax.set_title('复合泊松分布模拟')
plt.tight_layout(); plt.show()


## 2.2.2 复合二项分布
np.random.seed(123)
S_binomial = np.zeros(n_sim)
for i in range(n_sim):
    N = np.random.binomial(100, 0.01)
    X = np.random.gamma(shape=10, scale=5, size=N)  # rate=0.2 => scale=5
    S_binomial[i] = X.sum()

print(f"复合二项分布: E(S)={S_binomial.mean():.2f}, Var(S)={S_binomial.var():.1f}")


## 2.2.3 复合负二项分布
np.random.seed(123)
S_negbinom = np.zeros(n_sim)
for i in range(n_sim):
    N = np.random.negative_binomial(2.5, 0.5)
    X = np.random.gamma(shape=2, scale=500, size=N)
    S_negbinom[i] = X.sum()

print(f"复合负二项分布: E(S)={S_negbinom.mean():.1f}, Var(S)={S_negbinom.var():.1f}")


## 2.2.4 复合分布的数字特征（正态索赔强度）
for dist_name, N_dist in [('泊松', lambda: np.random.poisson(5, n_sim)),
                           ('二项', lambda: np.random.binomial(10, 0.5, n_sim)),
                           ('负二项', lambda: np.random.negative_binomial(2, 0.5, n_sim))]:
    N = N_dist()
    S = np.array([np.random.normal(1000, 10, n).sum() for n in N])
    ES = S.mean()
    varS = S.var()
    skewS = np.mean(S**3) - 3 * S.mean() * np.mean(S**2) + 2 * S.mean()**3
    print(f"\n复合{dist_name}分布（正态强度）: E(S)={ES:.1f}, Var(S)={varS:.1f}, skew={skewS:.1f}")


##############################################################################
# 2.3 短期个体风险模型
##############################################################################
np.random.seed(123)
S_individual = np.zeros(n_sim)
for k in range(n_sim):
    Ni = np.concatenate([np.random.binomial(1, 0.1, 50),
                          np.random.binomial(1, 0.2, 50)])
    Xi = np.concatenate([np.random.gamma(10, scale=50, size=50),    # rate=0.02
                          np.random.lognormal(5, 1, size=50)])
    S_individual[k] = np.sum(Ni * Xi)

print(f"\n个体风险模型: E(S)={S_individual.mean():.1f}, Var(S)={S_individual.var():.0f}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(S_individual, bins=50, density=True, alpha=0.5, color='grey')
kde = gaussian_kde(S_individual)
x_range = np.linspace(0, S_individual.max(), 200)
ax.plot(x_range, kde(x_range), 'r-')
ax.set_xlabel('S'); ax.set_ylabel('密度')
ax.set_title('个体风险模型模拟')
plt.tight_layout(); plt.show()


##############################################################################
# 2.4 聚合风险模型计算方法
##############################################################################

## 2.4.1 Panjer递推
def panjer_poisson(p, lam):
    """Panjer递推（泊松分布）"""
    cumul = np.exp(-lam * np.sum(p))
    f = [cumul]
    s = 0
    while cumul <= 0.99999999:
        s += 1
        m = min(s, len(p))
        last = lam / s * np.sum(np.arange(1, m+1) * p[:m] * f[s-m:s][::-1])
        f.append(last)
        cumul += last
    return np.array(f)

f_panjer = panjer_poisson(np.array([0.25, 0.5, 0.25]), 4) * np.exp(4)
print(f"\nPanjer递推结果: {f_panjer}")


## 2.4.2 FFT法
x_fft = np.array([0, 0.5, 0.4, 0.1] + [0]*40)
phi_x = np.fft.fft(x_fft)
phi_s = np.exp(3 * (phi_x - 1))
fs = np.real(np.fft.ifft(phi_s))
Fs = np.cumsum(fs)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(range(min(31, len(fs))), fs[:31], color='red')
axes[0].set_xlabel('s'); axes[0].set_ylabel('f(s)')
axes[0].set_title('FFT法：概率函数')
axes[1].step(range(min(31, len(Fs))), Fs[:31], where='post', color='red')
axes[1].set_xlabel('s'); axes[1].set_ylabel('F(s)')
axes[1].set_title('FFT法：分布函数')
plt.tight_layout(); plt.show()


## 2.4.3 随机模拟法
np.random.seed(42)
lam = 3; mu = 6; sigma = 1.5; u = 1000
s_sim = np.zeros(n_sim)
for i in range(n_sim):
    n = np.random.poisson(lam)
    s_sim[i] = np.sum(np.minimum(np.random.lognormal(mu, sigma, n), u))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(s_sim, bins=100, density=True, color='red', alpha=0.5)
axes[0].set_xlabel('s'); axes[0].set_title('随机模拟：频率直方图')
s_sorted = np.sort(s_sim)
axes[1].plot(s_sorted, np.cumsum(s_sorted)/s_sorted.sum(), 'r-')
axes[1].set_xlabel('s'); axes[1].set_title('Lorenz曲线')
plt.tight_layout(); plt.show()


## 2.4.4 随机模拟求累积损失的分布（例2.7）
np.random.seed(321)
d = 250; u = 1000
r_nb = 3; beta_nb = 2
alpha_g = 100; theta_g = 0.2
P = np.zeros(n_sim)

for i in range(n_sim):
    n = np.random.negative_binomial(r_nb, 1/(1+beta_nb))
    x = np.random.gamma(alpha_g, 1/theta_g, n)
    w = np.minimum(x, d)
    v = min(w.sum(), u)
    S = x.sum()
    P[i] = S - v

print(f"\n例2.7: E(P)={P.mean():.2f}, P95={np.percentile(P, 95):.2f}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(P, bins=50, density=True, color='grey', alpha=0.5)
ax.set_xlabel('累积赔款'); ax.set_ylabel('频率')
ax.set_title('保险人年度累积赔款')
plt.tight_layout(); plt.show()


##############################################################################
# 2.5 近似计算方法
##############################################################################

## 2.5.1 Tweedie分布模拟
np.random.seed(11)
lam_tw = 1; alpha_tw = 10; beta_tw = 2
Y = np.zeros(n_sim)
for i in range(n_sim):
    N = np.random.poisson(lam_tw)
    Y[i] = np.random.gamma(alpha_tw, 1/beta_tw, N).sum()

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(Y, bins=50, color='grey', alpha=0.5)
ax.set_title('Tweedie分布模拟')
plt.tight_layout(); plt.show()


## 2.5.2 近似计算比较
np.random.seed(123)
S_approx = np.zeros(n_sim)
for i in range(n_sim):
    N = np.random.binomial(1000, 0.001)
    S_approx[i] = N  # 每次索赔金额为1

ES = S_approx.mean()
varS = S_approx.var()

# 精确分布
p0 = 1 - stats.binom.cdf(3.5, 1000, 0.001)
p1 = 1 - stats.poisson.cdf(3.5, 1)

# 正态近似
p2 = 1 - stats.norm.cdf(3.5, ES, np.sqrt(varS))

# 平移伽马近似: S+1 ~ Gamma(4, 2)
p4 = 1 - stats.gamma.cdf(4.5, 4, scale=0.5)

# NP近似
mu_np = sigma_np = gamma_np = 1
a_np = -3/gamma_np + np.sqrt(9/gamma_np**2 + 1 + 6/gamma_np * (3.5 - mu_np) / sigma_np)
p5 = 1 - stats.norm.cdf(a_np)

print(f"\n近似计算比较 P(S>3.5):")
print(f"  二项分布(精确): {p0:.6f}")
print(f"  泊松分布(精确): {p1:.6f}")
print(f"  正态近似:       {p2:.6f}")
print(f"  平移伽马近似:   {p4:.6f}")
print(f"  NP近似:         {p5:.6f}")

##############################################################################
# 第2章 风险模型
# 对应教材：section2.tex
# 内容：短期聚合风险模型、复合分布、短期个体风险模型、
#       参数不确定性的影响、近似计算方法
##############################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


##############################################################################
# 2.1 短期聚合风险模型：卷积法
##############################################################################

## 2.1.1 卷积法计算累积索赔金额的分布
pn = np.array([0.3, 0.5, 0.2])           # 索赔次数概率
fx = np.array([0.2, 0.4, 0.2, 0.1, 0.1]) # 索赔强度概率

# 卷积法计算S的分布
max_s = 4 * 4
FS = np.zeros(max_s + 1)

# N=0
FS[0] = pn[0]
# N=1
for k in range(1, 5):
    FS[k] += pn[1] * fx[k-1]
# N=2
for k1 in range(1, 5):
    for k2 in range(1, 5):
        s = k1 + k2
        FS[s] += pn[2] * fx[k1-1] * fx[k2-1]

print("卷积法计算累积索赔金额分布:")
print(f"  P(S=0) = {FS[0]:.4f}")
for s in range(1, min(9, max_s+1)):
    if FS[s] > 0:
        print(f"  P(S={s}) = {FS[s]:.4f}")

# 绘图
fig, ax = plt.subplots(figsize=(8, 5))
ax.step(range(max_s+1), np.cumsum(FS), where='post')
ax.set_xlabel('s'); ax.set_ylabel('F(s)')
ax.set_title('累积索赔金额分布函数')
plt.tight_layout(); plt.show()


##############################################################################
# 2.2 复合分布的随机模拟
##############################################################################

## 2.2.1 复合泊松分布
np.random.seed(123)
n_sim = 10000
S_poisson = np.zeros(n_sim)
for i in range(n_sim):
    N = np.random.poisson(2.5)
    X = np.random.gamma(shape=2, scale=500, size=N)
    S_poisson[i] = X.sum()

print(f"\n复合泊松分布: E(S)={S_poisson.mean():.1f}, Var(S)={S_poisson.var():.1f}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(S_poisson, bins=range(0, 15001, 500), density=True, alpha=0.5, color='grey')
from scipy.stats import gaussian_kde
kde = gaussian_kde(S_poisson)
x_range = np.linspace(0, 15000, 200)
ax.plot(x_range, kde(x_range), 'r-')
ax.set_xlabel('S'); ax.set_ylabel('密度')
ax.set_title('复合泊松分布模拟')
plt.tight_layout(); plt.show()


## 2.2.2 复合二项分布
np.random.seed(123)
S_binomial = np.zeros(n_sim)
for i in range(n_sim):
    N = np.random.binomial(100, 0.01)
    X = np.random.gamma(shape=10, scale=5, size=N)  # rate=0.2 => scale=5
    S_binomial[i] = X.sum()

print(f"复合二项分布: E(S)={S_binomial.mean():.2f}, Var(S)={S_binomial.var():.1f}")


## 2.2.3 复合负二项分布
np.random.seed(123)
S_negbinom = np.zeros(n_sim)
for i in range(n_sim):
    N = np.random.negative_binomial(2.5, 0.5)
    X = np.random.gamma(shape=2, scale=500, size=N)
    S_negbinom[i] = X.sum()

print(f"复合负二项分布: E(S)={S_negbinom.mean():.1f}, Var(S)={S_negbinom.var():.1f}")


## 2.2.4 复合分布的数字特征（正态索赔强度）
for dist_name, N_dist in [('泊松', lambda: np.random.poisson(5, n_sim)),
                           ('二项', lambda: np.random.binomial(10, 0.5, n_sim)),
                           ('负二项', lambda: np.random.negative_binomial(2, 0.5, n_sim))]:
    N = N_dist()
    S = np.array([np.random.normal(1000, 10, n).sum() for n in N])
    ES = S.mean()
    varS = S.var()
    skewS = np.mean(S**3) - 3 * S.mean() * np.mean(S**2) + 2 * S.mean()**3
    print(f"\n复合{dist_name}分布（正态强度）: E(S)={ES:.1f}, Var(S)={varS:.1f}, skew={skewS:.1f}")


##############################################################################
# 2.3 短期个体风险模型
##############################################################################
np.random.seed(123)
S_individual = np.zeros(n_sim)
for k in range(n_sim):
    Ni = np.concatenate([np.random.binomial(1, 0.1, 50),
                          np.random.binomial(1, 0.2, 50)])
    Xi = np.concatenate([np.random.gamma(10, scale=50, size=50),    # rate=0.02
                          np.random.lognormal(5, 1, size=50)])
    S_individual[k] = np.sum(Ni * Xi)

print(f"\n个体风险模型: E(S)={S_individual.mean():.1f}, Var(S)={S_individual.var():.0f}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(S_individual, bins=50, density=True, alpha=0.5, color='grey')
kde = gaussian_kde(S_individual)
x_range = np.linspace(0, S_individual.max(), 200)
ax.plot(x_range, kde(x_range), 'r-')
ax.set_xlabel('S'); ax.set_ylabel('密度')
ax.set_title('个体风险模型模拟')
plt.tight_layout(); plt.show()


##############################################################################
# 2.4 聚合风险模型计算方法
##############################################################################

## 2.4.1 Panjer递推
def panjer_poisson(p, lam):
    """Panjer递推（泊松分布）"""
    cumul = np.exp(-lam * np.sum(p))
    f = [cumul]
    s = 0
    while cumul <= 0.99999999:
        s += 1
        m = min(s, len(p))
        last = lam / s * np.sum(np.arange(1, m+1) * p[:m] * f[s-m:s][::-1])
        f.append(last)
        cumul += last
    return np.array(f)

f_panjer = panjer_poisson(np.array([0.25, 0.5, 0.25]), 4) * np.exp(4)
print(f"\nPanjer递推结果: {f_panjer}")


## 2.4.2 FFT法
x_fft = np.array([0, 0.5, 0.4, 0.1] + [0]*40)
phi_x = np.fft.fft(x_fft)
phi_s = np.exp(3 * (phi_x - 1))
fs = np.real(np.fft.ifft(phi_s))
Fs = np.cumsum(fs)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(range(min(31, len(fs))), fs[:31], color='red')
axes[0].set_xlabel('s'); axes[0].set_ylabel('f(s)')
axes[0].set_title('FFT法：概率函数')
axes[1].step(range(min(31, len(Fs))), Fs[:31], where='post', color='red')
axes[1].set_xlabel('s'); axes[1].set_ylabel('F(s)')
axes[1].set_title('FFT法：分布函数')
plt.tight_layout(); plt.show()


## 2.4.3 随机模拟法
np.random.seed(42)
lam = 3; mu = 6; sigma = 1.5; u = 1000
s_sim = np.zeros(n_sim)
for i in range(n_sim):
    n = np.random.poisson(lam)
    s_sim[i] = np.sum(np.minimum(np.random.lognormal(mu, sigma, n), u))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(s_sim, bins=100, density=True, color='red', alpha=0.5)
axes[0].set_xlabel('s'); axes[0].set_title('随机模拟：频率直方图')
s_sorted = np.sort(s_sim)
axes[1].plot(s_sorted, np.cumsum(s_sorted)/s_sorted.sum(), 'r-')
axes[1].set_xlabel('s'); axes[1].set_title('Lorenz曲线')
plt.tight_layout(); plt.show()


## 2.4.4 随机模拟求累积损失的分布（例2.7）
np.random.seed(321)
d = 250; u = 1000
r_nb = 3; beta_nb = 2
alpha_g = 100; theta_g = 0.2
P = np.zeros(n_sim)

for i in range(n_sim):
    n = np.random.negative_binomial(r_nb, 1/(1+beta_nb))
    x = np.random.gamma(alpha_g, 1/theta_g, n)
    w = np.minimum(x, d)
    v = min(w.sum(), u)
    S = x.sum()
    P[i] = S - v

print(f"\n例2.7: E(P)={P.mean():.2f}, P95={np.percentile(P, 95):.2f}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(P, bins=50, density=True, color='grey', alpha=0.5)
ax.set_xlabel('累积赔款'); ax.set_ylabel('频率')
ax.set_title('保险人年度累积赔款')
plt.tight_layout(); plt.show()


##############################################################################
# 2.5 近似计算方法
##############################################################################

## 2.5.1 Tweedie分布模拟
np.random.seed(11)
lam_tw = 1; alpha_tw = 10; beta_tw = 2
Y = np.zeros(n_sim)
for i in range(n_sim):
    N = np.random.poisson(lam_tw)
    Y[i] = np.random.gamma(alpha_tw, 1/beta_tw, N).sum()

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(Y, bins=50, color='grey', alpha=0.5)
ax.set_title('Tweedie分布模拟')
plt.tight_layout(); plt.show()


## 2.5.2 近似计算比较
np.random.seed(123)
S_approx = np.zeros(n_sim)
for i in range(n_sim):
    N = np.random.binomial(1000, 0.001)
    S_approx[i] = N  # 每次索赔金额为1

ES = S_approx.mean()
varS = S_approx.var()

# 精确分布
p0 = 1 - stats.binom.cdf(3.5, 1000, 0.001)
p1 = 1 - stats.poisson.cdf(3.5, 1)

# 正态近似
p2 = 1 - stats.norm.cdf(3.5, ES, np.sqrt(varS))

# 平移伽马近似: S+1 ~ Gamma(4, 2)
p4 = 1 - stats.gamma.cdf(4.5, 4, scale=0.5)

# NP近似
mu_np = sigma_np = gamma_np = 1
a_np = -3/gamma_np + np.sqrt(9/gamma_np**2 + 1 + 6/gamma_np * (3.5 - mu_np) / sigma_np)
p5 = 1 - stats.norm.cdf(a_np)

print(f"\n近似计算比较 P(S>3.5):")
print(f"  二项分布(精确): {p0:.6f}")
print(f"  泊松分布(精确): {p1:.6f}")
print(f"  正态近似:       {p2:.6f}")
print(f"  平移伽马近似:   {p4:.6f}")
print(f"  NP近似:         {p5:.6f}")

##############################################################################
# 第2章 风险模型
# 对应教材：section2.tex
# 内容：短期聚合风险模型、复合分布、短期个体风险模型、
#       参数不确定性的影响、近似计算方法
##############################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


##############################################################################
# 2.1 短期聚合风险模型：卷积法
##############################################################################

## 2.4.4 随机模拟求累积损失的分布（例2.7）
np.random.seed(321)
d = 250; u = 1000
r_nb = 3; beta_nb = 2
alpha_g = 100; theta_g = 0.2
P = np.zeros(n_sim)

for i in range(n_sim):
    n = np.random.negative_binomial(r_nb, 1/(1+beta_nb))
    x = np.random.gamma(alpha_g, 1/theta_g, n)
    w = np.minimum(x, d)
    v = min(w.sum(), u)
    S = x.sum()
    P[i] = S - v

print(f"\n例2.7: E(P)={P.mean():.2f}, P95={np.percentile(P, 95):.2f}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(P, bins=50, density=True, color='grey', alpha=0.5)
ax.set_xlabel('累积赔款'); ax.set_ylabel('频率')
ax.set_title('保险人年度累积赔款')
plt.tight_layout(); plt.show()


##############################################################################
# 2.5 近似计算方法
##############################################################################

##############################################################################
# 第2章 风险模型
# 对应教材：section2.tex
# 内容：短期聚合风险模型、复合分布、短期个体风险模型、
#       参数不确定性的影响、近似计算方法
##############################################################################

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


##############################################################################
# 2.1 短期聚合风险模型：卷积法
##############################################################################

## 2.1.1 卷积法计算累积索赔金额的分布
pn = np.array([0.3, 0.5, 0.2])           # 索赔次数概率
fx = np.array([0.2, 0.4, 0.2, 0.1, 0.1]) # 索赔强度概率

# 卷积法计算S的分布
max_s = 4 * 4
FS = np.zeros(max_s + 1)

# N=0
FS[0] = pn[0]
# N=1
for k in range(1, 5):
    FS[k] += pn[1] * fx[k-1]
# N=2
for k1 in range(1, 5):
    for k2 in range(1, 5):
        s = k1 + k2
        FS[s] += pn[2] * fx[k1-1] * fx[k2-1]

print("卷积法计算累积索赔金额分布:")
print(f"  P(S=0) = {FS[0]:.4f}")
for s in range(1, min(9, max_s+1)):
    if FS[s] > 0:
        print(f"  P(S={s}) = {FS[s]:.4f}")

# 绘图
fig, ax = plt.subplots(figsize=(8, 5))
ax.step(range(max_s+1), np.cumsum(FS), where='post')
ax.set_xlabel('s'); ax.set_ylabel('F(s)')
ax.set_title('累积索赔金额分布函数')
plt.tight_layout(); plt.show()


##############################################################################
# 2.2 复合分布的随机模拟
##############################################################################

## 2.2.1 复合泊松分布
np.random.seed(123)
n_sim = 10000
S_poisson = np.zeros(n_sim)
for i in range(n_sim):
    N = np.random.poisson(2.5)
    X = np.random.gamma(shape=2, scale=500, size=N)
    S_poisson[i] = X.sum()

print(f"\n复合泊松分布: E(S)={S_poisson.mean():.1f}, Var(S)={S_poisson.var():.1f}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(S_poisson, bins=range(0, 15001, 500), density=True, alpha=0.5, color='grey')
from scipy.stats import gaussian_kde
kde = gaussian_kde(S_poisson)
x_range = np.linspace(0, 15000, 200)
ax.plot(x_range, kde(x_range), 'r-')
ax.set_xlabel('S'); ax.set_ylabel('密度')
ax.set_title('复合泊松分布模拟')
plt.tight_layout(); plt.show()


## 2.2.2 复合二项分布
np.random.seed(123)
S_binomial = np.zeros(n_sim)
for i in range(n_sim):
    N = np.random.binomial(100, 0.01)
    X = np.random.gamma(shape=10, scale=5, size=N)  # rate=0.2 => scale=5
    S_binomial[i] = X.sum()

print(f"复合二项分布: E(S)={S_binomial.mean():.2f}, Var(S)={S_binomial.var():.1f}")


## 2.2.3 复合负二项分布
np.random.seed(123)
S_negbinom = np.zeros(n_sim)
for i in range(n_sim):
    N = np.random.negative_binomial(2.5, 0.5)
    X = np.random.gamma(shape=2, scale=500, size=N)
    S_negbinom[i] = X.sum()

print(f"复合负二项分布: E(S)={S_negbinom.mean():.1f}, Var(S)={S_negbinom.var():.1f}")


## 2.2.4 复合分布的数字特征（正态索赔强度）
for dist_name, N_dist in [('泊松', lambda: np.random.poisson(5, n_sim)),
                           ('二项', lambda: np.random.binomial(10, 0.5, n_sim)),
                           ('负二项', lambda: np.random.negative_binomial(2, 0.5, n_sim))]:
    N = N_dist()
    S = np.array([np.random.normal(1000, 10, n).sum() for n in N])
    ES = S.mean()
    varS = S.var()
    skewS = np.mean(S**3) - 3 * S.mean() * np.mean(S**2) + 2 * S.mean()**3
    print(f"\n复合{dist_name}分布（正态强度）: E(S)={ES:.1f}, Var(S)={varS:.1f}, skew={skewS:.1f}")


##############################################################################
# 2.3 短期个体风险模型
##############################################################################
np.random.seed(123)
S_individual = np.zeros(n_sim)
for k in range(n_sim):
    Ni = np.concatenate([np.random.binomial(1, 0.1, 50),
                          np.random.binomial(1, 0.2, 50)])
    Xi = np.concatenate([np.random.gamma(10, scale=50, size=50),    # rate=0.02
                          np.random.lognormal(5, 1, size=50)])
    S_individual[k] = np.sum(Ni * Xi)

print(f"\n个体风险模型: E(S)={S_individual.mean():.1f}, Var(S)={S_individual.var():.0f}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(S_individual, bins=50, density=True, alpha=0.5, color='grey')
kde = gaussian_kde(S_individual)
x_range = np.linspace(0, S_individual.max(), 200)
ax.plot(x_range, kde(x_range), 'r-')
ax.set_xlabel('S'); ax.set_ylabel('密度')
ax.set_title('个体风险模型模拟')
plt.tight_layout(); plt.show()


##############################################################################
# 2.4 聚合风险模型计算方法
##############################################################################

## 2.4.1 Panjer递推
def panjer_poisson(p, lam):
    """Panjer递推（泊松分布）"""
    cumul = np.exp(-lam * np.sum(p))
    f = [cumul]
    s = 0
    while cumul <= 0.99999999:
        s += 1
        m = min(s, len(p))
        last = lam / s * np.sum(np.arange(1, m+1) * p[:m] * f[s-m:s][::-1])
        f.append(last)
        cumul += last
    return np.array(f)

f_panjer = panjer_poisson(np.array([0.25, 0.5, 0.25]), 4) * np.exp(4)
print(f"\nPanjer递推结果: {f_panjer}")


## 2.4.2 FFT法
x_fft = np.array([0, 0.5, 0.4, 0.1] + [0]*40)
phi_x = np.fft.fft(x_fft)
phi_s = np.exp(3 * (phi_x - 1))
fs = np.real(np.fft.ifft(phi_s))
Fs = np.cumsum(fs)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(range(min(31, len(fs))), fs[:31], color='red')
axes[0].set_xlabel('s'); axes[0].set_ylabel('f(s)')
axes[0].set_title('FFT法：概率函数')
axes[1].step(range(min(31, len(Fs))), Fs[:31], where='post', color='red')
axes[1].set_xlabel('s'); axes[1].set_ylabel('F(s)')
axes[1].set_title('FFT法：分布函数')
plt.tight_layout(); plt.show()


## 2.4.3 随机模拟法
np.random.seed(42)
lam = 3; mu = 6; sigma = 1.5; u = 1000
s_sim = np.zeros(n_sim)
for i in range(n_sim):
    n = np.random.poisson(lam)
    s_sim[i] = np.sum(np.minimum(np.random.lognormal(mu, sigma, n), u))

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].hist(s_sim, bins=100, density=True, color='red', alpha=0.5)
axes[0].set_xlabel('s'); axes[0].set_title('随机模拟：频率直方图')
s_sorted = np.sort(s_sim)
axes[1].plot(s_sorted, np.cumsum(s_sorted)/s_sorted.sum(), 'r-')
axes[1].set_xlabel('s'); axes[1].set_title('Lorenz曲线')
plt.tight_layout(); plt.show()


## 2.4.4 随机模拟求累积损失的分布（例2.7）
np.random.seed(321)
d = 250; u = 1000
r_nb = 3; beta_nb = 2
alpha_g = 100; theta_g = 0.2
P = np.zeros(n_sim)

for i in range(n_sim):
    n = np.random.negative_binomial(r_nb, 1/(1+beta_nb))
    x = np.random.gamma(alpha_g, 1/theta_g, n)
    w = np.minimum(x, d)
    v = min(w.sum(), u)
    S = x.sum()
    P[i] = S - v

print(f"\n例2.7: E(P)={P.mean():.2f}, P95={np.percentile(P, 95):.2f}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(P, bins=50, density=True, color='grey', alpha=0.5)
ax.set_xlabel('累积赔款'); ax.set_ylabel('频率')
ax.set_title('保险人年度累积赔款')
plt.tight_layout(); plt.show()


##############################################################################
# 2.5 近似计算方法
##############################################################################

## 2.5.1 Tweedie分布模拟
np.random.seed(11)
lam_tw = 1; alpha_tw = 10; beta_tw = 2
Y = np.zeros(n_sim)
for i in range(n_sim):
    N = np.random.poisson(lam_tw)
    Y[i] = np.random.gamma(alpha_tw, 1/beta_tw, N).sum()

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(Y, bins=50, color='grey', alpha=0.5)
ax.set_title('Tweedie分布模拟')
plt.tight_layout(); plt.show()


## 2.5.2 近似计算比较
np.random.seed(123)
S_approx = np.zeros(n_sim)
for i in range(n_sim):
    N = np.random.binomial(1000, 0.001)
    S_approx[i] = N  # 每次索赔金额为1

ES = S_approx.mean()
varS = S_approx.var()

# 精确分布
p0 = 1 - stats.binom.cdf(3.5, 1000, 0.001)
p1 = 1 - stats.poisson.cdf(3.5, 1)

# 正态近似
p2 = 1 - stats.norm.cdf(3.5, ES, np.sqrt(varS))

# 平移伽马近似: S+1 ~ Gamma(4, 2)
p4 = 1 - stats.gamma.cdf(4.5, 4, scale=0.5)

# NP近似
mu_np = sigma_np = gamma_np = 1
a_np = -3/gamma_np + np.sqrt(9/gamma_np**2 + 1 + 6/gamma_np * (3.5 - mu_np) / sigma_np)
p5 = 1 - stats.norm.cdf(a_np)

print(f"\n近似计算比较 P(S>3.5):")
print(f"  二项分布(精确): {p0:.6f}")
print(f"  泊松分布(精确): {p1:.6f}")
print(f"  正态近似:       {p2:.6f}")
print(f"  平移伽马近似:   {p4:.6f}")
print(f"  NP近似:         {p5:.6f}")