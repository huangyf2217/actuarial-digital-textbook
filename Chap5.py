# Chap5 Python代码
# 自动从chap5.html同步生成

##############################################################################
# 第5章 损失调整与再保险
# 对应教材：section5.tex
# 内容：通货膨胀影响、再保险类型、免赔额与限额、
#       不完整数据估计、再保险定价
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
# 5.1 通货膨胀对损失分布的影响
##############################################################################
mu, sigma = 5, 1
r = 0.1  # 通胀率
mu_new = mu + np.log(1 + r)

E_X = np.exp(mu + sigma**2/2)
E_Y = np.exp(mu_new + sigma**2/2)
print(f"通胀前后: E(X)={E_X:.2f}, E(Y)={E_Y:.2f}, 比值={E_Y/E_X:.4f}")

# 绘图
x = np.linspace(0.1, 500, 500)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, stats.lognorm.pdf(x, s=sigma, scale=np.exp(mu)), 'b-', label='原分布')
ax.plot(x, stats.lognorm.pdf(x, s=sigma, scale=np.exp(mu_new)), 'r--', label='通胀后')
ax.set_xlabel('x'); ax.set_ylabel('f(x)')
ax.set_title('通胀对损失分布的影响'); ax.legend()
plt.tight_layout(); plt.show()


##############################################################################
# 5.2 比例再保险
##############################################################################
np.random.seed(123)
n_sim = 10000
X = np.random.lognormal(mu, sigma, n_sim)
alpha = 0.7  # 自留比例

X_primary = alpha * X
X_reins = (1 - alpha) * X

print(f"\n比例再保险（自留比例0.7）:")
print(f"  原损失均值: {X.mean():.2f}")
print(f"  原保险人赔付均值: {X_primary.mean():.2f}")
print(f"  再保险人赔付均值: {X_reins.mean():.2f}")


##############################################################################
# 5.3 超额赔款再保险
##############################################################################

## 5.3.1 单个损失的再保险赔付
M, L = 1000, 5000
Y_reins = np.minimum(np.maximum(X - M, 0), L)
Y_primary = X - Y_reins

print(f"\n超额赔款再保险（M=1000, L=5000）:")
print(f"  原保险人赔付均值: {Y_primary.mean():.2f}")
print(f"  再保险人赔付均值: {Y_reins.mean():.2f}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(Y_reins[Y_reins > 0], bins=50, density=True, alpha=0.5, color='grey')
ax.set_xlabel('再保险赔付'); ax.set_ylabel('密度')
ax.set_title('超额赔款再保险赔付分布')
plt.tight_layout(); plt.show()


## 5.3.2 累积赔付的再保险
M_agg, L_agg = 5000, 20000
S_agg = np.zeros(n_sim)
for i in range(n_sim):
    N = np.random.poisson(5)
    S_agg[i] = np.random.lognormal(mu, sigma, N).sum()

Y_reins_agg = np.minimum(np.maximum(S_agg - M_agg, 0), L_agg)
Y_primary_agg = S_agg - Y_reins_agg

print(f"\n累积超额赔款再保险:")
print(f"  累积损失均值: {S_agg.mean():.2f}")
print(f"  再保险人赔付均值: {Y_reins_agg.mean():.2f}")
print(f"  P95: {np.percentile(S_agg, 95):.2f}")


##############################################################################
# 5.4 免赔额与限额的影响
##############################################################################
print(f"\n免赔额对期望赔付的影响:")
for d in [0, 50, 100, 200, 500, 1000]:
    E_X_above_d = np.exp(mu + sigma**2/2) * \
                  stats.norm.cdf((mu + sigma**2 - np.log(d))/sigma) if d > 0 else E_X
    P_above_d = 1 - stats.lognorm.cdf(d, s=sigma, scale=np.exp(mu)) if d > 0 else 1
    E_X_given_d = E_X_above_d / P_above_d if d > 0 else E_X
    print(f"  d={d}: E[X|X>d]={E_X_given_d:.2f}, P(X>d)={P_above_d:.4f}")

print(f"\n限额对期望赔付的影响:")
for u in [100, 500, 1000, 5000, 10000]:
    E_X_below_u = np.exp(mu + sigma**2/2) * \
                  stats.norm.cdf((mu + sigma**2 - np.log(u))/sigma)
    P_above_u = 1 - stats.lognorm.cdf(u, s=sigma, scale=np.exp(mu))
    E_min_Xu = E_X_below_u + u * P_above_u
    print(f"  u={u}: E[min(X,u)]={E_min_Xu:.2f}")


##############################################################################
# 5.5 不完整数据估计：截断数据
##############################################################################
np.random.seed(123)
d = 100
X_trunc = []
while len(X_trunc) < 1000:
    x = np.random.lognormal(mu, sigma)
    if x > d:
        X_trunc.append(x)
X_trunc = np.array(X_trunc)

# 截断数据MLE
log_x = np.log(X_trunc)
mu_hat = log_x.mean()
sigma_hat = log_x.std(ddof=1)
print(f"\n截断数据MLE: μ_hat={mu_hat:.4f}, σ_hat={sigma_hat:.4f}")


##############################################################################
# 5.6 不完整数据估计：删失数据
##############################################################################
np.random.seed(123)
u = 5000
X_full = np.random.lognormal(mu, sigma, 1000)
X_cens = np.minimum(X_full, u)
censored = (X_full > u).astype(int)

# 使用未删失部分估计
uncens = X_cens[censored == 0]
log_uncens = np.log(uncens)
mu_hat_c = log_uncens.mean()
sigma_hat_c = log_uncens.std(ddof=1)
print(f"\n删失数据MLE（仅未删失）: μ_hat={mu_hat_c:.4f}, σ_hat={sigma_hat_c:.4f}")
print(f"  删失比例: {censored.mean():.4f}")


##############################################################################
# 5.7 再保险定价
##############################################################################

## 5.7.1 溢额再保险定价
np.random.seed(123)
lam, alpha_g, beta_g = 100, 2, 0.01
M, L = 1000, 5000
S_reins = np.zeros(n_sim)
for i in range(n_sim):
    N = np.random.poisson(lam)
    X = np.random.gamma(alpha_g, 1/beta_g, N)
    S_reins[i] = np.minimum(np.maximum(X - M, 0), L).sum()

loading = 0.2
premium = S_reins.mean() * (1 + loading)
print(f"\n溢额再保险定价:")
print(f"  E[S_reins]={S_reins.mean():.2f}, Var={S_reins.var():.1f}")
print(f"  再保险保费(20%附加)={premium:.2f}")


## 5.7.2 比例再保险定价
alpha_quota = 0.3
S_reins_q = np.zeros(n_sim)
for i in range(n_sim):
    N = np.random.poisson(lam)
    X = np.random.gamma(alpha_g, 1/beta_g, N)
    S_reins_q[i] = (alpha_quota * X).sum()

premium_q = S_reins_q.mean() * 1.15
print(f"\n比例再保险定价:")
print(f"  E[S_reins]={S_reins_q.mean():.2f}")
print(f"  再保险保费(15%附加)={premium_q:.2f}")


##############################################################################
# 5.8 通货膨胀对再保险的影响
##############################################################################
print(f"\n通胀对超额赔款再保险的影响:")
for r in [0, 0.05, 0.10, 0.15, 0.20]:
    mu_new = mu + np.log(1 + r)
    np.random.seed(123)
    S_r = np.zeros(n_sim)
    for i in range(n_sim):
        N = np.random.poisson(100)
        X = np.random.lognormal(mu_new, sigma, N)
        S_r[i] = np.minimum(np.maximum(X - M, 0), L).sum()
    print(f"  r={r:.2f}: E[S_reins]={S_r.mean():.2f}")

##############################################################################
# 第5章 损失调整与再保险
# 对应教材：section5.tex
# 内容：通货膨胀影响、再保险类型、免赔额与限额、
#       不完整数据估计、再保险定价
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
# 5.1 通货膨胀对损失分布的影响
##############################################################################
mu, sigma = 5, 1
r = 0.1  # 通胀率
mu_new = mu + np.log(1 + r)

E_X = np.exp(mu + sigma**2/2)
E_Y = np.exp(mu_new + sigma**2/2)
print(f"通胀前后: E(X)={E_X:.2f}, E(Y)={E_Y:.2f}, 比值={E_Y/E_X:.4f}")

# 绘图
x = np.linspace(0.1, 500, 500)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, stats.lognorm.pdf(x, s=sigma, scale=np.exp(mu)), 'b-', label='原分布')
ax.plot(x, stats.lognorm.pdf(x, s=sigma, scale=np.exp(mu_new)), 'r--', label='通胀后')
ax.set_xlabel('x'); ax.set_ylabel('f(x)')
ax.set_title('通胀对损失分布的影响'); ax.legend()
plt.tight_layout(); plt.show()


##############################################################################
# 5.2 比例再保险
##############################################################################
np.random.seed(123)
n_sim = 10000
X = np.random.lognormal(mu, sigma, n_sim)
alpha = 0.7  # 自留比例

X_primary = alpha * X
X_reins = (1 - alpha) * X

print(f"\n比例再保险（自留比例0.7）:")
print(f"  原损失均值: {X.mean():.2f}")
print(f"  原保险人赔付均值: {X_primary.mean():.2f}")
print(f"  再保险人赔付均值: {X_reins.mean():.2f}")


##############################################################################
# 5.3 超额赔款再保险
##############################################################################

## 5.3.1 单个损失的再保险赔付
M, L = 1000, 5000
Y_reins = np.minimum(np.maximum(X - M, 0), L)
Y_primary = X - Y_reins

print(f"\n超额赔款再保险（M=1000, L=5000）:")
print(f"  原保险人赔付均值: {Y_primary.mean():.2f}")
print(f"  再保险人赔付均值: {Y_reins.mean():.2f}")

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(Y_reins[Y_reins > 0], bins=50, density=True, alpha=0.5, color='grey')
ax.set_xlabel('再保险赔付'); ax.set_ylabel('密度')
ax.set_title('超额赔款再保险赔付分布')
plt.tight_layout(); plt.show()


## 5.3.2 累积赔付的再保险
M_agg, L_agg = 5000, 20000
S_agg = np.zeros(n_sim)
for i in range(n_sim):
    N = np.random.poisson(5)
    S_agg[i] = np.random.lognormal(mu, sigma, N).sum()

Y_reins_agg = np.minimum(np.maximum(S_agg - M_agg, 0), L_agg)
Y_primary_agg = S_agg - Y_reins_agg

print(f"\n累积超额赔款再保险:")
print(f"  累积损失均值: {S_agg.mean():.2f}")
print(f"  再保险人赔付均值: {Y_reins_agg.mean():.2f}")
print(f"  P95: {np.percentile(S_agg, 95):.2f}")


##############################################################################
# 5.4 免赔额与限额的影响
##############################################################################
print(f"\n免赔额对期望赔付的影响:")
for d in [0, 50, 100, 200, 500, 1000]:
    E_X_above_d = np.exp(mu + sigma**2/2) * \
                  stats.norm.cdf((mu + sigma**2 - np.log(d))/sigma) if d > 0 else E_X
    P_above_d = 1 - stats.lognorm.cdf(d, s=sigma, scale=np.exp(mu)) if d > 0 else 1
    E_X_given_d = E_X_above_d / P_above_d if d > 0 else E_X
    print(f"  d={d}: E[X|X>d]={E_X_given_d:.2f}, P(X>d)={P_above_d:.4f}")

print(f"\n限额对期望赔付的影响:")
for u in [100, 500, 1000, 5000, 10000]:
    E_X_below_u = np.exp(mu + sigma**2/2) * \
                  stats.norm.cdf((mu + sigma**2 - np.log(u))/sigma)
    P_above_u = 1 - stats.lognorm.cdf(u, s=sigma, scale=np.exp(mu))
    E_min_Xu = E_X_below_u + u * P_above_u
    print(f"  u={u}: E[min(X,u)]={E_min_Xu:.2f}")


##############################################################################
# 5.5 不完整数据估计：截断数据
##############################################################################
np.random.seed(123)
d = 100
X_trunc = []
while len(X_trunc) < 1000:
    x = np.random.lognormal(mu, sigma)
    if x > d:
        X_trunc.append(x)
X_trunc = np.array(X_trunc)

# 截断数据MLE
log_x = np.log(X_trunc)
mu_hat = log_x.mean()
sigma_hat = log_x.std(ddof=1)
print(f"\n截断数据MLE: μ_hat={mu_hat:.4f}, σ_hat={sigma_hat:.4f}")


##############################################################################
# 5.6 不完整数据估计：删失数据
##############################################################################
np.random.seed(123)
u = 5000
X_full = np.random.lognormal(mu, sigma, 1000)
X_cens = np.minimum(X_full, u)
censored = (X_full > u).astype(int)

# 使用未删失部分估计
uncens = X_cens[censored == 0]
log_uncens = np.log(uncens)
mu_hat_c = log_uncens.mean()
sigma_hat_c = log_uncens.std(ddof=1)
print(f"\n删失数据MLE（仅未删失）: μ_hat={mu_hat_c:.4f}, σ_hat={sigma_hat_c:.4f}")
print(f"  删失比例: {censored.mean():.4f}")


##############################################################################
# 5.7 再保险定价
##############################################################################

## 5.7.1 溢额再保险定价
np.random.seed(123)
lam, alpha_g, beta_g = 100, 2, 0.01
M, L = 1000, 5000
S_reins = np.zeros(n_sim)
for i in range(n_sim):
    N = np.random.poisson(lam)
    X = np.random.gamma(alpha_g, 1/beta_g, N)
    S_reins[i] = np.minimum(np.maximum(X - M, 0), L).sum()

loading = 0.2
premium = S_reins.mean() * (1 + loading)
print(f"\n溢额再保险定价:")
print(f"  E[S_reins]={S_reins.mean():.2f}, Var={S_reins.var():.1f}")
print(f"  再保险保费(20%附加)={premium:.2f}")


## 5.7.2 比例再保险定价
alpha_quota = 0.3
S_reins_q = np.zeros(n_sim)
for i in range(n_sim):
    N = np.random.poisson(lam)
    X = np.random.gamma(alpha_g, 1/beta_g, N)
    S_reins_q[i] = (alpha_quota * X).sum()

premium_q = S_reins_q.mean() * 1.15
print(f"\n比例再保险定价:")
print(f"  E[S_reins]={S_reins_q.mean():.2f}")
print(f"  再保险保费(15%附加)={premium_q:.2f}")


##############################################################################
# 5.8 通货膨胀对再保险的影响
##############################################################################
print(f"\n通胀对超额赔款再保险的影响:")
for r in [0, 0.05, 0.10, 0.15, 0.20]:
    mu_new = mu + np.log(1 + r)
    np.random.seed(123)
    S_r = np.zeros(n_sim)
    for i in range(n_sim):
        N = np.random.poisson(100)
        X = np.random.lognormal(mu_new, sigma, N)
        S_r[i] = np.minimum(np.maximum(X - M, 0), L).sum()
    print(f"  r={r:.2f}: E[S_reins]={S_r.mean():.2f}")

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

# 5.1 通货膨胀对损失分布的影响
##############################################################################
mu, sigma = 5, 1
r = 0.1  # 通胀率
mu_new = mu + np.log(1 + r)

E_X = np.exp(mu + sigma**2/2)
E_Y = np.exp(mu_new + sigma**2/2)
print(f"通胀前后: E(X)={E_X:.2f}, E(Y)={E_Y:.2f}, 比值={E_Y/E_X:.4f}")

# 绘图
x = np.linspace(0.1, 500, 500)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, stats.lognorm.pdf(x, s=sigma, scale=np.exp(mu)), 'b-', label='原分布')
ax.plot(x, stats.lognorm.pdf(x, s=sigma, scale=np.exp(mu_new)), 'r--', label='通胀后')
ax.set_xlabel('x'); ax.set_ylabel('f(x)')
ax.set_title('通胀对损失分布的影响'); ax.legend()
plt.tight_layout(); plt.show()


##############################################################################
# 5.2 比例再保险
##############################################################################
np.random.seed(123)
n_sim = 10000
X = np.random.lognormal(mu, sigma, n_sim)
alpha = 0.7  # 自留比例

X_primary = alpha * X
X_reins = (1 - alpha) * X

print(f"\n比例再保险（自留比例0.7）:")

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

# 5.1 通货膨胀对损失分布的影响
##############################################################################
mu, sigma = 5, 1
r = 0.1  # 通胀率
mu_new = mu + np.log(1 + r)

E_X = np.exp(mu + sigma**2/2)
E_Y = np.exp(mu_new + sigma**2/2)
print(f"通胀前后: E(X)={E_X:.2f}, E(Y)={E_Y:.2f}, 比值={E_Y/E_X:.4f}")

# 绘图
x = np.linspace(0.1, 500, 500)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, stats.lognorm.pdf(x, s=sigma, scale=np.exp(mu)), 'b-', label='原分布')
ax.plot(x, stats.lognorm.pdf(x, s=sigma, scale=np.exp(mu_new)), 'r--', label='通胀后')
ax.set_xlabel('x'); ax.set_ylabel('f(x)')
ax.set_title('通胀对损失分布的影响'); ax.legend()
plt.tight_layout(); plt.show()


##############################################################################
# 5.2 比例再保险
##############################################################################
np.random.seed(123)
n_sim = 10000
X = np.random.lognormal(mu, sigma, n_sim)
alpha = 0.7  # 自留比例

X_primary = alpha * X
X_reins = (1 - alpha) * X

print(f"\n比例再保险（自留比例0.7）:")

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

# 5.1 通货膨胀对损失分布的影响
##############################################################################
mu, sigma = 5, 1
r = 0.1  # 通胀率
mu_new = mu + np.log(1 + r)

E_X = np.exp(mu + sigma**2/2)
E_Y = np.exp(mu_new + sigma**2/2)
print(f"通胀前后: E(X)={E_X:.2f}, E(Y)={E_Y:.2f}, 比值={E_Y/E_X:.4f}")

# 绘图
x = np.linspace(0.1, 500, 500)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, stats.lognorm.pdf(x, s=sigma, scale=np.exp(mu)), 'b-', label='原分布')
ax.plot(x, stats.lognorm.pdf(x, s=sigma, scale=np.exp(mu_new)), 'r--', label='通胀后')
ax.set_xlabel('x'); ax.set_ylabel('f(x)')
ax.set_title('通胀对损失分布的影响'); ax.legend()
plt.tight_layout(); plt.show()


##############################################################################
# 5.2 比例再保险
##############################################################################
np.random.seed(123)
n_sim = 10000
X = np.random.lognormal(mu, sigma, n_sim)
alpha = 0.7  # 自留比例

X_primary = alpha * X
X_reins = (1 - alpha) * X

print(f"\n比例再保险（自留比例0.7）:")

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

# 5.1 通货膨胀对损失分布的影响
##############################################################################
mu, sigma = 5, 1
r = 0.1  # 通胀率
mu_new = mu + np.log(1 + r)

E_X = np.exp(mu + sigma**2/2)
E_Y = np.exp(mu_new + sigma**2/2)
print(f"通胀前后: E(X)={E_X:.2f}, E(Y)={E_Y:.2f}, 比值={E_Y/E_X:.4f}")

# 绘图
x = np.linspace(0.1, 500, 500)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, stats.lognorm.pdf(x, s=sigma, scale=np.exp(mu)), 'b-', label='原分布')
ax.plot(x, stats.lognorm.pdf(x, s=sigma, scale=np.exp(mu_new)), 'r--', label='通胀后')
ax.set_xlabel('x'); ax.set_ylabel('f(x)')
ax.set_title('通胀对损失分布的影响'); ax.legend()
plt.tight_layout(); plt.show()


##############################################################################
# 5.2 比例再保险
##############################################################################
np.random.seed(123)
n_sim = 10000
X = np.random.lognormal(mu, sigma, n_sim)
alpha = 0.7  # 自留比例

X_primary = alpha * X
X_reins = (1 - alpha) * X

print(f"\n比例再保险（自留比例0.7）:")

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

# 5.1 通货膨胀对损失分布的影响
##############################################################################
mu, sigma = 5, 1
r = 0.1  # 通胀率
mu_new = mu + np.log(1 + r)

E_X = np.exp(mu + sigma**2/2)
E_Y = np.exp(mu_new + sigma**2/2)
print(f"通胀前后: E(X)={E_X:.2f}, E(Y)={E_Y:.2f}, 比值={E_Y/E_X:.4f}")

# 绘图
x = np.linspace(0.1, 500, 500)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, stats.lognorm.pdf(x, s=sigma, scale=np.exp(mu)), 'b-', label='原分布')
ax.plot(x, stats.lognorm.pdf(x, s=sigma, scale=np.exp(mu_new)), 'r--', label='通胀后')
ax.set_xlabel('x'); ax.set_ylabel('f(x)')
ax.set_title('通胀对损失分布的影响'); ax.legend()
plt.tight_layout(); plt.show()


##############################################################################
# 5.2 比例再保险
##############################################################################
np.random.seed(123)
n_sim = 10000
X = np.random.lognormal(mu, sigma, n_sim)
alpha = 0.7  # 自留比例

X_primary = alpha * X
X_reins = (1 - alpha) * X

print(f"\n比例再保险（自留比例0.7）:")

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

# 5.1 通货膨胀对损失分布的影响
##############################################################################
mu, sigma = 5, 1
r = 0.1  # 通胀率
mu_new = mu + np.log(1 + r)

E_X = np.exp(mu + sigma**2/2)
E_Y = np.exp(mu_new + sigma**2/2)
print(f"通胀前后: E(X)={E_X:.2f}, E(Y)={E_Y:.2f}, 比值={E_Y/E_X:.4f}")

# 绘图
x = np.linspace(0.1, 500, 500)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, stats.lognorm.pdf(x, s=sigma, scale=np.exp(mu)), 'b-', label='原分布')
ax.plot(x, stats.lognorm.pdf(x, s=sigma, scale=np.exp(mu_new)), 'r--', label='通胀后')
ax.set_xlabel('x'); ax.set_ylabel('f(x)')
ax.set_title('通胀对损失分布的影响'); ax.legend()
plt.tight_layout(); plt.show()


##############################################################################
# 5.2 比例再保险
##############################################################################
np.random.seed(123)
n_sim = 10000
X = np.random.lognormal(mu, sigma, n_sim)
alpha = 0.7  # 自留比例

X_primary = alpha * X
X_reins = (1 - alpha) * X

print(f"\n比例再保险（自留比例0.7）:")

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

# 5.1 通货膨胀对损失分布的影响
##############################################################################
mu, sigma = 5, 1
r = 0.1  # 通胀率
mu_new = mu + np.log(1 + r)

E_X = np.exp(mu + sigma**2/2)
E_Y = np.exp(mu_new + sigma**2/2)
print(f"通胀前后: E(X)={E_X:.2f}, E(Y)={E_Y:.2f}, 比值={E_Y/E_X:.4f}")

# 绘图
x = np.linspace(0.1, 500, 500)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, stats.lognorm.pdf(x, s=sigma, scale=np.exp(mu)), 'b-', label='原分布')
ax.plot(x, stats.lognorm.pdf(x, s=sigma, scale=np.exp(mu_new)), 'r--', label='通胀后')
ax.set_xlabel('x'); ax.set_ylabel('f(x)')
ax.set_title('通胀对损失分布的影响'); ax.legend()
plt.tight_layout(); plt.show()


##############################################################################
# 5.2 比例再保险
##############################################################################
np.random.seed(123)
n_sim = 10000
X = np.random.lognormal(mu, sigma, n_sim)
alpha = 0.7  # 自留比例

X_primary = alpha * X
X_reins = (1 - alpha) * X

print(f"\n比例再保险（自留比例0.7）:")

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

# 5.1 通货膨胀对损失分布的影响
##############################################################################
mu, sigma = 5, 1
r = 0.1  # 通胀率
mu_new = mu + np.log(1 + r)

E_X = np.exp(mu + sigma**2/2)
E_Y = np.exp(mu_new + sigma**2/2)
print(f"通胀前后: E(X)={E_X:.2f}, E(Y)={E_Y:.2f}, 比值={E_Y/E_X:.4f}")

# 绘图
x = np.linspace(0.1, 500, 500)
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x, stats.lognorm.pdf(x, s=sigma, scale=np.exp(mu)), 'b-', label='原分布')
ax.plot(x, stats.lognorm.pdf(x, s=sigma, scale=np.exp(mu_new)), 'r--', label='通胀后')
ax.set_xlabel('x'); ax.set_ylabel('f(x)')
ax.set_title('通胀对损失分布的影响'); ax.legend()
plt.tight_layout(); plt.show()


##############################################################################
# 5.2 比例再保险
##############################################################################
np.random.seed(123)
n_sim = 10000
X = np.random.lognormal(mu, sigma, n_sim)
alpha = 0.7  # 自留比例

X_primary = alpha * X
X_reins = (1 - alpha) * X

print(f"\n比例再保险（自留比例0.7）:")