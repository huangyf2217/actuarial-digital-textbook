# Chap7 Python代码
# 自动从chap7.html同步生成

##############################################################################
# 第7章 广义线性模型
# 对应教材：section7.tex
# 内容：指数族分布、连接函数、参数估计、模型诊断、
#       泊松回归、负二项回归、伽马回归、GAM与GAMLSS
##############################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Poisson, Gamma, Binomial, NegativeBinomial
from statsmodels.genmod.families.links import log, logit, identity
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


##############################################################################
# 7.1 指数族分布
##############################################################################
theta = np.array([-2, -1, 0, 1, 2])
b_normal = theta**2 / 2
b_poisson = np.exp(theta)
b_gamma = -np.log(-theta[theta < 0])  # 只对θ<0有定义
print("指数族分布的b(θ)函数:")
print(f"  正态: {b_normal}")
print(f"  泊松: {b_poisson}")
print(f"  伽马(θ<0): {b_gamma}")


##############################################################################
# 7.2 泊松回归
##############################################################################

## 7.2.1 生成模拟数据
np.random.seed(123)
n = 500
x1 = np.random.normal(0, 1, n)
x2 = np.random.binomial(1, 0.5, n)
eta = 1 + 0.5 * x1 + 0.3 * x2
lam = np.exp(eta)
y = np.random.poisson(lam)
df_poisson = pd.DataFrame({'y': y, 'x1': x1, 'x2': x2})

## 7.2.2 泊松回归模型
X = sm.add_constant(df_poisson[['x1', 'x2']])
model_poisson = GLM(df_poisson['y'], X, family=Poisson()).fit()
print(f"\n泊松回归:")
print(model_poisson.summary().tables[1])


##############################################################################
# 7.3 负二项回归
##############################################################################

## 7.3.1 生成过散布数据
np.random.seed(456)
x1_nb = np.random.normal(0, 1, n)
x2_nb = np.random.binomial(1, 0.5, n)
eta_nb = 1 + 0.5 * x1_nb + 0.3 * x2_nb
mu_nb = np.exp(eta_nb)
k = 2
p_nb = k / (k + mu_nb)
y_nb = np.random.negative_binomial(k, p_nb)
df_nb = pd.DataFrame({'y': y_nb, 'x1': x1_nb, 'x2': x2_nb})

## 7.3.2 负二项回归
X_nb = sm.add_constant(df_nb[['x1', 'x2']])
model_nb = smf.glm('y ~ x1 + x2', data=df_nb,
                    family=sm.families.NegativeBinomial(alpha=1/k)).fit()
print(f"\n负二项回归:")
print(model_nb.summary().tables[1])

## 7.3.3 泊松回归（过散布数据）
model_poisson_over = GLM(df_nb['y'], X_nb, family=Poisson()).fit()
print(f"\n泊松回归（过散布数据）:")
print(model_poisson_over.summary().tables[1])
print(f"  Pearson chi2: {model_poisson_over.pearson_chi2:.2f}")


##############################################################################
# 7.4 伽马回归
##############################################################################
np.random.seed(789)
x1_g = np.random.normal(0, 1, n)
x2_g = np.random.binomial(1, 0.5, n)
eta_g = 3 + 0.5 * x1_g - 0.3 * x2_g
mu_g = np.exp(eta_g)
phi = 0.5
shape = 1 / phi
scale = mu_g * phi
y_g = np.random.gamma(shape, scale)
df_gamma = pd.DataFrame({'y': y_g, 'x1': x1_g, 'x2': x2_g})

X_g = sm.add_constant(df_gamma[['x1', 'x2']])
model_gamma = GLM(df_gamma['y'], X_g, family=Gamma(link=log())).fit()
print(f"\n伽马回归:")
print(model_gamma.summary().tables[1])


##############################################################################
# 7.5 Logistic回归
##############################################################################
np.random.seed(101)
x1_l = np.random.normal(0, 1, n)
x2_l = np.random.binomial(1, 0.5, n)
eta_l = -1 + 0.8 * x1_l + 0.5 * x2_l
p_l = 1 / (1 + np.exp(-eta_l))
y_l = np.random.binomial(1, p_l)
df_logit = pd.DataFrame({'y': y_l, 'x1': x1_l, 'x2': x2_l})

X_l = sm.add_constant(df_logit[['x1', 'x2']])
model_logit = GLM(df_logit['y'], X_l, family=Binomial(link=logit())).fit()
print(f"\nLogistic回归:")
print(model_logit.summary().tables[1])


##############################################################################
# 7.6 模型诊断
##############################################################################

## 7.6.1 残差分析
residuals_dev = model_poisson.resid_deviance
fitted = model_poisson.fittedvalues

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(fitted, residuals_dev, alpha=0.5)
axes[0].axhline(y=0, color='r', ls='--')
axes[0].set_xlabel('拟合值'); axes[0].set_ylabel('Deviance残差')
axes[0].set_title('Deviance残差 vs 拟合值')

from scipy import stats
stats.probplot(residuals_dev, dist='norm', plot=axes[1])
axes[1].set_title('Deviance残差Q-Q图')
plt.tight_layout(); plt.show()


## 7.6.2 偏差分析
print(f"\n偏差分析:")
print(f"  Null deviance: {model_poisson.null_deviance:.4f}")
print(f"  Residual deviance: {model_poisson.deviance:.4f}")
print(f"  减少的偏差: {model_poisson.null_deviance - model_poisson.deviance:.4f}")


##############################################################################
# 7.7 偏差与模型比较
##############################################################################
y_ex = np.array([10, 20, 15, 25, 30], dtype=float)
mu_ex = np.array([12, 18, 17, 23, 28])

D_poisson = 2 * np.sum(y_ex * np.log(y_ex / mu_ex) - (y_ex - mu_ex))
df_ex = len(y_ex) - 2
p_value = 1 - stats.chi2.cdf(D_poisson, df_ex)
print(f"\n偏差比较: D={D_poisson:.4f}, df={df_ex}, p值={p_value:.4f}")


##############################################################################
# 7.8 保险定价应用
##############################################################################

## 7.8.1 车险索赔次数模型
np.random.seed(202)
n_policy = 1000
age_group = np.random.randint(1, 5, n_policy)
vehicle_age = np.random.randint(1, 4, n_policy)
gender = np.random.binomial(1, 0.5, n_policy)

eta_auto = -1 + 0.3 * (age_group == 1) - 0.2 * (age_group == 4) + \
           0.4 * (vehicle_age == 1) + 0.1 * gender
lam_auto = np.exp(eta_auto)
n_claims = np.random.poisson(lam_auto)

df_auto = pd.DataFrame({
    'n_claims': n_claims,
    'age_group': pd.Categorical(age_group),
    'vehicle_age': pd.Categorical(vehicle_age),
    'gender': pd.Categorical(gender)
})

model_auto = smf.glm('n_claims ~ age_group + vehicle_age + gender',
                      data=df_auto, family=Poisson()).fit()
print(f"\n车险索赔次数泊松回归:")
print(model_auto.summary().tables[1])


## 7.8.2 车险索赔金额模型
np.random.seed(303)
n_sev = 500
age_group_s = np.random.randint(1, 5, n_sev)
vehicle_age_s = np.random.randint(1, 4, n_sev)

eta_sev = 8 + 0.2 * (age_group_s == 1) - 0.1 * (age_group_s == 4) + \
          0.3 * (vehicle_age_s == 1)
mu_sev = np.exp(eta_sev)
phi_s = 0.5
claim_amount = np.random.gamma(1/phi_s, mu_sev * phi_s)

df_sev = pd.DataFrame({
    'claim_amount': claim_amount,
    'age_group': pd.Categorical(age_group_s),
    'vehicle_age': pd.Categorical(vehicle_age_s)
})

model_sev = smf.glm('claim_amount ~ age_group + vehicle_age',
                     data=df_sev, family=Gamma(link=log())).fit()
print(f"\n车险索赔金额伽马回归:")
print(model_sev.summary().tables[1])


##############################################################################
# 7.9 GAM（广义可加模型）
##############################################################################
np.random.seed(404)
n_gam = 500
x_gam = np.random.uniform(0, 10, n_gam)
eta_gam = 1 + 0.5 * np.sin(x_gam) + 0.3 * x_gam
lam_gam = np.exp(eta_gam)
y_gam = np.random.poisson(lam_gam)
df_gam = pd.DataFrame({'y': y_gam, 'x': x_gam})

from statsmodels.gam.api import GLMGam, BSplines
bs = BSplines(df_gam['x'], df=[6], degree=[3])
model_gam = GLMGam.from_formula('y ~ 1', data=df_gam,
                                  smoother=bs, family=Poisson()).fit()
print(f"\nGAM模型:")
print(model_gam.summary().tables[1])

# 绘图
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(x_gam, y_gam, alpha=0.3, label='数据')
x_sort = np.sort(x_gam)
idx = np.argsort(x_gam)
ax.plot(x_sort, model_gam.fittedvalues[idx], 'r-', label='GAM拟合')
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_title('GAM拟合'); ax.legend()
plt.tight_layout(); plt.show()

##############################################################################
# 第7章 广义线性模型
# 对应教材：section7.tex
# 内容：指数族分布、连接函数、参数估计、模型诊断、
#       泊松回归、负二项回归、伽马回归、GAM与GAMLSS
##############################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Poisson, Gamma, Binomial, NegativeBinomial
from statsmodels.genmod.families.links import log, logit, identity
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


##############################################################################
# 7.1 指数族分布
##############################################################################
theta = np.array([-2, -1, 0, 1, 2])
b_normal = theta**2 / 2
b_poisson = np.exp(theta)
b_gamma = -np.log(-theta[theta < 0])  # 只对θ<0有定义
print("指数族分布的b(θ)函数:")
print(f"  正态: {b_normal}")
print(f"  泊松: {b_poisson}")
print(f"  伽马(θ<0): {b_gamma}")


##############################################################################
# 7.2 泊松回归
##############################################################################

## 7.2.1 生成模拟数据
np.random.seed(123)
n = 500
x1 = np.random.normal(0, 1, n)
x2 = np.random.binomial(1, 0.5, n)
eta = 1 + 0.5 * x1 + 0.3 * x2
lam = np.exp(eta)
y = np.random.poisson(lam)
df_poisson = pd.DataFrame({'y': y, 'x1': x1, 'x2': x2})

## 7.2.2 泊松回归模型
X = sm.add_constant(df_poisson[['x1', 'x2']])
model_poisson = GLM(df_poisson['y'], X, family=Poisson()).fit()
print(f"\n泊松回归:")
print(model_poisson.summary().tables[1])


##############################################################################
# 7.3 负二项回归
##############################################################################

## 7.3.1 生成过散布数据
np.random.seed(456)
x1_nb = np.random.normal(0, 1, n)
x2_nb = np.random.binomial(1, 0.5, n)
eta_nb = 1 + 0.5 * x1_nb + 0.3 * x2_nb
mu_nb = np.exp(eta_nb)
k = 2
p_nb = k / (k + mu_nb)
y_nb = np.random.negative_binomial(k, p_nb)
df_nb = pd.DataFrame({'y': y_nb, 'x1': x1_nb, 'x2': x2_nb})

## 7.3.2 负二项回归
X_nb = sm.add_constant(df_nb[['x1', 'x2']])
model_nb = smf.glm('y ~ x1 + x2', data=df_nb,
                    family=sm.families.NegativeBinomial(alpha=1/k)).fit()
print(f"\n负二项回归:")
print(model_nb.summary().tables[1])

## 7.3.3 泊松回归（过散布数据）
model_poisson_over = GLM(df_nb['y'], X_nb, family=Poisson()).fit()
print(f"\n泊松回归（过散布数据）:")
print(model_poisson_over.summary().tables[1])
print(f"  Pearson chi2: {model_poisson_over.pearson_chi2:.2f}")


##############################################################################
# 7.4 伽马回归
##############################################################################
np.random.seed(789)
x1_g = np.random.normal(0, 1, n)
x2_g = np.random.binomial(1, 0.5, n)
eta_g = 3 + 0.5 * x1_g - 0.3 * x2_g
mu_g = np.exp(eta_g)
phi = 0.5
shape = 1 / phi
scale = mu_g * phi
y_g = np.random.gamma(shape, scale)
df_gamma = pd.DataFrame({'y': y_g, 'x1': x1_g, 'x2': x2_g})

X_g = sm.add_constant(df_gamma[['x1', 'x2']])
model_gamma = GLM(df_gamma['y'], X_g, family=Gamma(link=log())).fit()
print(f"\n伽马回归:")
print(model_gamma.summary().tables[1])


##############################################################################
# 7.5 Logistic回归
##############################################################################
np.random.seed(101)
x1_l = np.random.normal(0, 1, n)
x2_l = np.random.binomial(1, 0.5, n)
eta_l = -1 + 0.8 * x1_l + 0.5 * x2_l
p_l = 1 / (1 + np.exp(-eta_l))
y_l = np.random.binomial(1, p_l)
df_logit = pd.DataFrame({'y': y_l, 'x1': x1_l, 'x2': x2_l})

X_l = sm.add_constant(df_logit[['x1', 'x2']])
model_logit = GLM(df_logit['y'], X_l, family=Binomial(link=logit())).fit()
print(f"\nLogistic回归:")
print(model_logit.summary().tables[1])


##############################################################################
# 7.6 模型诊断
##############################################################################

## 7.6.1 残差分析
residuals_dev = model_poisson.resid_deviance
fitted = model_poisson.fittedvalues

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(fitted, residuals_dev, alpha=0.5)
axes[0].axhline(y=0, color='r', ls='--')
axes[0].set_xlabel('拟合值'); axes[0].set_ylabel('Deviance残差')
axes[0].set_title('Deviance残差 vs 拟合值')

from scipy import stats
stats.probplot(residuals_dev, dist='norm', plot=axes[1])
axes[1].set_title('Deviance残差Q-Q图')
plt.tight_layout(); plt.show()


## 7.6.2 偏差分析
print(f"\n偏差分析:")
print(f"  Null deviance: {model_poisson.null_deviance:.4f}")
print(f"  Residual deviance: {model_poisson.deviance:.4f}")
print(f"  减少的偏差: {model_poisson.null_deviance - model_poisson.deviance:.4f}")


##############################################################################
# 7.7 偏差与模型比较
##############################################################################
y_ex = np.array([10, 20, 15, 25, 30], dtype=float)
mu_ex = np.array([12, 18, 17, 23, 28])

D_poisson = 2 * np.sum(y_ex * np.log(y_ex / mu_ex) - (y_ex - mu_ex))
df_ex = len(y_ex) - 2
p_value = 1 - stats.chi2.cdf(D_poisson, df_ex)
print(f"\n偏差比较: D={D_poisson:.4f}, df={df_ex}, p值={p_value:.4f}")


##############################################################################
# 7.8 保险定价应用
##############################################################################

## 7.8.1 车险索赔次数模型
np.random.seed(202)
n_policy = 1000
age_group = np.random.randint(1, 5, n_policy)
vehicle_age = np.random.randint(1, 4, n_policy)
gender = np.random.binomial(1, 0.5, n_policy)

eta_auto = -1 + 0.3 * (age_group == 1) - 0.2 * (age_group == 4) + \
           0.4 * (vehicle_age == 1) + 0.1 * gender
lam_auto = np.exp(eta_auto)
n_claims = np.random.poisson(lam_auto)

df_auto = pd.DataFrame({
    'n_claims': n_claims,
    'age_group': pd.Categorical(age_group),
    'vehicle_age': pd.Categorical(vehicle_age),
    'gender': pd.Categorical(gender)
})

model_auto = smf.glm('n_claims ~ age_group + vehicle_age + gender',
                      data=df_auto, family=Poisson()).fit()
print(f"\n车险索赔次数泊松回归:")
print(model_auto.summary().tables[1])


## 7.8.2 车险索赔金额模型
np.random.seed(303)
n_sev = 500
age_group_s = np.random.randint(1, 5, n_sev)
vehicle_age_s = np.random.randint(1, 4, n_sev)

eta_sev = 8 + 0.2 * (age_group_s == 1) - 0.1 * (age_group_s == 4) + \
          0.3 * (vehicle_age_s == 1)
mu_sev = np.exp(eta_sev)
phi_s = 0.5
claim_amount = np.random.gamma(1/phi_s, mu_sev * phi_s)

df_sev = pd.DataFrame({
    'claim_amount': claim_amount,
    'age_group': pd.Categorical(age_group_s),
    'vehicle_age': pd.Categorical(vehicle_age_s)
})

model_sev = smf.glm('claim_amount ~ age_group + vehicle_age',
                     data=df_sev, family=Gamma(link=log())).fit()
print(f"\n车险索赔金额伽马回归:")
print(model_sev.summary().tables[1])


##############################################################################
# 7.9 GAM（广义可加模型）
##############################################################################
np.random.seed(404)
n_gam = 500
x_gam = np.random.uniform(0, 10, n_gam)
eta_gam = 1 + 0.5 * np.sin(x_gam) + 0.3 * x_gam
lam_gam = np.exp(eta_gam)
y_gam = np.random.poisson(lam_gam)
df_gam = pd.DataFrame({'y': y_gam, 'x': x_gam})

from statsmodels.gam.api import GLMGam, BSplines
bs = BSplines(df_gam['x'], df=[6], degree=[3])
model_gam = GLMGam.from_formula('y ~ 1', data=df_gam,
                                  smoother=bs, family=Poisson()).fit()
print(f"\nGAM模型:")
print(model_gam.summary().tables[1])

# 绘图
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(x_gam, y_gam, alpha=0.3, label='数据')
x_sort = np.sort(x_gam)
idx = np.argsort(x_gam)
ax.plot(x_sort, model_gam.fittedvalues[idx], 'r-', label='GAM拟合')
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_title('GAM拟合'); ax.legend()
plt.tight_layout(); plt.show()

##############################################################################
# 第7章 广义线性模型
# 对应教材：section7.tex
# 内容：指数族分布、连接函数、参数估计、模型诊断、
#       泊松回归、负二项回归、伽马回归、GAM与GAMLSS
##############################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Poisson, Gamma, Binomial, NegativeBinomial
from statsmodels.genmod.families.links import log, logit, identity
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


##############################################################################
# 7.1 指数族分布
##############################################################################
theta = np.array([-2, -1, 0, 1, 2])
b_normal = theta**2 / 2
b_poisson = np.exp(theta)
b_gamma = -np.log(-theta[theta < 0])  # 只对θ<0有定义
print("指数族分布的b(θ)函数:")
print(f"  正态: {b_normal}")
print(f"  泊松: {b_poisson}")
print(f"  伽马(θ<0): {b_gamma}")


##############################################################################
# 7.2 泊松回归
##############################################################################

## 7.2.1 生成模拟数据
np.random.seed(123)
n = 500
x1 = np.random.normal(0, 1, n)
x2 = np.random.binomial(1, 0.5, n)
eta = 1 + 0.5 * x1 + 0.3 * x2
lam = np.exp(eta)
y = np.random.poisson(lam)
df_poisson = pd.DataFrame({'y': y, 'x1': x1, 'x2': x2})

## 7.2.2 泊松回归模型
X = sm.add_constant(df_poisson[['x1', 'x2']])
model_poisson = GLM(df_poisson['y'], X, family=Poisson()).fit()
print(f"\n泊松回归:")
print(model_poisson.summary().tables[1])


##############################################################################
# 7.3 负二项回归
##############################################################################

## 7.3.1 生成过散布数据
np.random.seed(456)
x1_nb = np.random.normal(0, 1, n)
x2_nb = np.random.binomial(1, 0.5, n)
eta_nb = 1 + 0.5 * x1_nb + 0.3 * x2_nb
mu_nb = np.exp(eta_nb)
k = 2
p_nb = k / (k + mu_nb)
y_nb = np.random.negative_binomial(k, p_nb)
df_nb = pd.DataFrame({'y': y_nb, 'x1': x1_nb, 'x2': x2_nb})

## 7.3.2 负二项回归
X_nb = sm.add_constant(df_nb[['x1', 'x2']])
model_nb = smf.glm('y ~ x1 + x2', data=df_nb,
                    family=sm.families.NegativeBinomial(alpha=1/k)).fit()
print(f"\n负二项回归:")
print(model_nb.summary().tables[1])

## 7.3.3 泊松回归（过散布数据）
model_poisson_over = GLM(df_nb['y'], X_nb, family=Poisson()).fit()
print(f"\n泊松回归（过散布数据）:")
print(model_poisson_over.summary().tables[1])
print(f"  Pearson chi2: {model_poisson_over.pearson_chi2:.2f}")


##############################################################################
# 7.4 伽马回归
##############################################################################
np.random.seed(789)
x1_g = np.random.normal(0, 1, n)
x2_g = np.random.binomial(1, 0.5, n)
eta_g = 3 + 0.5 * x1_g - 0.3 * x2_g
mu_g = np.exp(eta_g)
phi = 0.5
shape = 1 / phi
scale = mu_g * phi
y_g = np.random.gamma(shape, scale)
df_gamma = pd.DataFrame({'y': y_g, 'x1': x1_g, 'x2': x2_g})

X_g = sm.add_constant(df_gamma[['x1', 'x2']])
model_gamma = GLM(df_gamma['y'], X_g, family=Gamma(link=log())).fit()
print(f"\n伽马回归:")
print(model_gamma.summary().tables[1])


##############################################################################
# 7.5 Logistic回归
##############################################################################
np.random.seed(101)
x1_l = np.random.normal(0, 1, n)
x2_l = np.random.binomial(1, 0.5, n)
eta_l = -1 + 0.8 * x1_l + 0.5 * x2_l
p_l = 1 / (1 + np.exp(-eta_l))
y_l = np.random.binomial(1, p_l)
df_logit = pd.DataFrame({'y': y_l, 'x1': x1_l, 'x2': x2_l})

X_l = sm.add_constant(df_logit[['x1', 'x2']])
model_logit = GLM(df_logit['y'], X_l, family=Binomial(link=logit())).fit()
print(f"\nLogistic回归:")
print(model_logit.summary().tables[1])


##############################################################################
# 7.6 模型诊断
##############################################################################

## 7.6.1 残差分析
residuals_dev = model_poisson.resid_deviance
fitted = model_poisson.fittedvalues

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(fitted, residuals_dev, alpha=0.5)
axes[0].axhline(y=0, color='r', ls='--')
axes[0].set_xlabel('拟合值'); axes[0].set_ylabel('Deviance残差')
axes[0].set_title('Deviance残差 vs 拟合值')

from scipy import stats
stats.probplot(residuals_dev, dist='norm', plot=axes[1])
axes[1].set_title('Deviance残差Q-Q图')
plt.tight_layout(); plt.show()


## 7.6.2 偏差分析
print(f"\n偏差分析:")
print(f"  Null deviance: {model_poisson.null_deviance:.4f}")
print(f"  Residual deviance: {model_poisson.deviance:.4f}")
print(f"  减少的偏差: {model_poisson.null_deviance - model_poisson.deviance:.4f}")


##############################################################################
# 7.7 偏差与模型比较
##############################################################################
y_ex = np.array([10, 20, 15, 25, 30], dtype=float)
mu_ex = np.array([12, 18, 17, 23, 28])

D_poisson = 2 * np.sum(y_ex * np.log(y_ex / mu_ex) - (y_ex - mu_ex))
df_ex = len(y_ex) - 2
p_value = 1 - stats.chi2.cdf(D_poisson, df_ex)
print(f"\n偏差比较: D={D_poisson:.4f}, df={df_ex}, p值={p_value:.4f}")


##############################################################################
# 7.8 保险定价应用
##############################################################################

## 7.8.1 车险索赔次数模型
np.random.seed(202)
n_policy = 1000
age_group = np.random.randint(1, 5, n_policy)
vehicle_age = np.random.randint(1, 4, n_policy)
gender = np.random.binomial(1, 0.5, n_policy)

eta_auto = -1 + 0.3 * (age_group == 1) - 0.2 * (age_group == 4) + \
           0.4 * (vehicle_age == 1) + 0.1 * gender
lam_auto = np.exp(eta_auto)
n_claims = np.random.poisson(lam_auto)

df_auto = pd.DataFrame({
    'n_claims': n_claims,
    'age_group': pd.Categorical(age_group),
    'vehicle_age': pd.Categorical(vehicle_age),
    'gender': pd.Categorical(gender)
})

model_auto = smf.glm('n_claims ~ age_group + vehicle_age + gender',
                      data=df_auto, family=Poisson()).fit()
print(f"\n车险索赔次数泊松回归:")
print(model_auto.summary().tables[1])


## 7.8.2 车险索赔金额模型
np.random.seed(303)
n_sev = 500
age_group_s = np.random.randint(1, 5, n_sev)
vehicle_age_s = np.random.randint(1, 4, n_sev)

eta_sev = 8 + 0.2 * (age_group_s == 1) - 0.1 * (age_group_s == 4) + \
          0.3 * (vehicle_age_s == 1)
mu_sev = np.exp(eta_sev)
phi_s = 0.5
claim_amount = np.random.gamma(1/phi_s, mu_sev * phi_s)

df_sev = pd.DataFrame({
    'claim_amount': claim_amount,
    'age_group': pd.Categorical(age_group_s),
    'vehicle_age': pd.Categorical(vehicle_age_s)
})

model_sev = smf.glm('claim_amount ~ age_group + vehicle_age',
                     data=df_sev, family=Gamma(link=log())).fit()
print(f"\n车险索赔金额伽马回归:")
print(model_sev.summary().tables[1])


##############################################################################
# 7.9 GAM（广义可加模型）
##############################################################################
np.random.seed(404)
n_gam = 500
x_gam = np.random.uniform(0, 10, n_gam)
eta_gam = 1 + 0.5 * np.sin(x_gam) + 0.3 * x_gam
lam_gam = np.exp(eta_gam)
y_gam = np.random.poisson(lam_gam)
df_gam = pd.DataFrame({'y': y_gam, 'x': x_gam})

from statsmodels.gam.api import GLMGam, BSplines
bs = BSplines(df_gam['x'], df=[6], degree=[3])
model_gam = GLMGam.from_formula('y ~ 1', data=df_gam,
                                  smoother=bs, family=Poisson()).fit()
print(f"\nGAM模型:")
print(model_gam.summary().tables[1])

# 绘图
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(x_gam, y_gam, alpha=0.3, label='数据')
x_sort = np.sort(x_gam)
idx = np.argsort(x_gam)
ax.plot(x_sort, model_gam.fittedvalues[idx], 'r-', label='GAM拟合')
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_title('GAM拟合'); ax.legend()
plt.tight_layout(); plt.show()

##############################################################################
# 第7章 广义线性模型
# 对应教材：section7.tex
# 内容：指数族分布、连接函数、参数估计、模型诊断、
#       泊松回归、负二项回归、伽马回归、GAM与GAMLSS
##############################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Poisson, Gamma, Binomial, NegativeBinomial
from statsmodels.genmod.families.links import log, logit, identity
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


##############################################################################
# 7.1 指数族分布
##############################################################################
theta = np.array([-2, -1, 0, 1, 2])
b_normal = theta**2 / 2
b_poisson = np.exp(theta)
b_gamma = -np.log(-theta[theta < 0])  # 只对θ<0有定义
print("指数族分布的b(θ)函数:")
print(f"  正态: {b_normal}")
print(f"  泊松: {b_poisson}")
print(f"  伽马(θ<0): {b_gamma}")


##############################################################################
# 7.2 泊松回归
##############################################################################

## 7.2.1 生成模拟数据
np.random.seed(123)
n = 500
x1 = np.random.normal(0, 1, n)
x2 = np.random.binomial(1, 0.5, n)
eta = 1 + 0.5 * x1 + 0.3 * x2
lam = np.exp(eta)
y = np.random.poisson(lam)
df_poisson = pd.DataFrame({'y': y, 'x1': x1, 'x2': x2})

## 7.2.2 泊松回归模型
X = sm.add_constant(df_poisson[['x1', 'x2']])
model_poisson = GLM(df_poisson['y'], X, family=Poisson()).fit()
print(f"\n泊松回归:")
print(model_poisson.summary().tables[1])


##############################################################################
# 7.3 负二项回归
##############################################################################

## 7.3.1 生成过散布数据
np.random.seed(456)
x1_nb = np.random.normal(0, 1, n)
x2_nb = np.random.binomial(1, 0.5, n)
eta_nb = 1 + 0.5 * x1_nb + 0.3 * x2_nb
mu_nb = np.exp(eta_nb)
k = 2
p_nb = k / (k + mu_nb)
y_nb = np.random.negative_binomial(k, p_nb)
df_nb = pd.DataFrame({'y': y_nb, 'x1': x1_nb, 'x2': x2_nb})

## 7.3.2 负二项回归
X_nb = sm.add_constant(df_nb[['x1', 'x2']])
model_nb = smf.glm('y ~ x1 + x2', data=df_nb,
                    family=sm.families.NegativeBinomial(alpha=1/k)).fit()
print(f"\n负二项回归:")
print(model_nb.summary().tables[1])

## 7.3.3 泊松回归（过散布数据）
model_poisson_over = GLM(df_nb['y'], X_nb, family=Poisson()).fit()
print(f"\n泊松回归（过散布数据）:")
print(model_poisson_over.summary().tables[1])
print(f"  Pearson chi2: {model_poisson_over.pearson_chi2:.2f}")


##############################################################################
# 7.4 伽马回归
##############################################################################
np.random.seed(789)
x1_g = np.random.normal(0, 1, n)
x2_g = np.random.binomial(1, 0.5, n)
eta_g = 3 + 0.5 * x1_g - 0.3 * x2_g
mu_g = np.exp(eta_g)
phi = 0.5
shape = 1 / phi
scale = mu_g * phi
y_g = np.random.gamma(shape, scale)
df_gamma = pd.DataFrame({'y': y_g, 'x1': x1_g, 'x2': x2_g})

X_g = sm.add_constant(df_gamma[['x1', 'x2']])
model_gamma = GLM(df_gamma['y'], X_g, family=Gamma(link=log())).fit()
print(f"\n伽马回归:")
print(model_gamma.summary().tables[1])


##############################################################################
# 7.5 Logistic回归
##############################################################################
np.random.seed(101)
x1_l = np.random.normal(0, 1, n)
x2_l = np.random.binomial(1, 0.5, n)
eta_l = -1 + 0.8 * x1_l + 0.5 * x2_l
p_l = 1 / (1 + np.exp(-eta_l))
y_l = np.random.binomial(1, p_l)
df_logit = pd.DataFrame({'y': y_l, 'x1': x1_l, 'x2': x2_l})

X_l = sm.add_constant(df_logit[['x1', 'x2']])
model_logit = GLM(df_logit['y'], X_l, family=Binomial(link=logit())).fit()
print(f"\nLogistic回归:")
print(model_logit.summary().tables[1])


##############################################################################
# 7.6 模型诊断
##############################################################################

## 7.6.1 残差分析
residuals_dev = model_poisson.resid_deviance
fitted = model_poisson.fittedvalues

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(fitted, residuals_dev, alpha=0.5)
axes[0].axhline(y=0, color='r', ls='--')
axes[0].set_xlabel('拟合值'); axes[0].set_ylabel('Deviance残差')
axes[0].set_title('Deviance残差 vs 拟合值')

from scipy import stats
stats.probplot(residuals_dev, dist='norm', plot=axes[1])
axes[1].set_title('Deviance残差Q-Q图')
plt.tight_layout(); plt.show()


## 7.6.2 偏差分析
print(f"\n偏差分析:")
print(f"  Null deviance: {model_poisson.null_deviance:.4f}")
print(f"  Residual deviance: {model_poisson.deviance:.4f}")
print(f"  减少的偏差: {model_poisson.null_deviance - model_poisson.deviance:.4f}")


##############################################################################
# 7.7 偏差与模型比较
##############################################################################
y_ex = np.array([10, 20, 15, 25, 30], dtype=float)
mu_ex = np.array([12, 18, 17, 23, 28])

D_poisson = 2 * np.sum(y_ex * np.log(y_ex / mu_ex) - (y_ex - mu_ex))
df_ex = len(y_ex) - 2
p_value = 1 - stats.chi2.cdf(D_poisson, df_ex)
print(f"\n偏差比较: D={D_poisson:.4f}, df={df_ex}, p值={p_value:.4f}")


##############################################################################
# 7.8 保险定价应用
##############################################################################

## 7.8.1 车险索赔次数模型
np.random.seed(202)
n_policy = 1000
age_group = np.random.randint(1, 5, n_policy)
vehicle_age = np.random.randint(1, 4, n_policy)
gender = np.random.binomial(1, 0.5, n_policy)

eta_auto = -1 + 0.3 * (age_group == 1) - 0.2 * (age_group == 4) + \
           0.4 * (vehicle_age == 1) + 0.1 * gender
lam_auto = np.exp(eta_auto)
n_claims = np.random.poisson(lam_auto)

df_auto = pd.DataFrame({
    'n_claims': n_claims,
    'age_group': pd.Categorical(age_group),
    'vehicle_age': pd.Categorical(vehicle_age),
    'gender': pd.Categorical(gender)
})

model_auto = smf.glm('n_claims ~ age_group + vehicle_age + gender',
                      data=df_auto, family=Poisson()).fit()
print(f"\n车险索赔次数泊松回归:")
print(model_auto.summary().tables[1])


## 7.8.2 车险索赔金额模型
np.random.seed(303)
n_sev = 500
age_group_s = np.random.randint(1, 5, n_sev)
vehicle_age_s = np.random.randint(1, 4, n_sev)

eta_sev = 8 + 0.2 * (age_group_s == 1) - 0.1 * (age_group_s == 4) + \
          0.3 * (vehicle_age_s == 1)
mu_sev = np.exp(eta_sev)
phi_s = 0.5
claim_amount = np.random.gamma(1/phi_s, mu_sev * phi_s)

df_sev = pd.DataFrame({
    'claim_amount': claim_amount,
    'age_group': pd.Categorical(age_group_s),
    'vehicle_age': pd.Categorical(vehicle_age_s)
})

model_sev = smf.glm('claim_amount ~ age_group + vehicle_age',
                     data=df_sev, family=Gamma(link=log())).fit()
print(f"\n车险索赔金额伽马回归:")
print(model_sev.summary().tables[1])


##############################################################################
# 7.9 GAM（广义可加模型）
##############################################################################
np.random.seed(404)
n_gam = 500
x_gam = np.random.uniform(0, 10, n_gam)
eta_gam = 1 + 0.5 * np.sin(x_gam) + 0.3 * x_gam
lam_gam = np.exp(eta_gam)
y_gam = np.random.poisson(lam_gam)
df_gam = pd.DataFrame({'y': y_gam, 'x': x_gam})

from statsmodels.gam.api import GLMGam, BSplines
bs = BSplines(df_gam['x'], df=[6], degree=[3])
model_gam = GLMGam.from_formula('y ~ 1', data=df_gam,
                                  smoother=bs, family=Poisson()).fit()
print(f"\nGAM模型:")
print(model_gam.summary().tables[1])

# 绘图
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(x_gam, y_gam, alpha=0.3, label='数据')
x_sort = np.sort(x_gam)
idx = np.argsort(x_gam)
ax.plot(x_sort, model_gam.fittedvalues[idx], 'r-', label='GAM拟合')
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_title('GAM拟合'); ax.legend()
plt.tight_layout(); plt.show()

##############################################################################
# 第7章 广义线性模型
# 对应教材：section7.tex
# 内容：指数族分布、连接函数、参数估计、模型诊断、
#       泊松回归、负二项回归、伽马回归、GAM与GAMLSS
##############################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Poisson, Gamma, Binomial, NegativeBinomial
from statsmodels.genmod.families.links import log, logit, identity
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


##############################################################################
# 7.1 指数族分布
##############################################################################
theta = np.array([-2, -1, 0, 1, 2])
b_normal = theta**2 / 2
b_poisson = np.exp(theta)
b_gamma = -np.log(-theta[theta < 0])  # 只对θ<0有定义
print("指数族分布的b(θ)函数:")
print(f"  正态: {b_normal}")
print(f"  泊松: {b_poisson}")
print(f"  伽马(θ<0): {b_gamma}")


##############################################################################
# 7.2 泊松回归
##############################################################################

## 7.2.1 生成模拟数据
np.random.seed(123)
n = 500
x1 = np.random.normal(0, 1, n)
x2 = np.random.binomial(1, 0.5, n)
eta = 1 + 0.5 * x1 + 0.3 * x2
lam = np.exp(eta)
y = np.random.poisson(lam)
df_poisson = pd.DataFrame({'y': y, 'x1': x1, 'x2': x2})

## 7.2.2 泊松回归模型
X = sm.add_constant(df_poisson[['x1', 'x2']])
model_poisson = GLM(df_poisson['y'], X, family=Poisson()).fit()
print(f"\n泊松回归:")
print(model_poisson.summary().tables[1])


##############################################################################
# 7.3 负二项回归
##############################################################################

## 7.3.1 生成过散布数据
np.random.seed(456)
x1_nb = np.random.normal(0, 1, n)
x2_nb = np.random.binomial(1, 0.5, n)
eta_nb = 1 + 0.5 * x1_nb + 0.3 * x2_nb
mu_nb = np.exp(eta_nb)
k = 2
p_nb = k / (k + mu_nb)
y_nb = np.random.negative_binomial(k, p_nb)
df_nb = pd.DataFrame({'y': y_nb, 'x1': x1_nb, 'x2': x2_nb})

## 7.3.2 负二项回归
X_nb = sm.add_constant(df_nb[['x1', 'x2']])
model_nb = smf.glm('y ~ x1 + x2', data=df_nb,
                    family=sm.families.NegativeBinomial(alpha=1/k)).fit()
print(f"\n负二项回归:")
print(model_nb.summary().tables[1])

## 7.3.3 泊松回归（过散布数据）
model_poisson_over = GLM(df_nb['y'], X_nb, family=Poisson()).fit()
print(f"\n泊松回归（过散布数据）:")
print(model_poisson_over.summary().tables[1])
print(f"  Pearson chi2: {model_poisson_over.pearson_chi2:.2f}")


##############################################################################
# 7.4 伽马回归
##############################################################################
np.random.seed(789)
x1_g = np.random.normal(0, 1, n)
x2_g = np.random.binomial(1, 0.5, n)
eta_g = 3 + 0.5 * x1_g - 0.3 * x2_g
mu_g = np.exp(eta_g)
phi = 0.5
shape = 1 / phi
scale = mu_g * phi
y_g = np.random.gamma(shape, scale)
df_gamma = pd.DataFrame({'y': y_g, 'x1': x1_g, 'x2': x2_g})

X_g = sm.add_constant(df_gamma[['x1', 'x2']])
model_gamma = GLM(df_gamma['y'], X_g, family=Gamma(link=log())).fit()
print(f"\n伽马回归:")
print(model_gamma.summary().tables[1])


##############################################################################
# 7.5 Logistic回归
##############################################################################
np.random.seed(101)
x1_l = np.random.normal(0, 1, n)
x2_l = np.random.binomial(1, 0.5, n)
eta_l = -1 + 0.8 * x1_l + 0.5 * x2_l
p_l = 1 / (1 + np.exp(-eta_l))
y_l = np.random.binomial(1, p_l)
df_logit = pd.DataFrame({'y': y_l, 'x1': x1_l, 'x2': x2_l})

X_l = sm.add_constant(df_logit[['x1', 'x2']])
model_logit = GLM(df_logit['y'], X_l, family=Binomial(link=logit())).fit()
print(f"\nLogistic回归:")
print(model_logit.summary().tables[1])


##############################################################################
# 7.6 模型诊断
##############################################################################

## 7.6.1 残差分析
residuals_dev = model_poisson.resid_deviance
fitted = model_poisson.fittedvalues

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(fitted, residuals_dev, alpha=0.5)
axes[0].axhline(y=0, color='r', ls='--')
axes[0].set_xlabel('拟合值'); axes[0].set_ylabel('Deviance残差')
axes[0].set_title('Deviance残差 vs 拟合值')

from scipy import stats
stats.probplot(residuals_dev, dist='norm', plot=axes[1])
axes[1].set_title('Deviance残差Q-Q图')
plt.tight_layout(); plt.show()


## 7.6.2 偏差分析
print(f"\n偏差分析:")
print(f"  Null deviance: {model_poisson.null_deviance:.4f}")
print(f"  Residual deviance: {model_poisson.deviance:.4f}")
print(f"  减少的偏差: {model_poisson.null_deviance - model_poisson.deviance:.4f}")


##############################################################################
# 7.7 偏差与模型比较
##############################################################################
y_ex = np.array([10, 20, 15, 25, 30], dtype=float)
mu_ex = np.array([12, 18, 17, 23, 28])

D_poisson = 2 * np.sum(y_ex * np.log(y_ex / mu_ex) - (y_ex - mu_ex))
df_ex = len(y_ex) - 2
p_value = 1 - stats.chi2.cdf(D_poisson, df_ex)
print(f"\n偏差比较: D={D_poisson:.4f}, df={df_ex}, p值={p_value:.4f}")


##############################################################################
# 7.8 保险定价应用
##############################################################################

## 7.8.1 车险索赔次数模型
np.random.seed(202)
n_policy = 1000
age_group = np.random.randint(1, 5, n_policy)
vehicle_age = np.random.randint(1, 4, n_policy)
gender = np.random.binomial(1, 0.5, n_policy)

eta_auto = -1 + 0.3 * (age_group == 1) - 0.2 * (age_group == 4) + \
           0.4 * (vehicle_age == 1) + 0.1 * gender
lam_auto = np.exp(eta_auto)
n_claims = np.random.poisson(lam_auto)

df_auto = pd.DataFrame({
    'n_claims': n_claims,
    'age_group': pd.Categorical(age_group),
    'vehicle_age': pd.Categorical(vehicle_age),
    'gender': pd.Categorical(gender)
})

model_auto = smf.glm('n_claims ~ age_group + vehicle_age + gender',
                      data=df_auto, family=Poisson()).fit()
print(f"\n车险索赔次数泊松回归:")
print(model_auto.summary().tables[1])


## 7.8.2 车险索赔金额模型
np.random.seed(303)
n_sev = 500
age_group_s = np.random.randint(1, 5, n_sev)
vehicle_age_s = np.random.randint(1, 4, n_sev)

eta_sev = 8 + 0.2 * (age_group_s == 1) - 0.1 * (age_group_s == 4) + \
          0.3 * (vehicle_age_s == 1)
mu_sev = np.exp(eta_sev)
phi_s = 0.5
claim_amount = np.random.gamma(1/phi_s, mu_sev * phi_s)

df_sev = pd.DataFrame({
    'claim_amount': claim_amount,
    'age_group': pd.Categorical(age_group_s),
    'vehicle_age': pd.Categorical(vehicle_age_s)
})

model_sev = smf.glm('claim_amount ~ age_group + vehicle_age',
                     data=df_sev, family=Gamma(link=log())).fit()
print(f"\n车险索赔金额伽马回归:")
print(model_sev.summary().tables[1])


##############################################################################
# 7.9 GAM（广义可加模型）
##############################################################################
np.random.seed(404)
n_gam = 500
x_gam = np.random.uniform(0, 10, n_gam)
eta_gam = 1 + 0.5 * np.sin(x_gam) + 0.3 * x_gam
lam_gam = np.exp(eta_gam)
y_gam = np.random.poisson(lam_gam)
df_gam = pd.DataFrame({'y': y_gam, 'x': x_gam})

from statsmodels.gam.api import GLMGam, BSplines
bs = BSplines(df_gam['x'], df=[6], degree=[3])
model_gam = GLMGam.from_formula('y ~ 1', data=df_gam,
                                  smoother=bs, family=Poisson()).fit()
print(f"\nGAM模型:")
print(model_gam.summary().tables[1])

# 绘图
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(x_gam, y_gam, alpha=0.3, label='数据')
x_sort = np.sort(x_gam)
idx = np.argsort(x_gam)
ax.plot(x_sort, model_gam.fittedvalues[idx], 'r-', label='GAM拟合')
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_title('GAM拟合'); ax.legend()
plt.tight_layout(); plt.show()

##############################################################################
# 第7章 广义线性模型
# 对应教材：section7.tex
# 内容：指数族分布、连接函数、参数估计、模型诊断、
#       泊松回归、负二项回归、伽马回归、GAM与GAMLSS
##############################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Poisson, Gamma, Binomial, NegativeBinomial
from statsmodels.genmod.families.links import log, logit, identity
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


##############################################################################
# 7.1 指数族分布
##############################################################################
theta = np.array([-2, -1, 0, 1, 2])
b_normal = theta**2 / 2
b_poisson = np.exp(theta)
b_gamma = -np.log(-theta[theta < 0])  # 只对θ<0有定义
print("指数族分布的b(θ)函数:")
print(f"  正态: {b_normal}")
print(f"  泊松: {b_poisson}")
print(f"  伽马(θ<0): {b_gamma}")


##############################################################################
# 7.2 泊松回归
##############################################################################

## 7.2.1 生成模拟数据
np.random.seed(123)
n = 500
x1 = np.random.normal(0, 1, n)
x2 = np.random.binomial(1, 0.5, n)
eta = 1 + 0.5 * x1 + 0.3 * x2
lam = np.exp(eta)
y = np.random.poisson(lam)
df_poisson = pd.DataFrame({'y': y, 'x1': x1, 'x2': x2})

## 7.2.2 泊松回归模型
X = sm.add_constant(df_poisson[['x1', 'x2']])
model_poisson = GLM(df_poisson['y'], X, family=Poisson()).fit()
print(f"\n泊松回归:")
print(model_poisson.summary().tables[1])


##############################################################################
# 7.3 负二项回归
##############################################################################

## 7.3.1 生成过散布数据
np.random.seed(456)
x1_nb = np.random.normal(0, 1, n)
x2_nb = np.random.binomial(1, 0.5, n)
eta_nb = 1 + 0.5 * x1_nb + 0.3 * x2_nb
mu_nb = np.exp(eta_nb)
k = 2
p_nb = k / (k + mu_nb)
y_nb = np.random.negative_binomial(k, p_nb)
df_nb = pd.DataFrame({'y': y_nb, 'x1': x1_nb, 'x2': x2_nb})

## 7.3.2 负二项回归
X_nb = sm.add_constant(df_nb[['x1', 'x2']])
model_nb = smf.glm('y ~ x1 + x2', data=df_nb,
                    family=sm.families.NegativeBinomial(alpha=1/k)).fit()
print(f"\n负二项回归:")
print(model_nb.summary().tables[1])

## 7.3.3 泊松回归（过散布数据）
model_poisson_over = GLM(df_nb['y'], X_nb, family=Poisson()).fit()
print(f"\n泊松回归（过散布数据）:")
print(model_poisson_over.summary().tables[1])
print(f"  Pearson chi2: {model_poisson_over.pearson_chi2:.2f}")


##############################################################################
# 7.4 伽马回归
##############################################################################
np.random.seed(789)
x1_g = np.random.normal(0, 1, n)
x2_g = np.random.binomial(1, 0.5, n)
eta_g = 3 + 0.5 * x1_g - 0.3 * x2_g
mu_g = np.exp(eta_g)
phi = 0.5
shape = 1 / phi
scale = mu_g * phi
y_g = np.random.gamma(shape, scale)
df_gamma = pd.DataFrame({'y': y_g, 'x1': x1_g, 'x2': x2_g})

X_g = sm.add_constant(df_gamma[['x1', 'x2']])
model_gamma = GLM(df_gamma['y'], X_g, family=Gamma(link=log())).fit()
print(f"\n伽马回归:")
print(model_gamma.summary().tables[1])


##############################################################################
# 7.5 Logistic回归
##############################################################################
np.random.seed(101)
x1_l = np.random.normal(0, 1, n)
x2_l = np.random.binomial(1, 0.5, n)
eta_l = -1 + 0.8 * x1_l + 0.5 * x2_l
p_l = 1 / (1 + np.exp(-eta_l))
y_l = np.random.binomial(1, p_l)
df_logit = pd.DataFrame({'y': y_l, 'x1': x1_l, 'x2': x2_l})

X_l = sm.add_constant(df_logit[['x1', 'x2']])
model_logit = GLM(df_logit['y'], X_l, family=Binomial(link=logit())).fit()
print(f"\nLogistic回归:")
print(model_logit.summary().tables[1])


##############################################################################
# 7.6 模型诊断
##############################################################################

## 7.6.1 残差分析
residuals_dev = model_poisson.resid_deviance
fitted = model_poisson.fittedvalues

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(fitted, residuals_dev, alpha=0.5)
axes[0].axhline(y=0, color='r', ls='--')
axes[0].set_xlabel('拟合值'); axes[0].set_ylabel('Deviance残差')
axes[0].set_title('Deviance残差 vs 拟合值')

from scipy import stats
stats.probplot(residuals_dev, dist='norm', plot=axes[1])
axes[1].set_title('Deviance残差Q-Q图')
plt.tight_layout(); plt.show()


## 7.6.2 偏差分析
print(f"\n偏差分析:")
print(f"  Null deviance: {model_poisson.null_deviance:.4f}")
print(f"  Residual deviance: {model_poisson.deviance:.4f}")
print(f"  减少的偏差: {model_poisson.null_deviance - model_poisson.deviance:.4f}")


##############################################################################
# 7.7 偏差与模型比较
##############################################################################
y_ex = np.array([10, 20, 15, 25, 30], dtype=float)
mu_ex = np.array([12, 18, 17, 23, 28])

D_poisson = 2 * np.sum(y_ex * np.log(y_ex / mu_ex) - (y_ex - mu_ex))
df_ex = len(y_ex) - 2
p_value = 1 - stats.chi2.cdf(D_poisson, df_ex)
print(f"\n偏差比较: D={D_poisson:.4f}, df={df_ex}, p值={p_value:.4f}")


##############################################################################
# 7.8 保险定价应用
##############################################################################

## 7.8.1 车险索赔次数模型
np.random.seed(202)
n_policy = 1000
age_group = np.random.randint(1, 5, n_policy)
vehicle_age = np.random.randint(1, 4, n_policy)
gender = np.random.binomial(1, 0.5, n_policy)

eta_auto = -1 + 0.3 * (age_group == 1) - 0.2 * (age_group == 4) + \
           0.4 * (vehicle_age == 1) + 0.1 * gender
lam_auto = np.exp(eta_auto)
n_claims = np.random.poisson(lam_auto)

df_auto = pd.DataFrame({
    'n_claims': n_claims,
    'age_group': pd.Categorical(age_group),
    'vehicle_age': pd.Categorical(vehicle_age),
    'gender': pd.Categorical(gender)
})

model_auto = smf.glm('n_claims ~ age_group + vehicle_age + gender',
                      data=df_auto, family=Poisson()).fit()
print(f"\n车险索赔次数泊松回归:")
print(model_auto.summary().tables[1])


## 7.8.2 车险索赔金额模型
np.random.seed(303)
n_sev = 500
age_group_s = np.random.randint(1, 5, n_sev)
vehicle_age_s = np.random.randint(1, 4, n_sev)

eta_sev = 8 + 0.2 * (age_group_s == 1) - 0.1 * (age_group_s == 4) + \
          0.3 * (vehicle_age_s == 1)
mu_sev = np.exp(eta_sev)
phi_s = 0.5
claim_amount = np.random.gamma(1/phi_s, mu_sev * phi_s)

df_sev = pd.DataFrame({
    'claim_amount': claim_amount,
    'age_group': pd.Categorical(age_group_s),
    'vehicle_age': pd.Categorical(vehicle_age_s)
})

model_sev = smf.glm('claim_amount ~ age_group + vehicle_age',
                     data=df_sev, family=Gamma(link=log())).fit()
print(f"\n车险索赔金额伽马回归:")
print(model_sev.summary().tables[1])


##############################################################################
# 7.9 GAM（广义可加模型）
##############################################################################
np.random.seed(404)
n_gam = 500
x_gam = np.random.uniform(0, 10, n_gam)
eta_gam = 1 + 0.5 * np.sin(x_gam) + 0.3 * x_gam
lam_gam = np.exp(eta_gam)
y_gam = np.random.poisson(lam_gam)
df_gam = pd.DataFrame({'y': y_gam, 'x': x_gam})

from statsmodels.gam.api import GLMGam, BSplines
bs = BSplines(df_gam['x'], df=[6], degree=[3])
model_gam = GLMGam.from_formula('y ~ 1', data=df_gam,
                                  smoother=bs, family=Poisson()).fit()
print(f"\nGAM模型:")
print(model_gam.summary().tables[1])

# 绘图
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(x_gam, y_gam, alpha=0.3, label='数据')
x_sort = np.sort(x_gam)
idx = np.argsort(x_gam)
ax.plot(x_sort, model_gam.fittedvalues[idx], 'r-', label='GAM拟合')
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_title('GAM拟合'); ax.legend()
plt.tight_layout(); plt.show()

##############################################################################
# 第7章 广义线性模型
# 对应教材：section7.tex
# 内容：指数族分布、连接函数、参数估计、模型诊断、
#       泊松回归、负二项回归、伽马回归、GAM与GAMLSS
##############################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Poisson, Gamma, Binomial, NegativeBinomial
from statsmodels.genmod.families.links import log, logit, identity
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


##############################################################################
# 7.1 指数族分布
##############################################################################
theta = np.array([-2, -1, 0, 1, 2])
b_normal = theta**2 / 2
b_poisson = np.exp(theta)
b_gamma = -np.log(-theta[theta < 0])  # 只对θ<0有定义
print("指数族分布的b(θ)函数:")
print(f"  正态: {b_normal}")
print(f"  泊松: {b_poisson}")
print(f"  伽马(θ<0): {b_gamma}")


##############################################################################
# 7.2 泊松回归
##############################################################################

## 7.2.1 生成模拟数据
np.random.seed(123)
n = 500
x1 = np.random.normal(0, 1, n)
x2 = np.random.binomial(1, 0.5, n)
eta = 1 + 0.5 * x1 + 0.3 * x2
lam = np.exp(eta)
y = np.random.poisson(lam)
df_poisson = pd.DataFrame({'y': y, 'x1': x1, 'x2': x2})

## 7.2.2 泊松回归模型
X = sm.add_constant(df_poisson[['x1', 'x2']])
model_poisson = GLM(df_poisson['y'], X, family=Poisson()).fit()
print(f"\n泊松回归:")
print(model_poisson.summary().tables[1])


##############################################################################
# 7.3 负二项回归
##############################################################################

## 7.3.1 生成过散布数据
np.random.seed(456)
x1_nb = np.random.normal(0, 1, n)
x2_nb = np.random.binomial(1, 0.5, n)
eta_nb = 1 + 0.5 * x1_nb + 0.3 * x2_nb
mu_nb = np.exp(eta_nb)
k = 2
p_nb = k / (k + mu_nb)
y_nb = np.random.negative_binomial(k, p_nb)
df_nb = pd.DataFrame({'y': y_nb, 'x1': x1_nb, 'x2': x2_nb})

## 7.3.2 负二项回归
X_nb = sm.add_constant(df_nb[['x1', 'x2']])
model_nb = smf.glm('y ~ x1 + x2', data=df_nb,
                    family=sm.families.NegativeBinomial(alpha=1/k)).fit()
print(f"\n负二项回归:")
print(model_nb.summary().tables[1])

## 7.3.3 泊松回归（过散布数据）
model_poisson_over = GLM(df_nb['y'], X_nb, family=Poisson()).fit()
print(f"\n泊松回归（过散布数据）:")
print(model_poisson_over.summary().tables[1])
print(f"  Pearson chi2: {model_poisson_over.pearson_chi2:.2f}")


##############################################################################
# 7.4 伽马回归
##############################################################################
np.random.seed(789)
x1_g = np.random.normal(0, 1, n)
x2_g = np.random.binomial(1, 0.5, n)
eta_g = 3 + 0.5 * x1_g - 0.3 * x2_g
mu_g = np.exp(eta_g)
phi = 0.5
shape = 1 / phi
scale = mu_g * phi
y_g = np.random.gamma(shape, scale)
df_gamma = pd.DataFrame({'y': y_g, 'x1': x1_g, 'x2': x2_g})

X_g = sm.add_constant(df_gamma[['x1', 'x2']])
model_gamma = GLM(df_gamma['y'], X_g, family=Gamma(link=log())).fit()
print(f"\n伽马回归:")
print(model_gamma.summary().tables[1])


##############################################################################
# 7.5 Logistic回归
##############################################################################
np.random.seed(101)
x1_l = np.random.normal(0, 1, n)
x2_l = np.random.binomial(1, 0.5, n)
eta_l = -1 + 0.8 * x1_l + 0.5 * x2_l
p_l = 1 / (1 + np.exp(-eta_l))
y_l = np.random.binomial(1, p_l)
df_logit = pd.DataFrame({'y': y_l, 'x1': x1_l, 'x2': x2_l})

X_l = sm.add_constant(df_logit[['x1', 'x2']])
model_logit = GLM(df_logit['y'], X_l, family=Binomial(link=logit())).fit()
print(f"\nLogistic回归:")
print(model_logit.summary().tables[1])


##############################################################################
# 7.6 模型诊断
##############################################################################

## 7.6.1 残差分析
residuals_dev = model_poisson.resid_deviance
fitted = model_poisson.fittedvalues

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(fitted, residuals_dev, alpha=0.5)
axes[0].axhline(y=0, color='r', ls='--')
axes[0].set_xlabel('拟合值'); axes[0].set_ylabel('Deviance残差')
axes[0].set_title('Deviance残差 vs 拟合值')

from scipy import stats
stats.probplot(residuals_dev, dist='norm', plot=axes[1])
axes[1].set_title('Deviance残差Q-Q图')
plt.tight_layout(); plt.show()


## 7.6.2 偏差分析
print(f"\n偏差分析:")
print(f"  Null deviance: {model_poisson.null_deviance:.4f}")
print(f"  Residual deviance: {model_poisson.deviance:.4f}")
print(f"  减少的偏差: {model_poisson.null_deviance - model_poisson.deviance:.4f}")


##############################################################################
# 7.7 偏差与模型比较
##############################################################################
y_ex = np.array([10, 20, 15, 25, 30], dtype=float)
mu_ex = np.array([12, 18, 17, 23, 28])

D_poisson = 2 * np.sum(y_ex * np.log(y_ex / mu_ex) - (y_ex - mu_ex))
df_ex = len(y_ex) - 2
p_value = 1 - stats.chi2.cdf(D_poisson, df_ex)
print(f"\n偏差比较: D={D_poisson:.4f}, df={df_ex}, p值={p_value:.4f}")


##############################################################################
# 7.8 保险定价应用
##############################################################################

## 7.8.1 车险索赔次数模型
np.random.seed(202)
n_policy = 1000
age_group = np.random.randint(1, 5, n_policy)
vehicle_age = np.random.randint(1, 4, n_policy)
gender = np.random.binomial(1, 0.5, n_policy)

eta_auto = -1 + 0.3 * (age_group == 1) - 0.2 * (age_group == 4) + \
           0.4 * (vehicle_age == 1) + 0.1 * gender
lam_auto = np.exp(eta_auto)
n_claims = np.random.poisson(lam_auto)

df_auto = pd.DataFrame({
    'n_claims': n_claims,
    'age_group': pd.Categorical(age_group),
    'vehicle_age': pd.Categorical(vehicle_age),
    'gender': pd.Categorical(gender)
})

model_auto = smf.glm('n_claims ~ age_group + vehicle_age + gender',
                      data=df_auto, family=Poisson()).fit()
print(f"\n车险索赔次数泊松回归:")
print(model_auto.summary().tables[1])


## 7.8.2 车险索赔金额模型
np.random.seed(303)
n_sev = 500
age_group_s = np.random.randint(1, 5, n_sev)
vehicle_age_s = np.random.randint(1, 4, n_sev)

eta_sev = 8 + 0.2 * (age_group_s == 1) - 0.1 * (age_group_s == 4) + \
          0.3 * (vehicle_age_s == 1)
mu_sev = np.exp(eta_sev)
phi_s = 0.5
claim_amount = np.random.gamma(1/phi_s, mu_sev * phi_s)

df_sev = pd.DataFrame({
    'claim_amount': claim_amount,
    'age_group': pd.Categorical(age_group_s),
    'vehicle_age': pd.Categorical(vehicle_age_s)
})

model_sev = smf.glm('claim_amount ~ age_group + vehicle_age',
                     data=df_sev, family=Gamma(link=log())).fit()
print(f"\n车险索赔金额伽马回归:")
print(model_sev.summary().tables[1])


##############################################################################
# 7.9 GAM（广义可加模型）
##############################################################################
np.random.seed(404)
n_gam = 500
x_gam = np.random.uniform(0, 10, n_gam)
eta_gam = 1 + 0.5 * np.sin(x_gam) + 0.3 * x_gam
lam_gam = np.exp(eta_gam)
y_gam = np.random.poisson(lam_gam)
df_gam = pd.DataFrame({'y': y_gam, 'x': x_gam})

from statsmodels.gam.api import GLMGam, BSplines
bs = BSplines(df_gam['x'], df=[6], degree=[3])
model_gam = GLMGam.from_formula('y ~ 1', data=df_gam,
                                  smoother=bs, family=Poisson()).fit()
print(f"\nGAM模型:")
print(model_gam.summary().tables[1])

# 绘图
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(x_gam, y_gam, alpha=0.3, label='数据')
x_sort = np.sort(x_gam)
idx = np.argsort(x_gam)
ax.plot(x_sort, model_gam.fittedvalues[idx], 'r-', label='GAM拟合')
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_title('GAM拟合'); ax.legend()
plt.tight_layout(); plt.show()

##############################################################################
# 第7章 广义线性模型
# 对应教材：section7.tex
# 内容：指数族分布、连接函数、参数估计、模型诊断、
#       泊松回归、负二项回归、伽马回归、GAM与GAMLSS
##############################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import Poisson, Gamma, Binomial, NegativeBinomial
from statsmodels.genmod.families.links import log, logit, identity
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


##############################################################################
# 7.1 指数族分布
##############################################################################
theta = np.array([-2, -1, 0, 1, 2])
b_normal = theta**2 / 2
b_poisson = np.exp(theta)
b_gamma = -np.log(-theta[theta < 0])  # 只对θ<0有定义
print("指数族分布的b(θ)函数:")
print(f"  正态: {b_normal}")
print(f"  泊松: {b_poisson}")
print(f"  伽马(θ<0): {b_gamma}")


##############################################################################
# 7.2 泊松回归
##############################################################################

## 7.2.1 生成模拟数据
np.random.seed(123)
n = 500
x1 = np.random.normal(0, 1, n)
x2 = np.random.binomial(1, 0.5, n)
eta = 1 + 0.5 * x1 + 0.3 * x2
lam = np.exp(eta)
y = np.random.poisson(lam)
df_poisson = pd.DataFrame({'y': y, 'x1': x1, 'x2': x2})

## 7.2.2 泊松回归模型
X = sm.add_constant(df_poisson[['x1', 'x2']])
model_poisson = GLM(df_poisson['y'], X, family=Poisson()).fit()
print(f"\n泊松回归:")
print(model_poisson.summary().tables[1])


##############################################################################
# 7.3 负二项回归
##############################################################################

## 7.3.1 生成过散布数据
np.random.seed(456)
x1_nb = np.random.normal(0, 1, n)
x2_nb = np.random.binomial(1, 0.5, n)
eta_nb = 1 + 0.5 * x1_nb + 0.3 * x2_nb
mu_nb = np.exp(eta_nb)
k = 2
p_nb = k / (k + mu_nb)
y_nb = np.random.negative_binomial(k, p_nb)
df_nb = pd.DataFrame({'y': y_nb, 'x1': x1_nb, 'x2': x2_nb})

## 7.3.2 负二项回归
X_nb = sm.add_constant(df_nb[['x1', 'x2']])
model_nb = smf.glm('y ~ x1 + x2', data=df_nb,
                    family=sm.families.NegativeBinomial(alpha=1/k)).fit()
print(f"\n负二项回归:")
print(model_nb.summary().tables[1])

## 7.3.3 泊松回归（过散布数据）
model_poisson_over = GLM(df_nb['y'], X_nb, family=Poisson()).fit()
print(f"\n泊松回归（过散布数据）:")
print(model_poisson_over.summary().tables[1])
print(f"  Pearson chi2: {model_poisson_over.pearson_chi2:.2f}")


##############################################################################
# 7.4 伽马回归
##############################################################################
np.random.seed(789)
x1_g = np.random.normal(0, 1, n)
x2_g = np.random.binomial(1, 0.5, n)
eta_g = 3 + 0.5 * x1_g - 0.3 * x2_g
mu_g = np.exp(eta_g)
phi = 0.5
shape = 1 / phi
scale = mu_g * phi
y_g = np.random.gamma(shape, scale)
df_gamma = pd.DataFrame({'y': y_g, 'x1': x1_g, 'x2': x2_g})

X_g = sm.add_constant(df_gamma[['x1', 'x2']])
model_gamma = GLM(df_gamma['y'], X_g, family=Gamma(link=log())).fit()
print(f"\n伽马回归:")
print(model_gamma.summary().tables[1])


##############################################################################
# 7.5 Logistic回归
##############################################################################
np.random.seed(101)
x1_l = np.random.normal(0, 1, n)
x2_l = np.random.binomial(1, 0.5, n)
eta_l = -1 + 0.8 * x1_l + 0.5 * x2_l
p_l = 1 / (1 + np.exp(-eta_l))
y_l = np.random.binomial(1, p_l)
df_logit = pd.DataFrame({'y': y_l, 'x1': x1_l, 'x2': x2_l})

X_l = sm.add_constant(df_logit[['x1', 'x2']])
model_logit = GLM(df_logit['y'], X_l, family=Binomial(link=logit())).fit()
print(f"\nLogistic回归:")
print(model_logit.summary().tables[1])


##############################################################################
# 7.6 模型诊断
##############################################################################

## 7.6.1 残差分析
residuals_dev = model_poisson.resid_deviance
fitted = model_poisson.fittedvalues

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(fitted, residuals_dev, alpha=0.5)
axes[0].axhline(y=0, color='r', ls='--')
axes[0].set_xlabel('拟合值'); axes[0].set_ylabel('Deviance残差')
axes[0].set_title('Deviance残差 vs 拟合值')

from scipy import stats
stats.probplot(residuals_dev, dist='norm', plot=axes[1])
axes[1].set_title('Deviance残差Q-Q图')
plt.tight_layout(); plt.show()


## 7.6.2 偏差分析
print(f"\n偏差分析:")
print(f"  Null deviance: {model_poisson.null_deviance:.4f}")
print(f"  Residual deviance: {model_poisson.deviance:.4f}")
print(f"  减少的偏差: {model_poisson.null_deviance - model_poisson.deviance:.4f}")


##############################################################################
# 7.7 偏差与模型比较
##############################################################################
y_ex = np.array([10, 20, 15, 25, 30], dtype=float)
mu_ex = np.array([12, 18, 17, 23, 28])

D_poisson = 2 * np.sum(y_ex * np.log(y_ex / mu_ex) - (y_ex - mu_ex))
df_ex = len(y_ex) - 2
p_value = 1 - stats.chi2.cdf(D_poisson, df_ex)
print(f"\n偏差比较: D={D_poisson:.4f}, df={df_ex}, p值={p_value:.4f}")


##############################################################################
# 7.8 保险定价应用
##############################################################################

## 7.8.1 车险索赔次数模型
np.random.seed(202)
n_policy = 1000
age_group = np.random.randint(1, 5, n_policy)
vehicle_age = np.random.randint(1, 4, n_policy)
gender = np.random.binomial(1, 0.5, n_policy)

eta_auto = -1 + 0.3 * (age_group == 1) - 0.2 * (age_group == 4) + \
           0.4 * (vehicle_age == 1) + 0.1 * gender
lam_auto = np.exp(eta_auto)
n_claims = np.random.poisson(lam_auto)

df_auto = pd.DataFrame({
    'n_claims': n_claims,
    'age_group': pd.Categorical(age_group),
    'vehicle_age': pd.Categorical(vehicle_age),
    'gender': pd.Categorical(gender)
})

model_auto = smf.glm('n_claims ~ age_group + vehicle_age + gender',
                      data=df_auto, family=Poisson()).fit()
print(f"\n车险索赔次数泊松回归:")
print(model_auto.summary().tables[1])


## 7.8.2 车险索赔金额模型
np.random.seed(303)
n_sev = 500
age_group_s = np.random.randint(1, 5, n_sev)
vehicle_age_s = np.random.randint(1, 4, n_sev)

eta_sev = 8 + 0.2 * (age_group_s == 1) - 0.1 * (age_group_s == 4) + \
          0.3 * (vehicle_age_s == 1)
mu_sev = np.exp(eta_sev)
phi_s = 0.5
claim_amount = np.random.gamma(1/phi_s, mu_sev * phi_s)

df_sev = pd.DataFrame({
    'claim_amount': claim_amount,
    'age_group': pd.Categorical(age_group_s),
    'vehicle_age': pd.Categorical(vehicle_age_s)
})

model_sev = smf.glm('claim_amount ~ age_group + vehicle_age',
                     data=df_sev, family=Gamma(link=log())).fit()
print(f"\n车险索赔金额伽马回归:")
print(model_sev.summary().tables[1])


##############################################################################
# 7.9 GAM（广义可加模型）
##############################################################################
np.random.seed(404)
n_gam = 500
x_gam = np.random.uniform(0, 10, n_gam)
eta_gam = 1 + 0.5 * np.sin(x_gam) + 0.3 * x_gam
lam_gam = np.exp(eta_gam)
y_gam = np.random.poisson(lam_gam)
df_gam = pd.DataFrame({'y': y_gam, 'x': x_gam})

from statsmodels.gam.api import GLMGam, BSplines
bs = BSplines(df_gam['x'], df=[6], degree=[3])
model_gam = GLMGam.from_formula('y ~ 1', data=df_gam,
                                  smoother=bs, family=Poisson()).fit()
print(f"\nGAM模型:")
print(model_gam.summary().tables[1])

# 绘图
fig, ax = plt.subplots(figsize=(10, 5))
ax.scatter(x_gam, y_gam, alpha=0.3, label='数据')
x_sort = np.sort(x_gam)
idx = np.argsort(x_gam)
ax.plot(x_sort, model_gam.fittedvalues[idx], 'r-', label='GAM拟合')
ax.set_xlabel('x'); ax.set_ylabel('y')
ax.set_title('GAM拟合'); ax.legend()
plt.tight_layout(); plt.show()