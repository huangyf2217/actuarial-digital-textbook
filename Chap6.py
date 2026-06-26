# Chap6 Python代码
# 自动从chap6.html同步生成

##############################################################################
# 第6章 时间序列分析
# 对应教材：section6.tex
# 内容：平稳性检验、AR/MA/ARMA/ARIMA模型、模型识别、参数估计、
#       诊断检验、外推预测、多元时间序列、非线性时间序列
##############################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.api import VAR
from arch import arch_model
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


##############################################################################
# 6.1 时间序列数据生成与可视化
##############################################################################
np.random.seed(123)
n = 200
e = np.random.normal(0, 1, n)

# AR(1)过程
y_ar1 = np.zeros(n)
for t in range(1, n):
    y_ar1[t] = 0.7 * y_ar1[t-1] + e[t]

# 白噪声
y_wn = e

# 随机游走
y_rw = np.cumsum(e)

fig, axes = plt.subplots(3, 1, figsize=(12, 10))
axes[0].plot(y_ar1); axes[0].set_title('AR(1)过程')
axes[1].plot(y_wn); axes[1].set_title('白噪声过程')
axes[2].plot(y_rw); axes[2].set_title('随机游走过程')
plt.tight_layout(); plt.show()


##############################################################################
# 6.2 平稳性检验
##############################################################################

## 6.2.1 ACF和PACF
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_acf(y_ar1, lags=20, ax=axes[0]); axes[0].set_title('AR(1) ACF')
plot_pacf(y_ar1, lags=20, ax=axes[1]); axes[1].set_title('AR(1) PACF')
plt.tight_layout(); plt.show()


## 6.2.2 ADF检验
adf_ar1 = adfuller(y_ar1)
adf_rw = adfuller(y_rw)
print(f"AR(1) ADF检验: 统计量={adf_ar1[0]:.4f}, p值={adf_ar1[1]:.4f}")
print(f"随机游走 ADF检验: 统计量={adf_rw[0]:.4f}, p值={adf_rw[1]:.4f}")


##############################################################################
# 6.3 AR模型
##############################################################################

## 6.3.1 AR(1)模型估计
model_ar1 = ARIMA(y_ar1, order=(1, 0, 0)).fit()
print(f"\nAR(1)模型:")
print(model_ar1.summary().tables[1])

## 6.3.2 AR(2)模型
np.random.seed(456)
n2 = 300
e2 = np.random.normal(0, 1, n2)
y_ar2 = np.zeros(n2)
for t in range(2, n2):
    y_ar2[t] = 0.3 * y_ar2[t-1] + 0.4 * y_ar2[t-2] + e2[t]

model_ar2 = ARIMA(y_ar2, order=(2, 0, 0)).fit()
print(f"\nAR(2)模型:")
print(model_ar2.summary().tables[1])


##############################################################################
# 6.4 MA模型
##############################################################################
np.random.seed(789)
n3 = 300
e3 = np.random.normal(0, 1, n3)
y_ma1 = np.zeros(n3)
for t in range(1, n3):
    y_ma1[t] = e3[t] + 0.6 * e3[t-1]

model_ma1 = ARIMA(y_ma1, order=(0, 0, 1)).fit()
print(f"\nMA(1)模型:")
print(model_ma1.summary().tables[1])


##############################################################################
# 6.5 ARMA模型
##############################################################################
np.random.seed(101)
n4 = 500
e4 = np.random.normal(0, 1, n4)
y_arma = np.zeros(n4)
for t in range(1, n4):
    y_arma[t] = 0.5 * y_arma[t-1] + e4[t] + 0.3 * e4[t-1]

model_arma = ARIMA(y_arma, order=(1, 0, 1)).fit()
print(f"\nARMA(1,1)模型:")
print(model_arma.summary().tables[1])


##############################################################################
# 6.6 ARIMA模型
##############################################################################
np.random.seed(202)
n5 = 200
e5 = np.random.normal(0, 1, n5)
y_arima = np.zeros(n5)
for t in range(1, n5):
    y_arima[t] = 0.5 + y_arima[t-1] + e5[t]  # 带漂移的随机游走

# 差分
dy = np.diff(y_arima)
adf_diff = adfuller(dy)
print(f"\n差分后ADF检验: 统计量={adf_diff[0]:.4f}, p值={adf_diff[1]:.4f}")

# ARIMA(0,1,0)
model_arima = ARIMA(y_arima, order=(0, 1, 0)).fit()
print(f"\nARIMA(0,1,0)模型:")
print(model_arima.summary().tables[1])


##############################################################################
# 6.7 模型诊断检验
##############################################################################
residuals = model_ar1.resid
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_acf(residuals, lags=20, ax=axes[0]); axes[0].set_title('残差ACF')
axes[1].hist(residuals, bins=30, density=True, alpha=0.5)
axes[1].set_title('残差直方图')
plt.tight_layout(); plt.show()

# Ljung-Box检验
from statsmodels.stats.diagnostic import acorr_ljungbox
lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
print(f"\nLjung-Box检验: {lb_test}")


##############################################################################
# 6.8 外推预测
##############################################################################
forecast = model_ar1.forecast(steps=20)
forecast_ci = model_ar1.get_forecast(steps=20).conf_int()

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(range(n), y_ar1, 'b-', label='观测值')
ax.plot(range(n, n+20), forecast, 'r-', label='预测')
ax.fill_between(range(n, n+20), forecast_ci[:, 0], forecast_ci[:, 1],
                alpha=0.3, color='red', label='95%置信区间')
ax.set_title('AR(1)模型预测'); ax.legend()
plt.tight_layout(); plt.show()


##############################################################################
# 6.9 季节性时间序列
##############################################################################
np.random.seed(303)
n6 = 240
t6 = np.arange(1, n6+1)
seasonal = 10 * np.sin(2 * np.pi * t6 / 12)
trend = 0.5 * t6
e6 = np.random.normal(0, 2, n6)
y_seasonal = trend + seasonal + 50 + e6

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(t6, y_seasonal)
ax.set_title('季节性时间序列')
plt.tight_layout(); plt.show()

# 季节差分
y_seasonal_diff = np.diff(y_seasonal, 12)
adf_season = adfuller(y_seasonal_diff)
print(f"\n季节差分后ADF: 统计量={adf_season[0]:.4f}, p值={adf_season[1]:.4f}")


##############################################################################
# 6.10 指数平滑预测
##############################################################################
# 简单指数平滑
model_es = ExponentialSmoothing(y_ar1, trend=None, seasonal=None).fit()
forecast_es = model_es.forecast(20)

# Holt-Winters
model_hw = ExponentialSmoothing(y_seasonal, trend='add', seasonal='add',
                                 seasonal_periods=12).fit()
forecast_hw = model_hw.forecast(12)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(range(n), y_ar1, 'b-', label='观测')
axes[0].plot(range(n, n+20), forecast_es, 'r-', label='预测')
axes[0].set_title('简单指数平滑'); axes[0].legend()
axes[1].plot(range(n6), y_seasonal, 'b-', label='观测')
axes[1].plot(range(n6, n6+12), forecast_hw, 'r-', label='预测')
axes[1].set_title('Holt-Winters'); axes[1].legend()
plt.tight_layout(); plt.show()


##############################################################################
# 6.11 非线性时间序列：ARCH/GARCH模型
##############################################################################
np.random.seed(404)
n7 = 500
e7 = np.random.normal(0, 1, n7)
y_arch = np.zeros(n7)
sigma2 = np.zeros(n7)
sigma2[0] = 1
for t in range(1, n7):
    y_arch[t] = np.sqrt(sigma2[t-1]) * e7[t]
    sigma2[t] = 0.1 + 0.8 * y_arch[t]**2

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(y_arch)
ax.set_title('ARCH(1)过程')
plt.tight_layout(); plt.show()

# GARCH模型估计
model_garch = arch_model(y_arch, vol='Garch', p=1, q=1).fit(disp='off')
print(f"\nGARCH(1,1)模型:")
print(model_garch.summary().tables[1])


##############################################################################
# 6.12 多元时间序列
##############################################################################
np.random.seed(505)
n8 = 200
e1_8 = np.random.normal(0, 1, n8)
e2_8 = np.random.normal(0, 1, n8)
y1_var = np.zeros(n8)
y2_var = np.zeros(n8)
for t in range(1, n8):
    y1_var[t] = 0.5 * y1_var[t-1] + e1_8[t]
    y2_var[t] = 0.3 * y2_var[t-1] + 0.4 * y1_var[t] + e2_8[t]

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(y1_var, label='y1')
ax.plot(y2_var, label='y2')
ax.set_title('二元时间序列'); ax.legend()
plt.tight_layout(); plt.show()

# VAR模型
df_var = pd.DataFrame({'y1': y1_var, 'y2': y2_var})
model_var = VAR(df_var).fit(1)
print(f"\nVAR(1)模型:")
print(model_var.summary())

# ARIMA模型拟合与预测
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf

# 模拟ARIMA数据
np.random.seed(123)
n = 100
e = np.random.normal(0, 1, n)
xt = np.zeros(n)
for t in range(1, n):
    xt[t] = 0.5 + xt[t-1] + e[t]  # 带漂移的随机游走

# 一阶差分
xt1 = np.diff(xt)

# ADF检验
adf_result = adfuller(xt1)
print(f"一阶差分后ADF检验: 统计量={adf_result[0]:.4f}, p值={adf_result[1]:.4f}")

# 拟合不同阶数的ARIMA模型
arima1 = ARIMA(xt, order=(1, 1, 1)).fit()
arima2 = ARIMA(xt, order=(2, 1, 1)).fit()
arima3 = ARIMA(xt, order=(1, 1, 2)).fit()
arima4 = ARIMA(xt, order=(2, 1, 2)).fit()

# AIC和BIC比较
print(f"ARIMA(1,1,1): AIC={arima1.aic:.2f}, BIC={arima1.bic:.2f}")
print(f"ARIMA(2,1,1): AIC={arima2.aic:.2f}, BIC={arima2.bic:.2f}")
print(f"ARIMA(1,1,2): AIC={arima3.aic:.2f}, BIC={arima3.bic:.2f}")
print(f"ARIMA(2,1,2): AIC={arima4.aic:.2f}, BIC={arima4.bic:.2f}")

# 外推预测
forecast = arima4.get_forecast(steps=10)
pred_mean = forecast.predicted_mean
pred_ci = forecast.conf_int(alpha=0.05)

# 绘制预测图
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(1, n+1), xt, 'k-', label='观测值')
ax.plot(range(n+1, n+11), pred_mean, 'r-', lw=2, label='预测值')
ax.fill_between(range(n+1, n+11), pred_ci[:, 0], pred_ci[:, 1],
                color='blue', alpha=0.2, label='95%置信区间')
ax.set_xlabel('t'); ax.set_ylabel('xt')
ax.set_title('ARIMA(2,1,2)模型预测')
ax.legend()
plt.tight_layout(); plt.show()

##############################################################################
# 第6章 时间序列分析
# 对应教材：section6.tex
# 内容：平稳性检验、AR/MA/ARMA/ARIMA模型、模型识别、参数估计、
#       诊断检验、外推预测、多元时间序列、非线性时间序列
##############################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.api import VAR
from arch import arch_model
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


##############################################################################
# 6.1 时间序列数据生成与可视化
##############################################################################
np.random.seed(123)
n = 200
e = np.random.normal(0, 1, n)

# AR(1)过程
y_ar1 = np.zeros(n)
for t in range(1, n):
    y_ar1[t] = 0.7 * y_ar1[t-1] + e[t]

# 白噪声
y_wn = e

# 随机游走
y_rw = np.cumsum(e)

fig, axes = plt.subplots(3, 1, figsize=(12, 10))
axes[0].plot(y_ar1); axes[0].set_title('AR(1)过程')
axes[1].plot(y_wn); axes[1].set_title('白噪声过程')
axes[2].plot(y_rw); axes[2].set_title('随机游走过程')
plt.tight_layout(); plt.show()


##############################################################################
# 6.2 平稳性检验
##############################################################################

## 6.2.1 ACF和PACF
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_acf(y_ar1, lags=20, ax=axes[0]); axes[0].set_title('AR(1) ACF')
plot_pacf(y_ar1, lags=20, ax=axes[1]); axes[1].set_title('AR(1) PACF')
plt.tight_layout(); plt.show()


## 6.2.2 ADF检验
adf_ar1 = adfuller(y_ar1)
adf_rw = adfuller(y_rw)
print(f"AR(1) ADF检验: 统计量={adf_ar1[0]:.4f}, p值={adf_ar1[1]:.4f}")
print(f"随机游走 ADF检验: 统计量={adf_rw[0]:.4f}, p值={adf_rw[1]:.4f}")


##############################################################################
# 6.3 AR模型
##############################################################################

## 6.3.1 AR(1)模型估计
model_ar1 = ARIMA(y_ar1, order=(1, 0, 0)).fit()
print(f"\nAR(1)模型:")
print(model_ar1.summary().tables[1])

## 6.3.2 AR(2)模型
np.random.seed(456)
n2 = 300
e2 = np.random.normal(0, 1, n2)
y_ar2 = np.zeros(n2)
for t in range(2, n2):
    y_ar2[t] = 0.3 * y_ar2[t-1] + 0.4 * y_ar2[t-2] + e2[t]

model_ar2 = ARIMA(y_ar2, order=(2, 0, 0)).fit()
print(f"\nAR(2)模型:")
print(model_ar2.summary().tables[1])


##############################################################################
# 6.4 MA模型
##############################################################################
np.random.seed(789)
n3 = 300
e3 = np.random.normal(0, 1, n3)
y_ma1 = np.zeros(n3)
for t in range(1, n3):
    y_ma1[t] = e3[t] + 0.6 * e3[t-1]

model_ma1 = ARIMA(y_ma1, order=(0, 0, 1)).fit()
print(f"\nMA(1)模型:")
print(model_ma1.summary().tables[1])


##############################################################################
# 6.5 ARMA模型
##############################################################################
np.random.seed(101)
n4 = 500
e4 = np.random.normal(0, 1, n4)
y_arma = np.zeros(n4)
for t in range(1, n4):
    y_arma[t] = 0.5 * y_arma[t-1] + e4[t] + 0.3 * e4[t-1]

model_arma = ARIMA(y_arma, order=(1, 0, 1)).fit()
print(f"\nARMA(1,1)模型:")
print(model_arma.summary().tables[1])


##############################################################################
# 6.6 ARIMA模型
##############################################################################
np.random.seed(202)
n5 = 200
e5 = np.random.normal(0, 1, n5)
y_arima = np.zeros(n5)
for t in range(1, n5):
    y_arima[t] = 0.5 + y_arima[t-1] + e5[t]  # 带漂移的随机游走

# 差分
dy = np.diff(y_arima)
adf_diff = adfuller(dy)
print(f"\n差分后ADF检验: 统计量={adf_diff[0]:.4f}, p值={adf_diff[1]:.4f}")

# ARIMA(0,1,0)
model_arima = ARIMA(y_arima, order=(0, 1, 0)).fit()
print(f"\nARIMA(0,1,0)模型:")
print(model_arima.summary().tables[1])


##############################################################################
# 6.7 模型诊断检验
##############################################################################
residuals = model_ar1.resid
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
plot_acf(residuals, lags=20, ax=axes[0]); axes[0].set_title('残差ACF')
axes[1].hist(residuals, bins=30, density=True, alpha=0.5)
axes[1].set_title('残差直方图')
plt.tight_layout(); plt.show()

# Ljung-Box检验
from statsmodels.stats.diagnostic import acorr_ljungbox
lb_test = acorr_ljungbox(residuals, lags=[10], return_df=True)
print(f"\nLjung-Box检验: {lb_test}")


##############################################################################
# 6.8 外推预测
##############################################################################
forecast = model_ar1.forecast(steps=20)
forecast_ci = model_ar1.get_forecast(steps=20).conf_int()

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(range(n), y_ar1, 'b-', label='观测值')
ax.plot(range(n, n+20), forecast, 'r-', label='预测')
ax.fill_between(range(n, n+20), forecast_ci[:, 0], forecast_ci[:, 1],
                alpha=0.3, color='red', label='95%置信区间')
ax.set_title('AR(1)模型预测'); ax.legend()
plt.tight_layout(); plt.show()


##############################################################################
# 6.9 季节性时间序列
##############################################################################
np.random.seed(303)
n6 = 240
t6 = np.arange(1, n6+1)
seasonal = 10 * np.sin(2 * np.pi * t6 / 12)
trend = 0.5 * t6
e6 = np.random.normal(0, 2, n6)
y_seasonal = trend + seasonal + 50 + e6

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(t6, y_seasonal)
ax.set_title('季节性时间序列')
plt.tight_layout(); plt.show()

# 季节差分
y_seasonal_diff = np.diff(y_seasonal, 12)
adf_season = adfuller(y_seasonal_diff)
print(f"\n季节差分后ADF: 统计量={adf_season[0]:.4f}, p值={adf_season[1]:.4f}")


##############################################################################
# 6.10 指数平滑预测
##############################################################################
# 简单指数平滑
model_es = ExponentialSmoothing(y_ar1, trend=None, seasonal=None).fit()
forecast_es = model_es.forecast(20)

# Holt-Winters
model_hw = ExponentialSmoothing(y_seasonal, trend='add', seasonal='add',
                                 seasonal_periods=12).fit()
forecast_hw = model_hw.forecast(12)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(range(n), y_ar1, 'b-', label='观测')
axes[0].plot(range(n, n+20), forecast_es, 'r-', label='预测')
axes[0].set_title('简单指数平滑'); axes[0].legend()
axes[1].plot(range(n6), y_seasonal, 'b-', label='观测')
axes[1].plot(range(n6, n6+12), forecast_hw, 'r-', label='预测')
axes[1].set_title('Holt-Winters'); axes[1].legend()
plt.tight_layout(); plt.show()


##############################################################################
# 6.11 非线性时间序列：ARCH/GARCH模型
##############################################################################
np.random.seed(404)
n7 = 500
e7 = np.random.normal(0, 1, n7)
y_arch = np.zeros(n7)
sigma2 = np.zeros(n7)
sigma2[0] = 1
for t in range(1, n7):
    y_arch[t] = np.sqrt(sigma2[t-1]) * e7[t]
    sigma2[t] = 0.1 + 0.8 * y_arch[t]**2

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(y_arch)
ax.set_title('ARCH(1)过程')
plt.tight_layout(); plt.show()

# GARCH模型估计
model_garch = arch_model(y_arch, vol='Garch', p=1, q=1).fit(disp='off')
print(f"\nGARCH(1,1)模型:")
print(model_garch.summary().tables[1])


##############################################################################
# 6.12 多元时间序列
##############################################################################
np.random.seed(505)
n8 = 200
e1_8 = np.random.normal(0, 1, n8)
e2_8 = np.random.normal(0, 1, n8)
y1_var = np.zeros(n8)
y2_var = np.zeros(n8)
for t in range(1, n8):
    y1_var[t] = 0.5 * y1_var[t-1] + e1_8[t]
    y2_var[t] = 0.3 * y2_var[t-1] + 0.4 * y1_var[t] + e2_8[t]

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(y1_var, label='y1')
ax.plot(y2_var, label='y2')
ax.set_title('二元时间序列'); ax.legend()
plt.tight_layout(); plt.show()

# VAR模型
df_var = pd.DataFrame({'y1': y1_var, 'y2': y2_var})
model_var = VAR(df_var).fit(1)
print(f"\nVAR(1)模型:")
print(model_var.summary())