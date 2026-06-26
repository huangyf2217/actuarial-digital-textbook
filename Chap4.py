# Chap4 Python代码
# 自动从chap4.html同步生成

##############################################################################
# 第4章 准备金评估
# 对应教材：section4.tex
# 内容：流量三角形、链梯法、案均赔款法、B-F法、通胀调整、
#       随机准备金模型（Mack链梯法、广义线性模型方法）
##############################################################################

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


##############################################################################
# 4.1 流量三角形与进展因子
##############################################################################

## 4.5.1 B-F法示例1
# 累积赔款流量三角形
triangle_bf1 = np.array([
    [2866, 3334, 3503, 3624, 3719, 3720],
    [3359, 3889, 4033, 4231, 4319, np.nan],
    [3848, 4503, 4779, 4946, np.nan, np.nan],
    [4673, 5422, 5676, np.nan, np.nan, np.nan],
    [5369, 6142, np.nan, np.nan, np.nan, np.nan],
    [5818, np.nan, np.nan, np.nan, np.nan, np.nan]
], dtype=float)

# 已赚保费
premium = np.array([4486, 5024, 5680, 6590, 7482, 8502])
# 预期赔付率
loss_ratio = 0.83

# 应用链梯法计算进展因子
result_cl = chain_ladder(triangle_bf1)
factors_bf = result_cl['factors']  # 1.158, 1.049, 1.039, 1.023, 1.000
print(f"\nB-F法进展因子: {factors_bf}")

# 计算各事故年的累积进展因子f
n_bf = triangle_bf1.shape[0]
f = np.zeros(n_bf)
for i in range(n_bf):
    # 从最后一个已知进展年到最大进展年的累积进展因子
    last_dev = n_bf - i  # 最后一个已知进展年（1-indexed）
    if last_dev < n_bf:
        f[i] = np.prod(factors_bf[last_dev-1:n_bf-1])
    else:
        f[i] = 1.0
print(f"累积进展因子: {f}")

# 最终赔款的初始估计 = 保费 * 赔付率
ultimate_initial = premium * loss_ratio
print(f"最终赔款初始估计: {ultimate_initial}")

# B-F法准备金 = 保费 * 赔付率 * (1 - 1/f)
reserve_bf = ultimate_initial * (1 - 1/f)
print(f"B-F法各事故年准备金: {reserve_bf}")

# 修正的最终赔款 = 已付赔款 + B-F法准备金
paid_bf = np.array([triangle_bf1[i, n_bf-1-i] for i in range(n_bf)])
ultimate_bf = paid_bf + reserve_bf
print(f"修正的最终赔款: {ultimate_bf}")

# 总准备金
print(f"B-F法总准备金: {reserve_bf.sum():.0f}")

##############################################################################
# 第4章 准备金评估
# 对应教材：section4.tex
# 内容：流量三角形、链梯法、案均赔款法、B-F法、通胀调整、
#       随机准备金模型（Mack链梯法、广义线性模型方法）
##############################################################################

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


##############################################################################
# 4.1 流量三角形与进展因子
##############################################################################

## 4.5.3 B-F法示例2
triangle_bf2 = np.array([
    [473, 620, 690, 715],
    [512, 660, 750, np.nan],
    [611, 700, np.nan, np.nan],
    [647, np.nan, np.nan, np.nan]
], dtype=float)

premium2 = np.array([860, 940, 980, 1020])
loss_ratio2 = 0.85

result_bf2 = BF_method(triangle_bf2, premium2, loss_ratio2)
print(f"\nB-F法示例2:")
print(f"  累积进展因子: {result_bf2['f']}")  # 1.0362, 1.1657, 1.4462
print(f"  各事故年准备金: {result_bf2['reserve']}")  # 0, 27.91, 118.41, 267.50
print(f"  总准备金: {result_bf2['total_reserve']:.2f}")  # 413.82


##############################################################################
# 4.6 随机准备金模型：Mack链梯法
##############################################################################