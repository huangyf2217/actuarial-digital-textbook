# Chap3 R代码
# 自动从chap3.html同步生成

# 工厂1份额60%，次品率10%；工厂2份额30%，次品率5%；工厂3份额10%，次品率15%
# 求发现次品时，该次品来自工厂3的概率
P_B <- c(0.6, 0.3, 0.1)              # 各工厂份额（先验概率）
P_A_given_B <- c(0.10, 0.05, 0.15)   # 各工厂次品率（条件概率）
P_A <- sum(P_A_given_B * P_B)        # 全概率公式
P_B3_given_A <- P_A_given_B[3] * P_B[3] / P_A
P_B3_given_A  # 结果为0.167

# 三类索赔占比80%、15%、5%，参数theta分别为100、1000、2500
# 索赔金额X的密度为f(x) = 2*theta^2/x^3, x > theta
# 求索赔金额为5000时，属于各类索赔的后验概率
theta <- c(100, 1000, 2500)
P_prior <- c(0.80, 0.15, 0.05)
x <- 5000
# P(A|B_i) ∝ f(x|theta_i) = 2*theta_i^2/x^3
P_A_given_B <- 2 * theta^2 / x^3
P_A <- sum(P_A_given_B * P_prior)
P_post <- P_A_given_B * P_prior / P_A
P_post  # 后验概率：S=0.0170, M=0.3188, L=0.6642

# X~Bin(n, theta)，theta~Beta(alpha, beta)
# 后验：Beta(sum(x) + alpha, n - sum(x) + beta)
alpha <- 2; beta <- 3
x <- c(1, 0, 1, 1, 0, 1, 1, 1, 0, 1)  # 10次实验，7次成功
n <- length(x)
alpha_post <- sum(x) + alpha
beta_post <- n - sum(x) + beta
# 后验期望
E_theta_post <- alpha_post / (alpha_post + beta_post)
# 先验 vs 后验对比
E_prior <- alpha / (alpha + beta)
c(先验期望 = E_prior, 后验期望 = E_theta_post, 样本比例 = mean(x))

##############################################################################
# 3.2 共轭先验分布与后验分布计算
##############################################################################

# 假设索赔次数X~Poi(mu)，mu的先验为Ga(alpha, lambda)
# 样本x1,...,xn，后验为Ga(sum(x)+alpha, n+lambda)
alpha <- 100; lambda <- 1  # 先验参数
x <- c(144, 144, 174, 148, 151, 156, 168, 147, 140, 161)  # 10年索赔次数
n <- length(x)
alpha_post <- sum(x) + alpha
lambda_post <- n + lambda
# 后验期望
E_mu_post <- alpha_post / lambda_post
E_mu_post

# X~N(theta, sigma1^2)，theta~N(mu, sigma2^2)
# 后验为N((sigma1^2*mu + n*sigma2^2*xbar)/(sigma1^2+n*sigma2^2), sigma1^2*sigma2^2/(sigma1^2+n*sigma2^2))
sigma1 <- 50; mu0 <- 300; sigma2 <- 20
n <- 10; xbar <- 270
mu_post <- (sigma1^2 * mu0 + n * sigma2^2 * xbar) / (sigma1^2 + n * sigma2^2)
sigma2_post <- sigma1^2 * sigma2^2 / (sigma1^2 + n * sigma2^2)
c(mu_post, sqrt(sigma2_post))  # 后验均值281.54，标准差12.40

# 已知theta的后验分布为Be(sum(x)+alpha, n-sum(x)+beta)
# 二次损失函数下，贝叶斯估计为后验期望
alpha <- 2; beta <- 3
sum_x <- 7; n <- 10
theta_hat <- (sum_x + alpha) / (n + alpha + beta)
theta_hat  # 二次损失下的贝叶斯估计

##############################################################################
# 3.3 贝叶斯估计（不同损失函数）
##############################################################################

# 后验分布为Ga(7, 13)
# (1) 二次损失：后验期望 = 7/13
mu_hat1 <- 7 / 13
# (2) 绝对损失：后验中位数
mu_hat2 <- qgamma(0.5, shape = 7, rate = 13)
# (3) 0/1损失：后验众数 = (7-1)/13
mu_hat3 <- (7 - 1) / 13
c(mu_hat1, mu_hat2, mu_hat3)  # 0.538, 0.513, 0.462

##############################################################################
# 3.4 信度保费与信度因子
##############################################################################

# X~Exp(1/mu)，mu~InvGamma(alpha, theta)
# theta=40, alpha=1.5, sum(x)=9826, n=100
theta <- 40; alpha <- 1.5
sum_x <- 9826; n <- 100
# 后验期望
mu_hat <- (sum_x + theta) / (n + alpha - 1)
# 信度因子
Z <- n / (n + alpha - 1)
c(mu_hat, Z)  # 98.1692, 0.9950

##############################################################################
# 3.5 贝叶斯信度模型
##############################################################################

# 保单组合10年索赔次数数据
x <- c(144, 144, 174, 148, 151, 156, 168, 147, 140, 161)
n <- length(x)

## 先验分布Ga(100, 1)
alpha1 <- 100; beta1 <- 1
Z1 <- numeric(n)
E_mu1 <- numeric(n)
for (k in 1:n) {
  Z1[k] <- k / (k + beta1)
  E_mu1[k] <- (sum(x[1:k]) + alpha1) / (k + beta1)
}

## 先验分布Ga(500, 5)
alpha2 <- 500; beta2 <- 5
Z2 <- numeric(n)
E_mu2 <- numeric(n)
for (k in 1:n) {
  Z2[k] <- k / (k + beta2)
  E_mu2[k] <- (sum(x[1:k]) + alpha2) / (k + beta2)
}

## 绘图：信度因子变化趋势
par(mfrow = c(1, 2))
plot(1:n, Z1, type = 'b', pch = 19, col = 'red',
     xlab = '年度', ylab = '信度因子 Z', main = 'Ga(100,1)',
     ylim = c(0, 1))
plot(1:n, Z2, type = 'b', pch = 19, col = 'blue',
     xlab = '年度', ylab = '信度因子 Z', main = 'Ga(500,5)',
     ylim = c(0, 1))

## 绘图：期望索赔次数估计值变化趋势
par(mfrow = c(1, 2))
plot(1:n, E_mu1, type = 'b', pch = 19, col = 'red',
     xlab = '年度', ylab = 'E(Lambda|x)', main = 'Ga(100,1)',
     ylim = c(100, 175))
abline(h = mean(x), lty = 2, col = 'black')
plot(1:n, E_mu2, type = 'b', pch = 19, col = 'blue',
     xlab = '年度', ylab = 'E(Lambda|x)', main = 'Ga(500,5)',
     ylim = c(100, 175))
abline(h = mean(x), lty = 2, col = 'black')

# X~Bernoulli(p)，p~Beta(a, b)
# 后验为Beta(sum(x)+a, n-sum(x)+b)
# 信度因子Z = n/(n+a+b)
a <- 2; b <- 3
x <- c(1, 0, 1, 1, 0, 1, 1, 1, 0, 1)
n <- length(x)
a_post <- sum(x) + a
b_post <- n - sum(x) + b
E_p_post <- a_post / (a_post + b_post)
Z <- n / (n + a + b)
c(E_p_post, Z)

# X~N(theta, sigma1^2)，theta~N(mu, sigma2^2)
# 信度因子Z = n*sigma2^2/(sigma1^2+n*sigma2^2) = n/(n+sigma1^2/sigma2^2)
sigma1 <- 50; mu0 <- 300; sigma2 <- 20
n <- 10; xbar <- 270

## (1) 先验概率P(mu < 270)
P_prior <- pnorm(270, mean = mu0, sd = sigma2)
P_prior  # 0.06681

## (2) 后验分布
mu_post <- (sigma1^2 * mu0 + n * sigma2^2 * xbar) / (sigma1^2 + n * sigma2^2)
sigma2_post <- sigma1^2 * sigma2^2 / (sigma1^2 + n * sigma2^2)
sigma_post <- sqrt(sigma2_post)

## 后验概率P(mu < 270 | x)
P_post <- pnorm(270, mean = mu_post, sd = sigma_post)
P_post  # 后验概率大于先验概率

## 信度因子
Z <- n * sigma2^2 / (sigma1^2 + n * sigma2^2)
Z

# 四个国家5年火灾保单索赔金额（万美元）
Y <- matrix(c(
  48, 53, 42, 50, 59,
  64, 71, 64, 73, 70,
  85, 54, 76, 65, 90,
  44, 52, 69, 55, 71
), nrow = 4, byrow = TRUE)

N <- nrow(Y); n <- ncol(Y)
X_bar_i <- rowMeans(Y)
X_bar <- mean(Y)

# 补充缺失项
s2_i <- apply(Y, 1, function(row) sum((row - mean(row))^2) / (n - 1))

E_m <- X_bar
E_s2 <- mean(s2_i)
Var_m <- var(X_bar_i) - E_s2 / n

Z <- n / (n + E_s2 / Var_m)
credibility_premium <- Z * X_bar_i + (1 - Z) * E_m

list(E_m = E_m, E_s2 = E_s2, Var_m = Var_m, Z = Z,
     X_bar_i = X_bar_i, s2_i = s2_i, premium = credibility_premium)

##############################################################################
# 3.6 经验贝叶斯信度：EBCT模型1（Bühlmann信度）
##############################################################################

# 五个车队，6年的索赔金额数据
# 数据矩阵：每行一个车队，每列一年
Y <- matrix(c(
  1250,  980, 1800, 2040, 1000, 1180,   # 车队A
  1700, 3080, 1700, 2820, 5760, 3480,   # 车队B
  2050, 3560, 2800, 1600, 4200, 2650,   # 车队C
  4690, 4370, 4800, 9070, 3770, 5250,   # 车队D
  7150, 3480, 5010, 4810, 8740, 7260    # 车队E
), nrow = 5, byrow = TRUE)

N <- nrow(Y)  # 风险数量
n <- ncol(Y)  # 年数

## 计算每个车队的均值
X_bar_i <- rowMeans(Y)
X_bar <- mean(Y)  # 总体均值

## E[m(theta)] = X_bar
E_m <- X_bar

## E[s^2(theta)] = (1/N) * sum_i (1/(n-1)) * sum_j (X_ij - X_bar_i)^2
E_s2 <- mean(apply(Y, 1, function(row) var(row) * (n-1) / (n-1)))
# 等价写法
E_s2 <- mean(apply(Y, 1, var))

## Var[m(theta)] = (1/(N-1)) * sum_i (X_bar_i - X_bar)^2 - E[s^2(theta)]/n
Var_m <- var(X_bar_i) * (N-1) / (N-1) - E_s2 / n
# 注意：var()在R中计算样本方差（除以n-1）
Var_m <- var(X_bar_i) - E_s2 / n

## 信度因子
Z <- n / (n + E_s2 / Var_m)
Z

## 各车队的信度保费
credibility_premium <- Z * X_bar_i + (1 - Z) * E_m
credibility_premium

##############################################################################
# 3.7 经验贝叶斯信度：EBCT模型2（Bühlmann-Straub信度）
##############################################################################

# 索赔金额Y_ij和风险单位数P_ij
Y <- matrix(c(
  1250,  980, 1800, 2040, 1000, 1180,
  1700, 3080, 1700, 2820, 5760, 3480,
  2050, 3560, 2800, 1600, 4200, 2650,
  4690, 4370, 4800, 9070, 3770, 5250,
  7150, 3480, 5010, 4810, 8740, 7260
), nrow = 5, byrow = TRUE)

P <- matrix(c(
  5, 5, 4, 6, 5, 5,
  11, 13, 10, 12, 15, 14,
  3, 4, 4, 3, 3, 2,
  9, 9, 8, 8, 9, 10,
  7, 7, 8, 8, 9, 10
), nrow = 5, byrow = TRUE)

N <- nrow(Y)  # 风险数量
n <- ncol(Y)  # 年数

## 计算X_ij = Y_ij / P_ij
X <- Y / P

## 每个风险的总风险单位数 P_bar_i
P_bar_i <- rowSums(P)
P_bar <- sum(P)  # 总风险单位数

## 每个风险的加权均值 X_bar_i = sum_j P_ij * X_ij / P_bar_i
X_bar_i <- rowSums(P * X) / P_bar_i

## 总体加权均值 X_bar = sum_ij P_ij * X_ij / P_bar
X_bar <- sum(P * X) / P_bar

## E[m(theta)] = X_bar
E_m <- X_bar

## E[s^2(theta)] = (1/N) * sum_i (1/(n-1)) * sum_j P_ij * (X_ij - X_bar_i)^2
E_s2 <- sum(rowSums(P * (X - X_bar_i)^2)) / (N * (n - 1))

## P* = (1/(N*n-1)) * sum_i P_bar_i * (1 - P_bar_i/P_bar)
P_star <- sum(P_bar_i * (1 - P_bar_i / P_bar)) / (N * n - 1)

## Var[m(theta)] = (1/P*) * [(1/(N*n-1)) * sum_ij P_ij*(X_ij-X_bar)^2 - E[s^2(theta)]]
Var_m <- (sum(P * (X - X_bar)^2) / (N * n - 1) - E_s2) / P_star

## 各风险的信度因子 Z_i = P_bar_i / (P_bar_i + E[s^2]/Var[m])
Z_i <- P_bar_i / (P_bar_i + E_s2 / Var_m)
Z_i

## 各风险的单位风险信度保费
credibility_premium_unit <- Z_i * X_bar_i + (1 - Z_i) * E_m
credibility_premium_unit

## 乘以2021年风险单位数得到各车队下一年度信度保费
P_2021 <- c(5, 14, 2, 10, 10)  # 假设的下年风险单位数
credibility_premium <- credibility_premium_unit * P_2021
credibility_premium

# 三个保险公司4年的索赔金额（百万元）和保单数量
Y <- matrix(c(
  14.2, 15.8, 22.7, 19.0,
  58.6, 63.1, 81.0, 64.2,
  123, 132, 161, 133
), nrow = 3, byrow = TRUE)

P <- matrix(c(
  163, 189, 252, 199,
  4435, 4761, 5576, 4581,
  16184, 17443, 20102, 18000
), nrow = 3, byrow = TRUE)

N <- nrow(Y)
n <- ncol(Y)

X <- Y / P
P_bar_i <- rowSums(P)
P_bar <- sum(P)
X_bar_i <- rowSums(P * X) / P_bar_i
X_bar <- sum(P * X) / P_bar

E_m <- X_bar
E_s2 <- sum(rowSums(P * (X - X_bar_i)^2)) / (N * (n - 1))
P_star <- sum(P_bar_i * (1 - P_bar_i / P_bar)) / (N * n - 1)
Var_m <- (sum(P * (X - X_bar)^2) / (N * n - 1) - E_s2) / P_star

## 保险公司B的信度因子
Z_B <- P_bar_i[2] / (P_bar_i[2] + E_s2 / Var_m)
Z_B  # 0.9992

## 保险公司B的单位风险信度保费
cred_B_unit <- Z_B * X_bar_i[2] + (1 - Z_B) * E_m
cred_B_unit  # 0.01369

## 下年风险单位4800的信度保费
cred_B <- cred_B_unit * 4800
cred_B  # 65.712

##############################################################################
# 3.8 EBCT模型计算函数封装
##############################################################################

# 验证EBCT模型2的假设条件：E(X_j|theta) = m(theta), P_j * Var(X_j|theta) = s^2(theta)
# 假设每张保单索赔额W_{j,k}相互独立，均值m(theta)，方差s^2(theta)
# Y_j = sum_k W_{j,k}, X_j = Y_j/P_j

# 模拟：假设3年，每年保单数不同
P <- c(100, 150, 120)  # 各年保单数
m_theta <- 5            # 每张保单期望索赔
s2_theta <- 25          # 每张保单索赔方差

set.seed(123)
# 模拟各年索赔
Y <- sapply(P, function(p) sum(rnorm(p, mean = m_theta, sd = sqrt(s2_theta))))
X <- Y / P

# 验证：E(X_j) ≈ m(theta)
mean(X)  # 应接近5

# 验证：P_j * Var(X_j) ≈ s^2(theta)
# 使用模拟验证
n_sim <- 1000
X_sim <- matrix(0, n_sim, length(P))
for (s in 1:n_sim) {
  Y_sim <- sapply(P, function(p) sum(rnorm(p, mean = m_theta, sd = sqrt(s2_theta))))
  X_sim[s, ] <- Y_sim / P
}
P_var_X <- apply(X_sim, 2, var) * P
c(理论值 = s2_theta, 模拟值 = mean(P_var_X))