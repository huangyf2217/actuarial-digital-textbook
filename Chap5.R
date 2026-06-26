# Chap5 R代码
# 自动从chap5.html同步生成

## 正态分布的不完全1阶矩和不完全2阶矩
norm1 <- function(mu,sigma,L,U){
  L1 <- (L-mu)/sigma
  U1 <- (U-mu)/sigma
  mu*(pnorm(U1)-pnorm(L1)) - sigma*(dnorm(U1)-dnorm(L1))
}
norm2 <- function(mu,sigma,L,U){
  L1 <- (L-mu)/sigma
  U1 <- (U-mu)/sigma
  (mu^2+sigma^2)*(pnorm(U1)-pnorm(L1)) - sigma*((2*mu+sigma*U1)*dnorm(U1)-(2*mu+sigma*L1)*dnorm(L1))
}

## W的期望和方差
EW1 <- function(mu,sigma,M){
  (norm1(mu,sigma,L=M,U=Inf) - M*(1-pnorm(M,mu,sigma))) / (1-pnorm(M,mu,sigma))
}
EW2 <- function(mu,sigma,M){
  (norm2(mu,sigma,L=M,U=Inf) - 2*M*norm1(mu,sigma,L=M,U=Inf) + (M^2)*(1-pnorm(M,mu,sigma))) / (1-pnorm(M,mu,sigma))
}
DW <- function(mu,sigma,M){
  EW2(mu,sigma,M) - EW1(mu,sigma,M)^2
}

## 对数正态分布的不完全k阶矩
lnorm.k <- function(mu,sigma,L,U,k){
  Lk <- (log(L)-mu)/sigma - k*sigma
  Uk <- (log(U)-mu)/sigma - k*sigma
  exp(k*mu+0.5*(k^2)*(sigma^2)) * (pnorm(Uk)-pnorm(Lk))
}

## W的期望和方差
EW1 <- function(mu,sigma,M){
  (lnorm.k(mu,sigma,L=M,U=Inf,k=1) - M*(1-plnorm(M,mu,sigma))) / (1-plnorm(M,mu,sigma))
}
EW2 <- function(mu,sigma,M){
  (lnorm.k(mu,sigma,L=M,U=Inf,k=2) - 2*M*lnorm.k(mu,sigma,L=M,U=Inf,k=1) + (M^2)*(1-plnorm(M,mu,sigma))) / (1-plnorm(M,mu,sigma))
}
DW <- function(mu,sigma,M){
  EW2(mu,sigma,M) - EW1(mu,sigma,M)^2
}

# 5.3.3 例5.1：再保险方案比较
# 对应教材：section5.tex 5.2.3节，例5.1
# -----------------------------------------------------------------
# 假设X~LN(mu=8.5, sigma^2=0.64)
mu <- 8.5
sigma <- 0.8  # sigma^2 = 0.64
M <- 25000

## (1) 不使用再保险：E(X) = exp(mu + sigma^2/2)
EX <- exp(mu + sigma^2 / 2)
EX  # 6768

## (2) 协议1：25%成数再保险，E(Y1) = 0.75 * E(X)
EX1 <- 0.75 * EX
EX1  # 5076

## (3) 协议2：无限额超额赔款，自留额M=25000
## E(Y2) = E(X ∧ M) = 不完全1阶矩 + M*(1-F(M))
EX2 <- lnorm.k(mu, sigma, 0, M, 1) + M * lnorm.k(mu, sigma, M, Inf, 0)
EX2  # 6558

# 5.3.4 例5.2：再保险方案下的方差
# 对应教材：section5.tex 5.2.3节，例5.2
# -----------------------------------------------------------------
## (1) Var(X) = E(X^2) - E(X)^2
EX_sq <- lnorm.k(mu, sigma, 0, Inf, 2)
VARX <- EX_sq - EX^2
VARX  # (6408)^2

## (2) 协议1：Var(Y1) = 0.75^2 * Var(X)
VARX1 <- 0.75^2 * VARX
VARX1  # (4806)^2

## (3) 协议2：Var(Y2) = E(Y2^2) - E(Y2)^2
EX2_SQ <- lnorm.k(mu, sigma, 0, M, 2) + M^2 * lnorm.k(mu, sigma, M, Inf, 0)
VARX2 <- EX2_SQ - EX2^2
VARX2  # (5303)^2

# 5.3.5 例5.3：有限额超额赔款再保险
# 对应教材：section5.tex 5.2.3节，例5.3
# -----------------------------------------------------------------
# 超额点M=25000，上限R=50000
R <- 50000
## E(Y2) = E(X∧M) + M*P(M<X≤R) + (R-M)*P(X>R) ... 实际为：
## E(Y2) = ∫₀ᴹ xf(x)dx + M∫ᴹᴿ f(x)dx + ∫ᴿ^∞ (x-R+M)f(x)dx
EX2_new <- lnorm.k(mu, sigma, 0, M, 1) +
            M * (plnorm(R, mu, sigma) - plnorm(M, mu, sigma)) +
            lnorm.k(mu, sigma, R, Inf, 1) -
            (R - M) * (1 - plnorm(R, mu, sigma))
EX2_new  # 6585

##############################################################################
# 5.4 通货膨胀的影响
##############################################################################

# 5.4.1 通胀对超额赔款再保险的影响
# 对应教材：section5.tex 5.2.4节，例5.4
# -----------------------------------------------------------------
# 假设X~Pa(alpha, lambda)，通胀率k=1.1，自留额M=500
# 通胀后X1 = k*X0 ~ Pa(alpha, k*lambda)
# E(Y) = lambda/(alpha-1) - (lambda/(lambda+M))^alpha * (lambda+M)/(alpha-1)

pareto_reinsurance_EY <- function(alpha, lambda, M) {
  lambda / (alpha - 1) -
    (lambda / (lambda + M))^alpha * (lambda + M) / (alpha - 1)
}

alpha <- 6
lambda0 <- 1000
M <- 500
k <- 1.1  # 通胀率10%

## 第0年
EY0 <- pareto_reinsurance_EY(alpha, lambda0, M)
EY0  # 173.66

## 第1年：lambda1 = k*lambda0 = 1100
EY1 <- pareto_reinsurance_EY(alpha, k * lambda0, M)
EY1  # 186.21

## 第2年：lambda2 = k^2*lambda0 = 1210
EY2 <- pareto_reinsurance_EY(alpha, k^2 * lambda0, M)
EY2  # 199.07

## 增长比例
(EY1 - EY0) / EY0  # 7.2%
(EY2 - EY1) / EY1  # 6.9%

##############################################################################
# 5.5 再保险下的累积赔付金额
##############################################################################

# 5.5.2 超额赔款再保险下的累积赔付
# 对应教材：section5.tex 5.3.2节，例5.5
# -----------------------------------------------------------------
# 假设N~Poi(lambda=10)，X~U(0,2000)，自留额M=1600
lambda <- 10
M <- 1600
a <- 0  # 均匀分布下限
b <- 2000  # 均匀分布上限

## 原保险人Y = min(X, M)
## E(Y) = ∫₀ᴹ x*f(x)dx + M*P(X>M)
EY <- integrate(function(x) x * dunif(x, a, b), 0, M)$value +
       M * (1 - punif(M, a, b))
EY  # 960

## E(Y^2) = ∫₀ᴹ x²*f(x)dx + M²*P(X>M)
EY2 <- integrate(function(x) x^2 * dunif(x, a, b), 0, M)$value +
        M^2 * (1 - punif(M, a, b))
EY2  # 1194666.7

## E(Y^3) = ∫₀ᴹ x³*f(x)dx + M³*P(X>M)
EY3 <- integrate(function(x) x^3 * dunif(x, a, b), 0, M)$value +
        M^3 * (1 - punif(M, a, b))
EY3  # 1.6384e9

## S_I的期望、方差、偏态系数
ES_I <- lambda * EY
VarS_I <- lambda * EY2
skewS_I <- lambda * EY3 / VarS_I^(3/2)
c(ES_I, VarS_I, skewS_I)  # 9600, 11946667, 0.397

## 再保险人Z = max(0, X-M)
## E(Z) = ∫ᴹᵇ (x-M)*f(x)dx
EZ <- integrate(function(x) (x - M) * dunif(x, a, b), M, b)$value
EZ  # 40

## E(Z^2)
EZ2 <- integrate(function(x) (x - M)^2 * dunif(x, a, b), M, b)$value
EZ2  # 10666.7

## E(Z^3)
EZ3 <- integrate(function(x) (x - M)^3 * dunif(x, a, b), M, b)$value
EZ3  # 3200000

## S_R的期望、方差、偏态系数
ES_R <- lambda * EZ
VarS_R <- lambda * EZ2
skewS_R <- lambda * EZ3 / VarS_R^(3/2)
c(ES_R, VarS_R, skewS_R)  # 400, 106667, 0.919

# 5.5.3 再保险人实际赔付次数和金额
# 对应教材：section5.tex 5.3.2节，例5.6
# -----------------------------------------------------------------
# 再保险人实际赔付次数 N_R = I_1 + I_2 + ... + I_N
# 其中I_i ~ Bernoulli(pi), pi = P(X > M)
# 若N~Poi(lambda)，则N_R~Poi(lambda*pi)

pi <- 1 - punif(M, a, b)  # P(X > M)
pi  # 0.2

lambda_R <- lambda * pi  # 再保险人实际赔付次数的泊松参数
lambda_R  # 2

## 示例数据验证
claims <- c(403, 1490, 1948, 443, 1866, 1704, 1221, 823)
# 再保险人对超过1600的索赔赔付
reinsurance_claims <- claims[claims > M] - M
reinsurance_claims  # 348, 266, 104
sum(reinsurance_claims)  # 718

##############################################################################
# 5.6 不完整数据的估计
##############################################################################

# 5.6.2 删失数据的极大似然估计
# 对应教材：section5.tex 5.4节，例5.7
# -----------------------------------------------------------------
# 假设X~Exp(lambda)，自留限额M=1000
# 90次赔付不超过1000，平均赔付82.9；10次赔付超过1000
# 似然函数：L(lambda) = lambda^90 * exp(-(10000 + 90*xbar)*lambda)

M_censor <- 1000
n_complete <- 90
n_censor <- 10
xbar <- 82.9

## MLE: lambda_hat = n_complete / (n_censor*M + n_complete*xbar)
lambda_hat <- n_complete / (n_censor * M_censor + n_complete * xbar)
lambda_hat  # 0.005154

## 验证：使用optim函数进行数值优化
negloglik <- function(lambda) {
  -n_complete * log(lambda) + (n_censor * M_censor + n_complete * xbar) * lambda
}
optim_result <- optim(par = 0.01, fn = negloglik, method = "BFGS")
optim_result$par  # 应接近lambda_hat

# 5.6.3 左删失随机变量
# 对应教材：section5.tex 5.4节，例5.8
# -----------------------------------------------------------------
# Y = (X-M)+ = max(0, X-M)
# E(Y) = E((X-M)+) = ∫ᴹ^∞ (x-M)*f(x)dx

## 示例：指数分布
E_left_censor <- function(lambda, M) {
  integrate(function(x) (x - M) * dexp(x, lambda), M, Inf)$value
}
E_left_censor(0.005, 1000)

##############################################################################
# 5.7 随机模拟：再保险效果比较
##############################################################################