# Chap1 R代码
# 自动从chap1.html同步生成

## 二项分布的概率分布列和分布函数
x <- seq(0, 9, 1)

# 概率分布列
plot(x, dbinom(x, size = 9, prob = 0.1), type = "l", col = 1, lty = 1,
     xlab = "k", ylab = "p(k)")
lines(x, dbinom(x, size = 9, prob = 0.2), col = 2, lty = 2)
lines(x, dbinom(x, size = 9, prob = 0.3), col = 3, lty = 3)
lines(x, dbinom(x, size = 9, prob = 0.5), col = 4, lty = 4)

# 累积分布函数
plot(x, pbinom(x, size = 9, prob = 0.1), type = "l", col = 1, lty = 1,
     xlab = "k", ylab = "F(k)")
lines(x, pbinom(x, size = 9, prob = 0.2), col = 2, lty = 2)
lines(x, pbinom(x, size = 9, prob = 0.3), col = 3, lty = 3)
lines(x, pbinom(x, size = 9, prob = 0.5), col = 4, lty = 4)

## 泊松分布的概率分布列和分布函数
x <- seq(0, 10, 1)

# 概率分布列
plot(x, dpois(x, lambda = 1), type = "l", col = 1, lty = 1,
     xlab = "k", ylab = "p(k)")
lines(x, dpois(x, lambda = 2), col = 2, lty = 2)
lines(x, dpois(x, lambda = 3), col = 3, lty = 3)
lines(x, dpois(x, lambda = 5), col = 4, lty = 4)

# 累积分布函数
plot(x, ppois(x, lambda = 1), type = "l", col = 1, lty = 1,
     xlab = "k", ylab = "F(k)")
lines(x, ppois(x, lambda = 2), col = 2, lty = 2)
lines(x, ppois(x, lambda = 3), col = 3, lty = 3)
lines(x, ppois(x, lambda = 5), col = 4, lty = 4)

## 负二项分布：size = r, prob = 1/(1+beta)
x <- seq(0, 10, 1)

# 概率分布列
plot(x, dnbinom(x, size = 2, prob = 1/(1+0.1)), type = "l", col = 1, lty = 1,
     xlab = "k", ylab = "p(k)")
lines(x, dnbinom(x, size = 2, prob = 1/(1+0.2)), col = 2, lty = 2)
lines(x, dnbinom(x, size = 2, prob = 1/(1+0.3)), col = 3, lty = 3)
lines(x, dnbinom(x, size = 2, prob = 1/(1+0.5)), col = 4, lty = 4)

# 累积分布函数
plot(x, pnbinom(x, size = 2, prob = 1/(1+0.1)), type = "l", col = 1, lty = 1,
     xlab = "k", ylab = "F(k)")
lines(x, pnbinom(x, size = 2, prob = 1/(1+0.2)), col = 2, lty = 2)
lines(x, pnbinom(x, size = 2, prob = 1/(1+0.3)), col = 3, lty = 3)
lines(x, pnbinom(x, size = 2, prob = 1/(1+0.5)), col = 4, lty = 4)

## 零截断负二项分布的概率
x <- 0:10
p <- dnbinom(x, 4, 0.7)       # 负二项分布的概率
p0 <- p[1]                    # 零点的概率
pt1 <- p[2:11] / (1 - p0)     # 零截断分布在非0点上的概率
pt <- c(0, pt1)               # 零截断分布的概率
com <- rbind(负二项 = p, 零截断负二项 = pt)
barplot(com, beside = TRUE, names.arg = 0:10, legend.text = TRUE)

## 零调整负二项分布的概率
pm0 <- 0.3                    # 调整后的零点概率
pm1 <- (1 - pm0) * p[2:11] / (1 - p0)
pm <- c(pm0, pm1)
com <- rbind(负二项 = p, 零调整负二项 = pm)
barplot(com, beside = TRUE, names.arg = 0:10, legend.text = TRUE)

## 指数分布的概率密度函数
x <- seq(0, 5, 0.01)
plot(x, dexp(x, rate = 0.5), type = "l", col = 1, lty = 1,
     xlab = "x", ylab = "f(x)")
lines(x, dexp(x, rate = 1), col = 2, lty = 2)
lines(x, dexp(x, rate = 2), col = 3, lty = 3)
lines(x, dexp(x, rate = 5), col = 4, lty = 4)

## 伽马分布概率密度函数：改变形状参数
x <- seq(0, 4, 0.001)
plot(x, dgamma(x, shape = 1, scale = 1), type = "l", col = 1, lty = 1,
     xlab = "x", ylab = "f(x)")
lines(x, dgamma(x, shape = 2, scale = 1), col = 2, lty = 2)
lines(x, dgamma(x, shape = 3, scale = 1), col = 3, lty = 3)
lines(x, dgamma(x, shape = 0.5, scale = 1), col = 4, lty = 4)

## 伽马分布概率密度函数：改变尺度参数
plot(x, dgamma(x, shape = 1, scale = 1), type = "l", col = 1, lty = 1,
     xlab = "x", ylab = "f(x)")
lines(x, dgamma(x, shape = 1, scale = 2), col = 2, lty = 2)
lines(x, dgamma(x, shape = 1, scale = 3), col = 3, lty = 3)
lines(x, dgamma(x, shape = 1, scale = 0.5), col = 4, lty = 4)

## 逆高斯分布概率密度函数（需 statmod 包）
x <- seq(0, 5, 0.01)

## 改变均值参数
plot(x, dinvgauss(x, mean = 1, shape = 1), type = "l", col = 1, lty = 1,
     xlab = "x", ylab = "f(x)")
lines(x, dinvgauss(x, mean = 2, shape = 1), col = 2, lty = 2)
lines(x, dinvgauss(x, mean = 5, shape = 1), col = 3, lty = 3)
lines(x, dinvgauss(x, mean = 0.5, shape = 1), col = 4, lty = 4)

## 改变形状参数
plot(x, dinvgauss(x, mean = 1, shape = 1), type = "l", col = 1, lty = 1,
     xlab = "x", ylab = "f(x)")
lines(x, dinvgauss(x, mean = 1, shape = 2), col = 2, lty = 2)
lines(x, dinvgauss(x, mean = 1, shape = 5), col = 3, lty = 3)
lines(x, dinvgauss(x, mean = 1, shape = 0.5), col = 4, lty = 4)

## 定义帕累托分布的概率密度函数
pareto.pdf <- function(x, alpha, lambda){
  alpha * lambda^alpha / (lambda + x)^(alpha + 1)
}

## 帕累托分布概率密度函数：改变形状参数
x <- seq(0, 3, 0.01)
plot(x, pareto.pdf(x, alpha = 1, lambda = 3), type = "l", col = 1, lty = 1,
     xlab = "x", ylab = "f(x)")
lines(x, pareto.pdf(x, alpha = 2, lambda = 3), col = 2, lty = 2)
lines(x, pareto.pdf(x, alpha = 5, lambda = 3), col = 3, lty = 3)
lines(x, pareto.pdf(x, alpha = 0.5, lambda = 3), col = 4, lty = 4)

## 帕累托分布概率密度函数：改变尺度参数
plot(x, pareto.pdf(x, alpha = 2, lambda = 1), type = "l", col = 1, lty = 1,
     xlab = "x", ylab = "f(x)")
lines(x, pareto.pdf(x, alpha = 2, lambda = 2), col = 2, lty = 2)
lines(x, pareto.pdf(x, alpha = 2, lambda = 5), col = 3, lty = 3)
lines(x, pareto.pdf(x, alpha = 2, lambda = 0.5), col = 4, lty = 4)

## 对数正态分布概率密度函数：改变均值参数
x <- seq(0, 7, 0.01)
plot(x, dlnorm(x, meanlog = 1, sdlog = 0.5), type = "l", col = 1, lty = 1,
     xlab = "x", ylab = "f(x)")
lines(x, dlnorm(x, meanlog = 1, sdlog = 3), col = 2, lty = 2)
lines(x, dlnorm(x, meanlog = 1, sdlog = 10), col = 3, lty = 3)
lines(x, dlnorm(x, meanlog = 1, sdlog = 1), col = 4, lty = 4)

## 对数正态分布概率密度函数：改变方差参数
plot(x, dlnorm(x, meanlog = 1, sdlog = 0.5), type = "l", col = 1, lty = 1,
     xlab = "x", ylab = "f(x)")
lines(x, dlnorm(x, meanlog = 1, sdlog = 3), col = 2, lty = 2)
lines(x, dlnorm(x, meanlog = 1, sdlog = 10), col = 3, lty = 3)
lines(x, dlnorm(x, meanlog = 1, sdlog = 1), col = 4, lty = 4)

## 威布尔分布概率密度函数：改变形状参数
x <- seq(0, 3, 0.01)
plot(x, dweibull(x, shape = 1, scale = 1), type = "l", col = 1, lty = 1,
     xlab = "x", ylab = "f(x)")
lines(x, dweibull(x, shape = 2, scale = 1), col = 2, lty = 2)
lines(x, dweibull(x, shape = 3, scale = 1), col = 3, lty = 3)
lines(x, dweibull(x, shape = 0.5, scale = 1), col = 4, lty = 4)

## 威布尔分布概率密度函数：改变尺度参数
plot(x, dweibull(x, shape = 1, scale = 1), type = "l", col = 1, lty = 1,
     xlab = "x", ylab = "f(x)")
lines(x, dweibull(x, shape = 1, scale = 2), col = 2, lty = 2)
lines(x, dweibull(x, shape = 1, scale = 3), col = 3, lty = 3)
lines(x, dweibull(x, shape = 1, scale = 0.5), col = 4, lty = 4)

## 定义混合分布的概率密度函数
f <- function(x){
  0.3 * dlnorm(x, meanlog = 1, sdlog = 2) + 0.7 * dlnorm(x, meanlog = 3, sdlog = 4)
}

curve(f, xlim = c(0, 1), ylim = c(0, 2), lty = 1, lwd = 2, col = 2)
curve(dlnorm(x, meanlog = 1, sdlog = 2), lty = 2, add = TRUE)
curve(dlnorm(x, meanlog = 3, sdlog = 4), lty = 3, add = TRUE)

set.seed(123)  # 设置随机种子以保证结果可重复
x <- rgamma(100, 2, 2.5)  # 从Ga(2, 2.5)中抽取100个随机数

## 极大似然法估计参数
fit1 <- fitdist(x, 'gamma', method = 'mle')
fit1  # 形状参数估计值约为2.19，速率参数估计值约为3.18
plot(fit1)

## 矩估计法估计参数
fit2 <- fitdist(x, 'gamma', method = 'mme')

## 分位数法估计参数
fit3 <- fitdist(x, 'gamma', method = 'qme', probs = c(1/3, 2/3))

## 最小距离法估计参数
fit4 <- fitdist(x, 'gamma', method = 'mge', gof = 'CvM')

summary(fit1)

## 卡方拟合优度检验：维修费用数据
lower <- c(0, 1000, 2000, 3000, 4000, 5000)
upper <- c(1000, 2000, 3000, 4000, 5000, Inf)
freq  <- c(250, 300, 250, 150, 100, 0)

## 指数分布的极大似然估计
f1 <- function(a) {-sum(freq * log(pexp(upper, a) - pexp(lower, a)))}
m1 <- optimize(f1, c(1e-5, 1e-3))

## 计算理论分布
q <- pexp(upper, m1$minimum) - pexp(lower, m1$minimum)
n <- sum(freq) * q

## 作卡方检验
chisq.test(freq, p = q)

## 泊松分布拟合检验
X <- 0:6
Y <- c(7, 10, 12, 8, 3, 2, 0)
q <- ppois(X, mean(rep(X, Y)))
n <- length(Y)
p <- rep(0, n)
p[1] <- q[1]
p[n] <- 1 - q[n-1]
for (i in 2:(n-1)) {
  p[i] <- q[i] - q[i-1]
}
chisq.test(Y, p = p)