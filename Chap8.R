# Chap8 R代码
# 自动从chap8.html同步生成

u1 <- u2 <- seq(0,1,0.01)

#独立Copula
f1 <- function(u1,u2) u1*u2
CI <- outer(u1,u2,f1)
persp3d(u1,u2,CI,col="blue") # Copula分布函数
contour(u1,u2,CI,col="blue") # 分布函数的等高线

# 同单调Copula
f2 <- Vectorize(function(u1,u2) min(u1,u2))
CU <- outer(u1,u2,f2)
persp3d(u1,u2,CU,col="blue") # Copula分布函数
contour(u1,u2,CU,col="blue") # 分布函数的等高线

# 反单调Copula
f3 <- Vectorize(function(u1,u2) max(u1+u2-1,0))
CL <- outer(u1,u2,f3)
persp3d(u1,u2,CL,col="blue") # Copula分布函数
contour(u1,u2,CL,col="blue") # 分布函数的等高线

library(gsl)mycop <- normalCopula(0.6) # 定义高斯Copula，参数为0.6
u <- rCopula(1000,mycop) # 从高斯Copula中模拟1000对(u1,u2)

persp(mycop,dCopula,col="lightblue") # Copula密度函数
plot(u) # 随机样本的分布
persp(mycop,pCopula,col="lightblue") # Copula分布函数
contour(mycop,pCopula) # 分布函数的等高线

library(gsl)mycop <- tCopula(0.6) # 定义T Copula，参数为0.6，自由度为默认值
u <- rCopula(1000,mycop) # 从高斯Copula中模拟1000对(u1,u2)

persp(mycop,dCopula,col="lightblue") # Copula密度函数
plot(u) # 随机样本的分布
persp(mycop,pCopula,col="lightblue") # Copula分布函数
contour(mycop,pCopula) # 分布函数的等高线

library(gsl)
# Clayton copula
mycop1 <- claytonCopula(0.6)
u1 <- rCopula(1000,mycop1)
persp(mycop1,dCopula,col="lightblue") # Copula密度函数
plot(u1) # 随机样本的分布
persp(mycop1,pCopula,col="lightblue") # Copula分布函数
contour(mycop1,pCopula) # 分布函数的等高线

# Frank copula
mycop2 <- frankCopula(0.6)  
u2 <- rCopula(1000, mycop2)
persp(mycop2,dCopula,col="lightblue") # Copula密度函数
plot(u2) # 随机样本的分布
persp(mycop2,pCopula,col="lightblue") # Copula分布函数
contour(mycop2,pCopula) # 分布函数的等高线

# Gumbel copula
mycop3 <- gumbelCopula(1.6)  
u3 <- rCopula(1000, mycop3)
persp(mycop3,dCopula,col="lightblue") # Copula密度函数
plot(u3) # 随机样本的分布
persp(mycop3,pCopula,col="lightblue") # Copula分布函数
contour(mycop3,pCopula) # 分布函数的等高线

library(gsl)
# 定义Copula模型及其参数
kendall.tau <- 0.4
copulas <- list(
  Gaussian = normalCopula(param=iTau(normalCopula(),kendall.tau)),
  t = tCopula(param=iTau(tCopula(),kendall.tau)),
  Clayton = claytonCopula(param=iTau(claytonCopula(),kendall.tau)),
  Frank = frankCopula(param=iTau(frankCopula(),kendall.tau)),
  Gumbel = gumbelCopula(param=iTau(gumbelCopula(),kendall.tau))
)

# 模拟(u1,u2)的联合样本，并转化回标准正态边际
set.seed(123)
data_list <- lapply(copulas,function(cop){
  u <- rCopula(5000, cop)
  qnorm(u)
})
plot(data_list[[i]]) # 分别绘图（i=1:5）

# 8.4.1 基于相依性测度的方法
# 对应教材：section8.tex 8.4.1节，例8.3
# -----------------------------------------------------------------
## 读取损失金额和ALAE数据
## loss_ALAE <- as.matrix(read.table("L_ALAE.txt", header = TRUE))

## 使用模拟数据演示
set.seed(101)
n_obs <- 100
# 使用Clayton Copula生成相依数据
true_cop <- claytonCopula(3.0)
u_obs <- rCopula(n_obs, true_cop)
loss_ALAE <- cbind(qgamma(u_obs[, 1], shape = 2, rate = 0.001),
                   qgamma(u_obs[, 2], shape = 3, rate = 0.0005))

## 计算Kendall秩相关系数
tau_kendall <- cor(loss_ALAE[, 1], loss_ALAE[, 2], method = "kendall")
tau_kendall

## 计算Copula参数的估计值
iTau(claytonCopula(), tau_kendall)
iTau(frankCopula(), tau_kendall)
iTau(gumbelCopula(), tau_kendall)

library(gsl)
## 计算Kendall秩相关系数
loss_ALAE <- as.matrix(read.table("L_ALAE.txt",header=T))
cor(loss_ALAE[,1],loss_ALAE[,2],method="kendall") # tau=0.7538462

## 计算Copula参数的估计值

iTau(claytonCopula(),0.7538462) # alpha=6.125002
iTau(frankCopula(),0.7538462) # alpha=14.39282
iTau(gumbelCopula(),0.7538462) # alpha=4.062501

## 计算Zi的经验观测值
compute_Z <- function(data){
  n <- nrow(data)
  Z <- numeric(n)
  for(i in 1:n){
    X.i1 <- data[i,1]
    X.i2 <- data[i,2]
    count <- which((data[-i,1]<X.i1) & (data[-i,2]<X.i2))
    Z[i] <- length(count) / (n-1)
  }
  return(Z)
}
Zi <- compute_Z(loss_ALAE)

## 给出K(z)的经验估计和参数估计
# K(z)的非参数估计
K.exp <- ecdf(Zi)
# Clayton Copula对应的K(z)
K1 <- function(z,alpha){
  f1 <- function(t) (t^(-alpha)-1)/alpha
  f2 <- function(t) -t^{-alpha-1}
  z - f1(z)/f2(z)
}
# Frank Copula对应的K(z)
K2 <- function(z,alpha){
  f1 <- function(t) -log((exp(-alpha*t)-1) / (exp(-alpha)-1))
  f2 <- function(t) alpha * exp(-alpha*t) / (exp(-alpha*t)-1)
  z - f1(z)/f2(z)
}
# Gumbel Copula对应的K(z)
K3 <- function(z,alpha){
  f1 <- function(t) (-log(t))^alpha
  f2 <- function(t) -alpha * (-log(t))^(alpha-1) / t
  z - f1(z)/f2(z)
}

## 计算K(z)的估计值与平方偏差和
Zi <- pmax(0.001, pmin(Zi, 0.999))
KZ.exp <- K.exp(Zi)
KZ.1 <- K1(Zi,alpha=6.125002)
KZ.2 <- K2(Zi,alpha=14.39282)
KZ.3 <- K3(Zi,alpha=4.062501)

sum((KZ.1-KZ.exp)^2) # 0.05739576
sum((KZ.2-KZ.exp)^2) # 0.08418203
sum((KZ.3-KZ.exp)^2) # 0.0952341


##############################################################################
# 8.2.5 应用复合函数生成Copula
##############################################################################
library(copula)

# 1. 拉普拉斯变换与阿基米德Copula的对应关系
# Clayton Copula: 潜变量gamma ~ Gamma(1/alpha, 1)
# 拉普拉斯变换: tau(s) = (1+s)^(-1/alpha)
# 拉普拉斯逆变换: tau^{-1}(t) = t^{-alpha} - 1

alpha <- 2
tau_inv <- function(t, alpha) t^(-alpha) - 1
psi_clayton <- function(t, alpha) (t^(-alpha) - 1) / alpha

# 验证：tau^{-1}(t) = alpha * psi(t)
t_vals <- seq(0.01, 0.99, 0.1)
cat("验证 tau^{-1}(t) = alpha * psi(t):\n")
for (t in t_vals) {
  cat(sprintf("  t=%.2f: tau_inv=%.4f, alpha*psi=%.4f\n",
              t, tau_inv(t, alpha), alpha * psi_clayton(t, alpha)))
}

# 2. 基于脆弱性模型的Copula模拟
set.seed(2024)
alpha <- 2
n <- 1000

gamma <- rgamma(n, shape = 1/alpha, rate = 1)
U1 <- runif(n)
U2 <- runif(n)

tau_laplace <- function(s, alpha) (1 + s)^(-1/alpha)

u1 <- tau_laplace(-log(U1) / gamma, alpha)
u2 <- tau_laplace(-log(U2) / gamma, alpha)

x1 <- qexp(u1)
x2 <- qexp(u2)

cat(sprintf("\n脆弱性模型模拟 (alpha=%.1f):\n", alpha))
cat(sprintf("  Kendall tau (经验) = %.4f\n", cor(x1, x2, method = "kendall")))
cat(sprintf("  Kendall tau (理论) = %.4f\n", alpha / (alpha + 2)))

# 3. 与直接使用rCopula的结果对比
cop_clayton <- claytonCopula(alpha)
u_direct <- rCopula(n, cop_clayton)
x1_direct <- qexp(u_direct[, 1])
x2_direct <- qexp(u_direct[, 2])

cat(sprintf("\nrCopula直接模拟 (alpha=%.1f):\n", alpha))
cat(sprintf("  Kendall tau (经验) = %.4f\n", cor(x1_direct, x2_direct, method = "kendall")))


##############################################################################
# 8.3.1 高斯Copula的模拟
##############################################################################
set.seed(123)

# 二元高斯Copula模拟
rho <- 0.6
cop_gauss <- normalCopula(rho, dim = 2)

# 方法1：使用rCopula函数直接模拟
n <- 2000
u <- rCopula(n, cop_gauss)
x1 <- qexp(u[, 1])
x2 <- qexp(u[, 2])

cat("二元高斯Copula模拟 (rho=0.6):\n")
cat("  Kendall tau (经验) =", round(cor(x1, x2, method = "kendall"), 4), "\n")
cat("  Kendall tau (理论) =", round(2/pi * asin(rho/2), 4), "\n")

# 方法2：手动Cholesky分解
Sigma <- matrix(c(1, rho, rho, 1), 2, 2)
B <- chol(Sigma)
Y <- matrix(rnorm(n * 2), n, 2)
Z <- Y %*% B
U_chol <- pnorm(Z)
x1_chol <- qexp(U_chol[, 1])
x2_chol <- qexp(U_chol[, 2])

cat("\nCholesky分解法模拟:\n")
cat("  Kendall tau (经验) =", round(cor(x1_chol, x2_chol, method = "kendall"), 4), "\n")

# 多元高斯Copula模拟 (d=3)
rho_mat <- matrix(c(1, 0.5, 0.3,
                     0.5, 1, 0.4,
                     0.3, 0.4, 1), 3, 3)
cop_multi <- normalCopula(P2p(rho_mat), dim = 3, dispstr = "un")
u_multi <- rCopula(n, cop_multi)

x1_m <- qexp(u_multi[, 1])
x2_m <- qgamma(u_multi[, 2], shape = 2)
x3_m <- qlnorm(u_multi[, 3], 0, 1)

cat("\n三元高斯Copula模拟:\n")
print(round(cor(cbind(x1_m, x2_m, x3_m)), 4))

# 从相依性测度反推rho
tau_target <- 0.4
rho_from_tau <- sin(pi * tau_target / 2) * 2
cat(sprintf("\n从Kendall tau=%.2f反推rho=%.4f\n", tau_target, rho_from_tau))

rho_target <- 0.5
rho_from_spearman <- 2 * sin(pi * rho_target / 6)
cat(sprintf("从Spearman rho=%.2f反推rho=%.4f\n", rho_target, rho_from_spearman))


##############################################################################
# 8.3.2 阿基米德Copula的模拟
##############################################################################
set.seed(2024)
n <- 2000

# 方法1：条件分布法（Clayton Copula）
alpha <- 2
u1 <- runif(n)
t <- runif(n)
u2 <- (1 - u1^(-alpha) + u1^(-alpha) * t^(-alpha/(1+alpha)))^(-1/alpha)

x1_clayton <- qexp(u1)
x2_clayton <- qexp(u2)

cat("Clayton Copula模拟 (条件分布法, alpha=2):\n")
cat("  Kendall tau (经验) =", round(cor(x1_clayton, x2_clayton, method = "kendall"), 4), "\n")
cat("  Kendall tau (理论) =", round(alpha / (alpha + 2), 4), "\n")

# 方法1b：Frank Copula条件分布法
alpha_f <- 5
u1_f <- runif(n)
t_f <- runif(n)
u2_f <- -1/alpha_f * log(1 + (1 - exp(-alpha_f)) * t_f /
         (exp(-alpha_f * u1_f) - t_f * (exp(-alpha_f * u1_f) - 1) -
          exp(-alpha_f) + t_f * (exp(-alpha_f) - 1)))

x1_frank <- qexp(u1_f)
x2_frank <- qexp(u2_f)

cat("\nFrank Copula模拟 (条件分布法, alpha=5):\n")
cat("  Kendall tau (经验) =", round(cor(x1_frank, x2_frank, method = "kendall"), 4), "\n")

# 方法2：脆弱性模型法
gamma <- rgamma(n, shape = 1/alpha, rate = 1)
U1 <- runif(n)
U2 <- runif(n)

u1_frail <- tau_laplace(-log(U1) / gamma, alpha)
u2_frail <- tau_laplace(-log(U2) / gamma, alpha)

x1_frail <- qexp(u1_frail)
x2_frail <- qexp(u2_frail)

cat("\nClayton Copula模拟 (脆弱性模型法, alpha=2):\n")
cat("  Kendall tau (经验) =", round(cor(x1_frail, x2_frail, method = "kendall"), 4), "\n")

# 方法3：rCopula函数直接模拟
cop_clayton <- claytonCopula(alpha, dim = 2)
u_r <- rCopula(n, cop_clayton)
x1_r <- qexp(u_r[, 1])
x2_r <- qexp(u_r[, 2])

cat("\nClayton Copula模拟 (rCopula函数, alpha=2):\n")
cat("  Kendall tau (经验) =", round(cor(x1_r, x2_r, method = "kendall"), 4), "\n")

# Gumbel Copula模拟
alpha_g <- 3
cop_gumbel <- gumbelCopula(alpha_g, dim = 2)
u_gumbel <- rCopula(n, cop_gumbel)
x1_gumbel <- qexp(u_gumbel[, 1])
x2_gumbel <- qexp(u_gumbel[, 2])

cat("\nGumbel Copula模拟 (rCopula函数, alpha=3):\n")
cat("  Kendall tau (经验) =", round(cor(x1_gumbel, x2_gumbel, method = "kendall"), 4), "\n")
cat("  Kendall tau (理论) =", round(1 - 1/alpha_g, 4), "\n")

# 比较三种方法的效率
cat("\n三种模拟方法比较 (Clayton, alpha=2):\n")
cat("  条件分布法: tau =", round(cor(x1_clayton, x2_clayton, method = "kendall"), 4), "\n")
cat("  脆弱性模型法: tau =", round(cor(x1_frail, x2_frail, method = "kendall"), 4), "\n")
cat("  rCopula函数: tau =", round(cor(x1_r, x2_r, method = "kendall"), 4), "\n")