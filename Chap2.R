# Chap2 R代码
# 自动从chap2.html同步生成

# 2.1.1 卷积法计算累积索赔金额的分布
# 对应教材：section2.tex 2.1.2节，例2.1
# -----------------------------------------------------------------

pn <- c(0.3, 0.5, 0.2)              # 索赔次数的概率分布
fx <- c(0.2, 0.4, 0.2, 0.1, 0.1)    # 索赔强度的概率分布
FS <- aggregateDist("convolution", model.freq = pn, model.sev = fx, x.scale = 100)
plot(FS)  # 绘制S的分布函数

## 累积损失的概率
diff(FS)  # 譬如，累积损失等于0的概率为0.408，累积损失等于800的概率为0.02
S <- seq(from = 0, to = 800, by = 100)
cbind(S, pS = diff(FS))  # 展示S取不同值的概率

##############################################################################
# 2.2 复合分布的随机模拟
##############################################################################

pn <- c(0.3,0.5,0.2) # 定义索赔次数的概率分布
fx <- c(0.2,0.4,0.2,0.1,0.1) # 定义索赔强度的概率分布
FS <- aggregateDist("convolution", model.freq=pn, model.sev=fx, x.scale=100)
plot(FS) # 绘制S的分布函数

S <- seq(from=0, to=800, by=100)
cbind(S, pS=diff(FS)) # 展示S取不同值的概率

set.seed(123)
N <- rpois(10000,lambda=2.5) # 模拟10000个场景下的索赔次数
S <- NULL
for(n in N){
  # 在每个场景下，模拟n次索赔的金额并加总
  Xi <- rgamma(n,shape=2,scale=500)
  S <- c(S,sum(Xi))
}
mean(S) # 计算S的期望，结果为2473
var(S) # 计算S的方差，结果为3660613
hist(S,breaks=seq(0,15000,500),freq = F) # S的频率直方图
lines(density(S),col="red") # S的经验密度函数

set.seed(123)
N <- rbinom(10000,size=100,prob=0.01) # 模拟10000个场景下的索赔次数
S <- NULL
for(n in N){
  # 在每个场景下，模拟n次索赔的金额并加总
  Xi <- rgamma(n,shape=10,rate=0.2)
  S <- c(S,sum(Xi))
}
mean(S) # 计算S的期望，结果为49.45
var(S) # 计算S的方差，结果为2631
hist(S,freq = F,ylim=c(0,0.02)) # S的频率直方图
lines(density(S),col="red") # S的经验密度函数

set.seed(123)
N <- rnbinom(10000,size=2.5,prob=0.5) # 模拟10000个场景下的索赔次数
S <- NULL
for(n in N){
  # 在每个场景下，模拟n次索赔的金额并加总
  Xi <- rgamma(n,shape=2,scale=500)
  S <- c(S,sum(Xi))
}
mean(S) # 计算S的期望，结果为2492
var(S) # 计算S的方差，结果为6233531
hist(S,breaks=seq(0,20000,500),freq = F) # S的频率直方图
lines(density(S),col="red") # S的经验密度函数

# 2.4.5 随机模拟求累积损失的分布（例2.7）
# -----------------------------------------------------------------
set.seed(321)  # 设定随机种子
iter <- 10000  # 模拟次数
d <- 250; u <- 1000  # 免赔额和限额
r <- 3; beta <- 2    # 负二项分布的参数
alpha <- 100; theta <- 0.2  # 伽马分布的参数
P <- NULL  # 保险人的年度累积赔款

# 开始模拟
for (i in 1:iter) {
  n <- rnbinom(1, size = r, mu = r * beta)  # 模拟损失次数
  x <- rgamma(n, shape = alpha, rate = theta)  # 模拟每次事故的损失额
  w <- pmin(x, d)  # 保单持有人对每次损失的自负金额
  v <- min(sum(w), u)  # 保单持有人自负的总金额
  S <- sum(x)  # 保单持有人的总损失
  P[i] <- S - v  # 保单持有人的年度累积赔款
}

hist(P, breaks = 50, col = 'grey', prob = TRUE, main = '',
     ylab = '频率', xlab = '累积赔款')
mean(P); quantile(P, 0.95)

##############################################################################
# 2.5 近似计算方法
##############################################################################

set.seed(123)
S <- NULL
# 模拟10000个场景下的个体索赔情况
for(k in 1:10000){
  # 模拟100个个体的索赔次数
  Ni <- c(rbinom(50,size=1,prob=0.1),rbinom(50,size=1,prob=0.2))
  # 模拟100个个体的索赔金额
  Xi <- c(rgamma(50,shape=10,rate=0.02),
          rlnorm(50,meanlog=5,sdlog=1))
  # 模拟100个个体的索赔情况，求和计算总索赔金额
  Yi <- Ni*Xi
  S <- c(S,sum(Yi))
}
mean(S) # 计算S的期望，结果为4967
var(S) # 计算S的方差，结果为2716003
hist(S,freq = F) # S的频率直方图
lines(density(S),col="red") # S的经验密度函数