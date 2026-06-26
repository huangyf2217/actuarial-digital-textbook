# Chap9 R代码
# 自动从chap9.html同步生成

# 9.1.1 VaR和TVaR的计算
# 对应教材：section9.tex 9.1节，例9.1
# -----------------------------------------------------------------
## 假设X~Ga(alpha=2, theta=500)
## 定义法计算VaR和TVaR
VaR <- function(alpha) qgamma(alpha, shape = 2, scale = 500)

TVaR <- function(alpha) {
  ll <- VaR(alpha)
  integrate(f = function(x) {
    x * dgamma(x, shape = 2, scale = 500)
  }, lower = ll, upper = Inf)$value / (1 - alpha)
}

VaR1 <- VaR(0.95)   # 2371.932
TVaR1 <- TVaR(0.95)  # 2958.982

## 随机模拟法
set.seed(111)
x <- rgamma(10000, shape = 2, scale = 500)
VaR2 <- quantile(x, 0.95)       # 2375.909
TVaR2 <- mean(x[x >= VaR2])     # 2928.817

##############################################################################
# 9.2 区块最大值模型
##############################################################################

## 定义法
VaR <- function(alpha) qgamma(alpha,shape=2,scale=500)
TVaR <- function(alpha){
  ll <- VaR(alpha)
  integrate(f = function(x){
    x*dgamma(x,shape=2,scale=500)
  },lower=ll,upper=Inf)$value / (1-alpha)
}
VaR1 <- VaR(0.95) # 2371.932
TVaR1 <- TVaR(0.95) # 2958.982

## 随机模拟法
set.seed(111)
x <- rgamma(10000,shape=2,scale=500)
VaR2 <- quantile(x,0.95) # 2375.909
TVaR2 <- mean(x[x>=VaR2]) # 2928.817

library(evir)mu <- 0; sigma = 1; alpha = 1
x <- seq(-5,5,0.01)
## 定义极值Weibull分布的密度函数和分布函数
dwei <- Vectorize(function(x,mu,sigma,alpha){
  if(x<=mu){
    (alpha/sigma) * ((mu-x)/sigma)^(alpha-1) * exp(-((mu-x)/sigma)^alpha)
  }else return(0)
})
pwei <- Vectorize(function(x,mu,sigma,alpha){
  if(x<=mu){
    exp(-(-(x-mu)/sigma)^alpha)
  }else return(1)
})
## 概率密度函数比较
gx.gumbel <- dgumbel(x,loc=mu,scale=sigma)
gx.frechet <- dfrechet(x,loc=mu,scale=sigma,shape=alpha)
gx.weibull <- dwei(x,mu,sigma,alpha)
plot(x,gx.gumbel,type="l",lty=1,col="black",
     ylim=c(0,1),ylab="g(x)",main="概率密度函数")
lines(x,gx.frechet,lty=2,col = "red")
lines(x,gx.weibull,lty=3,col = "blue")
## 分布函数比较
Gx.gumbel <- pgumbel(x,loc=mu,scale=sigma)
Gx.frechet <- pfrechet(x,loc=mu,scale=sigma,shape=alpha)
Gx.weibull <- pwei(x,mu,sigma,alpha)
plot(x,Gx.gumbel,type="l",lty=1,col="black",
     ylim=c(0,1),ylab="G(x)",main="累积分布函数")
lines(x,Gx.frechet,lty=2,col = "red")
lines(x,Gx.weibull,lty=3,col = "blue")

library(evir)
# 9.2.2 例9.6：降雨量数据的区块最大值模型
# 对应教材：section9.tex 9.2.4节，例9.6
# -----------------------------------------------------------------
## 模拟50年的每日降雨量数据
set.seed(123)
daily_rainy <- rbinom(50 * 365, size = 1, prob = 0.3)  # 每日是否下雨
daily_rainfall <- daily_rainy * rgamma(50 * 365, 2, 0.1)  # 每日降雨量
yearly_rainfall <- matrix(daily_rainfall, nrow = 50, byrow = TRUE)
annual_maxima <- apply(yearly_rainfall, 1, max)  # 提取区块最大值

## 拟合广义极值分布

gev_fit <- fgev(annual_maxima)
cbind(gev_fit$estimate, gev_fit$std.err)
#             [,1]       [,2]
# loc   64.6352122 1.49018255
# scale  9.5418725 1.09663900
# shape  0.0867381 0.09057615  # 形状参数大于0，属于Fréchet分布

loc.fit <- gev_fit$estimate["loc"]
scale.fit <- gev_fit$estimate["scale"]
shape.fit <- gev_fit$estimate["shape"]

## 拟合优度检验：残差分析
qqplot(annual_maxima,
       qgev(ppoints(50), loc = loc.fit, scale = scale.fit, shape = shape.fit),
       xlab = "Theoretical Quantiles", ylab = "Sample Quantiles")
abline(0, 1, col = "red")

## 经验分布与拟合分布比较
hist(annual_maxima, breaks = 10, probability = TRUE)
curve(dgev(x, loc = loc.fit, scale = scale.fit, shape = shape.fit),
      add = TRUE, col = "red", lwd = 2)

## 计算100年一遇事件的重现水平
return_level <- qgev(1 - 1/100, loc = loc.fit, scale = scale.fit, shape = shape.fit)
return_level  # 118.5771

##############################################################################
# 9.3 超阈值模型
##############################################################################

library(evir)
## 模拟50年的每日降雨量数据
set.seed(123)
daily_rainy <- rbinom(50*365,size=1,prob=0.3) # 每日是否下雨
daily_rainfall <- daily_rainy * rgamma(50*365,2,0.1) # 每日降雨量
yearly_rainfall <- matrix(daily_rainfall,nrow=50,byrow=TRUE)
annual_maxima <- apply(yearly_rainfall,1,max) # 提取区块最大值

## 拟合广义极值分布

gev_fit <- fgev(annual_maxima)
cbind(gev_fit$estimate,gev_fit$std.err)
#             [,1]       [,2]
# loc   64.6352122 1.49018255
# scale  9.5418725 1.09663900
# shape  0.0867381 0.09057615 # 形状参数大于0，属于Fréchet分布
loc.fit <- gev_fit$estimate["loc"]
scale.fit <- gev_fit$estimate["scale"]
shape.fit <- gev_fit$estimate["shape"]

## 拟合优度检验
# 残差分析
qqplot(annual_maxima,qgev(ppoints(50),loc=loc.fit,scale=scale.fit,shape=shape.fit),xlab="Theoretical_Quantiles",ylab="Sample_Quantiles"); abline(0, 1, col = "red")
# 经验分布与拟合分布比较
hist(annual_maxima,breaks=10,probability=TRUE)
curve(dgev(x,loc=loc.fit,scale=scale.fit,shape=shape.fit),add=TRUE,col="red",lwd=2)

## 计算100年一遇事件的重现水平
return_level <- qgev(1-1/100,loc=loc.fit,scale=scale.fit,shape=shape.fit)
return_level # 118.5771

library(evir)
## 广义帕累托概率密度函数：改变尺度参数
x <- seq(0.001,5,0.001)
plot(x,dgpd(x,scale=1,shape=1),type="l",col=1,lty=1,xlab="x",ylab="f(x)")
lines(x,dgpd(x,scale=2,shape=1),col=2,lty=2)
lines(x,dgpd(x,scale=5,shape=1),col=3,lty=3)
lines(x,dgpd(x,scale=10,shape=1),col=4,lty=4)

## 广义帕累托概率密度函数：改变形状参数
plot(x,dgpd(x,scale=1,shape=1),type="l",col=1,lty=1,xlab="x",ylab="f(x)")
lines(x,dgpd(x,scale=1,shape=5),col=2,lty=2)
lines(x,dgpd(x,scale=1,shape=-0.2),col=3,lty=3)
lines(x,dgpd(x,scale=1,shape=-0.5),col=4,lty=4)

library(evir)
# 9.3.2 例9.7：丹麦火灾数据的POT模型
# 对应教材：section9.tex 9.3.5节，例9.7
# -----------------------------------------------------------------

data("danish")

## 绘制经验平均超越函数
cal.en <- Vectorize(function(v) {
  mean(danish[danish > v] - v)
})
v <- seq(0, 50, 1)
en.v <- cal.en(v)
plot(v, en.v, ylab = "经验平均超越函数", main = "经验平均超越函数")

## 绘制Hill图
cal.Hill <- Vectorize(function(k) {
  x <- sort(danish, decreasing = TRUE)
  mean(log(x[1:k])) - log(x[k + 1])
})
k <- seq(1, length(danish) - 1, 10)
ksai.k <- cal.Hill(k)
plot(k, ksai.k, ylab = "Hill估计量", type = "l", main = "Hill图")
sort(danish, decreasing = TRUE)[500]  # 3.135

## 确定阈值后，分别估计两个部分的参数
u <- 10
x_low <- danish[danish <= u]
y_high <- danish[danish > u] - u
p.hat <- length(x_low) / length(danish)  # 0.9497

## 估计阈值以下部分的伽马分布参数
loglik.x <- function(param, x, u) {
  shape <- param[1]  # alpha
  scale <- param[2]  # theta
  li <- dgamma(x, shape = shape, scale = scale) /
        pgamma(u, shape = shape, scale = scale)
  sum(log(li))
}

gamma.param <- constrOptim(
  theta = c(1, 1), f = loglik.x, grad = NULL,
  ui = diag(2), ci = c(0, 0), x = x_low, u = u,
  control = list(fnscale = -1))$par
# alpha.hat=3.396, theta.hat=0.674

## 估计阈值以上超越值的GPD参数
loglik.y <- function(param, y) {
  scale <- param[1]  # sigma
  shape <- param[2]  # xi
  sum(evd::dgpd(y, scale = scale, shape = shape, log = TRUE))
}

gpd.param <- constrOptim(
  theta = c(10, 0), f = loglik.y, grad = NULL,
  ui = c(1, 0), ci = c(0), y = y_high,
  control = list(fnscale = -1))$par
# sigma.hat=6.975, xi.hat=0.497

## 合并两部分的概率密度函数
hx <- Vectorize(function(x) {
  if (x > 0 & x <= u) {
    p.hat * dgamma(x, shape = gamma.param[1], scale = gamma.param[2]) /
            pgamma(u, shape = gamma.param[1], scale = gamma.param[2])
  } else if (x > u) {
    (1 - p.hat) * evd::dgpd(x - u, scale = gpd.param[1], shape = gpd.param[2])
  }
})

## 绘图比较
par(mfrow = c(1, 2))
hist(danish, breaks = 100, freq = FALSE, xlim = c(0, 10), ylim = c(0, 0.35),
     main = "danish[0-10]")
lines(seq(0.1, 10, 0.1), hx(seq(0.1, 10, 0.1)), col = "red")

hist(danish, breaks = 100, freq = FALSE, xlim = c(10, 100), ylim = c(0, 0.015),
     main = "danish[10-100]")
lines(seq(10.1, 100, 0.1), hx(seq(10.1, 100, 0.1)), col = "red")
par(mfrow = c(1, 1))

##############################################################################
# 9.4 广义极值分布的性质
##############################################################################

library(evir)data("danish")

## 绘制经验平均超越函数
cal.en <- Vectorize(function(v){mean(danish[danish>v]-v)})
v <- seq(0,50,1)
en.v <- cal.en(v)
plot(v,en.v,ylab="经验平均超越函数")

## 绘制Hill图
cal.Hill <- Vectorize(function(k){
  x <- sort(danish,decreasing=TRUE)
  mean(log(x[1:k])) - log(x[k+1])
})
k <- seq(1,length(danish)-1,10)
ksai.k <- cal.Hill(k)
plot(k,ksai.k,ylab="Hill估计量",type="l")
sort(danish,decreasing=TRUE)[500] # 3.135

## 确定阈值后，分别估计两个部分的参数
u <- 10
x <- danish[danish<=u]
y <- danish[danish>u]-u
p.hat <- length(x)/length(danish) # 0.9497

loglik.x <- function(param,x,u){
  shape <- param[1] # alpha
  scale <- param[2] # theta
  li <- dgamma(x,shape=shape,scale=scale)/pgamma(u,shape=shape,scale=scale)
  sum(log(li))
}
gamma.param <- constrOptim(theta=c(1,1),f=loglik.x,grad=NULL,ui=diag(2),ci=c(0,0),x=x,u=u,control=list(fnscale=-1))$par
# alpha.hat=3.396, theta.hat=0.674

loglik.y <- function(param,y){
  scale <- param[1] # sigma
  shape <- param[2] # ksai
  sum(evd::dgpd(y,scale=scale,shape=shape,log=TRUE))
}
gpd.param <- constrOptim(theta=c(10,0),f=loglik.y,grad=NULL,ui=c(1,0),ci=c(0),y=y,control=list(fnscale=-1))$par
# sigma.hat=6.975, ksai.hat=0.497

## 合并两部分的概率密度函数
hx <- Vectorize(function(x){
  if(x>0&x<=u){
    p.hat*dgamma(x,shape=gamma.param[1],scale=gamma.param[2])/pgamma(u,shape=gamma.param[1],scale=gamma.param[2])
  }else if(x>u){
    (1-p.hat)*dgpd(x-u,scale=gpd.param[1],shape=gpd.param[2])
  }
})
hist(danish,breaks=100,freq=F,xlim=c(0,10),ylim=c(0,0.35),main="danish[0-10]")
lines(seq(0.1,10,0.1),hx(seq(0.1,10,0.1)),col="red")
hist(danish,breaks=100,freq=F,xlim=c(10,100),ylim=c(0,0.015),main="danish[10-100]")
lines(seq(10.1,100,0.1),hx(seq(10.1,100,0.1)),col="red")