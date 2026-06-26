# Chap7 R代码
# 自动从chap7.html同步生成

mlogit <- function(exposure = 1){
  linkfun <- function(mu){
    eta <- if(any(exposure-mu<=0)) log(mu/abs(mu-exposure)) else log(mu/(exposure-mu))
    eta
  }
  linkinv <- function(eta){
    thresh <- -log(.Machine$double.eps)
    eta <- pmin(thresh,pmax(eta,-thresh))
    exposure * exp(eta) / (1+exp(eta))
  }
  mu.eta <- function(eta){
    thresh <- -log(.Machine$double.eps)
    res <- rep(.Machine$double.eps,length(eta))
    res[abs(eta)<thresh] <- (exposure * exp(eta) / (1 + exp(eta))^2)[abs(eta) < thresh]
    res
  }
  valideta <- function(eta) TRUE
  link <- paste("logexp(",exposure,")",sep="") 
  structure(list(linkfun=linkfun,linkinv=linkinv,mu.eta=mu.eta,valideta=valideta,name=link), class="link-glm")
}

library(goftest)
library(insuranceData)
library(CASdatasets)data("ukautocoll") # 使用CASdatasets的英国车险数据集
head(ukautocoll) # 自变量为 Age（8个水平）和 Vehicle_Use（4个水平）
##     Age Vehicle_Use Severity Claim_Count
## 1 17-20    Pleasure   250.48          21
## 2 17-20  DriveShort   274.78          40
## 3 17-20   DriveLong   244.52          23
## 4 17-20    Business   797.80           5
## 5 21-24    Pleasure   213.71          63
## 6 21-24  DriveShort   298.60         171

## 定义泊松回归模型的似然函数
loglikelihood <- function(pars,y,X){
  beta <- pars[1:ncol(X)]
  mu <- exp(X
  lnL <- sum(dpois(y,lambda=mu,log=T))
  return(lnL)
}
## 使用optim函数，进行极大似然估计
# 共11个参数，1个截距项+7个年龄水平+3个车辆用途水平，初始值设为2
# fnscale设为-1，表示求极大化问题
beta.hat <- optim(fn=loglikelihood,par=rep(2,11),
                  y=ukautocoll$Claim_Count,
                  X=model.matrix(~Age+Vehicle_Use,data=ukautocoll),
                  control=list(fnscale=-1),hessian=T)
beta.hat$par # 展示回归系数估计结果

library(goftest)
library(insuranceData)
library(CASdatasets)data(dataCar); dataCar <- dataCar[,c(3,4,5,2,8,10,7)]
dataCar$agecat <- factor(dataCar$agecat,levels=c(1:6))
dataCar$veh_age <- factor(dataCar$veh_age,levels=c(1:4))
#   clm numclaims claimcst0  exposure gender agecat veh_age
# 1   0         0         0 0.3039014      F      2       3
# 2   0         0         0 0.6488706      F      4       2
# 3   0         0         0 0.5694730      F      2       2
# 4   0         0         0 0.3175907      F      2       2
# 5   0         0         0 0.6488706      F      2       4
# 6   0         0         0 0.8542094      M      4       3

## 索赔次数：泊松回归模型
Poi.mod <- glm(numclaims~gender+agecat+veh_age, offset=log(exposure), family=poisson(link="log"), data=dataCar)
summary(Poi.mod)
#             Estimate Std. Error z value Pr(>|z|)    
# (Intercept) -1.55595    0.05378 -28.930  < 2e-16 ***
# genderM     -0.01632    0.02888  -0.565 0.572116    
# agecat2     -0.16038    0.05395  -2.973 0.002952 ** 
# agecat3     -0.21421    0.05247  -4.083 4.45e-05 ***
# agecat4     -0.24660    0.05248  -4.699 2.61e-06 ***
# agecat5     -0.46389    0.05875  -7.895 2.90e-15 ***
# agecat6     -0.45607    0.06694  -6.813 9.56e-12 ***
# veh_age2     0.04419    0.04338   1.019 0.308364    
# veh_age3    -0.07670    0.04285  -1.790 0.073440 .  
# veh_age4    -0.14581    0.04408  -3.308 0.000941 ***

# 剔除不显著的gender变量，重新建模
Poi.mod1 <- glm(numclaims~agecat+veh_age, offset=log(exposure),
                family=poisson(link="log"), data=dataCar)
summary(Poi.mod1)
#             Estimate Std. Error z value Pr(>|z|)    
# (Intercept) -1.56279    0.05242 -29.813  < 2e-16 ***
# agecat2     -0.15999    0.05395  -2.966 0.003019 ** 
# agecat3     -0.21383    0.05246  -4.076 4.58e-05 ***
# agecat4     -0.24651    0.05247  -4.698 2.63e-06 ***
# agecat5     -0.46439    0.05875  -7.905 2.68e-15 ***
# agecat6     -0.45714    0.06691  -6.832 8.38e-12 ***
# veh_age2     0.04453    0.04337   1.027 0.304554    
# veh_age3    -0.07677    0.04285  -1.792 0.073191 .  
# veh_age4    -0.14687    0.04404  -3.335 0.000854 ***
anova(Poi.mod,Poi.mod1,test="Chisq") 
# 检验p值为0.572，两模型无显著差异

# 剔除veh_age变量，重新建模
Poi.mod2 <- glm(numclaims~agecat, offset=log(exposure),
                family=poisson(link="log"), data=dataCar)
summary(Poi.mod2)
#             Estimate Std. Error z value Pr(>|z|)    
# (Intercept) -1.60458    0.04364 -36.766  < 2e-16 ***
# agecat2     -0.16900    0.05390  -3.136  0.00172 ** 
# agecat3     -0.22507    0.05240  -4.295 1.75e-05 ***
# agecat4     -0.25600    0.05243  -4.883 1.05e-06 ***
# agecat5     -0.47235    0.05872  -8.044 8.68e-16 ***
# agecat6     -0.46832    0.06685  -7.006 2.46e-12 ***
anova(Poi.mod1,Poi.mod2,test="Chisq") 
# 检验p值<0.05，模型2的偏差明显增大，应使用模型1

library(goftest)
library(insuranceData)
library(CASdatasets)Poi.res1 <- residuals(Poi.mod1, type="deviance")
set.seed(111)
ad.test(Poi.res1, null="pnorm", estimated=TRUE) # 偏差残差的正态检验
#   Anderson-Darling test of goodness-of-fit
#   Braun's adjustment using 260 groups
#   Null hypothesis: Normal distribution
#   Parameters assumed to have been estimated from data
# 
# data:  Poi.res1
# Anmax = 78.386, p-value = 0.0005975

qqnorm(Poi.res1); qqline(Poi.res1, col="red") # 绘制残差的正态Q-Q图

library(goftest)
library(insuranceData)
library(CASdatasets)NB.mod <- glm.nb(numclaims~agecat+veh_age+offset(log(exposure)), data=dataCar)
summary(NB.mod)
#             Estimate Std. Error z value Pr(>|z|)    
# (Intercept) -1.55968    0.05376 -29.014  < 2e-16 ***
# agecat2     -0.16375    0.05532  -2.960  0.00308 ** 
# agecat3     -0.21649    0.05378  -4.026 5.69e-05 ***
# agecat4     -0.24952    0.05378  -4.639 3.49e-06 ***
# agecat5     -0.46810    0.06008  -7.791 6.62e-15 ***
# agecat6     -0.46153    0.06837  -6.751 1.47e-11 ***
# veh_age2     0.04660    0.04440   1.050  0.29393    
# veh_age3    -0.07486    0.04381  -1.709  0.08751 .  
# veh_age4    -0.14367    0.04499  -3.193  0.00141 ** 

NB.res <- residuals(NB.mod, type="deviance")
set.seed(111)
ad.test(NB.res, null="pnorm", estimated=TRUE)
#   Anderson-Darling test of goodness-of-fit
#   Braun's adjustment using 260 groups
# 	Null hypothesis: Normal distribution
# 	Parameters assumed to have been estimated from data
# 
# data:  NB.res
# Anmax = 78.668, p-value = 0.0005975

qqnorm(NB.res); qqline(NB.res, col="red")

## 索赔强度：伽马回归模型
loss.dataCar <- subset(dataCar, claimcst0>0)
loss.dataCar$severity <- loss.dataCar$claimcst0 / loss.dataCar$numclaims

GA.mod <- glm(severity~gender+agecat+veh_age, weights=loss.dataCar$numclaims, family=Gamma(link="log"), data=loss.dataCar)
summary(GA.mod)
#             Estimate Std. Error t value Pr(>|t|)    
# (Intercept)  7.66392    0.09878  77.586  < 2e-16 ***
# genderM      0.17375    0.05283   3.289 0.001014 ** 
# agecat2     -0.22295    0.09848  -2.264 0.023632 *  
# agecat3     -0.31614    0.09588  -3.297 0.000983 ***
# agecat4     -0.31700    0.09591  -3.305 0.000956 ***
# agecat5     -0.42079    0.10729  -3.922 8.91e-05 ***
# agecat6     -0.36918    0.12218  -3.022 0.002528 ** 
# veh_age2     0.04328    0.07924   0.546 0.584985    
# veh_age3     0.08014    0.07828   1.024 0.305985    
# veh_age4     0.15264    0.08053   1.896 0.058080 . 

GA.res <- residuals(GA.mod, type="deviance")
set.seed(111)
ad.test(GA.res, null="pnorm", estimated=TRUE)
#   Anderson-Darling test of goodness-of-fit
#   Braun's adjustment using 68 groups
#   Null hypothesis: Normal distribution
#   Parameters assumed to have been estimated from data
# 
# data:  GA.res
# Anmax = 37.194, p-value = 0.0005998

merge(AIC(Poi.mod1,NB.mod),BIC(Poi.mod1,NB.mod))
#   df      AIC      BIC
# 1  9 34840.93 34923.05
# 2 10 34801.65 34892.90