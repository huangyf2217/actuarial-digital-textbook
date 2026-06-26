# Chap10 R代码
# 自动从chap10.html同步生成

# 10.1.1 例10.1：鸢尾花数据集的主成分分析
# 对应教材：section10.tex 10.2.1节，例10.1
# -----------------------------------------------------------------
data(iris)
pca_model <- prcomp(iris[, 1:4], scale = TRUE)  # 标准化后进行主成分分析
summary(pca_model)
#                           PC1    PC2     PC3     PC4
# Standard deviation     1.7084 0.9560 0.38309 0.14393
# Proportion of Variance 0.7296 0.2285 0.03669 0.00518
# Cumulative Proportion  0.7296 0.9581 0.99482 1.00000

pca_model
# Rotation (n x k) = (4 x 4): # 特征矩阵
#                     PC1         PC2        PC3        PC4
# Sepal.Length  0.5210659 -0.37741762  0.7195664  0.2612863
# Sepal.Width  -0.2693474 -0.92329566 -0.2443818 -0.1235096
# Petal.Length  0.5804131 -0.02449161 -0.1421264 -0.8014492
# Petal.Width   0.5648565 -0.06694199 -0.6342727  0.5235971

## 提取样本在每个主成分上的得分值：两种方法结果一致
X.pca1 <- as.matrix(scale(iris[, 1:4])) %*% pca_model$rotation
X.pca2 <- pca_model$x

## 绘制前2个主成分与因变量的关系
par(mfrow = c(1, 2))
plot(pca_model$x[, 1], col = iris[, 5], ylab = "PC1")
plot(pca_model$x[, 2], col = iris[, 5], ylab = "PC2")
par(mfrow = c(1, 1))

##############################################################################
# 10.2 无监督学习：因子分析
##############################################################################

data(iris)
pca_model <- prcomp(iris[,1:4],scale=TRUE) # 标准化后进行主成分分析
summary(pca_model)
#                           PC1    PC2     PC3     PC4
# Standard deviation     1.7084 0.9560 0.38309 0.14393
# Proportion of Variance 0.7296 0.2285 0.03669 0.00518
# Cumulative Proportion  0.7296 0.9581 0.99482 1.00000
pca_model
# Rotation (n x k) = (4 x 4): # 特征矩阵
#                     PC1         PC2        PC3        PC4
# Sepal.Length  0.5210659 -0.37741762  0.7195664  0.2612863
# Sepal.Width  -0.2693474 -0.92329566 -0.2443818 -0.1235096
# Petal.Length  0.5804131 -0.02449161 -0.1421264 -0.8014492
# Petal.Width   0.5648565 -0.06694199 -0.6342727  0.5235971

## 提取样本在每个主成分上的得分值：两种方法结果一致
X.pca1 <- as.matrix(scale(iris[,1:4])) 
X.pca2 <- pca_model$x

## 绘制前2个主成分与因变量的关系：可以发现，第一个主成分已经能够解释因变量和自变量的变化规律
par(mfrow = c(1, 2))
plot(pca_model$x[,1],col=iris[,5],ylab = 'PC1')
plot(pca_model$x[,2],col=iris[,5],ylab = 'PC2')

library(Metrics)
library(ModelMetrics)
library(caret)
library(data.table)
library(ggplot2)
library(glmnet)
library(gridExtra)
library(lightgbm)
library(pdp)
library(psych)
library(randomForest)
library(reshape2)
library(rpart)
library(rpart.plot)
# 10.2.1 例10.2：五大人格特质数据的因子分析
# 对应教材：section10.tex 10.2.2节，例10.2
# -----------------------------------------------------------------

bfi <- as.data.frame(bfi)[, 1:25]
bfi <- na.omit(bfi)  # 剔除缺失值
bfi <- scale(bfi)    # 变量标准化

## 计算相关系数矩阵的特征值
cor_matrix <- cor(bfi)
eigenvalues <- eigen(cor_matrix)$values
plot(eigenvalues, type = "b", xlab = "m", ylab = "Eigenvalues")  # 碎石图

## 因子分析：未旋转
factanal_result <- factanal(bfi, factors = 5, rotation = "none")
print(factanal_result)  # Uniquenesses表示特殊因子的方差，Loadings表示载荷矩阵

## 因子分析：使用Varimax旋转，并计算因子得分
factanal_rotated <- factanal(bfi, factors = 5, rotation = "varimax", scores = "regression")
print(factanal_rotated)
loadings <- factanal_rotated$loadings  # 载荷矩阵
apply(loadings, 1, function(x) { which.max(abs(x)) })  # 查看特征和因子的对应关系
sample.scores <- factanal_rotated$scores  # 因子得分（个体人格评分）

##############################################################################
# 10.3 无监督学习：聚类分析
##############################################################################

library(Metrics)
library(ModelMetrics)
library(caret)
library(data.table)
library(ggplot2)
library(glmnet)
library(gridExtra)
library(lightgbm)
library(pdp)
library(psych)
library(randomForest)
library(reshape2)
library(rpart)
library(rpart.plot)bfi <- as.data.frame(bfi)[,1:25]
bfi <- na.omit(bfi) # 剔除缺失值
bfi <- scale(bfi) # 变量标准化

## 计算相关系数矩阵的特征值
cor_matrix <- cor(bfi)
eigenvalues <- eigen(cor_matrix)$values
plot(eigenvalues,type="b",xlab="m",ylab="Eigenvalues") # 碎石图

## 因子分析：未旋转
factanal_result <- factanal(bfi,factors=5,rotation="none")
print(factanal_result) # Uniquenesses表示特殊因子的方差，Loadings表示载荷矩阵

## 因子分析：使用Varimax旋转，并计算因子得分
factanal_rotated <- factanal(bfi,factors=5,rotation="varimax",scores="regression")
print(factanal_rotated)
loadings <- factanal_rotated$loadings # 载荷矩阵
apply(loadings,1,function(x){which.max(abs(x))}) # 查看特征和因子的对应关系：含义相似的题项都属于同一因子
sample.scores <- factanal_rotated$scores # 因子得分（个体人格评分）

# 10.3.1 例10.3：鸢尾花数据集的聚类分析
# 对应教材：section10.tex 10.2.3节，例10.3
# -----------------------------------------------------------------
data(iris)
iris_data <- scale(iris[, 1:4])  # 提取自变量并标准化

## K-均值聚类
set.seed(123)
kmeans_result <- kmeans(iris_data, centers = 3)
kmeans_result$centers  # 提取簇的中心
#   Sepal.Length Sepal.Width Petal.Length Petal.Width
# 1  -1.01119138  0.85041372   -1.3006301  -1.2507035
# 2  -0.05005221 -0.88042696    0.3465767   0.2805873
# 3   1.13217737  0.08812645    0.9928284   1.0141287
kmeans_cluster <- kmeans_result$cluster  # 提取聚类结果

## 层次聚类
dist_matrix <- dist(iris_data, method = "euclidean")
hclust_result <- hclust(dist_matrix, method = "complete")
plot(hclust_result, cex = 0.9)  # 绘制层次树
hclust_cluster <- cutree(hclust_result, k = 3)  # 将样本划分成3簇
rbind(apply(iris_data[hclust_cluster == 1, ], 2, mean),
      apply(iris_data[hclust_cluster == 2, ], 2, mean),
      apply(iris_data[hclust_cluster == 3, ], 2, mean))

## 将两种聚类结果与样本的真实类别进行比较
cluster_comp <- cbind(iris$Species, kmeans_cluster, hclust_cluster)

##############################################################################
# 10.4 监督学习：目标函数与评价准则
##############################################################################

data(iris)
iris_data <- scale(iris[,1:4]) # 提取自变量并标准化

## K-均值聚类
set.seed(123)
kmeans_result <- kmeans(iris_data,centers=3)
kmeans_result$centers # 提取簇的中心
#   Sepal.Length Sepal.Width Petal.Length Petal.Width
# 1  -1.01119138  0.85041372   -1.3006301  -1.2507035
# 2  -0.05005221 -0.88042696    0.3465767   0.2805873
# 3   1.13217737  0.08812645    0.9928284   1.0141287
kmeans_cluster <- kmeans_result$cluster # 提取聚类结果

## 层次聚类
dist_matrix <- dist(iris_data,method="euclidean")
hclust_result <- hclust(dist_matrix,method="complete")
plot(hclust_result,cex = 0.9) # 绘制层次树
hclust_cluster <- cutree(hclust_result,k=3) # 将样本划分成3簇，提取聚类结果，并近似计算每个簇的中心
rbind(apply(iris_data[hclust_cluster==1,],2,mean),
      apply(iris_data[hclust_cluster==2,],2,mean),
      apply(iris_data[hclust_cluster==3,],2,mean))
#      Sepal.Length Sepal.Width Petal.Length Petal.Width
# [1,]   -0.9987207   0.9032290  -1.29875725 -1.25214931
# [2,]   -0.3995253  -1.3551557   0.06155712 -0.03738991
# [3,]    0.7600769  -0.1523959   0.80729525  0.80847629

## 将两种聚类结果与样本的真实类别进行比较
cluster_comp <- cbind(iris$Species,kmeans_cluster,hclust_cluster)

# 0-1损失函数
zeroone.loss <- function(y.hat,y) ifelse(y.hat==y,0,1)

# 交叉熵损失函数
cross.entropy <- function(prob,y) -y*log(prob) - (1-y)*log(1-prob)

# Hinge损失函数
Hinge.loss <- function(score,y) max(0, 1-y*scores)

# 平均绝对误差
MAE <- function(y,y.hat) mean(abs(y-y.hat))

# 均方误差
MSE <- function(y,y.hat) mean((y-y.hat)^2)

# Huber损失函数
Huber.loss <- function(y,y.hat,delta){
  if(abs(y-y.hat)<=delta) 0.5*(y-y.hat)^2
  else delta*abs(y-y.hat) -0.5*delta^2
}

# Log-Cosh损失函数
LogCosh.loss <- function(y,y.hat) log(cosh(y.hat-y))

# MSLE
MSLE <- function(y, y.hat) mean((log(1+y)-log(1+y.hat))^2)

library(Metrics)
library(ModelMetrics)
library(caret)
library(data.table)
library(ggplot2)
library(glmnet)
library(gridExtra)
library(lightgbm)
library(pdp)
library(psych)
library(randomForest)
library(reshape2)
library(rpart)
library(rpart.plot)
# 10.5.1 例10.8：岭回归和LASSO回归的比较
# 对应教材：section10.tex 10.3.2节，例10.8
# -----------------------------------------------------------------

## 生成模拟数据
set.seed(221)
X <- matrix(rnorm(1000, mean = 0, sd = 1), nrow = 100, ncol = 10)
colnames(X) <- paste('x', c(1:10), sep = '')
beta <- c(2, -2, 1, -1, 0.5, -0.5, 0, 0, 0, 0)
mu <- X %*% beta
y <- rnorm(100, mean = mu, sd = 1)

## 训练岭回归模型
ridge.mod <- glmnet(x = X, y = y, family = "gaussian", alpha = 0,
                    standardize = FALSE, intercept = FALSE)
ridge.lambda <- ridge.mod$lambda  # 模型尝试的正则参数值
ridge.coef <- t(as.matrix(ridge.mod$beta))  # 每个正则参数对应的结果
ridge.solution.path <- data.table(lambda = ridge.lambda, ridge.coef)

## 训练LASSO回归模型
lasso.mod <- glmnet(x = X, y = y, family = "gaussian", alpha = 1,
                    standardize = FALSE, intercept = FALSE)
lasso.lambda <- lasso.mod$lambda
lasso.coef <- t(as.matrix(lasso.mod$beta))
lasso.solution.path <- data.table(lambda = lasso.lambda, lasso.coef)

## 绘制参数求解路径
ridge.solution.path <- melt(ridge.solution.path, id.vars = 'lambda',
                            measure.vars = paste('x', c(1:10), sep = ''))
p.ridge <- ggplot(data = ridge.solution.path) +
  geom_line(aes(x = lambda, y = value, color = variable)) +
  xlim(0, 100) + ylab('regression_coefficients')

lasso.solution.path <- melt(lasso.solution.path, id.vars = 'lambda',
                            measure.vars = paste('x', c(1:10), sep = ''))
p.lasso <- ggplot(data = lasso.solution.path) +
  geom_line(aes(x = lambda, y = value, color = variable)) +
  xlim(0, 2.5) + ylab('regression_coefficients')

grid.arrange(p.ridge, p.lasso, nrow = 1)

library(Metrics)
library(ModelMetrics)
library(caret)
library(data.table)
library(ggplot2)
library(glmnet)
library(gridExtra)
library(lightgbm)
library(pdp)
library(psych)
library(randomForest)
library(reshape2)
library(rpart)
library(rpart.plot)
## 生成模拟数据
set.seed(221)
X <- matrix(rnorm(1000,mean=0,sd=1),nrow=100,ncol=10)
colnames(X) <- paste('x',c(1:10),sep='')
beta <- c(2,-2,1,-1,0.5,-0.5,0,0,0,0)
mu <- X 
y <- rnorm(100,mean=mu,sd=1)

## 训练岭回归模型
ridge.mod <- glmnet(x=X,y=y,family="gaussian",alpha=0,standardize=F,intercept=F) 
ridge.lambda <- ridge.mod$lambda # 模型尝试的正则参数值
ridge.coef <- t(as.matrix(ridge.mod$beta)) # 每个正则参数对应的结果
ridge.solution.path <- data.table(lambda=ridge.lambda,ridge.coef)

## 训练LASSO回归模型
lasso.mod <- glmnet(x=X,y=y,family="gaussian",alpha=1,standardize=F,intercept=F)
lasso.lambda <- lasso.mod$lambda # 模型尝试的正则参数值
lasso.coef <- t(as.matrix(lasso.mod$beta)) # 每个正则参数对应的结果
lasso.solution.path <- data.table(lambda=lasso.lambda,lasso.coef)

## 绘制参数求解路径
ridge.solution.path <- melt(ridge.solution.path,id.vars='lambda',measure.vars=paste('x',c(1:10),sep=''))
p.ridge <- ggplot(data=ridge.solution.path) + 
  geom_line(aes(x=lambda,y=value,color=variable)) + 
  xlim(0,100) + ylab('regression_coefficients')
lasso.solution.path <- melt(lasso.solution.path,id.vars='lambda',measure.vars= paste('x',c(1:10),sep=''))
p.lasso <- ggplot(data=lasso.solution.path) + 
  geom_line(aes(x=lambda,y=value,color=variable)) + 
  xlim(0, 2.5) + ylab('regression_coefficients')
grid.arrange(p.ridge, p.lasso, nrow = 1)

library(Metrics)
library(ModelMetrics)
library(caret)
library(data.table)
library(ggplot2)
library(glmnet)
library(gridExtra)
library(lightgbm)
library(pdp)
library(psych)
library(randomForest)
library(reshape2)
library(rpart)
library(rpart.plot)
# 10.5.2 例10.9：glmnet建模应用（心血管疾病数据）
# 对应教材：section10.tex 10.3.2节，例10.9
# -----------------------------------------------------------------

## 数据预处理和异常值清洗
## 数据来源：Kaggle Cardiovascular Disease dataset
## https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset
## 若本地已有数据文件 cardiovascular_disease_dataset.csv 则读取之；
## 否则生成符合教材数据结构的模拟数据用于演示代码流程。
data_file <- 'cardiovascular_disease_dataset.csv'
if (file.exists(data_file)) {
  dat <- fread(data_file)
} else {
  warning("未找到 'cardiovascular_disease_dataset.csv'，使用模拟数据演示。\n",
          "如需复现教材结果，请从 Kaggle 下载：\n",
          "https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset\n",
          "下载后将 cardio_train.csv 重命名为 cardiovascular_disease_dataset.csv\n",
          "并放置于工作目录：", getwd())
  set.seed(2024)
  n_sim <- 7000  # 模拟7000条记录，与原数据规模接近
  dat <- data.table::data.table(
    age       = round(runif(n_sim, 14000, 25000)),         # 年龄（天）
    gender    = sample(1:2, n_sim, replace = TRUE),        # 性别
    height    = round(rnorm(n_sim, 170, 8)),               # 身高（cm）
    weight    = round(rnorm(n_sim, 75, 12)),               # 体重（kg）
    ap_hi     = round(rnorm(n_sim, 125, 15)),              # 收缩压
    ap_lo     = round(rnorm(n_sim, 80, 10)),               # 舒张压
    cholesterol = sample(1:3, n_sim, replace = TRUE, prob = c(0.7, 0.2, 0.1)),
    gluc      = sample(1:3, n_sim, replace = TRUE, prob = c(0.8, 0.15, 0.05)),
    smoke     = sample(0:1, n_sim, replace = TRUE, prob = c(0.9, 0.1)),
    alco      = sample(0:1, n_sim, replace = TRUE, prob = c(0.95, 0.05)),
    active    = sample(0:1, n_sim, replace = TRUE, prob = c(0.2, 0.8)),
    cardio    = sample(0:1, n_sim, replace = TRUE, prob = c(0.5, 0.5))
  )
}
dat$age <- dat$age / 365
dat$height <- dat$height / 100
dat <- subset(dat, height >= 1.4 & height <= 2.1)
dat <- subset(dat, weight >= 40 & weight <= 200)
dat <- subset(dat, ap_hi >= 70 & ap_hi <= 240)
dat <- subset(dat, ap_lo >= 30 & ap_lo <= 140)
dat$gender <- factor(dat$gender, levels = c(1, 2))
dat$smoke <- factor(dat$smoke, levels = c(0, 1))
dat$alco <- factor(dat$alco, levels = c(0, 1))
dat$active <- factor(dat$active, levels = c(0, 1))
dat$cardio <- factor(dat$cardio, levels = c(0, 1))
dat$cholesterol <- factor(dat$cholesterol, levels = c(1, 2, 3))
dat$gluc <- factor(dat$gluc, levels = c(1, 2, 3))

## 随机抽取1000个训练集和测试集
set.seed(221)
idx <- sample.int(nrow(dat), 2000)
train_data <- dat[idx[1:1000], ]  # 训练集
X.train <- makeX(train_data[, -12])  # 独热编码
X.train <- scale(X.train)  # 标准化
y.train <- train_data$cardio
test_data <- dat[idx[1001:2000], ]  # 测试集
X.test <- makeX(test_data[, -12])
X.test <- scale(X.test)
y.test <- test_data$cardio

## 交叉验证选择正则参数
cv.mod <- cv.glmnet(x = X.train, y = y.train, family = 'binomial',
                    alpha = 1, standardize = FALSE, intercept = FALSE,
                    type.measure = 'auc', nfolds = 10)
plot(cv.mod)
cv.mod$lambda.min  # 0.008683118
cv.mod$lambda.1se  # 0.0888744

## 在train_data上训练模型
lasso.mod <- glmnet(x = X.train, y = y.train, family = 'binomial', alpha = 1,
                    lambda = cv.mod$lambda.1se, standardize = FALSE, intercept = FALSE)
coef(lasso.mod)  # 回归系数估计值

## 在test_data上进行评价
y.pred <- predict(lasso.mod, newx = X.test, type = 'response')
auc(actual = y.test, predicted = y.pred)  # 0.7685578

##############################################################################
# 10.6 监督学习：决策树、随机森林和梯度提升树
##############################################################################

library(Metrics)
library(ModelMetrics)
library(caret)
library(data.table)
library(ggplot2)
library(glmnet)
library(gridExtra)
library(lightgbm)
library(pdp)
library(psych)
library(randomForest)
library(reshape2)
library(rpart)
library(rpart.plot)
## 数据预处理和异常值清洗
dat <- fread('cardiovascular_disease_dataset.csv')
dat$age <- dat$age/365
dat$height <- dat$height/100
dat <- subset(dat,height>=1.4 & height<=2.1)
dat <- subset(dat, weight>=40 & weight<=200)
dat <- subset(dat, ap_hi>=70 & ap_hi<=240)
dat <- subset(dat, ap_lo>=30 & ap_lo<=140)
dat$gender <- factor(dat$gender,levels=c(1,2))
dat$smoke <- factor(dat$smoke,levels=c(0,1))
dat$alco <- factor(dat$alco,levels=c(0,1))
dat$active <- factor(dat$active,levels=c(0,1))
dat$cardio <- factor(dat$cardio,levels=c(0,1))
dat$cholesterol <- factor(dat$cholesterol,levels=c(1,2,3))
dat$gluc <- factor(dat$gluc,levels=c(1,2,3))

## 随机抽取1000个训练集和测试集
set.seed(221)
idx <- sample.int(nrow(dat),2000)
train_data <- dat[idx[1:1000],] # 训练集
X.train <- makeX(train_data[,-12]) # 独热编码
X.train <- scale(X.train) # 标准化
y.train <- train_data$cardio
test_data <- dat[idx[1001:2000],] # 测试集
X.test <- makeX(test_data[,-12]) # 独热编码
X.test <- scale(X.test) # 标准化
y.test <- test_data$cardio

## 交叉验证选择正则参数
cv.mod <- cv.glmnet(x=X.train,y=y.train,family='binomial',alpha=1,standardize=F,intercept=F,type.measure='auc',nfolds=10)
plot(cv.mod)
cv.mod$lambda.min # 0.008683118
cv.mod$lambda.1se # 0.0888744
## 在train_data上训练模型
lasso.mod <- glmnet(x=X.train,y=y.train,family='binomial',alpha=1,lambda=cv.mod$lambda.1se,standardize=F,intercept=F)
coef(lasso.mod) # 回归系数估计值

## 在test_data上进行评价
y.pred <- predict(lasso.mod,newx=X.test,type='response')
auc(actual=y.test,predicted=y.pred) # 0.7685578

library(Metrics)
library(ModelMetrics)
library(caret)
library(data.table)
library(ggplot2)
library(glmnet)
library(gridExtra)
library(lightgbm)
library(pdp)
library(psych)
library(randomForest)
library(reshape2)
library(rpart)
library(rpart.plot)
# 10.6.1 例10.10：决策树建模示例（GermanCredit数据）
# 对应教材：section10.tex 10.3.3节，例10.10
# -----------------------------------------------------------------

data(GermanCredit)
dat <- GermanCredit[, c(10, 5, 1, 2, 6)]

## 决策树建模
mod <- rpart(Class ~ ., data = dat)
rpart.plot(mod)
sample.probs <- predict(mod, newdata = dat, type = 'prob')  # 类别概率
sample.class <- predict(mod, newdata = dat, type = 'vector')  # 预测类别
table(actual = dat$Class, predicted = sample.class)  # 输出混淆矩阵
#       predicted
# actual   1   2
#   Bad   56 244
#   Good  21 679

library(Metrics)
library(ModelMetrics)
library(caret)
library(data.table)
library(ggplot2)
library(glmnet)
library(gridExtra)
library(lightgbm)
library(pdp)
library(psych)
library(randomForest)
library(reshape2)
library(rpart)
library(rpart.plot)data(GermanCredit)
dat <- GermanCredit[,c(10,5,1,2,6)]

## 决策树建模
mod <- rpart(Class~.,data=dat)
rpart.plot(mod)
sample.probs <- predict(mod,newdata=dat,type='prob') # 类别概率
sample.class <- predict(mod,newdata=dat,type='vector') # 预测类别
table(actual=dat$Class,predicted=sample.class) # 输出混淆矩阵
#       predicted
# actual   1   2
#   Bad   56 244
#   Good  21 679

library(Metrics)
library(ModelMetrics)
library(caret)
library(data.table)
library(ggplot2)
library(glmnet)
library(gridExtra)
library(lightgbm)
library(pdp)
library(psych)
library(randomForest)
library(reshape2)
library(rpart)
library(rpart.plot)
# 10.6.2 例10.11：随机森林建模示例
# 对应教材：section10.tex 10.3.3节，例10.11
# -----------------------------------------------------------------

data(GermanCredit)
dat <- GermanCredit[, c(10, 5, 1, 2, 6)]

## 随机森林建模
set.seed(123)
mod <- randomForest(Class ~ ., data = dat, ntree = 100, replace = TRUE,
                    importance = TRUE, proximity = TRUE)
sample.probs <- predict(mod, newdata = dat, type = 'prob')  # 类别概率
sample.class <- predict(mod, newdata = dat, type = 'response')  # 预测类别
table(actual = dat$Class, predicted = sample.class)  # 混淆矩阵
#       predicted
# actual  Bad Good
#   Bad  297    3
#  Good    1  699

## 提取袋外误差的变化趋势，判断结果稳定性
oob.err <- mod$err.rate[, 1]  # 分类问题默认使用accuracy作为指标
plot(x = c(1:100), y = oob.err, xlab = 'trees', ylab = 'OOB_error', type = 'l')

## 绘制变量重要性图
imp <- importance(mod)  # 提取重要性结果
varImpPlot(mod)  # 绘图

## 绘制样本邻近图
prox.matrix <- mod$proximity  # 提取邻近矩阵
MDSplot(mod, fac = mod$y, k = 2, palette = c('blue', 'red'))  # 绘图

library(Metrics)
library(ModelMetrics)
library(caret)
library(data.table)
library(ggplot2)
library(glmnet)
library(gridExtra)
library(lightgbm)
library(pdp)
library(psych)
library(randomForest)
library(reshape2)
library(rpart)
library(rpart.plot)data(GermanCredit)
dat <- GermanCredit[,c(10,5,1,2,6)]

## 随机森林建模
set.seed(123)
mod <- randomForest(Class~.,data=dat,ntree=100,replace=TRUE,importance=TRUE,proximity=TRUE)
sample.probs <- predict(mod,newdata=dat,type='prob') # 类别概率
sample.class <- predict(mod,newdata=dat,type='response') # 预测类别
table(actua=dat$Class,predicted=sample.class) # 混淆矩阵
#       predicted
# actua  Bad Good
#   Bad  297    3
#  Good    1  699

# 提取袋外误差的变化趋势，判断结果稳定性
oob.err <- mod$err.rate[,1] # 分类问题默认使用accuracy作为指标
plot(x=c(1:100),y=oob.err,xlab='trees',ylab='OOB_error',type='l')

# 绘制变量重要性图
imp <- importance(mod) # 提取重要性结果
varImpPlot(mod) # 绘图

# 绘制样本邻近图
prox.matrix = mod$proximity # 提取邻近矩阵
MDSplot(mod,fac=mod$y,k=2,palette=c('blue','red')) # 绘图

library(Metrics)
library(ModelMetrics)
library(caret)
library(data.table)
library(ggplot2)
library(glmnet)
library(gridExtra)
library(lightgbm)
library(pdp)
library(psych)
library(randomForest)
library(reshape2)
library(rpart)
library(rpart.plot)
# 10.6.3 例10.12：提升树建模示例
# 对应教材：section10.tex 10.3.3节，例10.12
# -----------------------------------------------------------------

data(GermanCredit)
dat <- GermanCredit[, c(10, 5, 1, 2, 6)]

## 梯度提升树建模
dtrain <- lgb.Dataset(
  data = as.matrix(dat[, 2:5]),
  label = as.numeric(dat$Class) - 1
)
mod <- lgb.train(data = dtrain, nrounds = 100, obj = 'binary',
                 params = list(learning_rate = 0.1))
sample.probs <- predict(mod, as.matrix(dat[, 2:5]))  # 类别概率
sample.class <- predict(mod, as.matrix(dat[, 2:5]), type = 'class')  # 预测类别
sample.class <- factor(sample.class, labels = levels(dat$Class))
table(actual = dat$Class, predicted = sample.class)  # 混淆矩阵
#       predicted
# actual Bad Good
#   Bad  250   50
#   Good  15  685

## 绘制变量重要性图
imp.data <- lgb.importance(mod, percentage = TRUE)  # 提取重要性结果
lgb.plot.importance(imp.data, measure = 'Gain')  # 绘图

##############################################################################
# 10.7 模型选择与应用：交叉验证与超参数调优
##############################################################################

library(Metrics)
library(ModelMetrics)
library(caret)
library(data.table)
library(ggplot2)
library(glmnet)
library(gridExtra)
library(lightgbm)
library(pdp)
library(psych)
library(randomForest)
library(reshape2)
library(rpart)
library(rpart.plot)data(GermanCredit)
dat <- GermanCredit[,c(10,5,1,2,6)]

## 梯度提升树建模
dtrain <- lgb.Dataset(
  data = as.matrix(dat[,2:5]), 
  label = as.numeric(dat$Class)-1
)
mod <- lgb.train(data=dtrain,nrounds=100,obj='binary',params=list(learning_rate=0.1))
sample.probs <- predict(mod,as.matrix(dat[,2:5])) # 类别概率
sample.class <- predict(mod,as.matrix(dat[,2:5]),type='class') # 预测类别
sample.class <- factor(sample.class,labels=levels(dat$Class))
table(actual=dat$Class,predicted=sample.class) # 混淆矩阵
#       predicted
# actual Bad Good
#   Bad  250   50
#   Good  15  685

# 绘制变量重要性图
imp.data <- lgb.importance(mod,percentage=T) # 提取重要性结果
lgb.plot.importance(imp.data,measure='Gain') # 绘图

library(Metrics)
library(ModelMetrics)
library(caret)
library(data.table)
library(ggplot2)
library(glmnet)
library(gridExtra)
library(lightgbm)
library(pdp)
library(psych)
library(randomForest)
library(reshape2)
library(rpart)
library(rpart.plot)
## 生成模拟数据
set.seed(221)
X <- matrix(rnorm(10000*10,mean=0,sd=1),nrow=10000,ncol=10)
colnames(X) <- paste('x',c(1:10),sep='')
beta <- c(2,-2,1,-1,0.5,-0.5,0,0,0,0)
mu <- X 
y <- rnorm(10000,mean=mu,sd=1)
dat <- data.frame(cbind(y,X))

## 双重交叉验证
tune.lambdas <- seq(0,0.02,0.001) # 超参数的范围
outer_folds <- createFolds(y=dat$y,k=5) # 确定外层数据划分
for(k in 1:5){
  train_data <- dat[-outer_folds[[k]],] # 训练集
  test_data <- dat[outer_folds[[k]],] # 测试集
  inner_folds <- createFolds(y=train_data$y,k=5) # 确定内层数据划分

  # 开始内循环，记录每个超参数的结果
  rmse.metrics <- matrix(0,nrow=5,ncol=length(tune.lambdas))
  for(m in 1:5){
    calib_data <- train_data[-inner_folds[[m]],] # 校准集
    valid_data <- train_data[inner_folds[[m]],] # 验证集
    
    for(i in 1:length(tune.lambdas)){
      # 尝试不同超参数
      mod <- glmnet(x=calib_data[,-1],y=calib_data$y,family='gaussian',alpha=1,lambda=tune.lambdas[i],standardize=F,intercept=F)
      # 计算模型在验证集上的表现
      valid.y.pred <- as.matrix(valid_data[,-1]) 
      rmse.metrics[m,i] <- rmse(actual=valid_data$y,predicted=valid.y.pred)
    }
  }
  
  # 寻找内层交叉验证指标最优的超参数
  tune.lambdas.rmse <- apply(rmse.metrics,2,mean)
  choose.lambda <- tune.lambdas[which.min(tune.lambdas.rmse)]
  
  # 应用最优超参数，重新训练模型并评价
  mod <- glmnet(x=train_data[,-1],y=train_data$y,family='gaussian',alpha=1,lambda=choose.lambda,standardize=F,intercept=F)
  test.y.pred <- as.matrix(test_data[,-1]) 
  test.rmse <- rmse(actual=test_data$y,predicted=test.y.pred)
  print(paste('outer-loop',k,'choose-lambda',choose.lambda,'test-rmse',round(test.rmse,2)))
}

library(Metrics)
library(ModelMetrics)
library(caret)
library(data.table)
library(ggplot2)
library(glmnet)
library(gridExtra)
library(lightgbm)
library(pdp)
library(psych)
library(randomForest)
library(reshape2)
library(rpart)
library(rpart.plot)
# 10.8.1 部分依赖图（PDP）示例
# 对应教材：section10.tex 10.4.2节，例10.14
# -----------------------------------------------------------------

## 模拟数据：自变量独立情况
set.seed(221)
n <- 1000
x1 <- runif(n, 0, 2 * pi)
x2 <- runif(n, 0, 2 * pi)
x3 <- runif(n, 0, 2 * pi)
y_indep <- sin(3 * x1) + x2 + x3 + rnorm(n, 0, 0.1)
dat_indep <- data.frame(y = y_indep, x1 = x1, x2 = x2, x3 = x3)

## 训练随机森林模型
rf_indep <- randomForest(y ~ ., data = dat_indep, ntree = 100)

## 绘制部分依赖图
par(mfrow = c(1, 3))
partialPlot(rf_indep, pred.data = dat_indep, x.var = "x1", main = "x1的PDP")
partialPlot(rf_indep, pred.data = dat_indep, x.var = "x2", main = "x2的PDP")
partialPlot(rf_indep, pred.data = dat_indep, x.var = "x3", main = "x3的PDP")
par(mfrow = c(1, 1))

## 模拟数据：自变量相关情况（x1和x2相关系数0.9）
set.seed(222)

Sigma <- matrix(c(1, 0.9, 0, 0.9, 1, 0, 0, 0, 1), 3, 3)
X_corr <- mvrnorm(n, mu = rep(0, 3), Sigma = Sigma)
x1.c <- X_corr[, 1] + pi  # 平移到类似范围
x2.c <- X_corr[, 2] + pi
x3.c <- X_corr[, 3] + pi
y_corr <- sin(3 * x1.c) + x2.c + x3.c + rnorm(n, 0, 0.1)
dat_corr <- data.frame(y = y_corr, x1 = x1.c, x2 = x2.c, x3 = x3.c)

## 训练随机森林模型
rf_corr <- randomForest(y ~ ., data = dat_corr, ntree = 100)

## 绘制部分依赖图（变量相关时PDP可能不准确）
par(mfrow = c(1, 3))
partialPlot(rf_corr, pred.data = dat_corr, x.var = "x1", main = "x1的PDP（相关）")
partialPlot(rf_corr, pred.data = dat_corr, x.var = "x2", main = "x2的PDP（相关）")
partialPlot(rf_corr, pred.data = dat_corr, x.var = "x3", main = "x3的PDP（相关）")
par(mfrow = c(1, 1))