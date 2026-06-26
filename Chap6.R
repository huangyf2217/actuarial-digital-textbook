# Chap6 R代码
# 自动从chap6.html同步生成

data(npext,package="urca") # 加载urca包的npext数据
xt <- ts(data=npext$cpi,start=npext$year[1]) # 提取CPI构建时间序列
plot(xt) # 时序图
acf(xt) # ACF图
# 根据图像，xt显然为非平稳序列，这一结果也可通过ADF检验得出

adf.test(xt) # 检验p值为0.7374，接受原假设，认为序列非平稳

## 进行一阶差分，再次判断
xt_diff1 <- diff(xt,differences=1,lag=1) # 调整differences参数，可进行高阶差分；调整lag参数，可进行季节性差分
plot(xt_diff1)
acf(xt_diff1)
adf.test(xt_diff1) # 检验p值为0.01，拒接原假设，认为差分后序列平稳

# 6.3.2 例6.15：ARIMA(2,1,1)模型的拟合与预测
# 对应教材：section6.tex 6.3.4节，例6.15
# -----------------------------------------------------------------
set.seed(123)
xt <- arima.sim(n = 100, model = list(ar = c(0.5, 0.3), ma = 0.4, d = 1))
plot(xt, main = "ARIMA(2,1,1)模拟数据")

## ACF检验
acf(xt, main = "原始序列ACF")  # ACF结果显示非平稳

## 一阶差分
xt1 <- diff(xt)
acf(xt1, main = "一阶差分后ACF")  # 一阶差分后平稳

## 拟合不同阶数的ARIMA模型
arima1 <- arima(xt, order = c(1, 1, 1))
arima2 <- arima(xt, order = c(2, 1, 1))
arima3 <- arima(xt, order = c(1, 1, 2))
arima4 <- arima(xt, order = c(2, 1, 2))

## AIC和BIC比较
AIC(arima1, arima2, arima3, arima4)  # AIC结果：模型4最优
BIC(arima1, arima2, arima3, arima4)  # BIC结果：模型1最优

## 诊断检验
tsdiag(arima4)  # SACF均为0，且Ljung-Box检验通过

## 外推预测
xt_pred <- predict(arima4, 10)  # 向前10步预测
print(xt_pred)

## 绘制预测图
plot(xt, xlim = c(1, 110), main = "ARIMA(2,1,2)模型预测")
lines(101:110, xt_pred$pred, col = "red", lwd = 2)
lines(101:110, xt_pred$pred + 2 * xt_pred$se, col = "blue", lty = 2)
lines(101:110, xt_pred$pred - 2 * xt_pred$se, col = "blue", lty = 2)
legend("topleft", legend = c("预测值", "95%置信区间"),
       col = c("red", "blue"), lty = c(1, 2), lwd = c(2, 1))

##############################################################################
# 6.4 模型拟合与应用（Box-Jenkins方法）
##############################################################################

set.seed(123)
xt <- arima.sim(n=100, model=list(ar=c(0.5,0.3),ma=0.4,d=1))
plot(xt)
acf(xt) # ACF结果显示非平稳
xt1 <- diff(xt)
acf(xt1) # 一阶差分后平稳
## 拟合模型
arima1 <- arima(xt, order=c(1,1,1))
arima2 <- arima(xt, order=c(2,1,1))
arima3 <- arima(xt, order=c(1,1,2))
arima4 <- arima(xt, order=c(2,1,2))
AIC(arima1,arima2,arima3,arima4) # AIC结果：模型4最优
BIC(arima1,arima2,arima3,arima4) # BIC结果：模型1最优
## 诊断检验
tsdiag(arima4) # SACF均为0，且Ljung-Box检验通过
## 外推预测
xt_pred <- predict(arima4,10) # 向前10步预测