# Chap4 R代码
# 自动从chap4.html同步生成

# 4.5.1 B-F法示例1
# 对应教材：section4.tex 4.4节，例4.3
# -----------------------------------------------------------------
# 累积赔款流量三角形
triangle_bf1 <- matrix(c(
  2866, 3334, 3503, 3624, 3719, 3720,
  3359, 3889, 4033, 4231, 4319,   NA,
  3848, 4503, 4779, 4946,   NA,   NA,
  4673, 5422, 5676,   NA,   NA,   NA,
  5369, 6142,   NA,   NA,   NA,   NA,
  5818,   NA,   NA,   NA,   NA,   NA
), nrow = 6, byrow = TRUE)

# 已赚保费
premium <- c(4486, 5024, 5680, 6590, 7482, 8502)
# 预期赔付率
loss_ratio <- 0.83

# 应用链梯法计算进展因子
result_cl <- chain_ladder(triangle_bf1)
factors <- result_cl$factors  # 1.158, 1.049, 1.039, 1.023, 1.000

# 计算各事故年的累积进展因子f
n <- nrow(triangle_bf1)
f <- numeric(n)
for (i in 1:n) {
  # 从最后一个已知进展年到最大进展年的累积进展因子
  last_dev <- n - i + 1  # 最后一个已知进展年
  if (last_dev < n) {
    f[i] <- prod(factors[last_dev:(n-1)])
  } else {
    f[i] <- 1
  }
}
f

# 最终赔款的初始估计 = 保费 * 赔付率
ultimate_initial <- premium * loss_ratio
ultimate_initial

# B-F法准备金 = 保费 * 赔付率 * (1 - 1/f)
reserve_bf <- ultimate_initial * (1 - 1/f)
reserve_bf

# 修正的最终赔款 = 已付赔款 + B-F法准备金
paid_bf <- diag(triangle_bf1[, n:1])
ultimate_bf <- paid_bf + reserve_bf
ultimate_bf

# 总准备金
sum(reserve_bf)

# 4.5.3 B-F法示例2
# 对应教材：section4.tex 4.4节，例4.4
# -----------------------------------------------------------------
triangle_bf2 <- matrix(c(
  473, 620, 690, 715,
  512, 660, 750,  NA,
  611, 700,  NA,  NA,
  647,  NA,  NA,  NA
), nrow = 4, byrow = TRUE)

premium2 <- c(860, 940, 980, 1020)
loss_ratio2 <- 0.85

result_bf2 <- BF_method(triangle_bf2, premium2, loss_ratio2)
result_bf2$f  # 1.0362, 1.1657, 1.4462
result_bf2$reserve  # 0, 27.91, 118.41, 267.50
result_bf2$total_reserve  # 413.82

##############################################################################
# 4.6 随机准备金模型：Mack链梯法
##############################################################################