/* Chap5 SAS代码 */
/* 自动从chap5.html同步生成 */

/****************************************************************************/
/* 第5章 损失调整与再保险                                                   */
/* 对应教材：section5.tex                                                   */
/* 内容：通货膨胀影响、再保险类型、免赔额与限额、                           */
/*       不完整数据估计、再保险定价                                         */
/****************************************************************************/

/****************************************************************************/
/* 5.1 通货膨胀对损失分布的影响                                             */
/****************************************************************************/
proc iml;
  /* 原始损失分布：对数正态LN(μ=5, σ=1) */
  mu = 5; sigma = 1;
  /* 通胀率r=10% */
  r = 0.1;
  /* 通胀后分布：LN(μ+ln(1+r), σ) */
  mu_new = mu + log(1 + r);
  print "通胀前后参数:";
  print "原 μ=" mu "σ=" sigma;
  print "通胀后 μ=" mu_new "σ=" sigma;

  /* 均值和方差变化 */
  E_X = exp(mu + sigma**2/2);
  E_Y = exp(mu_new + sigma**2/2);
  print "E(X)=" E_X "E(Y)=" E_Y "比值=" E_Y/E_X;
quit;


/****************************************************************************/
/* 5.2 比例再保险                                                           */
/****************************************************************************/
proc iml;
  /* 原保险人自留比例α */
  alpha = 0.7;
  /* 原损失X */
  call streaminit(123);
  X = j(10000, 1, 0);
  do i = 1 to 10000;
    X[i] = exp(rand('NORMAL', 5, 1));
  end;

  /* 原保险人赔付 */
  X_primary = alpha * X;
  /* 再保险人赔付 */
  X_reins = (1 - alpha) * X;

  print "比例再保险（自留比例0.7）";
  print "原损失均值:" mean(X);
  print "原保险人赔付均值:" mean(X_primary);
  print "再保险人赔付均值:" mean(X_reins);
quit;


/****************************************************************************/
/* 5.3 超额赔款再保险                                                       */
/****************************************************************************/

/* 5.3.1 单个损失的再保险赔付 */
proc iml;
  call streaminit(123);
  M = 1000;  /* 免赔额 */
  L = 5000;  /* 限额 */

  X = j(10000, 1, 0);
  Y_reins = j(10000, 1, 0);
  Y_primary = j(10000, 1, 0);
  do i = 1 to 10000;
    X[i] = exp(rand('NORMAL', 5, 1));
    /* 再保险人赔付：min(max(X-M, 0), L) */
    Y_reins[i] = min(max(X[i] - M, 0), L);
    /* 原保险人赔付：X - Y_reins */
    Y_primary[i] = X[i] - Y_reins[i];
  end;

  print "超额赔款再保险（M=1000, L=5000）";
  print "原损失均值:" mean(X);
  print "原保险人赔付均值:" mean(Y_primary);
  print "再保险人赔付均值:" mean(Y_reins);
quit;


/* 5.3.2 累积赔付的再保险 */
data aggregate_reins;
  call streaminit(123);
  M_agg = 5000;   /* 累积免赔额 */
  L_agg = 20000;  /* 累积限额 */
  do sim = 1 to 10000;
    N = rand('POISSON', 5);
    S = 0;
    do j = 1 to N;
      S = S + exp(rand('NORMAL', 5, 1));
    end;
    /* 再保险人赔付 */
    Y_reins = min(max(S - M_agg, 0), L_agg);
    Y_primary = S - Y_reins;
    output;
  end;
run;

title "累积超额赔款再保险";
proc means data=aggregate_reins mean std p95;
  var S Y_reins Y_primary;
run;
title;


/****************************************************************************/
/* 5.4 免赔额与限额的影响                                                   */
/****************************************************************************/
proc iml;
  /* 对数正态分布LN(μ=5, σ=1) */
  mu = 5; sigma = 1;

  /* 不同免赔额下的期望赔付 */
  d_values = {0, 50, 100, 200, 500, 1000};
  print "免赔额对期望赔付的影响";
  do i = 1 to nrow(d_values);
    d = d_values[i];
    /* E[X | X>d] = E[X; X>d] / P(X>d) */
    /* 对于对数正态：E[X; X>d] = exp(μ+σ²/2) * Φ((μ+σ²-ln d)/σ) */
    E_X_above_d = exp(mu + sigma**2/2) *
                  cdf('NORMAL', (mu + sigma**2 - log(d))/sigma);
    P_above_d = 1 - cdf('LOGNORMAL', d, mu, sigma);
    E_X_given_d = E_X_above_d / P_above_d;
    print "d=" d "E[X|X>d]=" E_X_given_d "P(X>d)=" P_above_d;
  end;

  /* 不同限额下的期望赔付 */
  u_values = {100, 500, 1000, 5000, 10000};
  print "限额对期望赔付的影响";
  do i = 1 to nrow(u_values);
    u = u_values[i];
    /* E[min(X,u)] = E[X; X≤u] + u*P(X>u) */
    E_X_below_u = exp(mu + sigma**2/2) *
                  cdf('NORMAL', (mu + sigma**2 - log(u))/sigma);
    P_above_u = 1 - cdf('LOGNORMAL', u, mu, sigma);
    E_min_Xu = E_X_below_u + u * P_above_u;
    print "u=" u "E[min(X,u)]=" E_min_Xu;
  end;
quit;


/****************************************************************************/
/* 5.5 不完整数据估计：截断数据                                             */
/****************************************************************************/
proc iml;
  /* 生成截断数据：只观测大于d的损失 */
  call streaminit(123);
  d = 100;
  n = 1000;
  X_trunc = j(n, 1, 0);
  count = 0;
  do until(count >= n);
    x = exp(rand('NORMAL', 5, 1));
    if x > d then do;
      count = count + 1;
      X_trunc[count] = x;
    end;
  end;

  /* 截断数据的MLE估计 */
  /* 对于对数正态，截断后似然函数：
     L = ∏ f(x_i) / (1 - F(d)) */
  log_x = log(X_trunc);
  mu_hat = mean(log_x);
  sigma_hat = std(log_x);
  print "截断数据MLE（对数正态）";
  print "μ_hat =" mu_hat "σ_hat =" sigma_hat;
quit;


/****************************************************************************/
/* 5.6 不完整数据估计：删失数据                                             */
/****************************************************************************/
proc iml;
  /* 生成删失数据：超过u的损失只记录u */
  call streaminit(123);
  u = 5000;
  n = 1000;
  X_cens = j(n, 1, 0);
  censored = j(n, 1, 0);
  do i = 1 to n;
    x = exp(rand('NORMAL', 5, 1));
    if x > u then do;
      X_cens[i] = u;
      censored[i] = 1;
    end;
    else X_cens[i] = x;
  end;

  /* 删失数据的MLE：使用生存分析 */
  /* 简化：仅展示未删失部分的估计 */
  uncens_idx = loc(censored = 0);
  X_uncens = X_cens[uncens_idx];
  log_x = log(X_uncens);
  mu_hat = mean(log_x);
  sigma_hat = std(log_x);
  print "删失数据MLE（对数正态，仅未删失部分）";
  print "μ_hat =" mu_hat "σ_hat =" sigma_hat;
  print "删失比例:" n - nrow(uncens_idx);
quit;


/****************************************************************************/
/* 5.7 再保险定价                                                           */
/****************************************************************************/

/* 5.7.1 溢额再保险定价 */
proc iml;
  /* 假设索赔次数N~Poisson(λ=100) */
  /* 单个索赔金额X~Gamma(α=2, β=0.01) */
  lambda = 100;
  alpha = 2; beta = 0.01;

  /* 再保险人每次赔付：min(max(X-M, 0), L) */
  M = 1000; L = 5000;

  /* 模拟 */
  call streaminit(123);
  n_sim = 10000;
  S_reins = j(n_sim, 1, 0);
  do i = 1 to n_sim;
    N = rand('POISSON', lambda);
    S = 0;
    do j = 1 to N;
      X = rand('GAMMA', alpha, 1/beta);
      S = S + min(max(X - M, 0), L);
    end;
    S_reins[i] = S;
  end;

  E_S_reins = mean(S_reins);
  Var_S_reins = var(S_reins);
  /* 再保险保费 = E[S] + 风险附加 */
  loading = 0.2;
  premium = E_S_reins * (1 + loading);
  print "溢额再保险定价";
  print "E[S_reins] =" E_S_reins;
  print "Var[S_reins] =" Var_S_reins;
  print "再保险保费(20%附加) =" premium;
quit;


/* 5.7.2 比例再保险定价 */
proc iml;
  lambda = 100;
  alpha = 2; beta = 0.01;
  alpha_quota = 0.3;  /* 再保险人承担30% */

  call streaminit(123);
  n_sim = 10000;
  S_reins = j(n_sim, 1, 0);
  do i = 1 to n_sim;
    N = rand('POISSON', lambda);
    S = 0;
    do j = 1 to N;
      X = rand('GAMMA', alpha, 1/beta);
      S = S + alpha_quota * X;
    end;
    S_reins[i] = S;
  end;

  E_S_reins = mean(S_reins);
  loading = 0.15;
  premium = E_S_reins * (1 + loading);
  print "比例再保险定价";
  print "E[S_reins] =" E_S_reins;
  print "再保险保费(15%附加) =" premium;
quit;


/****************************************************************************/
/* 5.8 通货膨胀对再保险的影响                                               */
/****************************************************************************/
proc iml;
  /* 原损失X~LN(5,1)，通胀率r */
  r_values = {0, 0.05, 0.10, 0.15, 0.20};
  M = 1000; L = 5000;

  print "通胀对超额赔款再保险的影响";
  do i = 1 to nrow(r_values);
    r = r_values[i];
    mu_new = 5 + log(1 + r);

    call streaminit(123);
    n_sim = 10000;
    S_reins = j(n_sim, 1, 0);
    do k = 1 to n_sim;
      N = rand('POISSON', 100);
      S = 0;
      do j = 1 to N;
        X = exp(rand('NORMAL', mu_new, 1));
        S = S + min(max(X - M, 0), L);
      end;
      S_reins[k] = S;
    end;

    E_S = mean(S_reins);
    print "r=" r "E[S_reins]=" E_S;
  end;
quit;

/****************************************************************************/
/* 第5章 损失调整与再保险                                                   */
/* 对应教材：section5.tex                                                   */
/* 内容：通货膨胀影响、再保险类型、免赔额与限额、                           */
/*       不完整数据估计、再保险定价                                         */
/****************************************************************************/

/****************************************************************************/
/* 5.1 通货膨胀对损失分布的影响                                             */
/****************************************************************************/
proc iml;
  /* 原始损失分布：对数正态LN(μ=5, σ=1) */
  mu = 5; sigma = 1;
  /* 通胀率r=10% */
  r = 0.1;
  /* 通胀后分布：LN(μ+ln(1+r), σ) */
  mu_new = mu + log(1 + r);
  print "通胀前后参数:";
  print "原 μ=" mu "σ=" sigma;
  print "通胀后 μ=" mu_new "σ=" sigma;

  /* 均值和方差变化 */
  E_X = exp(mu + sigma**2/2);
  E_Y = exp(mu_new + sigma**2/2);
  print "E(X)=" E_X "E(Y)=" E_Y "比值=" E_Y/E_X;
quit;


/****************************************************************************/
/* 5.2 比例再保险                                                           */
/****************************************************************************/
proc iml;
  /* 原保险人自留比例α */
  alpha = 0.7;
  /* 原损失X */
  call streaminit(123);
  X = j(10000, 1, 0);
  do i = 1 to 10000;
    X[i] = exp(rand('NORMAL', 5, 1));
  end;

  /* 原保险人赔付 */
  X_primary = alpha * X;
  /* 再保险人赔付 */
  X_reins = (1 - alpha) * X;

  print "比例再保险（自留比例0.7）";
  print "原损失均值:" mean(X);
  print "原保险人赔付均值:" mean(X_primary);
  print "再保险人赔付均值:" mean(X_reins);
quit;


/****************************************************************************/
/* 5.3 超额赔款再保险                                                       */
/****************************************************************************/

/* 5.3.1 单个损失的再保险赔付 */
proc iml;
  call streaminit(123);
  M = 1000;  /* 免赔额 */
  L = 5000;  /* 限额 */

  X = j(10000, 1, 0);
  Y_reins = j(10000, 1, 0);
  Y_primary = j(10000, 1, 0);
  do i = 1 to 10000;
    X[i] = exp(rand('NORMAL', 5, 1));
    /* 再保险人赔付：min(max(X-M, 0), L) */
    Y_reins[i] = min(max(X[i] - M, 0), L);
    /* 原保险人赔付：X - Y_reins */
    Y_primary[i] = X[i] - Y_reins[i];
  end;

  print "超额赔款再保险（M=1000, L=5000）";
  print "原损失均值:" mean(X);
  print "原保险人赔付均值:" mean(Y_primary);
  print "再保险人赔付均值:" mean(Y_reins);
quit;


/* 5.3.2 累积赔付的再保险 */
data aggregate_reins;
  call streaminit(123);
  M_agg = 5000;   /* 累积免赔额 */
  L_agg = 20000;  /* 累积限额 */
  do sim = 1 to 10000;
    N = rand('POISSON', 5);
    S = 0;
    do j = 1 to N;
      S = S + exp(rand('NORMAL', 5, 1));
    end;
    /* 再保险人赔付 */
    Y_reins = min(max(S - M_agg, 0), L_agg);
    Y_primary = S - Y_reins;
    output;
  end;
run;

title "累积超额赔款再保险";
proc means data=aggregate_reins mean std p95;
  var S Y_reins Y_primary;
run;
title;


/****************************************************************************/
/* 5.4 免赔额与限额的影响                                                   */
/****************************************************************************/
proc iml;
  /* 对数正态分布LN(μ=5, σ=1) */
  mu = 5; sigma = 1;

  /* 不同免赔额下的期望赔付 */
  d_values = {0, 50, 100, 200, 500, 1000};
  print "免赔额对期望赔付的影响";
  do i = 1 to nrow(d_values);
    d = d_values[i];
    /* E[X | X>d] = E[X; X>d] / P(X>d) */
    /* 对于对数正态：E[X; X>d] = exp(μ+σ²/2) * Φ((μ+σ²-ln d)/σ) */
    E_X_above_d = exp(mu + sigma**2/2) *
                  cdf('NORMAL', (mu + sigma**2 - log(d))/sigma);
    P_above_d = 1 - cdf('LOGNORMAL', d, mu, sigma);
    E_X_given_d = E_X_above_d / P_above_d;
    print "d=" d "E[X|X>d]=" E_X_given_d "P(X>d)=" P_above_d;
  end;

  /* 不同限额下的期望赔付 */
  u_values = {100, 500, 1000, 5000, 10000};
  print "限额对期望赔付的影响";
  do i = 1 to nrow(u_values);
    u = u_values[i];
    /* E[min(X,u)] = E[X; X≤u] + u*P(X>u) */
    E_X_below_u = exp(mu + sigma**2/2) *
                  cdf('NORMAL', (mu + sigma**2 - log(u))/sigma);
    P_above_u = 1 - cdf('LOGNORMAL', u, mu, sigma);
    E_min_Xu = E_X_below_u + u * P_above_u;
    print "u=" u "E[min(X,u)]=" E_min_Xu;
  end;
quit;


/****************************************************************************/
/* 5.5 不完整数据估计：截断数据                                             */
/****************************************************************************/
proc iml;
  /* 生成截断数据：只观测大于d的损失 */
  call streaminit(123);
  d = 100;
  n = 1000;
  X_trunc = j(n, 1, 0);
  count = 0;
  do until(count >= n);
    x = exp(rand('NORMAL', 5, 1));
    if x > d then do;
      count = count + 1;
      X_trunc[count] = x;
    end;
  end;

  /* 截断数据的MLE估计 */
  /* 对于对数正态，截断后似然函数：
     L = ∏ f(x_i) / (1 - F(d)) */
  log_x = log(X_trunc);
  mu_hat = mean(log_x);
  sigma_hat = std(log_x);
  print "截断数据MLE（对数正态）";
  print "μ_hat =" mu_hat "σ_hat =" sigma_hat;
quit;


/****************************************************************************/
/* 5.6 不完整数据估计：删失数据                                             */
/****************************************************************************/
proc iml;
  /* 生成删失数据：超过u的损失只记录u */
  call streaminit(123);
  u = 5000;
  n = 1000;
  X_cens = j(n, 1, 0);
  censored = j(n, 1, 0);
  do i = 1 to n;
    x = exp(rand('NORMAL', 5, 1));
    if x > u then do;
      X_cens[i] = u;
      censored[i] = 1;
    end;
    else X_cens[i] = x;
  end;

  /* 删失数据的MLE：使用生存分析 */
  /* 简化：仅展示未删失部分的估计 */
  uncens_idx = loc(censored = 0);
  X_uncens = X_cens[uncens_idx];
  log_x = log(X_uncens);
  mu_hat = mean(log_x);
  sigma_hat = std(log_x);
  print "删失数据MLE（对数正态，仅未删失部分）";
  print "μ_hat =" mu_hat "σ_hat =" sigma_hat;
  print "删失比例:" n - nrow(uncens_idx);
quit;


/****************************************************************************/
/* 5.7 再保险定价                                                           */
/****************************************************************************/

/* 5.7.1 溢额再保险定价 */
proc iml;
  /* 假设索赔次数N~Poisson(λ=100) */
  /* 单个索赔金额X~Gamma(α=2, β=0.01) */
  lambda = 100;
  alpha = 2; beta = 0.01;

  /* 再保险人每次赔付：min(max(X-M, 0), L) */
  M = 1000; L = 5000;

  /* 模拟 */
  call streaminit(123);
  n_sim = 10000;
  S_reins = j(n_sim, 1, 0);
  do i = 1 to n_sim;
    N = rand('POISSON', lambda);
    S = 0;
    do j = 1 to N;
      X = rand('GAMMA', alpha, 1/beta);
      S = S + min(max(X - M, 0), L);
    end;
    S_reins[i] = S;
  end;

  E_S_reins = mean(S_reins);
  Var_S_reins = var(S_reins);
  /* 再保险保费 = E[S] + 风险附加 */
  loading = 0.2;
  premium = E_S_reins * (1 + loading);
  print "溢额再保险定价";
  print "E[S_reins] =" E_S_reins;
  print "Var[S_reins] =" Var_S_reins;
  print "再保险保费(20%附加) =" premium;
quit;


/* 5.7.2 比例再保险定价 */
proc iml;
  lambda = 100;
  alpha = 2; beta = 0.01;
  alpha_quota = 0.3;  /* 再保险人承担30% */

  call streaminit(123);
  n_sim = 10000;
  S_reins = j(n_sim, 1, 0);
  do i = 1 to n_sim;
    N = rand('POISSON', lambda);
    S = 0;
    do j = 1 to N;
      X = rand('GAMMA', alpha, 1/beta);
      S = S + alpha_quota * X;
    end;
    S_reins[i] = S;
  end;

  E_S_reins = mean(S_reins);
  loading = 0.15;
  premium = E_S_reins * (1 + loading);
  print "比例再保险定价";
  print "E[S_reins] =" E_S_reins;
  print "再保险保费(15%附加) =" premium;
quit;


/****************************************************************************/
/* 5.8 通货膨胀对再保险的影响                                               */
/****************************************************************************/
proc iml;
  /* 原损失X~LN(5,1)，通胀率r */
  r_values = {0, 0.05, 0.10, 0.15, 0.20};
  M = 1000; L = 5000;

  print "通胀对超额赔款再保险的影响";
  do i = 1 to nrow(r_values);
    r = r_values[i];
    mu_new = 5 + log(1 + r);

    call streaminit(123);
    n_sim = 10000;
    S_reins = j(n_sim, 1, 0);
    do k = 1 to n_sim;
      N = rand('POISSON', 100);
      S = 0;
      do j = 1 to N;
        X = exp(rand('NORMAL', mu_new, 1));
        S = S + min(max(X - M, 0), L);
      end;
      S_reins[k] = S;
    end;

    E_S = mean(S_reins);
    print "r=" r "E[S_reins]=" E_S;
  end;
quit;

proc iml;
  /* 原始损失分布：对数正态LN(μ=5, σ=1) */
  mu = 5; sigma = 1;
  /* 通胀率r=10% */
  r = 0.1;
  /* 通胀后分布：LN(μ+ln(1+r), σ) */
  mu_new = mu + log(1 + r);
  print "通胀前后参数:";
  print "原 μ=" mu "σ=" sigma;
  print "通胀后 μ=" mu_new "σ=" sigma;

  /* 均值和方差变化 */
  E_X = exp(mu + sigma**2/2);
  E_Y = exp(mu_new + sigma**2/2);
  print "E(X)=" E_X "E(Y)=" E_Y "比值=" E_Y/E_X;
quit;


/****************************************************************************/
/* 5.2 比例再保险                                                           */
/****************************************************************************/
proc iml;
  /* 原保险人自留比例α */
  alpha = 0.7;
  /* 原损失X */
  call streaminit(123);
  X = j(10000, 1, 0);
  do i = 1 to 10000;
    X[i] = exp(rand('NORMAL', 5, 1));
  end;

  /* 原保险人赔付 */

proc iml;
  /* 原始损失分布：对数正态LN(μ=5, σ=1) */
  mu = 5; sigma = 1;
  /* 通胀率r=10% */
  r = 0.1;
  /* 通胀后分布：LN(μ+ln(1+r), σ) */
  mu_new = mu + log(1 + r);
  print "通胀前后参数:";
  print "原 μ=" mu "σ=" sigma;
  print "通胀后 μ=" mu_new "σ=" sigma;

  /* 均值和方差变化 */
  E_X = exp(mu + sigma**2/2);
  E_Y = exp(mu_new + sigma**2/2);
  print "E(X)=" E_X "E(Y)=" E_Y "比值=" E_Y/E_X;
quit;


/****************************************************************************/
/* 5.2 比例再保险                                                           */
/****************************************************************************/
proc iml;
  /* 原保险人自留比例α */
  alpha = 0.7;
  /* 原损失X */
  call streaminit(123);
  X = j(10000, 1, 0);
  do i = 1 to 10000;
    X[i] = exp(rand('NORMAL', 5, 1));
  end;

  /* 原保险人赔付 */

proc iml;
  /* 原始损失分布：对数正态LN(μ=5, σ=1) */
  mu = 5; sigma = 1;
  /* 通胀率r=10% */
  r = 0.1;
  /* 通胀后分布：LN(μ+ln(1+r), σ) */
  mu_new = mu + log(1 + r);
  print "通胀前后参数:";
  print "原 μ=" mu "σ=" sigma;
  print "通胀后 μ=" mu_new "σ=" sigma;

  /* 均值和方差变化 */
  E_X = exp(mu + sigma**2/2);
  E_Y = exp(mu_new + sigma**2/2);
  print "E(X)=" E_X "E(Y)=" E_Y "比值=" E_Y/E_X;
quit;


/****************************************************************************/
/* 5.2 比例再保险                                                           */
/****************************************************************************/
proc iml;
  /* 原保险人自留比例α */
  alpha = 0.7;
  /* 原损失X */
  call streaminit(123);
  X = j(10000, 1, 0);
  do i = 1 to 10000;
    X[i] = exp(rand('NORMAL', 5, 1));
  end;

  /* 原保险人赔付 */

proc iml;
  /* 原始损失分布：对数正态LN(μ=5, σ=1) */
  mu = 5; sigma = 1;
  /* 通胀率r=10% */
  r = 0.1;
  /* 通胀后分布：LN(μ+ln(1+r), σ) */
  mu_new = mu + log(1 + r);
  print "通胀前后参数:";
  print "原 μ=" mu "σ=" sigma;
  print "通胀后 μ=" mu_new "σ=" sigma;

  /* 均值和方差变化 */
  E_X = exp(mu + sigma**2/2);
  E_Y = exp(mu_new + sigma**2/2);
  print "E(X)=" E_X "E(Y)=" E_Y "比值=" E_Y/E_X;
quit;


/****************************************************************************/
/* 5.2 比例再保险                                                           */
/****************************************************************************/
proc iml;
  /* 原保险人自留比例α */
  alpha = 0.7;
  /* 原损失X */
  call streaminit(123);
  X = j(10000, 1, 0);
  do i = 1 to 10000;
    X[i] = exp(rand('NORMAL', 5, 1));
  end;

  /* 原保险人赔付 */

proc iml;
  /* 原始损失分布：对数正态LN(μ=5, σ=1) */
  mu = 5; sigma = 1;
  /* 通胀率r=10% */
  r = 0.1;
  /* 通胀后分布：LN(μ+ln(1+r), σ) */
  mu_new = mu + log(1 + r);
  print "通胀前后参数:";
  print "原 μ=" mu "σ=" sigma;
  print "通胀后 μ=" mu_new "σ=" sigma;

  /* 均值和方差变化 */
  E_X = exp(mu + sigma**2/2);
  E_Y = exp(mu_new + sigma**2/2);
  print "E(X)=" E_X "E(Y)=" E_Y "比值=" E_Y/E_X;
quit;


/****************************************************************************/
/* 5.2 比例再保险                                                           */
/****************************************************************************/
proc iml;
  /* 原保险人自留比例α */
  alpha = 0.7;
  /* 原损失X */
  call streaminit(123);
  X = j(10000, 1, 0);
  do i = 1 to 10000;
    X[i] = exp(rand('NORMAL', 5, 1));
  end;

  /* 原保险人赔付 */

proc iml;
  /* 原始损失分布：对数正态LN(μ=5, σ=1) */
  mu = 5; sigma = 1;
  /* 通胀率r=10% */
  r = 0.1;
  /* 通胀后分布：LN(μ+ln(1+r), σ) */
  mu_new = mu + log(1 + r);
  print "通胀前后参数:";
  print "原 μ=" mu "σ=" sigma;
  print "通胀后 μ=" mu_new "σ=" sigma;

  /* 均值和方差变化 */
  E_X = exp(mu + sigma**2/2);
  E_Y = exp(mu_new + sigma**2/2);
  print "E(X)=" E_X "E(Y)=" E_Y "比值=" E_Y/E_X;
quit;


/****************************************************************************/
/* 5.2 比例再保险                                                           */
/****************************************************************************/
proc iml;
  /* 原保险人自留比例α */
  alpha = 0.7;
  /* 原损失X */
  call streaminit(123);
  X = j(10000, 1, 0);
  do i = 1 to 10000;
    X[i] = exp(rand('NORMAL', 5, 1));
  end;

  /* 原保险人赔付 */

proc iml;
  /* 原始损失分布：对数正态LN(μ=5, σ=1) */
  mu = 5; sigma = 1;
  /* 通胀率r=10% */
  r = 0.1;
  /* 通胀后分布：LN(μ+ln(1+r), σ) */
  mu_new = mu + log(1 + r);
  print "通胀前后参数:";
  print "原 μ=" mu "σ=" sigma;
  print "通胀后 μ=" mu_new "σ=" sigma;

  /* 均值和方差变化 */
  E_X = exp(mu + sigma**2/2);
  E_Y = exp(mu_new + sigma**2/2);
  print "E(X)=" E_X "E(Y)=" E_Y "比值=" E_Y/E_X;
quit;


/****************************************************************************/
/* 5.2 比例再保险                                                           */
/****************************************************************************/
proc iml;
  /* 原保险人自留比例α */
  alpha = 0.7;
  /* 原损失X */
  call streaminit(123);
  X = j(10000, 1, 0);
  do i = 1 to 10000;
    X[i] = exp(rand('NORMAL', 5, 1));
  end;

  /* 原保险人赔付 */

proc iml;
  /* 原始损失分布：对数正态LN(μ=5, σ=1) */
  mu = 5; sigma = 1;
  /* 通胀率r=10% */
  r = 0.1;
  /* 通胀后分布：LN(μ+ln(1+r), σ) */
  mu_new = mu + log(1 + r);
  print "通胀前后参数:";
  print "原 μ=" mu "σ=" sigma;
  print "通胀后 μ=" mu_new "σ=" sigma;

  /* 均值和方差变化 */
  E_X = exp(mu + sigma**2/2);
  E_Y = exp(mu_new + sigma**2/2);
  print "E(X)=" E_X "E(Y)=" E_Y "比值=" E_Y/E_X;
quit;


/****************************************************************************/
/* 5.2 比例再保险                                                           */
/****************************************************************************/
proc iml;
  /* 原保险人自留比例α */
  alpha = 0.7;
  /* 原损失X */
  call streaminit(123);
  X = j(10000, 1, 0);
  do i = 1 to 10000;
    X[i] = exp(rand('NORMAL', 5, 1));
  end;

  /* 原保险人赔付 */