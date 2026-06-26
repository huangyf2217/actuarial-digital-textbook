/* Chap9 SAS代码 */
/* 自动从chap9.html同步生成 */

do i = 1 to nrow(p_values);
    p = p_values[i];
    VaR = quantile('NORMAL', p, mu, sigma);
    /* TVaR = μ + σ * φ(Φ^{-1}(p)) / (1-p) */
    z = quantile('NORMAL', p);
    TVaR = mu + sigma * pdf('NORMAL', z) / (1 - p);
    print "p=" p "VaR=" VaR "TVaR=" TVaR;
  end;

  /* 对数正态分布的风险度量 */
  mu_ln = 0; sigma_ln = 1;
  print "对数正态分布的风险度量";
  do i = 1 to nrow(p_values);
    p = p_values[i];
    VaR = quantile('LOGNORMAL', p, mu_ln, sigma_ln);
    print "p=" p "VaR=" VaR;
  end;
quit;


/****************************************************************************/
/* 9.2 广义极值分布（GEV）                                                  */
/****************************************************************************/
proc iml;
  /* GEV分布函数:
     - ξ > 0 (Frechet): F(x) = exp(-(1+ξx/σ)^(-1/ξ)), 1+ξx/σ > 0
     - ξ = 0 (Gumbel): F(x) = exp(-exp(-x/σ))
     - ξ < 0 (Weibull): F(x) = exp(-(1+ξx/σ)^(-1/ξ)), 1+ξx/σ > 0
  */

  /* 绘制不同ξ的GEV密度 */
  x = do(-4, 4, 0.1)`;

/****************************************************************************/
/* 第9章 极值理论                                                           */
/* 对应教材：section9.tex                                                   */
/* 内容：风险度量、广义极值分布（GEV）、广义帕累托分布（GPD）、              */
/*       阈值选择、参数估计、应用                                           */
/****************************************************************************/

/****************************************************************************/
/* 9.1 风险度量：VaR与TVaR                                                  */
/****************************************************************************/
proc iml;
  /* VaR_p = F^{-1}(p) */
  /* TVaR_p = E[X | X > VaR_p] */

  /* 正态分布的风险度量 */
  mu = 0; sigma = 1;
  p_values = {0.95, 0.99, 0.999};

  print "正态分布的风险度量";
  do i = 1 to nrow(p_values);
    p = p_values[i];
    VaR = quantile('NORMAL', p, mu, sigma);
    /* TVaR = μ + σ * φ(Φ^{-1}(p)) / (1-p) */
    z = quantile('NORMAL', p);
    TVaR = mu + sigma * pdf('NORMAL', z) / (1 - p);
    print "p=" p "VaR=" VaR "TVaR=" TVaR;
  end;

  /* 对数正态分布的风险度量 */
  mu_ln = 0; sigma_ln = 1;
  print "对数正态分布的风险度量";
  do i = 1 to nrow(p_values);
    p = p_values[i];
    VaR = quantile('LOGNORMAL', p, mu_ln, sigma_ln);
    print "p=" p "VaR=" VaR;
  end;
quit;


/****************************************************************************/
/* 9.2 广义极值分布（GEV）                                                  */
/****************************************************************************/
proc iml;
  /* GEV分布函数:
     - ξ > 0 (Frechet): F(x) = exp(-(1+ξx/σ)^(-1/ξ)), 1+ξx/σ > 0
     - ξ = 0 (Gumbel): F(x) = exp(-exp(-x/σ))
     - ξ < 0 (Weibull): F(x) = exp(-(1+ξx/σ)^(-1/ξ)), 1+ξx/σ > 0
  */

  /* 绘制不同ξ的GEV密度 */
  x = do(-4, 4, 0.1)`;
  sigma = 1;

  /* ξ = 0.5 (Frechet) */
  xi1 = 0.5;
  f1 = j(nrow(x), 1, 0);
  do i = 1 to nrow(x);
    if 1 + xi1 * x[i] / sigma > 0 then do;
      t = (1 + xi1 * x[i] / sigma)##(-1/xi1);
      f1[i] = t##(xi1 + 1) * exp(-t) / sigma;
    end;
  end;

  /* ξ = 0 (Gumbel) */
  xi2 = 0;
  f2 = j(nrow(x), 1, 0);
  do i = 1 to nrow(x);
    t = exp(-x[i] / sigma);
    f2[i] = t * exp(-t) / sigma;
  end;

  /* ξ = -0.5 (Weibull) */
  xi3 = -0.5;
  f3 = j(nrow(x), 1, 0);
  do i = 1 to nrow(x);
    if 1 + xi3 * x[i] / sigma > 0 then do;
      t = (1 + xi3 * x[i] / sigma)##(-1/xi3);
      f3[i] = t##(xi3 + 1) * exp(-t) / sigma;
    end;
  end;

  create gev_data from x [colname={'x'}];
  append from x;
  close;
  create gev_f1 from f1 [colname={'f1'}];
  append from f1;
  close;
  create gev_f2 from f2 [colname={'f2'}];
  append from f2;
  close;
  create gev_f3 from f3 [colname={'f3'}];
    append from f3;
  close;
quit;

data gev_plot;
  merge gev_data gev_f1 gev_f2 gev_f3;
run;

title "GEV分布密度函数";
proc sgplot data=gev_plot;
  series x=x y=f1 / legendlabel="ξ=0.5 (Frechet)";
  series x=x y=f2 / legendlabel="ξ=0 (Gumbel)";
  series x=x y=f3 / legendlabel="ξ=-0.5 (Weibull)";
  yaxis label="f(x)";
run;
title;


/****************************************************************************/
/* 9.3 区块最大值法（Block Maxima）                                         */
/****************************************************************************/
proc iml;
  /* 生成模拟数据 */
  call streaminit(123);
  n_blocks = 100;
  block_size = 50;

  maxima = j(n_blocks, 1, 0);
  do b = 1 to n_blocks;
    max_val = -1e30;
    do i = 1 to block_size;
      x = rand('EXPONENTIAL', 1);  /* 指数分布 */
      if x > max_val then max_val = x;
    end;
    maxima[b] = max_val;
  end;

  print "区块最大值法";
  print "均值:" mean(maxima);
  print "标准差:" std(maxima);
  print "最大值:" max(maxima);

  create maxima_data from maxima [colname={'maxima'}];
  append from maxima;
  close;
quit;

title "区块最大值直方图";
proc sgplot data=maxima_data;
  histogram maxima / nbins=20;
  density maxima / type=kernel;
run;
title;


/****************************************************************************/
/* 9.4 GEV参数估计                                                          */
/****************************************************************************/
proc iml;
  /* 使用极大似然估计GEV参数 */
  use maxima_data;
  read all var {maxima} into x;
  close maxima_data;

  /* 负对数似然函数 */
  start gev_nll(parms) global(x);
    mu = parms[1];
    sigma = exp(parms[2]);
    xi = parms[3];

    n = nrow(x);
    nll = 0;
    do i = 1 to n;
      z = (x[i] - mu) / sigma;
      if 1 + xi * z <= 0 then return(1e10);
      if abs(xi) < 1e-8 then do;
        t = exp(-z);
      end;
      else do;
        t = (1 + xi * z)##(-1/xi);
      end;
      nll = nll + log(sigma) + (1 + 1/xi) * log(t) + t;
    end;
    return(nll);
  finish;

  /* 优化 */
  init = {3, 0, 0.1};
  opt = {1, 0};
  call nlpnra(rc, result, "gev_nll", init, opt);

  mu_hat = result[1];
  sigma_hat = exp(result[2]);
  xi_hat = result[3];
  print "GEV参数MLE估计";
  print "μ_hat =" mu_hat;
  print "σ_hat =" sigma_hat;
  print "ξ_hat =" xi_hat;
quit;


/****************************************************************************/
/* 9.5 广义帕累托分布（GPD）                                                */
/****************************************************************************/
proc iml;
  /* GPD分布函数:
     - ξ ≠ 0: G(x) = 1 - (1 + ξx/σ)^(-1/ξ), x > 0 (ξ≥0) 或 0 < x < -σ/ξ (ξ<0)
     - ξ = 0: G(x) = 1 - exp(-x/σ)
  */

  /* 绘制不同ξ的GPD密度 */
  x = do(0, 5, 0.05)`;
  sigma = 1;

  /* ξ = 0.5 */
  xi1 = 0.5;
  f1 = (1/sigma) * (1 + xi1 * x / sigma)##(-1/xi1 - 1);

  /* ξ = 0 */
  xi2 = 0;
  f2 = (1/sigma) * exp(-x / sigma);

  /* ξ = -0.5 */
  xi3 = -0.5;
  f3 = j(nrow(x), 1, 0);
  do i = 1 to nrow(x);
    if 1 + xi3 * x[i] / sigma > 0 then
      f3[i] = (1/sigma) * (1 + xi3 * x[i] / sigma)##(-1/xi3 - 1);
  end;

  create gpd_data from x [colname={'x'}];
  append from x;
  close;
  create gpd_f1 from f1 [colname={'f1'}];
  append from f1;
  close;
  create gpd_f2 from f2 [colname={'f2'}];
  append from f2;
  close;
  create gpd_f3 from f3 [colname={'f3'}];
  append from f3;
  close;
quit;

data gpd_plot;
  merge gpd_data gpd_f1 gpd_f2 gpd_f3;
run;

title "GPD分布密度函数";
proc sgplot data=gpd_plot;
  series x=x y=f1 / legendlabel="ξ=0.5";
  series x=x y=f2 / legendlabel="ξ=0";
  series x=x y=f3 / legendlabel="ξ=-0.5";
  yaxis label="f(x)";
run;
title;


/****************************************************************************/
/* 9.6 阈值选择（POT方法）                                                  */
/****************************************************************************/
proc iml;
  /* 生成模拟数据 */
  call streaminit(456);
  n = 1000;
  x = j(n, 1, 0);
  do i = 1 to n;
    x[i] = rand('EXPONENTIAL', 1);
  end;

  /* 不同阈值下的超出量 */
  u_values = {0.5, 1.0, 1.5, 2.0, 2.5};
  do i = 1 to nrow(u_values);
    u = u_values[i];
    excess = x[loc(x > u)] - u;
    n_excess = nrow(excess);
    if n_excess > 0 then do;
      mean_excess = mean(excess);
      print "u=" u "超出数=" n_excess "平均超出量=" mean_excess;
    end;
  end;

  /* 平均超出量图 */
  u_grid = do(0, 3, 0.1)`;
  me = j(nrow(u_grid), 1, 0);
  do i = 1 to nrow(u_grid);
    u = u_grid[i];
    excess = x[loc(x > u)] - u;
    if nrow(excess) > 0 then me[i] = mean(excess);
  end;

  create me_data from u_grid [colname={'u'}];
  append from u_grid;
  close;
  create me_values from me [colname={'me'}];
  append from me;
  close;
quit;

data me_plot;
  merge me_data me_values;
run;

title "平均超出量图";
proc sgplot data=me_plot;
  scatter x=u y=me;
  xaxis label="阈值 u";
  yaxis label="平均超出量";
run;
title;


/****************************************************************************/
/* 9.7 GPD参数估计                                                          */
/****************************************************************************/
proc iml;
  /* 生成GPD数据 */
  call streaminit(789);
  n = 1000;
  sigma_true = 1;
  xi_true = 0.3;

  x = j(n, 1, 0);
  do i = 1 to n;
    u = rand('UNIFORM');
    if abs(xi_true) < 1e-8 then
      x[i] = -sigma_true * log(1 - u);
    else
      x[i] = sigma_true / xi_true * ((1 - u)##(-xi_true) - 1);
  end;

  /* MLE估计 */
  start gpd_nll(parms) global(x);
    sigma = exp(parms[1]);
    xi = parms[2];
    n = nrow(x);
    nll = 0;
    do i = 1 to n;
      if 1 + xi * x[i] / sigma <= 0 then return(1e10);
      if abs(xi) < 1e-8 then
        nll = nll + log(sigma) + x[i] / sigma;
      else
        nll = nll + log(sigma) + (1 + 1/xi) * log(1 + xi * x[i] / sigma);
    end;
    return(nll);
  finish;

  init = {0, 0.3};
  opt = {1, 0};
  call nlpnra(rc, result, "gpd_nll", init, opt);

  sigma_hat = exp(result[1]);
  xi_hat = result[2];
  print "GPD参数MLE估计";
  print "σ_true =" sigma_true "σ_hat =" sigma_hat;
  print "ξ_true =" xi_true "ξ_hat =" xi_hat;
quit;


/****************************************************************************/
/* 9.8 极值理论应用：VaR估计                                                */
/****************************************************************************/
proc iml;
  /* 使用GPD估计高置信水平VaR */
  call streaminit(101);
  n = 1000;
  x = j(n, 1, 0);
  do i = 1 to n;
    x[i] = rand('EXPONENTIAL', 1);
  end;

  /* 选择阈值 */
  u = 2;
  excess = x[loc(x > u)] - u;
  n_excess = nrow(excess);
  n_total = nrow(x);

  /* GPD参数估计（简化：使用矩估计） */
  xbar = mean(excess);
  s2 = var(excess);
  xi_hat = 0.5 * (xbar**2 / s2 - 1);
  sigma_hat = 0.5 * xbar * (xbar**2 / s2 + 1);

  /* VaR估计 */
  p_values = {0.95, 0.99, 0.999};
  do i = 1 to nrow(p_values);
    p = p_values[i];
    /* VaR_p = u + σ/ξ * [(n/n_u * (1-p))^(-ξ) - 1] */
    VaR = u + sigma_hat / xi_hat *
          ((n_total / n_excess * (1 - p))##(-xi_hat) - 1);
    print "p=" p "VaR_GPD =" VaR;
  end;

  /* 与经验VaR比较 */
  x_sorted = x;
  call sort(x_sorted, 1);
  do i = 1 to nrow(p_values);
    p = p_values[i];
    idx = ceil(p * n_total);
    VaR_emp = x_sorted[idx];
    print "p=" p "VaR_emp =" VaR_emp;
  end;
quit;

/****************************************************************************/
/* 第9章 极值理论                                                           */
/* 对应教材：section9.tex                                                   */
/* 内容：风险度量、广义极值分布（GEV）、广义帕累托分布（GPD）、              */
/*       阈值选择、参数估计、应用                                           */
/****************************************************************************/

/****************************************************************************/
/* 9.1 风险度量：VaR与TVaR                                                  */
/****************************************************************************/
proc iml;
  /* VaR_p = F^{-1}(p) */
  /* TVaR_p = E[X | X > VaR_p] */

  /* 正态分布的风险度量 */
  mu = 0; sigma = 1;
  p_values = {0.95, 0.99, 0.999};

  print "正态分布的风险度量";
  do i = 1 to nrow(p_values);
    p = p_values[i];
    VaR = quantile('NORMAL', p, mu, sigma);
    /* TVaR = μ + σ * φ(Φ^{-1}(p)) / (1-p) */
    z = quantile('NORMAL', p);
    TVaR = mu + sigma * pdf('NORMAL', z) / (1 - p);
    print "p=" p "VaR=" VaR "TVaR=" TVaR;
  end;

  /* 对数正态分布的风险度量 */
  mu_ln = 0; sigma_ln = 1;
  print "对数正态分布的风险度量";
  do i = 1 to nrow(p_values);
    p = p_values[i];
    VaR = quantile('LOGNORMAL', p, mu_ln, sigma_ln);
    print "p=" p "VaR=" VaR;
  end;
quit;


/****************************************************************************/
/* 9.2 广义极值分布（GEV）                                                  */
/****************************************************************************/
proc iml;
  /* GEV分布函数:
     - ξ > 0 (Frechet): F(x) = exp(-(1+ξx/σ)^(-1/ξ)), 1+ξx/σ > 0
     - ξ = 0 (Gumbel): F(x) = exp(-exp(-x/σ))
     - ξ < 0 (Weibull): F(x) = exp(-(1+ξx/σ)^(-1/ξ)), 1+ξx/σ > 0
  */

  /* 绘制不同ξ的GEV密度 */
  x = do(-4, 4, 0.1)`;
  sigma = 1;

  /* ξ = 0.5 (Frechet) */
  xi1 = 0.5;
  f1 = j(nrow(x), 1, 0);
  do i = 1 to nrow(x);
    if 1 + xi1 * x[i] / sigma > 0 then do;
      t = (1 + xi1 * x[i] / sigma)##(-1/xi1);
      f1[i] = t##(xi1 + 1) * exp(-t) / sigma;
    end;
  end;

  /* ξ = 0 (Gumbel) */
  xi2 = 0;
  f2 = j(nrow(x), 1, 0);
  do i = 1 to nrow(x);
    t = exp(-x[i] / sigma);
    f2[i] = t * exp(-t) / sigma;
  end;

  /* ξ = -0.5 (Weibull) */
  xi3 = -0.5;
  f3 = j(nrow(x), 1, 0);
  do i = 1 to nrow(x);
    if 1 + xi3 * x[i] / sigma > 0 then do;
      t = (1 + xi3 * x[i] / sigma)##(-1/xi3);
      f3[i] = t##(xi3 + 1) * exp(-t) / sigma;
    end;
  end;

  create gev_data from x [colname={'x'}];
  append from x;
  close;
  create gev_f1 from f1 [colname={'f1'}];
  append from f1;
  close;
  create gev_f2 from f2 [colname={'f2'}];
  append from f2;
  close;
  create gev_f3 from f3 [colname={'f3'}];
    append from f3;
  close;
quit;

data gev_plot;
  merge gev_data gev_f1 gev_f2 gev_f3;
run;

title "GEV分布密度函数";
proc sgplot data=gev_plot;
  series x=x y=f1 / legendlabel="ξ=0.5 (Frechet)";
  series x=x y=f2 / legendlabel="ξ=0 (Gumbel)";
  series x=x y=f3 / legendlabel="ξ=-0.5 (Weibull)";
  yaxis label="f(x)";
run;
title;


/****************************************************************************/
/* 9.3 区块最大值法（Block Maxima）                                         */
/****************************************************************************/
proc iml;
  /* 生成模拟数据 */
  call streaminit(123);
  n_blocks = 100;
  block_size = 50;

  maxima = j(n_blocks, 1, 0);
  do b = 1 to n_blocks;
    max_val = -1e30;
    do i = 1 to block_size;
      x = rand('EXPONENTIAL', 1);  /* 指数分布 */
      if x > max_val then max_val = x;
    end;
    maxima[b] = max_val;
  end;

  print "区块最大值法";
  print "均值:" mean(maxima);
  print "标准差:" std(maxima);
  print "最大值:" max(maxima);

  create maxima_data from maxima [colname={'maxima'}];
  append from maxima;
  close;
quit;

title "区块最大值直方图";
proc sgplot data=maxima_data;
  histogram maxima / nbins=20;
  density maxima / type=kernel;
run;
title;


/****************************************************************************/
/* 9.4 GEV参数估计                                                          */
/****************************************************************************/
proc iml;
  /* 使用极大似然估计GEV参数 */
  use maxima_data;
  read all var {maxima} into x;
  close maxima_data;

  /* 负对数似然函数 */
  start gev_nll(parms) global(x);
    mu = parms[1];
    sigma = exp(parms[2]);
    xi = parms[3];

    n = nrow(x);
    nll = 0;
    do i = 1 to n;
      z = (x[i] - mu) / sigma;
      if 1 + xi * z <= 0 then return(1e10);
      if abs(xi) < 1e-8 then do;
        t = exp(-z);
      end;
      else do;
        t = (1 + xi * z)##(-1/xi);
      end;
      nll = nll + log(sigma) + (1 + 1/xi) * log(t) + t;
    end;
    return(nll);
  finish;

  /* 优化 */
  init = {3, 0, 0.1};
  opt = {1, 0};
  call nlpnra(rc, result, "gev_nll", init, opt);

  mu_hat = result[1];
  sigma_hat = exp(result[2]);
  xi_hat = result[3];
  print "GEV参数MLE估计";
  print "μ_hat =" mu_hat;
  print "σ_hat =" sigma_hat;
  print "ξ_hat =" xi_hat;
quit;


/****************************************************************************/
/* 9.5 广义帕累托分布（GPD）                                                */
/****************************************************************************/
proc iml;
  /* GPD分布函数:
     - ξ ≠ 0: G(x) = 1 - (1 + ξx/σ)^(-1/ξ), x > 0 (ξ≥0) 或 0 < x < -σ/ξ (ξ<0)
     - ξ = 0: G(x) = 1 - exp(-x/σ)
  */

  /* 绘制不同ξ的GPD密度 */
  x = do(0, 5, 0.05)`;
  sigma = 1;

  /* ξ = 0.5 */
  xi1 = 0.5;
  f1 = (1/sigma) * (1 + xi1 * x / sigma)##(-1/xi1 - 1);

  /* ξ = 0 */
  xi2 = 0;
  f2 = (1/sigma) * exp(-x / sigma);

  /* ξ = -0.5 */
  xi3 = -0.5;
  f3 = j(nrow(x), 1, 0);
  do i = 1 to nrow(x);
    if 1 + xi3 * x[i] / sigma > 0 then
      f3[i] = (1/sigma) * (1 + xi3 * x[i] / sigma)##(-1/xi3 - 1);
  end;

  create gpd_data from x [colname={'x'}];
  append from x;
  close;
  create gpd_f1 from f1 [colname={'f1'}];
  append from f1;
  close;
  create gpd_f2 from f2 [colname={'f2'}];
  append from f2;
  close;
  create gpd_f3 from f3 [colname={'f3'}];
  append from f3;
  close;
quit;

data gpd_plot;
  merge gpd_data gpd_f1 gpd_f2 gpd_f3;
run;

title "GPD分布密度函数";
proc sgplot data=gpd_plot;
  series x=x y=f1 / legendlabel="ξ=0.5";
  series x=x y=f2 / legendlabel="ξ=0";
  series x=x y=f3 / legendlabel="ξ=-0.5";
  yaxis label="f(x)";
run;
title;


/****************************************************************************/
/* 9.6 阈值选择（POT方法）                                                  */
/****************************************************************************/
proc iml;
  /* 生成模拟数据 */
  call streaminit(456);
  n = 1000;
  x = j(n, 1, 0);
  do i = 1 to n;
    x[i] = rand('EXPONENTIAL', 1);
  end;

  /* 不同阈值下的超出量 */
  u_values = {0.5, 1.0, 1.5, 2.0, 2.5};
  do i = 1 to nrow(u_values);
    u = u_values[i];
    excess = x[loc(x > u)] - u;
    n_excess = nrow(excess);
    if n_excess > 0 then do;
      mean_excess = mean(excess);
      print "u=" u "超出数=" n_excess "平均超出量=" mean_excess;
    end;
  end;

  /* 平均超出量图 */
  u_grid = do(0, 3, 0.1)`;
  me = j(nrow(u_grid), 1, 0);
  do i = 1 to nrow(u_grid);
    u = u_grid[i];
    excess = x[loc(x > u)] - u;
    if nrow(excess) > 0 then me[i] = mean(excess);
  end;

  create me_data from u_grid [colname={'u'}];
  append from u_grid;
  close;
  create me_values from me [colname={'me'}];
  append from me;
  close;
quit;

data me_plot;
  merge me_data me_values;
run;

title "平均超出量图";
proc sgplot data=me_plot;
  scatter x=u y=me;
  xaxis label="阈值 u";
  yaxis label="平均超出量";
run;
title;


/****************************************************************************/
/* 9.7 GPD参数估计                                                          */
/****************************************************************************/
proc iml;
  /* 生成GPD数据 */
  call streaminit(789);
  n = 1000;
  sigma_true = 1;
  xi_true = 0.3;

  x = j(n, 1, 0);
  do i = 1 to n;
    u = rand('UNIFORM');
    if abs(xi_true) < 1e-8 then
      x[i] = -sigma_true * log(1 - u);
    else
      x[i] = sigma_true / xi_true * ((1 - u)##(-xi_true) - 1);
  end;

  /* MLE估计 */
  start gpd_nll(parms) global(x);
    sigma = exp(parms[1]);
    xi = parms[2];
    n = nrow(x);
    nll = 0;
    do i = 1 to n;
      if 1 + xi * x[i] / sigma <= 0 then return(1e10);
      if abs(xi) < 1e-8 then
        nll = nll + log(sigma) + x[i] / sigma;
      else
        nll = nll + log(sigma) + (1 + 1/xi) * log(1 + xi * x[i] / sigma);
    end;
    return(nll);
  finish;

  init = {0, 0.3};
  opt = {1, 0};
  call nlpnra(rc, result, "gpd_nll", init, opt);

  sigma_hat = exp(result[1]);
  xi_hat = result[2];
  print "GPD参数MLE估计";
  print "σ_true =" sigma_true "σ_hat =" sigma_hat;
  print "ξ_true =" xi_true "ξ_hat =" xi_hat;
quit;


/****************************************************************************/
/* 9.8 极值理论应用：VaR估计                                                */
/****************************************************************************/
proc iml;
  /* 使用GPD估计高置信水平VaR */
  call streaminit(101);
  n = 1000;
  x = j(n, 1, 0);
  do i = 1 to n;
    x[i] = rand('EXPONENTIAL', 1);
  end;

  /* 选择阈值 */
  u = 2;
  excess = x[loc(x > u)] - u;
  n_excess = nrow(excess);
  n_total = nrow(x);

  /* GPD参数估计（简化：使用矩估计） */
  xbar = mean(excess);
  s2 = var(excess);
  xi_hat = 0.5 * (xbar**2 / s2 - 1);
  sigma_hat = 0.5 * xbar * (xbar**2 / s2 + 1);

  /* VaR估计 */
  p_values = {0.95, 0.99, 0.999};
  do i = 1 to nrow(p_values);
    p = p_values[i];
    /* VaR_p = u + σ/ξ * [(n/n_u * (1-p))^(-ξ) - 1] */
    VaR = u + sigma_hat / xi_hat *
          ((n_total / n_excess * (1 - p))##(-xi_hat) - 1);
    print "p=" p "VaR_GPD =" VaR;
  end;

  /* 与经验VaR比较 */
  x_sorted = x;
  call sort(x_sorted, 1);
  do i = 1 to nrow(p_values);
    p = p_values[i];
    idx = ceil(p * n_total);
    VaR_emp = x_sorted[idx];
    print "p=" p "VaR_emp =" VaR_emp;
  end;
quit;

do i = 1 to nrow(p_values);
    p = p_values[i];
    VaR = quantile('NORMAL', p, mu, sigma);
    /* TVaR = μ + σ * φ(Φ^{-1}(p)) / (1-p) */
    z = quantile('NORMAL', p);
    TVaR = mu + sigma * pdf('NORMAL', z) / (1 - p);
    print "p=" p "VaR=" VaR "TVaR=" TVaR;
  end;

  /* 对数正态分布的风险度量 */
  mu_ln = 0; sigma_ln = 1;
  print "对数正态分布的风险度量";
  do i = 1 to nrow(p_values);
    p = p_values[i];
    VaR = quantile('LOGNORMAL', p, mu_ln, sigma_ln);
    print "p=" p "VaR=" VaR;
  end;
quit;


/****************************************************************************/
/* 9.2 广义极值分布（GEV）                                                  */
/****************************************************************************/
proc iml;
  /* GEV分布函数:
     - ξ > 0 (Frechet): F(x) = exp(-(1+ξx/σ)^(-1/ξ)), 1+ξx/σ > 0
     - ξ = 0 (Gumbel): F(x) = exp(-exp(-x/σ))
     - ξ < 0 (Weibull): F(x) = exp(-(1+ξx/σ)^(-1/ξ)), 1+ξx/σ > 0
  */

  /* 绘制不同ξ的GEV密度 */
  x = do(-4, 4, 0.1)`;

/****************************************************************************/
/* 第9章 极值理论                                                           */
/* 对应教材：section9.tex                                                   */
/* 内容：风险度量、广义极值分布（GEV）、广义帕累托分布（GPD）、              */
/*       阈值选择、参数估计、应用                                           */
/****************************************************************************/

/****************************************************************************/
/* 9.1 风险度量：VaR与TVaR                                                  */
/****************************************************************************/
proc iml;
  /* VaR_p = F^{-1}(p) */
  /* TVaR_p = E[X | X > VaR_p] */

  /* 正态分布的风险度量 */
  mu = 0; sigma = 1;
  p_values = {0.95, 0.99, 0.999};

  print "正态分布的风险度量";
  do i = 1 to nrow(p_values);
    p = p_values[i];
    VaR = quantile('NORMAL', p, mu, sigma);
    /* TVaR = μ + σ * φ(Φ^{-1}(p)) / (1-p) */
    z = quantile('NORMAL', p);
    TVaR = mu + sigma * pdf('NORMAL', z) / (1 - p);
    print "p=" p "VaR=" VaR "TVaR=" TVaR;
  end;

  /* 对数正态分布的风险度量 */
  mu_ln = 0; sigma_ln = 1;
  print "对数正态分布的风险度量";
  do i = 1 to nrow(p_values);
    p = p_values[i];
    VaR = quantile('LOGNORMAL', p, mu_ln, sigma_ln);
    print "p=" p "VaR=" VaR;
  end;
quit;


/****************************************************************************/
/* 9.2 广义极值分布（GEV）                                                  */
/****************************************************************************/
proc iml;
  /* GEV分布函数:
     - ξ > 0 (Frechet): F(x) = exp(-(1+ξx/σ)^(-1/ξ)), 1+ξx/σ > 0
     - ξ = 0 (Gumbel): F(x) = exp(-exp(-x/σ))
     - ξ < 0 (Weibull): F(x) = exp(-(1+ξx/σ)^(-1/ξ)), 1+ξx/σ > 0
  */

  /* 绘制不同ξ的GEV密度 */
  x = do(-4, 4, 0.1)`;
  sigma = 1;

  /* ξ = 0.5 (Frechet) */
  xi1 = 0.5;
  f1 = j(nrow(x), 1, 0);
  do i = 1 to nrow(x);
    if 1 + xi1 * x[i] / sigma > 0 then do;
      t = (1 + xi1 * x[i] / sigma)##(-1/xi1);
      f1[i] = t##(xi1 + 1) * exp(-t) / sigma;
    end;
  end;

  /* ξ = 0 (Gumbel) */
  xi2 = 0;
  f2 = j(nrow(x), 1, 0);
  do i = 1 to nrow(x);
    t = exp(-x[i] / sigma);
    f2[i] = t * exp(-t) / sigma;
  end;

  /* ξ = -0.5 (Weibull) */
  xi3 = -0.5;
  f3 = j(nrow(x), 1, 0);
  do i = 1 to nrow(x);
    if 1 + xi3 * x[i] / sigma > 0 then do;
      t = (1 + xi3 * x[i] / sigma)##(-1/xi3);
      f3[i] = t##(xi3 + 1) * exp(-t) / sigma;
    end;
  end;

  create gev_data from x [colname={'x'}];
  append from x;
  close;
  create gev_f1 from f1 [colname={'f1'}];
  append from f1;
  close;
  create gev_f2 from f2 [colname={'f2'}];
  append from f2;
  close;
  create gev_f3 from f3 [colname={'f3'}];
    append from f3;
  close;
quit;

data gev_plot;
  merge gev_data gev_f1 gev_f2 gev_f3;
run;

title "GEV分布密度函数";
proc sgplot data=gev_plot;
  series x=x y=f1 / legendlabel="ξ=0.5 (Frechet)";
  series x=x y=f2 / legendlabel="ξ=0 (Gumbel)";
  series x=x y=f3 / legendlabel="ξ=-0.5 (Weibull)";
  yaxis label="f(x)";
run;
title;


/****************************************************************************/
/* 9.3 区块最大值法（Block Maxima）                                         */
/****************************************************************************/
proc iml;
  /* 生成模拟数据 */
  call streaminit(123);
  n_blocks = 100;
  block_size = 50;

  maxima = j(n_blocks, 1, 0);
  do b = 1 to n_blocks;
    max_val = -1e30;
    do i = 1 to block_size;
      x = rand('EXPONENTIAL', 1);  /* 指数分布 */
      if x > max_val then max_val = x;
    end;
    maxima[b] = max_val;
  end;

  print "区块最大值法";
  print "均值:" mean(maxima);
  print "标准差:" std(maxima);
  print "最大值:" max(maxima);

  create maxima_data from maxima [colname={'maxima'}];
  append from maxima;
  close;
quit;

title "区块最大值直方图";
proc sgplot data=maxima_data;
  histogram maxima / nbins=20;
  density maxima / type=kernel;
run;
title;


/****************************************************************************/
/* 9.4 GEV参数估计                                                          */
/****************************************************************************/
proc iml;
  /* 使用极大似然估计GEV参数 */
  use maxima_data;
  read all var {maxima} into x;
  close maxima_data;

  /* 负对数似然函数 */
  start gev_nll(parms) global(x);
    mu = parms[1];
    sigma = exp(parms[2]);
    xi = parms[3];

    n = nrow(x);
    nll = 0;
    do i = 1 to n;
      z = (x[i] - mu) / sigma;
      if 1 + xi * z <= 0 then return(1e10);
      if abs(xi) < 1e-8 then do;
        t = exp(-z);
      end;
      else do;
        t = (1 + xi * z)##(-1/xi);
      end;
      nll = nll + log(sigma) + (1 + 1/xi) * log(t) + t;
    end;
    return(nll);
  finish;

  /* 优化 */
  init = {3, 0, 0.1};
  opt = {1, 0};
  call nlpnra(rc, result, "gev_nll", init, opt);

  mu_hat = result[1];
  sigma_hat = exp(result[2]);
  xi_hat = result[3];
  print "GEV参数MLE估计";
  print "μ_hat =" mu_hat;
  print "σ_hat =" sigma_hat;
  print "ξ_hat =" xi_hat;
quit;


/****************************************************************************/
/* 9.5 广义帕累托分布（GPD）                                                */
/****************************************************************************/
proc iml;
  /* GPD分布函数:
     - ξ ≠ 0: G(x) = 1 - (1 + ξx/σ)^(-1/ξ), x > 0 (ξ≥0) 或 0 < x < -σ/ξ (ξ<0)
     - ξ = 0: G(x) = 1 - exp(-x/σ)
  */

  /* 绘制不同ξ的GPD密度 */
  x = do(0, 5, 0.05)`;
  sigma = 1;

  /* ξ = 0.5 */
  xi1 = 0.5;
  f1 = (1/sigma) * (1 + xi1 * x / sigma)##(-1/xi1 - 1);

  /* ξ = 0 */
  xi2 = 0;
  f2 = (1/sigma) * exp(-x / sigma);

  /* ξ = -0.5 */
  xi3 = -0.5;
  f3 = j(nrow(x), 1, 0);
  do i = 1 to nrow(x);
    if 1 + xi3 * x[i] / sigma > 0 then
      f3[i] = (1/sigma) * (1 + xi3 * x[i] / sigma)##(-1/xi3 - 1);
  end;

  create gpd_data from x [colname={'x'}];
  append from x;
  close;
  create gpd_f1 from f1 [colname={'f1'}];
  append from f1;
  close;
  create gpd_f2 from f2 [colname={'f2'}];
  append from f2;
  close;
  create gpd_f3 from f3 [colname={'f3'}];
  append from f3;
  close;
quit;

data gpd_plot;
  merge gpd_data gpd_f1 gpd_f2 gpd_f3;
run;

title "GPD分布密度函数";
proc sgplot data=gpd_plot;
  series x=x y=f1 / legendlabel="ξ=0.5";
  series x=x y=f2 / legendlabel="ξ=0";
  series x=x y=f3 / legendlabel="ξ=-0.5";
  yaxis label="f(x)";
run;
title;


/****************************************************************************/
/* 9.6 阈值选择（POT方法）                                                  */
/****************************************************************************/
proc iml;
  /* 生成模拟数据 */
  call streaminit(456);
  n = 1000;
  x = j(n, 1, 0);
  do i = 1 to n;
    x[i] = rand('EXPONENTIAL', 1);
  end;

  /* 不同阈值下的超出量 */
  u_values = {0.5, 1.0, 1.5, 2.0, 2.5};
  do i = 1 to nrow(u_values);
    u = u_values[i];
    excess = x[loc(x > u)] - u;
    n_excess = nrow(excess);
    if n_excess > 0 then do;
      mean_excess = mean(excess);
      print "u=" u "超出数=" n_excess "平均超出量=" mean_excess;
    end;
  end;

  /* 平均超出量图 */
  u_grid = do(0, 3, 0.1)`;
  me = j(nrow(u_grid), 1, 0);
  do i = 1 to nrow(u_grid);
    u = u_grid[i];
    excess = x[loc(x > u)] - u;
    if nrow(excess) > 0 then me[i] = mean(excess);
  end;

  create me_data from u_grid [colname={'u'}];
  append from u_grid;
  close;
  create me_values from me [colname={'me'}];
  append from me;
  close;
quit;

data me_plot;
  merge me_data me_values;
run;

title "平均超出量图";
proc sgplot data=me_plot;
  scatter x=u y=me;
  xaxis label="阈值 u";
  yaxis label="平均超出量";
run;
title;


/****************************************************************************/
/* 9.7 GPD参数估计                                                          */
/****************************************************************************/
proc iml;
  /* 生成GPD数据 */
  call streaminit(789);
  n = 1000;
  sigma_true = 1;
  xi_true = 0.3;

  x = j(n, 1, 0);
  do i = 1 to n;
    u = rand('UNIFORM');
    if abs(xi_true) < 1e-8 then
      x[i] = -sigma_true * log(1 - u);
    else
      x[i] = sigma_true / xi_true * ((1 - u)##(-xi_true) - 1);
  end;

  /* MLE估计 */
  start gpd_nll(parms) global(x);
    sigma = exp(parms[1]);
    xi = parms[2];
    n = nrow(x);
    nll = 0;
    do i = 1 to n;
      if 1 + xi * x[i] / sigma <= 0 then return(1e10);
      if abs(xi) < 1e-8 then
        nll = nll + log(sigma) + x[i] / sigma;
      else
        nll = nll + log(sigma) + (1 + 1/xi) * log(1 + xi * x[i] / sigma);
    end;
    return(nll);
  finish;

  init = {0, 0.3};
  opt = {1, 0};
  call nlpnra(rc, result, "gpd_nll", init, opt);

  sigma_hat = exp(result[1]);
  xi_hat = result[2];
  print "GPD参数MLE估计";
  print "σ_true =" sigma_true "σ_hat =" sigma_hat;
  print "ξ_true =" xi_true "ξ_hat =" xi_hat;
quit;


/****************************************************************************/
/* 9.8 极值理论应用：VaR估计                                                */
/****************************************************************************/
proc iml;
  /* 使用GPD估计高置信水平VaR */
  call streaminit(101);
  n = 1000;
  x = j(n, 1, 0);
  do i = 1 to n;
    x[i] = rand('EXPONENTIAL', 1);
  end;

  /* 选择阈值 */
  u = 2;
  excess = x[loc(x > u)] - u;
  n_excess = nrow(excess);
  n_total = nrow(x);

  /* GPD参数估计（简化：使用矩估计） */
  xbar = mean(excess);
  s2 = var(excess);
  xi_hat = 0.5 * (xbar**2 / s2 - 1);
  sigma_hat = 0.5 * xbar * (xbar**2 / s2 + 1);

  /* VaR估计 */
  p_values = {0.95, 0.99, 0.999};
  do i = 1 to nrow(p_values);
    p = p_values[i];
    /* VaR_p = u + σ/ξ * [(n/n_u * (1-p))^(-ξ) - 1] */
    VaR = u + sigma_hat / xi_hat *
          ((n_total / n_excess * (1 - p))##(-xi_hat) - 1);
    print "p=" p "VaR_GPD =" VaR;
  end;

  /* 与经验VaR比较 */
  x_sorted = x;
  call sort(x_sorted, 1);
  do i = 1 to nrow(p_values);
    p = p_values[i];
    idx = ceil(p * n_total);
    VaR_emp = x_sorted[idx];
    print "p=" p "VaR_emp =" VaR_emp;
  end;
quit;

/****************************************************************************/
/* 第9章 极值理论                                                           */
/* 对应教材：section9.tex                                                   */
/* 内容：风险度量、广义极值分布（GEV）、广义帕累托分布（GPD）、              */
/*       阈值选择、参数估计、应用                                           */
/****************************************************************************/

/****************************************************************************/
/* 9.1 风险度量：VaR与TVaR                                                  */
/****************************************************************************/
proc iml;
  /* VaR_p = F^{-1}(p) */
  /* TVaR_p = E[X | X > VaR_p] */

  /* 正态分布的风险度量 */
  mu = 0; sigma = 1;
  p_values = {0.95, 0.99, 0.999};

  print "正态分布的风险度量";
  do i = 1 to nrow(p_values);
    p = p_values[i];
    VaR = quantile('NORMAL', p, mu, sigma);
    /* TVaR = μ + σ * φ(Φ^{-1}(p)) / (1-p) */
    z = quantile('NORMAL', p);
    TVaR = mu + sigma * pdf('NORMAL', z) / (1 - p);
    print "p=" p "VaR=" VaR "TVaR=" TVaR;
  end;

  /* 对数正态分布的风险度量 */
  mu_ln = 0; sigma_ln = 1;
  print "对数正态分布的风险度量";
  do i = 1 to nrow(p_values);
    p = p_values[i];
    VaR = quantile('LOGNORMAL', p, mu_ln, sigma_ln);
    print "p=" p "VaR=" VaR;
  end;
quit;


/****************************************************************************/
/* 9.2 广义极值分布（GEV）                                                  */
/****************************************************************************/
proc iml;
  /* GEV分布函数:
     - ξ > 0 (Frechet): F(x) = exp(-(1+ξx/σ)^(-1/ξ)), 1+ξx/σ > 0
     - ξ = 0 (Gumbel): F(x) = exp(-exp(-x/σ))
     - ξ < 0 (Weibull): F(x) = exp(-(1+ξx/σ)^(-1/ξ)), 1+ξx/σ > 0
  */

  /* 绘制不同ξ的GEV密度 */
  x = do(-4, 4, 0.1)`;
  sigma = 1;

  /* ξ = 0.5 (Frechet) */
  xi1 = 0.5;
  f1 = j(nrow(x), 1, 0);
  do i = 1 to nrow(x);
    if 1 + xi1 * x[i] / sigma > 0 then do;
      t = (1 + xi1 * x[i] / sigma)##(-1/xi1);
      f1[i] = t##(xi1 + 1) * exp(-t) / sigma;
    end;
  end;

  /* ξ = 0 (Gumbel) */
  xi2 = 0;
  f2 = j(nrow(x), 1, 0);
  do i = 1 to nrow(x);
    t = exp(-x[i] / sigma);
    f2[i] = t * exp(-t) / sigma;
  end;

  /* ξ = -0.5 (Weibull) */
  xi3 = -0.5;
  f3 = j(nrow(x), 1, 0);
  do i = 1 to nrow(x);
    if 1 + xi3 * x[i] / sigma > 0 then do;
      t = (1 + xi3 * x[i] / sigma)##(-1/xi3);
      f3[i] = t##(xi3 + 1) * exp(-t) / sigma;
    end;
  end;

  create gev_data from x [colname={'x'}];
  append from x;
  close;
  create gev_f1 from f1 [colname={'f1'}];
  append from f1;
  close;
  create gev_f2 from f2 [colname={'f2'}];
  append from f2;
  close;
  create gev_f3 from f3 [colname={'f3'}];
    append from f3;
  close;
quit;

data gev_plot;
  merge gev_data gev_f1 gev_f2 gev_f3;
run;

title "GEV分布密度函数";
proc sgplot data=gev_plot;
  series x=x y=f1 / legendlabel="ξ=0.5 (Frechet)";
  series x=x y=f2 / legendlabel="ξ=0 (Gumbel)";
  series x=x y=f3 / legendlabel="ξ=-0.5 (Weibull)";
  yaxis label="f(x)";
run;
title;


/****************************************************************************/
/* 9.3 区块最大值法（Block Maxima）                                         */
/****************************************************************************/
proc iml;
  /* 生成模拟数据 */
  call streaminit(123);
  n_blocks = 100;
  block_size = 50;

  maxima = j(n_blocks, 1, 0);
  do b = 1 to n_blocks;
    max_val = -1e30;
    do i = 1 to block_size;
      x = rand('EXPONENTIAL', 1);  /* 指数分布 */
      if x > max_val then max_val = x;
    end;
    maxima[b] = max_val;
  end;

  print "区块最大值法";
  print "均值:" mean(maxima);
  print "标准差:" std(maxima);
  print "最大值:" max(maxima);

  create maxima_data from maxima [colname={'maxima'}];
  append from maxima;
  close;
quit;

title "区块最大值直方图";
proc sgplot data=maxima_data;
  histogram maxima / nbins=20;
  density maxima / type=kernel;
run;
title;


/****************************************************************************/
/* 9.4 GEV参数估计                                                          */
/****************************************************************************/
proc iml;
  /* 使用极大似然估计GEV参数 */
  use maxima_data;
  read all var {maxima} into x;
  close maxima_data;

  /* 负对数似然函数 */
  start gev_nll(parms) global(x);
    mu = parms[1];
    sigma = exp(parms[2]);
    xi = parms[3];

    n = nrow(x);
    nll = 0;
    do i = 1 to n;
      z = (x[i] - mu) / sigma;
      if 1 + xi * z <= 0 then return(1e10);
      if abs(xi) < 1e-8 then do;
        t = exp(-z);
      end;
      else do;
        t = (1 + xi * z)##(-1/xi);
      end;
      nll = nll + log(sigma) + (1 + 1/xi) * log(t) + t;
    end;
    return(nll);
  finish;

  /* 优化 */
  init = {3, 0, 0.1};
  opt = {1, 0};
  call nlpnra(rc, result, "gev_nll", init, opt);

  mu_hat = result[1];
  sigma_hat = exp(result[2]);
  xi_hat = result[3];
  print "GEV参数MLE估计";
  print "μ_hat =" mu_hat;
  print "σ_hat =" sigma_hat;
  print "ξ_hat =" xi_hat;
quit;


/****************************************************************************/
/* 9.5 广义帕累托分布（GPD）                                                */
/****************************************************************************/
proc iml;
  /* GPD分布函数:
     - ξ ≠ 0: G(x) = 1 - (1 + ξx/σ)^(-1/ξ), x > 0 (ξ≥0) 或 0 < x < -σ/ξ (ξ<0)
     - ξ = 0: G(x) = 1 - exp(-x/σ)
  */

  /* 绘制不同ξ的GPD密度 */
  x = do(0, 5, 0.05)`;
  sigma = 1;

  /* ξ = 0.5 */
  xi1 = 0.5;
  f1 = (1/sigma) * (1 + xi1 * x / sigma)##(-1/xi1 - 1);

  /* ξ = 0 */
  xi2 = 0;
  f2 = (1/sigma) * exp(-x / sigma);

  /* ξ = -0.5 */
  xi3 = -0.5;
  f3 = j(nrow(x), 1, 0);
  do i = 1 to nrow(x);
    if 1 + xi3 * x[i] / sigma > 0 then
      f3[i] = (1/sigma) * (1 + xi3 * x[i] / sigma)##(-1/xi3 - 1);
  end;

  create gpd_data from x [colname={'x'}];
  append from x;
  close;
  create gpd_f1 from f1 [colname={'f1'}];
  append from f1;
  close;
  create gpd_f2 from f2 [colname={'f2'}];
  append from f2;
  close;
  create gpd_f3 from f3 [colname={'f3'}];
  append from f3;
  close;
quit;

data gpd_plot;
  merge gpd_data gpd_f1 gpd_f2 gpd_f3;
run;

title "GPD分布密度函数";
proc sgplot data=gpd_plot;
  series x=x y=f1 / legendlabel="ξ=0.5";
  series x=x y=f2 / legendlabel="ξ=0";
  series x=x y=f3 / legendlabel="ξ=-0.5";
  yaxis label="f(x)";
run;
title;


/****************************************************************************/
/* 9.6 阈值选择（POT方法）                                                  */
/****************************************************************************/
proc iml;
  /* 生成模拟数据 */
  call streaminit(456);
  n = 1000;
  x = j(n, 1, 0);
  do i = 1 to n;
    x[i] = rand('EXPONENTIAL', 1);
  end;

  /* 不同阈值下的超出量 */
  u_values = {0.5, 1.0, 1.5, 2.0, 2.5};
  do i = 1 to nrow(u_values);
    u = u_values[i];
    excess = x[loc(x > u)] - u;
    n_excess = nrow(excess);
    if n_excess > 0 then do;
      mean_excess = mean(excess);
      print "u=" u "超出数=" n_excess "平均超出量=" mean_excess;
    end;
  end;

  /* 平均超出量图 */
  u_grid = do(0, 3, 0.1)`;
  me = j(nrow(u_grid), 1, 0);
  do i = 1 to nrow(u_grid);
    u = u_grid[i];
    excess = x[loc(x > u)] - u;
    if nrow(excess) > 0 then me[i] = mean(excess);
  end;

  create me_data from u_grid [colname={'u'}];
  append from u_grid;
  close;
  create me_values from me [colname={'me'}];
  append from me;
  close;
quit;

data me_plot;
  merge me_data me_values;
run;

title "平均超出量图";
proc sgplot data=me_plot;
  scatter x=u y=me;
  xaxis label="阈值 u";
  yaxis label="平均超出量";
run;
title;


/****************************************************************************/
/* 9.7 GPD参数估计                                                          */
/****************************************************************************/
proc iml;
  /* 生成GPD数据 */
  call streaminit(789);
  n = 1000;
  sigma_true = 1;
  xi_true = 0.3;

  x = j(n, 1, 0);
  do i = 1 to n;
    u = rand('UNIFORM');
    if abs(xi_true) < 1e-8 then
      x[i] = -sigma_true * log(1 - u);
    else
      x[i] = sigma_true / xi_true * ((1 - u)##(-xi_true) - 1);
  end;

  /* MLE估计 */
  start gpd_nll(parms) global(x);
    sigma = exp(parms[1]);
    xi = parms[2];
    n = nrow(x);
    nll = 0;
    do i = 1 to n;
      if 1 + xi * x[i] / sigma <= 0 then return(1e10);
      if abs(xi) < 1e-8 then
        nll = nll + log(sigma) + x[i] / sigma;
      else
        nll = nll + log(sigma) + (1 + 1/xi) * log(1 + xi * x[i] / sigma);
    end;
    return(nll);
  finish;

  init = {0, 0.3};
  opt = {1, 0};
  call nlpnra(rc, result, "gpd_nll", init, opt);

  sigma_hat = exp(result[1]);
  xi_hat = result[2];
  print "GPD参数MLE估计";
  print "σ_true =" sigma_true "σ_hat =" sigma_hat;
  print "ξ_true =" xi_true "ξ_hat =" xi_hat;
quit;


/****************************************************************************/
/* 9.8 极值理论应用：VaR估计                                                */
/****************************************************************************/
proc iml;
  /* 使用GPD估计高置信水平VaR */
  call streaminit(101);
  n = 1000;
  x = j(n, 1, 0);
  do i = 1 to n;
    x[i] = rand('EXPONENTIAL', 1);
  end;

  /* 选择阈值 */
  u = 2;
  excess = x[loc(x > u)] - u;
  n_excess = nrow(excess);
  n_total = nrow(x);

  /* GPD参数估计（简化：使用矩估计） */
  xbar = mean(excess);
  s2 = var(excess);
  xi_hat = 0.5 * (xbar**2 / s2 - 1);
  sigma_hat = 0.5 * xbar * (xbar**2 / s2 + 1);

  /* VaR估计 */
  p_values = {0.95, 0.99, 0.999};
  do i = 1 to nrow(p_values);
    p = p_values[i];
    /* VaR_p = u + σ/ξ * [(n/n_u * (1-p))^(-ξ) - 1] */
    VaR = u + sigma_hat / xi_hat *
          ((n_total / n_excess * (1 - p))##(-xi_hat) - 1);
    print "p=" p "VaR_GPD =" VaR;
  end;

  /* 与经验VaR比较 */
  x_sorted = x;
  call sort(x_sorted, 1);
  do i = 1 to nrow(p_values);
    p = p_values[i];
    idx = ceil(p * n_total);
    VaR_emp = x_sorted[idx];
    print "p=" p "VaR_emp =" VaR_emp;
  end;
quit;

do i = 1 to nrow(p_values);
    p = p_values[i];
    VaR = quantile('NORMAL', p, mu, sigma);
    /* TVaR = μ + σ * φ(Φ^{-1}(p)) / (1-p) */
    z = quantile('NORMAL', p);
    TVaR = mu + sigma * pdf('NORMAL', z) / (1 - p);
    print "p=" p "VaR=" VaR "TVaR=" TVaR;
  end;

  /* 对数正态分布的风险度量 */
  mu_ln = 0; sigma_ln = 1;
  print "对数正态分布的风险度量";
  do i = 1 to nrow(p_values);
    p = p_values[i];
    VaR = quantile('LOGNORMAL', p, mu_ln, sigma_ln);
    print "p=" p "VaR=" VaR;
  end;
quit;


/****************************************************************************/
/* 9.2 广义极值分布（GEV）                                                  */
/****************************************************************************/
proc iml;
  /* GEV分布函数:
     - ξ > 0 (Frechet): F(x) = exp(-(1+ξx/σ)^(-1/ξ)), 1+ξx/σ > 0
     - ξ = 0 (Gumbel): F(x) = exp(-exp(-x/σ))
     - ξ < 0 (Weibull): F(x) = exp(-(1+ξx/σ)^(-1/ξ)), 1+ξx/σ > 0
  */

  /* 绘制不同ξ的GEV密度 */
  x = do(-4, 4, 0.1)`;

/****************************************************************************/
/* 第9章 极值理论                                                           */
/* 对应教材：section9.tex                                                   */
/* 内容：风险度量、广义极值分布（GEV）、广义帕累托分布（GPD）、              */
/*       阈值选择、参数估计、应用                                           */
/****************************************************************************/

/****************************************************************************/
/* 9.1 风险度量：VaR与TVaR                                                  */
/****************************************************************************/
proc iml;
  /* VaR_p = F^{-1}(p) */
  /* TVaR_p = E[X | X > VaR_p] */

  /* 正态分布的风险度量 */
  mu = 0; sigma = 1;
  p_values = {0.95, 0.99, 0.999};

  print "正态分布的风险度量";
  do i = 1 to nrow(p_values);
    p = p_values[i];
    VaR = quantile('NORMAL', p, mu, sigma);
    /* TVaR = μ + σ * φ(Φ^{-1}(p)) / (1-p) */
    z = quantile('NORMAL', p);
    TVaR = mu + sigma * pdf('NORMAL', z) / (1 - p);
    print "p=" p "VaR=" VaR "TVaR=" TVaR;
  end;

  /* 对数正态分布的风险度量 */
  mu_ln = 0; sigma_ln = 1;
  print "对数正态分布的风险度量";
  do i = 1 to nrow(p_values);
    p = p_values[i];
    VaR = quantile('LOGNORMAL', p, mu_ln, sigma_ln);
    print "p=" p "VaR=" VaR;
  end;
quit;


/****************************************************************************/
/* 9.2 广义极值分布（GEV）                                                  */
/****************************************************************************/
proc iml;
  /* GEV分布函数:
     - ξ > 0 (Frechet): F(x) = exp(-(1+ξx/σ)^(-1/ξ)), 1+ξx/σ > 0
     - ξ = 0 (Gumbel): F(x) = exp(-exp(-x/σ))
     - ξ < 0 (Weibull): F(x) = exp(-(1+ξx/σ)^(-1/ξ)), 1+ξx/σ > 0
  */

  /* 绘制不同ξ的GEV密度 */
  x = do(-4, 4, 0.1)`;
  sigma = 1;

  /* ξ = 0.5 (Frechet) */
  xi1 = 0.5;
  f1 = j(nrow(x), 1, 0);
  do i = 1 to nrow(x);
    if 1 + xi1 * x[i] / sigma > 0 then do;
      t = (1 + xi1 * x[i] / sigma)##(-1/xi1);
      f1[i] = t##(xi1 + 1) * exp(-t) / sigma;
    end;
  end;

  /* ξ = 0 (Gumbel) */
  xi2 = 0;
  f2 = j(nrow(x), 1, 0);
  do i = 1 to nrow(x);
    t = exp(-x[i] / sigma);
    f2[i] = t * exp(-t) / sigma;
  end;

  /* ξ = -0.5 (Weibull) */
  xi3 = -0.5;
  f3 = j(nrow(x), 1, 0);
  do i = 1 to nrow(x);
    if 1 + xi3 * x[i] / sigma > 0 then do;
      t = (1 + xi3 * x[i] / sigma)##(-1/xi3);
      f3[i] = t##(xi3 + 1) * exp(-t) / sigma;
    end;
  end;

  create gev_data from x [colname={'x'}];
  append from x;
  close;
  create gev_f1 from f1 [colname={'f1'}];
  append from f1;
  close;
  create gev_f2 from f2 [colname={'f2'}];
  append from f2;
  close;
  create gev_f3 from f3 [colname={'f3'}];
    append from f3;
  close;
quit;

data gev_plot;
  merge gev_data gev_f1 gev_f2 gev_f3;
run;

title "GEV分布密度函数";
proc sgplot data=gev_plot;
  series x=x y=f1 / legendlabel="ξ=0.5 (Frechet)";
  series x=x y=f2 / legendlabel="ξ=0 (Gumbel)";
  series x=x y=f3 / legendlabel="ξ=-0.5 (Weibull)";
  yaxis label="f(x)";
run;
title;


/****************************************************************************/
/* 9.3 区块最大值法（Block Maxima）                                         */
/****************************************************************************/
proc iml;
  /* 生成模拟数据 */
  call streaminit(123);
  n_blocks = 100;
  block_size = 50;

  maxima = j(n_blocks, 1, 0);
  do b = 1 to n_blocks;
    max_val = -1e30;
    do i = 1 to block_size;
      x = rand('EXPONENTIAL', 1);  /* 指数分布 */
      if x > max_val then max_val = x;
    end;
    maxima[b] = max_val;
  end;

  print "区块最大值法";
  print "均值:" mean(maxima);
  print "标准差:" std(maxima);
  print "最大值:" max(maxima);

  create maxima_data from maxima [colname={'maxima'}];
  append from maxima;
  close;
quit;

title "区块最大值直方图";
proc sgplot data=maxima_data;
  histogram maxima / nbins=20;
  density maxima / type=kernel;
run;
title;


/****************************************************************************/
/* 9.4 GEV参数估计                                                          */
/****************************************************************************/
proc iml;
  /* 使用极大似然估计GEV参数 */
  use maxima_data;
  read all var {maxima} into x;
  close maxima_data;

  /* 负对数似然函数 */
  start gev_nll(parms) global(x);
    mu = parms[1];
    sigma = exp(parms[2]);
    xi = parms[3];

    n = nrow(x);
    nll = 0;
    do i = 1 to n;
      z = (x[i] - mu) / sigma;
      if 1 + xi * z <= 0 then return(1e10);
      if abs(xi) < 1e-8 then do;
        t = exp(-z);
      end;
      else do;
        t = (1 + xi * z)##(-1/xi);
      end;
      nll = nll + log(sigma) + (1 + 1/xi) * log(t) + t;
    end;
    return(nll);
  finish;

  /* 优化 */
  init = {3, 0, 0.1};
  opt = {1, 0};
  call nlpnra(rc, result, "gev_nll", init, opt);

  mu_hat = result[1];
  sigma_hat = exp(result[2]);
  xi_hat = result[3];
  print "GEV参数MLE估计";
  print "μ_hat =" mu_hat;
  print "σ_hat =" sigma_hat;
  print "ξ_hat =" xi_hat;
quit;


/****************************************************************************/
/* 9.5 广义帕累托分布（GPD）                                                */
/****************************************************************************/
proc iml;
  /* GPD分布函数:
     - ξ ≠ 0: G(x) = 1 - (1 + ξx/σ)^(-1/ξ), x > 0 (ξ≥0) 或 0 < x < -σ/ξ (ξ<0)
     - ξ = 0: G(x) = 1 - exp(-x/σ)
  */

  /* 绘制不同ξ的GPD密度 */
  x = do(0, 5, 0.05)`;
  sigma = 1;

  /* ξ = 0.5 */
  xi1 = 0.5;
  f1 = (1/sigma) * (1 + xi1 * x / sigma)##(-1/xi1 - 1);

  /* ξ = 0 */
  xi2 = 0;
  f2 = (1/sigma) * exp(-x / sigma);

  /* ξ = -0.5 */
  xi3 = -0.5;
  f3 = j(nrow(x), 1, 0);
  do i = 1 to nrow(x);
    if 1 + xi3 * x[i] / sigma > 0 then
      f3[i] = (1/sigma) * (1 + xi3 * x[i] / sigma)##(-1/xi3 - 1);
  end;

  create gpd_data from x [colname={'x'}];
  append from x;
  close;
  create gpd_f1 from f1 [colname={'f1'}];
  append from f1;
  close;
  create gpd_f2 from f2 [colname={'f2'}];
  append from f2;
  close;
  create gpd_f3 from f3 [colname={'f3'}];
  append from f3;
  close;
quit;

data gpd_plot;
  merge gpd_data gpd_f1 gpd_f2 gpd_f3;
run;

title "GPD分布密度函数";
proc sgplot data=gpd_plot;
  series x=x y=f1 / legendlabel="ξ=0.5";
  series x=x y=f2 / legendlabel="ξ=0";
  series x=x y=f3 / legendlabel="ξ=-0.5";
  yaxis label="f(x)";
run;
title;


/****************************************************************************/
/* 9.6 阈值选择（POT方法）                                                  */
/****************************************************************************/
proc iml;
  /* 生成模拟数据 */
  call streaminit(456);
  n = 1000;
  x = j(n, 1, 0);
  do i = 1 to n;
    x[i] = rand('EXPONENTIAL', 1);
  end;

  /* 不同阈值下的超出量 */
  u_values = {0.5, 1.0, 1.5, 2.0, 2.5};
  do i = 1 to nrow(u_values);
    u = u_values[i];
    excess = x[loc(x > u)] - u;
    n_excess = nrow(excess);
    if n_excess > 0 then do;
      mean_excess = mean(excess);
      print "u=" u "超出数=" n_excess "平均超出量=" mean_excess;
    end;
  end;

  /* 平均超出量图 */
  u_grid = do(0, 3, 0.1)`;
  me = j(nrow(u_grid), 1, 0);
  do i = 1 to nrow(u_grid);
    u = u_grid[i];
    excess = x[loc(x > u)] - u;
    if nrow(excess) > 0 then me[i] = mean(excess);
  end;

  create me_data from u_grid [colname={'u'}];
  append from u_grid;
  close;
  create me_values from me [colname={'me'}];
  append from me;
  close;
quit;

data me_plot;
  merge me_data me_values;
run;

title "平均超出量图";
proc sgplot data=me_plot;
  scatter x=u y=me;
  xaxis label="阈值 u";
  yaxis label="平均超出量";
run;
title;


/****************************************************************************/
/* 9.7 GPD参数估计                                                          */
/****************************************************************************/
proc iml;
  /* 生成GPD数据 */
  call streaminit(789);
  n = 1000;
  sigma_true = 1;
  xi_true = 0.3;

  x = j(n, 1, 0);
  do i = 1 to n;
    u = rand('UNIFORM');
    if abs(xi_true) < 1e-8 then
      x[i] = -sigma_true * log(1 - u);
    else
      x[i] = sigma_true / xi_true * ((1 - u)##(-xi_true) - 1);
  end;

  /* MLE估计 */
  start gpd_nll(parms) global(x);
    sigma = exp(parms[1]);
    xi = parms[2];
    n = nrow(x);
    nll = 0;
    do i = 1 to n;
      if 1 + xi * x[i] / sigma <= 0 then return(1e10);
      if abs(xi) < 1e-8 then
        nll = nll + log(sigma) + x[i] / sigma;
      else
        nll = nll + log(sigma) + (1 + 1/xi) * log(1 + xi * x[i] / sigma);
    end;
    return(nll);
  finish;

  init = {0, 0.3};
  opt = {1, 0};
  call nlpnra(rc, result, "gpd_nll", init, opt);

  sigma_hat = exp(result[1]);
  xi_hat = result[2];
  print "GPD参数MLE估计";
  print "σ_true =" sigma_true "σ_hat =" sigma_hat;
  print "ξ_true =" xi_true "ξ_hat =" xi_hat;
quit;


/****************************************************************************/
/* 9.8 极值理论应用：VaR估计                                                */
/****************************************************************************/
proc iml;
  /* 使用GPD估计高置信水平VaR */
  call streaminit(101);
  n = 1000;
  x = j(n, 1, 0);
  do i = 1 to n;
    x[i] = rand('EXPONENTIAL', 1);
  end;

  /* 选择阈值 */
  u = 2;
  excess = x[loc(x > u)] - u;
  n_excess = nrow(excess);
  n_total = nrow(x);

  /* GPD参数估计（简化：使用矩估计） */
  xbar = mean(excess);
  s2 = var(excess);
  xi_hat = 0.5 * (xbar**2 / s2 - 1);
  sigma_hat = 0.5 * xbar * (xbar**2 / s2 + 1);

  /* VaR估计 */
  p_values = {0.95, 0.99, 0.999};
  do i = 1 to nrow(p_values);
    p = p_values[i];
    /* VaR_p = u + σ/ξ * [(n/n_u * (1-p))^(-ξ) - 1] */
    VaR = u + sigma_hat / xi_hat *
          ((n_total / n_excess * (1 - p))##(-xi_hat) - 1);
    print "p=" p "VaR_GPD =" VaR;
  end;

  /* 与经验VaR比较 */
  x_sorted = x;
  call sort(x_sorted, 1);
  do i = 1 to nrow(p_values);
    p = p_values[i];
    idx = ceil(p * n_total);
    VaR_emp = x_sorted[idx];
    print "p=" p "VaR_emp =" VaR_emp;
  end;
quit;