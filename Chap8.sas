/* Chap8 SAS代码 */
/* 自动从chap8.html同步生成 */

/****************************************************************************/
/* 第8章 相依风险与Copula                                                   */
/* 对应教材：section8.tex                                                   */
/* 内容：Copula概念、常见Copula函数、参数估计、尾部相关性、                  */
/*       多元Copula、 vine Copula                                            */
/****************************************************************************/

/****************************************************************************/
/* 8.1 Copula基本概念                                                       */
/****************************************************************************/
proc iml;
  /* Sklar定理：F(x,y) = C(F_X(x), F_Y(y)) */
  /* Copula是将边缘分布连接为联合分布的函数 */

  /* 独立Copula: C(u,v) = u*v */
  u = {0.1, 0.3, 0.5, 0.7, 0.9};
  v = {0.2, 0.4, 0.6, 0.8, 1.0};
  C_indep = u # v;
  print "独立Copula" u v C_indep;
quit;


/****************************************************************************/
/* 8.2 高斯Copula                                                           */
/****************************************************************************/
proc iml;
  /* 高斯Copula: C(u,v;ρ) = Φ_ρ(Φ⁻¹(u), Φ⁻¹(v)) */
  rho = 0.5;
  u = {0.1, 0.3, 0.5, 0.7, 0.9};
  v = {0.2, 0.4, 0.6, 0.8, 1.0};

  /* 转换为标准正态 */
  z1 = quantile('NORMAL', u);
  z2 = quantile('NORMAL', v);

  /* 二元正态CDF */
  C_gaussian = j(nrow(u), 1, 0);
  do i = 1 to nrow(u);
    C_gaussian[i] = probbnrm(z1[i], z2[i], rho);
  end;

  print "高斯Copula (ρ=0.5)" u v C_gaussian;
quit;


/* 8.2.1 高斯Copula模拟 */
proc iml;
  call streaminit(123);
  rho = 0.7;
  n = 1000;

  /* 生成相关正态随机数 */
  z = j(n, 2, 0);
  do i = 1 to n;
    e1 = rand('NORMAL', 0, 1);
    e2 = rand('NORMAL', 0, 1);
    z[i, 1] = e1;
    z[i, 2] = rho * e1 + sqrt(1 - rho**2) * e2;
  end;

  /* 转换为均匀分布 */
  u = cdf('NORMAL', z, 0, 1);

  /* 转换为任意边缘分布（这里用指数分布） */
  x1 = quantile('EXPONENTIAL', u[, 1], 1);
  x2 = quantile('EXPONENTIAL', u[, 2], 1);

  /* 计算相关系数 */
  corr_emp = corr(x1 || x2);
  print "高斯Copula模拟";
  print "经验相关系数:" corr_emp;

  /* 绘图数据 */
  create gauss_copula from x1 [colname={'x1'}];
  append from x1;
  close;
  create gauss_copula2 from x2 [colname={'x2'}];
  append from x2;
  close;
quit;

data gauss_plot;
  merge gauss_copula gauss_copula2;
  i = _n_;
run;

title "高斯Copula模拟散点图";
proc sgplot data=gauss_plot;
  scatter x=x1 y=x2;
run;
title;


/****************************************************************************/
/* 8.3 T Copula                                                             */
/****************************************************************************/
proc iml;
  call streaminit(456);
  rho = 0.5;
  df = 4;
  n = 1000;

  z = j(n, 2, 0);
  do i = 1 to n;
    e1 = rand('NORMAL', 0, 1);
    e2 = rand('NORMAL', 0, 1);
    z[i, 1] = e1;
    z[i, 2] = rho * e1 + sqrt(1 - rho**2) * e2;
    /* T分布 */
    w = rand('CHISQUARE', df) / df;
    z[i, ] = z[i, ] / sqrt(w);
  end;

  /* 转换为均匀分布 */
  u = cdf('T', z, df);

  /* 转换为指数分布 */
  x1 = quantile('EXPONENTIAL', u[, 1], 1);
  x2 = quantile('EXPONENTIAL', u[, 2], 1);

  corr_emp = corr(x1 || x2);
  print "T Copula模拟 (df=4)";
  print "经验相关系数:" corr_emp;
quit;


/****************************************************************************/
/* 8.4 阿基米德Copula                                                       */
/****************************************************************************/
proc iml;
  /* 8.4.1 Gumbel Copula: C(u,v;θ) = exp(-[(-ln u)^θ + (-ln v)^θ]^(1/θ)) */
  theta = 2;
  u = {0.1, 0.3, 0.5, 0.7, 0.9};
  v = {0.2, 0.4, 0.6, 0.8, 1.0};

  C_gumbel = exp(-((-log(u))##theta + (-log(v))##theta)##(1/theta));
  print "Gumbel Copula (θ=2)" u v C_gumbel;

  /* 8.4.2 Clayton Copula: C(u,v;θ) = (u^(-θ) + v^(-θ) - 1)^(-1/θ) */
  theta_c = 2;
  C_clayton = (u##(-theta_c) + v##(-theta_c) - 1)##(-1/theta_c);
  print "Clayton Copula (θ=2)" u v C_clayton;

  /* 8.4.3 Frank Copula */
  theta_f = 5;
  C_frank = -1/theta_f * log(1 +
             (exp(-theta_f * u) - 1) * (exp(-theta_f * v) - 1) /
             (exp(-theta_f) - 1));
  print "Frank Copula (θ=5)" u v C_frank;
quit;


/* 8.4.4 Clayton Copula模拟 */
proc iml;
  call streaminit(789);
  theta = 2;
  n = 1000;

  x1 = j(n, 1, 0);
  x2 = j(n, 1, 0);
  do i = 1 to n;
    u1 = rand('UNIFORM');
    t = rand('UNIFORM');
    /* Clayton Copula的条件分布法 */
    u2 = (1 - u1##(-theta) + u1##(-theta) * t##(-theta/(1+theta)))##(-1/theta);
    /* 转换为指数分布 */
    x1[i] = quantile('EXPONENTIAL', u1, 1);
    x2[i] = quantile('EXPONENTIAL', u2, 1);
  end;

  corr_emp = corr(x1 || x2);
  print "Clayton Copula模拟 (θ=2)";
  print "经验相关系数:" corr_emp;

  create clayton_data from x1 [colname={'x1'}];
  append from x1;
  close;
  create clayton_data2 from x2 [colname={'x2'}];
  append from x2;
  close;
quit;

data clayton_plot;
  merge clayton_data clayton_data2;
run;

title "Clayton Copula模拟散点图";
proc sgplot data=clayton_plot;
  scatter x=x1 y=x2;
run;
title;


/****************************************************************************/
/* 8.5 尾部相关性                                                           */
/****************************************************************************/
proc iml;
  /* 尾部相关性: λ(U) = P(V>u | U>u) as u→1 */

  /* 高斯Copula的尾部相关性 */
  rho_values = {0.0, 0.3, 0.5, 0.7, 0.9};
  print "高斯Copula的尾部相关性（渐近为0）";
  do i = 1 to nrow(rho_values);
    rho = rho_values[i];
    /* 高斯Copula的上下尾相关性都为0（ρ<1时） */
    print "ρ=" rho "λ_upper=0 λ_lower=0";
  end;

  /* T Copula的尾部相关性 */
  print "T Copula的尾部相关性";
  do i = 1 to nrow(rho_values);
    rho = rho_values[i];
    df = 4;
    /* T Copula尾部相关性公式 */
    lambda = 2 * tinv(1 - 0.5, df + 1) * (1 - rho) /
             sqrt((df + 1) * (1 - rho**2));
    print "ρ=" rho "λ=" lambda;
  end;

  /* Clayton Copula的下尾相关性 */
  print "Clayton Copula的下尾相关性";
  theta_values = {1, 2, 5, 10};
  do i = 1 to nrow(theta_values);
    theta = theta_values[i];
    lambda_lower = 2##(-1/theta);
    print "θ=" theta "λ_lower=" lambda_lower;
  end;

  /* Gumbel Copula的上尾相关性 */
  print "Gumbel Copula的上尾相关性";
  do i = 1 to nrow(theta_values);
    theta = theta_values[i];
    lambda_upper = 2 - 2##(1/theta);
    print "θ=" theta "λ_upper=" lambda_upper;
  end;
quit;


/****************************************************************************/
/* 8.6 Copula参数估计                                                       */
/****************************************************************************/
proc iml;
  /* 生成Clayton Copula数据 */
  call streaminit(123);
  theta_true = 2;
  n = 1000;

  x1 = j(n, 1, 0);
  x2 = j(n, 1, 0);
  do i = 1 to n;
    u1 = rand('UNIFORM');
    t = rand('UNIFORM');
    u2 = (1 - u1##(-theta_true) + u1##(-theta_true) * t##(-theta_true/(1+theta_true)))##(-1/theta_true);
    x1[i] = quantile('EXPONENTIAL', u1, 1);
    x2[i] = quantile('EXPONENTIAL', u2, 1);
  end;

  /* 方法1：矩估计（基于Kendall tau） */
  /* Kendall tau = θ / (θ + 2) for Clayton */
  tau = 0;
  do i = 1 to n-1;
    do j = i+1 to n;
      a = sign(x1[i] - x1[j]) * sign(x2[i] - x2[j]);
      tau = tau + a;
    end;
  end;
  tau = tau / (n * (n-1) / 2);
  theta_hat = 2 * tau / (1 - tau);
  print "Clayton Copula参数估计";
  print "Kendall tau =" tau;
  print "θ_hat（矩估计）=" theta_hat;
quit;


/****************************************************************************/
/* 8.7 多元Copula应用                                                       */
/****************************************************************************/
proc iml;
  /* 三元高斯Copula */
  call streaminit(202);
  n = 1000;

  /* 相关矩阵 */
  Sigma = {1.0 0.5 0.3,
           0.5 1.0 0.4,
           0.3 0.4 1.0};

  /* 生成多元正态 */
  z = j(n, 3, 0);
  do i = 1 to n;
    e = rand('NORMAL', 0, 1) || rand('NORMAL', 0, 1) || rand('NORMAL', 0, 1);
    L = root(Sigma);
    z[i, ] = t(L * t(e));
  end;

  /* 转换为均匀分布 */
  u = cdf('NORMAL', z, 0, 1);

  /* 转换为不同边缘分布 */
  x1 = quantile('EXPONENTIAL', u[, 1], 1);      /* 指数分布 */
  x2 = quantile('GAMMA', u[, 2], 2, 1);         /* 伽马分布 */
  x3 = quantile('LOGNORMAL', u[, 3], 0, 1);     /* 对数正态分布 */

  corr_emp = corr(x1 || x2 || x3);
  print "三元高斯Copula";
  print "经验相关矩阵:" corr_emp;
quit;

/****************************************************************************/
/* 第8章 相依风险与Copula                                                   */
/* 对应教材：section8.tex                                                   */
/* 内容：Copula概念、常见Copula函数、参数估计、尾部相关性、                  */
/*       多元Copula、 vine Copula                                            */
/****************************************************************************/

/****************************************************************************/
/* 8.1 Copula基本概念                                                       */
/****************************************************************************/
proc iml;
  /* Sklar定理：F(x,y) = C(F_X(x), F_Y(y)) */
  /* Copula是将边缘分布连接为联合分布的函数 */

  /* 独立Copula: C(u,v) = u*v */
  u = {0.1, 0.3, 0.5, 0.7, 0.9};
  v = {0.2, 0.4, 0.6, 0.8, 1.0};
  C_indep = u # v;
  print "独立Copula" u v C_indep;
quit;


/****************************************************************************/
/* 8.2 高斯Copula                                                           */
/****************************************************************************/
proc iml;
  /* 高斯Copula: C(u,v;ρ) = Φ_ρ(Φ⁻¹(u), Φ⁻¹(v)) */
  rho = 0.5;
  u = {0.1, 0.3, 0.5, 0.7, 0.9};
  v = {0.2, 0.4, 0.6, 0.8, 1.0};

  /* 转换为标准正态 */
  z1 = quantile('NORMAL', u);
  z2 = quantile('NORMAL', v);

  /* 二元正态CDF */
  C_gaussian = j(nrow(u), 1, 0);
  do i = 1 to nrow(u);
    C_gaussian[i] = probbnrm(z1[i], z2[i], rho);
  end;

  print "高斯Copula (ρ=0.5)" u v C_gaussian;
quit;


/* 8.2.1 高斯Copula模拟 */
proc iml;
  call streaminit(123);
  rho = 0.7;
  n = 1000;

  /* 生成相关正态随机数 */
  z = j(n, 2, 0);
  do i = 1 to n;
    e1 = rand('NORMAL', 0, 1);
    e2 = rand('NORMAL', 0, 1);
    z[i, 1] = e1;
    z[i, 2] = rho * e1 + sqrt(1 - rho**2) * e2;
  end;

  /* 转换为均匀分布 */
  u = cdf('NORMAL', z, 0, 1);

  /* 转换为任意边缘分布（这里用指数分布） */
  x1 = quantile('EXPONENTIAL', u[, 1], 1);
  x2 = quantile('EXPONENTIAL', u[, 2], 1);

  /* 计算相关系数 */
  corr_emp = corr(x1 || x2);
  print "高斯Copula模拟";
  print "经验相关系数:" corr_emp;

  /* 绘图数据 */
  create gauss_copula from x1 [colname={'x1'}];
  append from x1;
  close;
  create gauss_copula2 from x2 [colname={'x2'}];
  append from x2;
  close;
quit;

data gauss_plot;
  merge gauss_copula gauss_copula2;
  i = _n_;
run;

title "高斯Copula模拟散点图";
proc sgplot data=gauss_plot;
  scatter x=x1 y=x2;
run;
title;


/****************************************************************************/
/* 8.3 T Copula                                                             */
/****************************************************************************/
proc iml;
  call streaminit(456);
  rho = 0.5;
  df = 4;
  n = 1000;

  z = j(n, 2, 0);
  do i = 1 to n;
    e1 = rand('NORMAL', 0, 1);
    e2 = rand('NORMAL', 0, 1);
    z[i, 1] = e1;
    z[i, 2] = rho * e1 + sqrt(1 - rho**2) * e2;
    /* T分布 */
    w = rand('CHISQUARE', df) / df;
    z[i, ] = z[i, ] / sqrt(w);
  end;

  /* 转换为均匀分布 */
  u = cdf('T', z, df);

  /* 转换为指数分布 */
  x1 = quantile('EXPONENTIAL', u[, 1], 1);
  x2 = quantile('EXPONENTIAL', u[, 2], 1);

  corr_emp = corr(x1 || x2);
  print "T Copula模拟 (df=4)";
  print "经验相关系数:" corr_emp;
quit;


/****************************************************************************/
/* 8.4 阿基米德Copula                                                       */
/****************************************************************************/
proc iml;
  /* 8.4.1 Gumbel Copula: C(u,v;θ) = exp(-[(-ln u)^θ + (-ln v)^θ]^(1/θ)) */
  theta = 2;
  u = {0.1, 0.3, 0.5, 0.7, 0.9};
  v = {0.2, 0.4, 0.6, 0.8, 1.0};

  C_gumbel = exp(-((-log(u))##theta + (-log(v))##theta)##(1/theta));
  print "Gumbel Copula (θ=2)" u v C_gumbel;

  /* 8.4.2 Clayton Copula: C(u,v;θ) = (u^(-θ) + v^(-θ) - 1)^(-1/θ) */
  theta_c = 2;
  C_clayton = (u##(-theta_c) + v##(-theta_c) - 1)##(-1/theta_c);
  print "Clayton Copula (θ=2)" u v C_clayton;

  /* 8.4.3 Frank Copula */
  theta_f = 5;
  C_frank = -1/theta_f * log(1 +
             (exp(-theta_f * u) - 1) * (exp(-theta_f * v) - 1) /
             (exp(-theta_f) - 1));
  print "Frank Copula (θ=5)" u v C_frank;
quit;


/* 8.4.4 Clayton Copula模拟 */
proc iml;
  call streaminit(789);
  theta = 2;
  n = 1000;

  x1 = j(n, 1, 0);
  x2 = j(n, 1, 0);
  do i = 1 to n;
    u1 = rand('UNIFORM');
    t = rand('UNIFORM');
    /* Clayton Copula的条件分布法 */
    u2 = (1 - u1##(-theta) + u1##(-theta) * t##(-theta/(1+theta)))##(-1/theta);
    /* 转换为指数分布 */
    x1[i] = quantile('EXPONENTIAL', u1, 1);
    x2[i] = quantile('EXPONENTIAL', u2, 1);
  end;

  corr_emp = corr(x1 || x2);
  print "Clayton Copula模拟 (θ=2)";
  print "经验相关系数:" corr_emp;

  create clayton_data from x1 [colname={'x1'}];
  append from x1;
  close;
  create clayton_data2 from x2 [colname={'x2'}];
  append from x2;
  close;
quit;

data clayton_plot;
  merge clayton_data clayton_data2;
run;

title "Clayton Copula模拟散点图";
proc sgplot data=clayton_plot;
  scatter x=x1 y=x2;
run;
title;


/****************************************************************************/
/* 8.5 尾部相关性                                                           */
/****************************************************************************/
proc iml;
  /* 尾部相关性: λ(U) = P(V>u | U>u) as u→1 */

  /* 高斯Copula的尾部相关性 */
  rho_values = {0.0, 0.3, 0.5, 0.7, 0.9};
  print "高斯Copula的尾部相关性（渐近为0）";
  do i = 1 to nrow(rho_values);
    rho = rho_values[i];
    /* 高斯Copula的上下尾相关性都为0（ρ<1时） */
    print "ρ=" rho "λ_upper=0 λ_lower=0";
  end;

  /* T Copula的尾部相关性 */
  print "T Copula的尾部相关性";
  do i = 1 to nrow(rho_values);
    rho = rho_values[i];
    df = 4;
    /* T Copula尾部相关性公式 */
    lambda = 2 * tinv(1 - 0.5, df + 1) * (1 - rho) /
             sqrt((df + 1) * (1 - rho**2));
    print "ρ=" rho "λ=" lambda;
  end;

  /* Clayton Copula的下尾相关性 */
  print "Clayton Copula的下尾相关性";
  theta_values = {1, 2, 5, 10};
  do i = 1 to nrow(theta_values);
    theta = theta_values[i];
    lambda_lower = 2##(-1/theta);
    print "θ=" theta "λ_lower=" lambda_lower;
  end;

  /* Gumbel Copula的上尾相关性 */
  print "Gumbel Copula的上尾相关性";
  do i = 1 to nrow(theta_values);
    theta = theta_values[i];
    lambda_upper = 2 - 2##(1/theta);
    print "θ=" theta "λ_upper=" lambda_upper;
  end;
quit;


/****************************************************************************/
/* 8.6 Copula参数估计                                                       */
/****************************************************************************/
proc iml;
  /* 生成Clayton Copula数据 */
  call streaminit(123);
  theta_true = 2;
  n = 1000;

  x1 = j(n, 1, 0);
  x2 = j(n, 1, 0);
  do i = 1 to n;
    u1 = rand('UNIFORM');
    t = rand('UNIFORM');
    u2 = (1 - u1##(-theta_true) + u1##(-theta_true) * t##(-theta_true/(1+theta_true)))##(-1/theta_true);
    x1[i] = quantile('EXPONENTIAL', u1, 1);
    x2[i] = quantile('EXPONENTIAL', u2, 1);
  end;

  /* 方法1：矩估计（基于Kendall tau） */
  /* Kendall tau = θ / (θ + 2) for Clayton */
  tau = 0;
  do i = 1 to n-1;
    do j = i+1 to n;
      a = sign(x1[i] - x1[j]) * sign(x2[i] - x2[j]);
      tau = tau + a;
    end;
  end;
  tau = tau / (n * (n-1) / 2);
  theta_hat = 2 * tau / (1 - tau);
  print "Clayton Copula参数估计";
  print "Kendall tau =" tau;
  print "θ_hat（矩估计）=" theta_hat;
quit;


/****************************************************************************/
/* 8.7 多元Copula应用                                                       */
/****************************************************************************/
proc iml;
  /* 三元高斯Copula */
  call streaminit(202);
  n = 1000;

  /* 相关矩阵 */
  Sigma = {1.0 0.5 0.3,
           0.5 1.0 0.4,
           0.3 0.4 1.0};

  /* 生成多元正态 */
  z = j(n, 3, 0);
  do i = 1 to n;
    e = rand('NORMAL', 0, 1) || rand('NORMAL', 0, 1) || rand('NORMAL', 0, 1);
    L = root(Sigma);
    z[i, ] = t(L * t(e));
  end;

  /* 转换为均匀分布 */
  u = cdf('NORMAL', z, 0, 1);

  /* 转换为不同边缘分布 */
  x1 = quantile('EXPONENTIAL', u[, 1], 1);      /* 指数分布 */
  x2 = quantile('GAMMA', u[, 2], 2, 1);         /* 伽马分布 */
  x3 = quantile('LOGNORMAL', u[, 3], 0, 1);     /* 对数正态分布 */

  corr_emp = corr(x1 || x2 || x3);
  print "三元高斯Copula";
  print "经验相关矩阵:" corr_emp;
quit;

/****************************************************************************/
/* 第8章 相依风险与Copula                                                   */
/* 对应教材：section8.tex                                                   */
/* 内容：Copula概念、常见Copula函数、参数估计、尾部相关性、                  */
/*       多元Copula、 vine Copula                                            */
/****************************************************************************/

/****************************************************************************/
/* 8.1 Copula基本概念                                                       */
/****************************************************************************/
proc iml;
  /* Sklar定理：F(x,y) = C(F_X(x), F_Y(y)) */
  /* Copula是将边缘分布连接为联合分布的函数 */

  /* 独立Copula: C(u,v) = u*v */
  u = {0.1, 0.3, 0.5, 0.7, 0.9};
  v = {0.2, 0.4, 0.6, 0.8, 1.0};
  C_indep = u # v;
  print "独立Copula" u v C_indep;
quit;


/****************************************************************************/
/* 8.2 高斯Copula                                                           */
/****************************************************************************/
proc iml;
  /* 高斯Copula: C(u,v;ρ) = Φ_ρ(Φ⁻¹(u), Φ⁻¹(v)) */
  rho = 0.5;
  u = {0.1, 0.3, 0.5, 0.7, 0.9};
  v = {0.2, 0.4, 0.6, 0.8, 1.0};

  /* 转换为标准正态 */
  z1 = quantile('NORMAL', u);
  z2 = quantile('NORMAL', v);

  /* 二元正态CDF */
  C_gaussian = j(nrow(u), 1, 0);
  do i = 1 to nrow(u);
    C_gaussian[i] = probbnrm(z1[i], z2[i], rho);
  end;

  print "高斯Copula (ρ=0.5)" u v C_gaussian;
quit;


/* 8.2.1 高斯Copula模拟 */
proc iml;
  call streaminit(123);
  rho = 0.7;
  n = 1000;

  /* 生成相关正态随机数 */
  z = j(n, 2, 0);
  do i = 1 to n;
    e1 = rand('NORMAL', 0, 1);
    e2 = rand('NORMAL', 0, 1);
    z[i, 1] = e1;
    z[i, 2] = rho * e1 + sqrt(1 - rho**2) * e2;
  end;

  /* 转换为均匀分布 */
  u = cdf('NORMAL', z, 0, 1);

  /* 转换为任意边缘分布（这里用指数分布） */
  x1 = quantile('EXPONENTIAL', u[, 1], 1);
  x2 = quantile('EXPONENTIAL', u[, 2], 1);

  /* 计算相关系数 */
  corr_emp = corr(x1 || x2);
  print "高斯Copula模拟";
  print "经验相关系数:" corr_emp;

  /* 绘图数据 */
  create gauss_copula from x1 [colname={'x1'}];
  append from x1;
  close;
  create gauss_copula2 from x2 [colname={'x2'}];
  append from x2;
  close;
quit;

data gauss_plot;
  merge gauss_copula gauss_copula2;
  i = _n_;
run;

title "高斯Copula模拟散点图";
proc sgplot data=gauss_plot;
  scatter x=x1 y=x2;
run;
title;


/****************************************************************************/
/* 8.3 T Copula                                                             */
/****************************************************************************/
proc iml;
  call streaminit(456);
  rho = 0.5;
  df = 4;
  n = 1000;

  z = j(n, 2, 0);
  do i = 1 to n;
    e1 = rand('NORMAL', 0, 1);
    e2 = rand('NORMAL', 0, 1);
    z[i, 1] = e1;
    z[i, 2] = rho * e1 + sqrt(1 - rho**2) * e2;
    /* T分布 */
    w = rand('CHISQUARE', df) / df;
    z[i, ] = z[i, ] / sqrt(w);
  end;

  /* 转换为均匀分布 */
  u = cdf('T', z, df);

  /* 转换为指数分布 */
  x1 = quantile('EXPONENTIAL', u[, 1], 1);
  x2 = quantile('EXPONENTIAL', u[, 2], 1);

  corr_emp = corr(x1 || x2);
  print "T Copula模拟 (df=4)";
  print "经验相关系数:" corr_emp;
quit;


/****************************************************************************/
/* 8.4 阿基米德Copula                                                       */
/****************************************************************************/
proc iml;
  /* 8.4.1 Gumbel Copula: C(u,v;θ) = exp(-[(-ln u)^θ + (-ln v)^θ]^(1/θ)) */
  theta = 2;
  u = {0.1, 0.3, 0.5, 0.7, 0.9};
  v = {0.2, 0.4, 0.6, 0.8, 1.0};

  C_gumbel = exp(-((-log(u))##theta + (-log(v))##theta)##(1/theta));
  print "Gumbel Copula (θ=2)" u v C_gumbel;

  /* 8.4.2 Clayton Copula: C(u,v;θ) = (u^(-θ) + v^(-θ) - 1)^(-1/θ) */
  theta_c = 2;
  C_clayton = (u##(-theta_c) + v##(-theta_c) - 1)##(-1/theta_c);
  print "Clayton Copula (θ=2)" u v C_clayton;

  /* 8.4.3 Frank Copula */
  theta_f = 5;
  C_frank = -1/theta_f * log(1 +
             (exp(-theta_f * u) - 1) * (exp(-theta_f * v) - 1) /
             (exp(-theta_f) - 1));
  print "Frank Copula (θ=5)" u v C_frank;
quit;


/* 8.4.4 Clayton Copula模拟 */
proc iml;
  call streaminit(789);
  theta = 2;
  n = 1000;

  x1 = j(n, 1, 0);
  x2 = j(n, 1, 0);
  do i = 1 to n;
    u1 = rand('UNIFORM');
    t = rand('UNIFORM');
    /* Clayton Copula的条件分布法 */
    u2 = (1 - u1##(-theta) + u1##(-theta) * t##(-theta/(1+theta)))##(-1/theta);
    /* 转换为指数分布 */
    x1[i] = quantile('EXPONENTIAL', u1, 1);
    x2[i] = quantile('EXPONENTIAL', u2, 1);
  end;

  corr_emp = corr(x1 || x2);
  print "Clayton Copula模拟 (θ=2)";
  print "经验相关系数:" corr_emp;

  create clayton_data from x1 [colname={'x1'}];
  append from x1;
  close;
  create clayton_data2 from x2 [colname={'x2'}];
  append from x2;
  close;
quit;

data clayton_plot;
  merge clayton_data clayton_data2;
run;

title "Clayton Copula模拟散点图";
proc sgplot data=clayton_plot;
  scatter x=x1 y=x2;
run;
title;


/****************************************************************************/
/* 8.5 尾部相关性                                                           */
/****************************************************************************/
proc iml;
  /* 尾部相关性: λ(U) = P(V>u | U>u) as u→1 */

  /* 高斯Copula的尾部相关性 */
  rho_values = {0.0, 0.3, 0.5, 0.7, 0.9};
  print "高斯Copula的尾部相关性（渐近为0）";
  do i = 1 to nrow(rho_values);
    rho = rho_values[i];
    /* 高斯Copula的上下尾相关性都为0（ρ<1时） */
    print "ρ=" rho "λ_upper=0 λ_lower=0";
  end;

  /* T Copula的尾部相关性 */
  print "T Copula的尾部相关性";
  do i = 1 to nrow(rho_values);
    rho = rho_values[i];
    df = 4;
    /* T Copula尾部相关性公式 */
    lambda = 2 * tinv(1 - 0.5, df + 1) * (1 - rho) /
             sqrt((df + 1) * (1 - rho**2));
    print "ρ=" rho "λ=" lambda;
  end;

  /* Clayton Copula的下尾相关性 */
  print "Clayton Copula的下尾相关性";
  theta_values = {1, 2, 5, 10};
  do i = 1 to nrow(theta_values);
    theta = theta_values[i];
    lambda_lower = 2##(-1/theta);
    print "θ=" theta "λ_lower=" lambda_lower;
  end;

  /* Gumbel Copula的上尾相关性 */
  print "Gumbel Copula的上尾相关性";
  do i = 1 to nrow(theta_values);
    theta = theta_values[i];
    lambda_upper = 2 - 2##(1/theta);
    print "θ=" theta "λ_upper=" lambda_upper;
  end;
quit;


/****************************************************************************/
/* 8.6 Copula参数估计                                                       */
/****************************************************************************/
proc iml;
  /* 生成Clayton Copula数据 */
  call streaminit(123);
  theta_true = 2;
  n = 1000;

  x1 = j(n, 1, 0);
  x2 = j(n, 1, 0);
  do i = 1 to n;
    u1 = rand('UNIFORM');
    t = rand('UNIFORM');
    u2 = (1 - u1##(-theta_true) + u1##(-theta_true) * t##(-theta_true/(1+theta_true)))##(-1/theta_true);
    x1[i] = quantile('EXPONENTIAL', u1, 1);
    x2[i] = quantile('EXPONENTIAL', u2, 1);
  end;

  /* 方法1：矩估计（基于Kendall tau） */
  /* Kendall tau = θ / (θ + 2) for Clayton */
  tau = 0;
  do i = 1 to n-1;
    do j = i+1 to n;
      a = sign(x1[i] - x1[j]) * sign(x2[i] - x2[j]);
      tau = tau + a;
    end;
  end;
  tau = tau / (n * (n-1) / 2);
  theta_hat = 2 * tau / (1 - tau);
  print "Clayton Copula参数估计";
  print "Kendall tau =" tau;
  print "θ_hat（矩估计）=" theta_hat;
quit;


/****************************************************************************/
/* 8.7 多元Copula应用                                                       */
/****************************************************************************/
proc iml;
  /* 三元高斯Copula */
  call streaminit(202);
  n = 1000;

  /* 相关矩阵 */
  Sigma = {1.0 0.5 0.3,
           0.5 1.0 0.4,
           0.3 0.4 1.0};

  /* 生成多元正态 */
  z = j(n, 3, 0);
  do i = 1 to n;
    e = rand('NORMAL', 0, 1) || rand('NORMAL', 0, 1) || rand('NORMAL', 0, 1);
    L = root(Sigma);
    z[i, ] = t(L * t(e));
  end;

  /* 转换为均匀分布 */
  u = cdf('NORMAL', z, 0, 1);

  /* 转换为不同边缘分布 */
  x1 = quantile('EXPONENTIAL', u[, 1], 1);      /* 指数分布 */
  x2 = quantile('GAMMA', u[, 2], 2, 1);         /* 伽马分布 */
  x3 = quantile('LOGNORMAL', u[, 3], 0, 1);     /* 对数正态分布 */

  corr_emp = corr(x1 || x2 || x3);
  print "三元高斯Copula";
  print "经验相关矩阵:" corr_emp;
quit;

/****************************************************************************/
/* 第8章 相依风险与Copula                                                   */
/* 对应教材：section8.tex                                                   */
/* 内容：Copula概念、常见Copula函数、参数估计、尾部相关性、                  */
/*       多元Copula、 vine Copula                                            */
/****************************************************************************/

/****************************************************************************/
/* 8.1 Copula基本概念                                                       */
/****************************************************************************/
proc iml;
  /* Sklar定理：F(x,y) = C(F_X(x), F_Y(y)) */
  /* Copula是将边缘分布连接为联合分布的函数 */

  /* 独立Copula: C(u,v) = u*v */
  u = {0.1, 0.3, 0.5, 0.7, 0.9};
  v = {0.2, 0.4, 0.6, 0.8, 1.0};
  C_indep = u # v;
  print "独立Copula" u v C_indep;
quit;


/****************************************************************************/
/* 8.2 高斯Copula                                                           */
/****************************************************************************/
proc iml;
  /* 高斯Copula: C(u,v;ρ) = Φ_ρ(Φ⁻¹(u), Φ⁻¹(v)) */
  rho = 0.5;
  u = {0.1, 0.3, 0.5, 0.7, 0.9};
  v = {0.2, 0.4, 0.6, 0.8, 1.0};

  /* 转换为标准正态 */
  z1 = quantile('NORMAL', u);
  z2 = quantile('NORMAL', v);

  /* 二元正态CDF */
  C_gaussian = j(nrow(u), 1, 0);
  do i = 1 to nrow(u);
    C_gaussian[i] = probbnrm(z1[i], z2[i], rho);
  end;

  print "高斯Copula (ρ=0.5)" u v C_gaussian;
quit;


/* 8.2.1 高斯Copula模拟 */
proc iml;
  call streaminit(123);
  rho = 0.7;
  n = 1000;

  /* 生成相关正态随机数 */
  z = j(n, 2, 0);
  do i = 1 to n;
    e1 = rand('NORMAL', 0, 1);
    e2 = rand('NORMAL', 0, 1);
    z[i, 1] = e1;
    z[i, 2] = rho * e1 + sqrt(1 - rho**2) * e2;
  end;

  /* 转换为均匀分布 */
  u = cdf('NORMAL', z, 0, 1);

  /* 转换为任意边缘分布（这里用指数分布） */
  x1 = quantile('EXPONENTIAL', u[, 1], 1);
  x2 = quantile('EXPONENTIAL', u[, 2], 1);

  /* 计算相关系数 */
  corr_emp = corr(x1 || x2);
  print "高斯Copula模拟";
  print "经验相关系数:" corr_emp;

  /* 绘图数据 */
  create gauss_copula from x1 [colname={'x1'}];
  append from x1;
  close;
  create gauss_copula2 from x2 [colname={'x2'}];
  append from x2;
  close;
quit;

data gauss_plot;
  merge gauss_copula gauss_copula2;
  i = _n_;
run;

title "高斯Copula模拟散点图";
proc sgplot data=gauss_plot;
  scatter x=x1 y=x2;
run;
title;


/****************************************************************************/
/* 8.3 T Copula                                                             */
/****************************************************************************/
proc iml;
  call streaminit(456);
  rho = 0.5;
  df = 4;
  n = 1000;

  z = j(n, 2, 0);
  do i = 1 to n;
    e1 = rand('NORMAL', 0, 1);
    e2 = rand('NORMAL', 0, 1);
    z[i, 1] = e1;
    z[i, 2] = rho * e1 + sqrt(1 - rho**2) * e2;
    /* T分布 */
    w = rand('CHISQUARE', df) / df;
    z[i, ] = z[i, ] / sqrt(w);
  end;

  /* 转换为均匀分布 */
  u = cdf('T', z, df);

  /* 转换为指数分布 */
  x1 = quantile('EXPONENTIAL', u[, 1], 1);
  x2 = quantile('EXPONENTIAL', u[, 2], 1);

  corr_emp = corr(x1 || x2);
  print "T Copula模拟 (df=4)";
  print "经验相关系数:" corr_emp;
quit;


/****************************************************************************/
/* 8.4 阿基米德Copula                                                       */
/****************************************************************************/
proc iml;
  /* 8.4.1 Gumbel Copula: C(u,v;θ) = exp(-[(-ln u)^θ + (-ln v)^θ]^(1/θ)) */
  theta = 2;
  u = {0.1, 0.3, 0.5, 0.7, 0.9};
  v = {0.2, 0.4, 0.6, 0.8, 1.0};

  C_gumbel = exp(-((-log(u))##theta + (-log(v))##theta)##(1/theta));
  print "Gumbel Copula (θ=2)" u v C_gumbel;

  /* 8.4.2 Clayton Copula: C(u,v;θ) = (u^(-θ) + v^(-θ) - 1)^(-1/θ) */
  theta_c = 2;
  C_clayton = (u##(-theta_c) + v##(-theta_c) - 1)##(-1/theta_c);
  print "Clayton Copula (θ=2)" u v C_clayton;

  /* 8.4.3 Frank Copula */
  theta_f = 5;
  C_frank = -1/theta_f * log(1 +
             (exp(-theta_f * u) - 1) * (exp(-theta_f * v) - 1) /
             (exp(-theta_f) - 1));
  print "Frank Copula (θ=5)" u v C_frank;
quit;


/* 8.4.4 Clayton Copula模拟 */
proc iml;
  call streaminit(789);
  theta = 2;
  n = 1000;

  x1 = j(n, 1, 0);
  x2 = j(n, 1, 0);
  do i = 1 to n;
    u1 = rand('UNIFORM');
    t = rand('UNIFORM');
    /* Clayton Copula的条件分布法 */
    u2 = (1 - u1##(-theta) + u1##(-theta) * t##(-theta/(1+theta)))##(-1/theta);
    /* 转换为指数分布 */
    x1[i] = quantile('EXPONENTIAL', u1, 1);
    x2[i] = quantile('EXPONENTIAL', u2, 1);
  end;

  corr_emp = corr(x1 || x2);
  print "Clayton Copula模拟 (θ=2)";
  print "经验相关系数:" corr_emp;

  create clayton_data from x1 [colname={'x1'}];
  append from x1;
  close;
  create clayton_data2 from x2 [colname={'x2'}];
  append from x2;
  close;
quit;

data clayton_plot;
  merge clayton_data clayton_data2;
run;

title "Clayton Copula模拟散点图";
proc sgplot data=clayton_plot;
  scatter x=x1 y=x2;
run;
title;


/****************************************************************************/
/* 8.5 尾部相关性                                                           */
/****************************************************************************/
proc iml;
  /* 尾部相关性: λ(U) = P(V>u | U>u) as u→1 */

  /* 高斯Copula的尾部相关性 */
  rho_values = {0.0, 0.3, 0.5, 0.7, 0.9};
  print "高斯Copula的尾部相关性（渐近为0）";
  do i = 1 to nrow(rho_values);
    rho = rho_values[i];
    /* 高斯Copula的上下尾相关性都为0（ρ<1时） */
    print "ρ=" rho "λ_upper=0 λ_lower=0";
  end;

  /* T Copula的尾部相关性 */
  print "T Copula的尾部相关性";
  do i = 1 to nrow(rho_values);
    rho = rho_values[i];
    df = 4;
    /* T Copula尾部相关性公式 */
    lambda = 2 * tinv(1 - 0.5, df + 1) * (1 - rho) /
             sqrt((df + 1) * (1 - rho**2));
    print "ρ=" rho "λ=" lambda;
  end;

  /* Clayton Copula的下尾相关性 */
  print "Clayton Copula的下尾相关性";
  theta_values = {1, 2, 5, 10};
  do i = 1 to nrow(theta_values);
    theta = theta_values[i];
    lambda_lower = 2##(-1/theta);
    print "θ=" theta "λ_lower=" lambda_lower;
  end;

  /* Gumbel Copula的上尾相关性 */
  print "Gumbel Copula的上尾相关性";
  do i = 1 to nrow(theta_values);
    theta = theta_values[i];
    lambda_upper = 2 - 2##(1/theta);
    print "θ=" theta "λ_upper=" lambda_upper;
  end;
quit;


/****************************************************************************/
/* 8.6 Copula参数估计                                                       */
/****************************************************************************/
proc iml;
  /* 生成Clayton Copula数据 */
  call streaminit(123);
  theta_true = 2;
  n = 1000;

  x1 = j(n, 1, 0);
  x2 = j(n, 1, 0);
  do i = 1 to n;
    u1 = rand('UNIFORM');
    t = rand('UNIFORM');
    u2 = (1 - u1##(-theta_true) + u1##(-theta_true) * t##(-theta_true/(1+theta_true)))##(-1/theta_true);
    x1[i] = quantile('EXPONENTIAL', u1, 1);
    x2[i] = quantile('EXPONENTIAL', u2, 1);
  end;

  /* 方法1：矩估计（基于Kendall tau） */
  /* Kendall tau = θ / (θ + 2) for Clayton */
  tau = 0;
  do i = 1 to n-1;
    do j = i+1 to n;
      a = sign(x1[i] - x1[j]) * sign(x2[i] - x2[j]);
      tau = tau + a;
    end;
  end;
  tau = tau / (n * (n-1) / 2);
  theta_hat = 2 * tau / (1 - tau);
  print "Clayton Copula参数估计";
  print "Kendall tau =" tau;
  print "θ_hat（矩估计）=" theta_hat;
quit;


/****************************************************************************/
/* 8.7 多元Copula应用                                                       */
/****************************************************************************/
proc iml;
  /* 三元高斯Copula */
  call streaminit(202);
  n = 1000;

  /* 相关矩阵 */
  Sigma = {1.0 0.5 0.3,
           0.5 1.0 0.4,
           0.3 0.4 1.0};

  /* 生成多元正态 */
  z = j(n, 3, 0);
  do i = 1 to n;
    e = rand('NORMAL', 0, 1) || rand('NORMAL', 0, 1) || rand('NORMAL', 0, 1);
    L = root(Sigma);
    z[i, ] = t(L * t(e));
  end;

  /* 转换为均匀分布 */
  u = cdf('NORMAL', z, 0, 1);

  /* 转换为不同边缘分布 */
  x1 = quantile('EXPONENTIAL', u[, 1], 1);      /* 指数分布 */
  x2 = quantile('GAMMA', u[, 2], 2, 1);         /* 伽马分布 */
  x3 = quantile('LOGNORMAL', u[, 3], 0, 1);     /* 对数正态分布 */

  corr_emp = corr(x1 || x2 || x3);
  print "三元高斯Copula";
  print "经验相关矩阵:" corr_emp;
quit;

/****************************************************************************/
/* 第8章 相依风险与Copula                                                   */
/* 对应教材：section8.tex                                                   */
/* 内容：Copula概念、常见Copula函数、参数估计、尾部相关性、                  */
/*       多元Copula、 vine Copula                                            */
/****************************************************************************/

/****************************************************************************/
/* 8.1 Copula基本概念                                                       */
/****************************************************************************/
proc iml;
  /* Sklar定理：F(x,y) = C(F_X(x), F_Y(y)) */
  /* Copula是将边缘分布连接为联合分布的函数 */

  /* 独立Copula: C(u,v) = u*v */
  u = {0.1, 0.3, 0.5, 0.7, 0.9};
  v = {0.2, 0.4, 0.6, 0.8, 1.0};
  C_indep = u # v;
  print "独立Copula" u v C_indep;
quit;


/****************************************************************************/
/* 8.2 高斯Copula                                                           */
/****************************************************************************/
proc iml;
  /* 高斯Copula: C(u,v;ρ) = Φ_ρ(Φ⁻¹(u), Φ⁻¹(v)) */
  rho = 0.5;
  u = {0.1, 0.3, 0.5, 0.7, 0.9};
  v = {0.2, 0.4, 0.6, 0.8, 1.0};

  /* 转换为标准正态 */
  z1 = quantile('NORMAL', u);
  z2 = quantile('NORMAL', v);

  /* 二元正态CDF */
  C_gaussian = j(nrow(u), 1, 0);
  do i = 1 to nrow(u);
    C_gaussian[i] = probbnrm(z1[i], z2[i], rho);
  end;

  print "高斯Copula (ρ=0.5)" u v C_gaussian;
quit;


/* 8.2.1 高斯Copula模拟 */
proc iml;
  call streaminit(123);
  rho = 0.7;
  n = 1000;

  /* 生成相关正态随机数 */
  z = j(n, 2, 0);
  do i = 1 to n;
    e1 = rand('NORMAL', 0, 1);
    e2 = rand('NORMAL', 0, 1);
    z[i, 1] = e1;
    z[i, 2] = rho * e1 + sqrt(1 - rho**2) * e2;
  end;

  /* 转换为均匀分布 */
  u = cdf('NORMAL', z, 0, 1);

  /* 转换为任意边缘分布（这里用指数分布） */
  x1 = quantile('EXPONENTIAL', u[, 1], 1);
  x2 = quantile('EXPONENTIAL', u[, 2], 1);

  /* 计算相关系数 */
  corr_emp = corr(x1 || x2);
  print "高斯Copula模拟";
  print "经验相关系数:" corr_emp;

  /* 绘图数据 */
  create gauss_copula from x1 [colname={'x1'}];
  append from x1;
  close;
  create gauss_copula2 from x2 [colname={'x2'}];
  append from x2;
  close;
quit;

data gauss_plot;
  merge gauss_copula gauss_copula2;
  i = _n_;
run;

title "高斯Copula模拟散点图";
proc sgplot data=gauss_plot;
  scatter x=x1 y=x2;
run;
title;


/****************************************************************************/
/* 8.3 T Copula                                                             */
/****************************************************************************/
proc iml;
  call streaminit(456);
  rho = 0.5;
  df = 4;
  n = 1000;

  z = j(n, 2, 0);
  do i = 1 to n;
    e1 = rand('NORMAL', 0, 1);
    e2 = rand('NORMAL', 0, 1);
    z[i, 1] = e1;
    z[i, 2] = rho * e1 + sqrt(1 - rho**2) * e2;
    /* T分布 */
    w = rand('CHISQUARE', df) / df;
    z[i, ] = z[i, ] / sqrt(w);
  end;

  /* 转换为均匀分布 */
  u = cdf('T', z, df);

  /* 转换为指数分布 */
  x1 = quantile('EXPONENTIAL', u[, 1], 1);
  x2 = quantile('EXPONENTIAL', u[, 2], 1);

  corr_emp = corr(x1 || x2);
  print "T Copula模拟 (df=4)";
  print "经验相关系数:" corr_emp;
quit;


/****************************************************************************/
/* 8.4 阿基米德Copula                                                       */
/****************************************************************************/
proc iml;
  /* 8.4.1 Gumbel Copula: C(u,v;θ) = exp(-[(-ln u)^θ + (-ln v)^θ]^(1/θ)) */
  theta = 2;
  u = {0.1, 0.3, 0.5, 0.7, 0.9};
  v = {0.2, 0.4, 0.6, 0.8, 1.0};

  C_gumbel = exp(-((-log(u))##theta + (-log(v))##theta)##(1/theta));
  print "Gumbel Copula (θ=2)" u v C_gumbel;

  /* 8.4.2 Clayton Copula: C(u,v;θ) = (u^(-θ) + v^(-θ) - 1)^(-1/θ) */
  theta_c = 2;
  C_clayton = (u##(-theta_c) + v##(-theta_c) - 1)##(-1/theta_c);
  print "Clayton Copula (θ=2)" u v C_clayton;

  /* 8.4.3 Frank Copula */
  theta_f = 5;
  C_frank = -1/theta_f * log(1 +
             (exp(-theta_f * u) - 1) * (exp(-theta_f * v) - 1) /
             (exp(-theta_f) - 1));
  print "Frank Copula (θ=5)" u v C_frank;
quit;


/* 8.4.4 Clayton Copula模拟 */
proc iml;
  call streaminit(789);
  theta = 2;
  n = 1000;

  x1 = j(n, 1, 0);
  x2 = j(n, 1, 0);
  do i = 1 to n;
    u1 = rand('UNIFORM');
    t = rand('UNIFORM');
    /* Clayton Copula的条件分布法 */
    u2 = (1 - u1##(-theta) + u1##(-theta) * t##(-theta/(1+theta)))##(-1/theta);
    /* 转换为指数分布 */
    x1[i] = quantile('EXPONENTIAL', u1, 1);
    x2[i] = quantile('EXPONENTIAL', u2, 1);
  end;

  corr_emp = corr(x1 || x2);
  print "Clayton Copula模拟 (θ=2)";
  print "经验相关系数:" corr_emp;

  create clayton_data from x1 [colname={'x1'}];
  append from x1;
  close;
  create clayton_data2 from x2 [colname={'x2'}];
  append from x2;
  close;
quit;

data clayton_plot;
  merge clayton_data clayton_data2;
run;

title "Clayton Copula模拟散点图";
proc sgplot data=clayton_plot;
  scatter x=x1 y=x2;
run;
title;


/****************************************************************************/
/* 8.5 尾部相关性                                                           */
/****************************************************************************/
proc iml;
  /* 尾部相关性: λ(U) = P(V>u | U>u) as u→1 */

  /* 高斯Copula的尾部相关性 */
  rho_values = {0.0, 0.3, 0.5, 0.7, 0.9};
  print "高斯Copula的尾部相关性（渐近为0）";
  do i = 1 to nrow(rho_values);
    rho = rho_values[i];
    /* 高斯Copula的上下尾相关性都为0（ρ<1时） */
    print "ρ=" rho "λ_upper=0 λ_lower=0";
  end;

  /* T Copula的尾部相关性 */
  print "T Copula的尾部相关性";
  do i = 1 to nrow(rho_values);
    rho = rho_values[i];
    df = 4;
    /* T Copula尾部相关性公式 */
    lambda = 2 * tinv(1 - 0.5, df + 1) * (1 - rho) /
             sqrt((df + 1) * (1 - rho**2));
    print "ρ=" rho "λ=" lambda;
  end;

  /* Clayton Copula的下尾相关性 */
  print "Clayton Copula的下尾相关性";
  theta_values = {1, 2, 5, 10};
  do i = 1 to nrow(theta_values);
    theta = theta_values[i];
    lambda_lower = 2##(-1/theta);
    print "θ=" theta "λ_lower=" lambda_lower;
  end;

  /* Gumbel Copula的上尾相关性 */
  print "Gumbel Copula的上尾相关性";
  do i = 1 to nrow(theta_values);
    theta = theta_values[i];
    lambda_upper = 2 - 2##(1/theta);
    print "θ=" theta "λ_upper=" lambda_upper;
  end;
quit;


/****************************************************************************/
/* 8.6 Copula参数估计                                                       */
/****************************************************************************/
proc iml;
  /* 生成Clayton Copula数据 */
  call streaminit(123);
  theta_true = 2;
  n = 1000;

  x1 = j(n, 1, 0);
  x2 = j(n, 1, 0);
  do i = 1 to n;
    u1 = rand('UNIFORM');
    t = rand('UNIFORM');
    u2 = (1 - u1##(-theta_true) + u1##(-theta_true) * t##(-theta_true/(1+theta_true)))##(-1/theta_true);
    x1[i] = quantile('EXPONENTIAL', u1, 1);
    x2[i] = quantile('EXPONENTIAL', u2, 1);
  end;

  /* 方法1：矩估计（基于Kendall tau） */
  /* Kendall tau = θ / (θ + 2) for Clayton */
  tau = 0;
  do i = 1 to n-1;
    do j = i+1 to n;
      a = sign(x1[i] - x1[j]) * sign(x2[i] - x2[j]);
      tau = tau + a;
    end;
  end;
  tau = tau / (n * (n-1) / 2);
  theta_hat = 2 * tau / (1 - tau);
  print "Clayton Copula参数估计";
  print "Kendall tau =" tau;
  print "θ_hat（矩估计）=" theta_hat;
quit;


/****************************************************************************/
/* 8.7 多元Copula应用                                                       */
/****************************************************************************/
proc iml;
  /* 三元高斯Copula */
  call streaminit(202);
  n = 1000;

  /* 相关矩阵 */
  Sigma = {1.0 0.5 0.3,
           0.5 1.0 0.4,
           0.3 0.4 1.0};

  /* 生成多元正态 */
  z = j(n, 3, 0);
  do i = 1 to n;
    e = rand('NORMAL', 0, 1) || rand('NORMAL', 0, 1) || rand('NORMAL', 0, 1);
    L = root(Sigma);
    z[i, ] = t(L * t(e));
  end;

  /* 转换为均匀分布 */
  u = cdf('NORMAL', z, 0, 1);

  /* 转换为不同边缘分布 */
  x1 = quantile('EXPONENTIAL', u[, 1], 1);      /* 指数分布 */
  x2 = quantile('GAMMA', u[, 2], 2, 1);         /* 伽马分布 */
  x3 = quantile('LOGNORMAL', u[, 3], 0, 1);     /* 对数正态分布 */

  corr_emp = corr(x1 || x2 || x3);
  print "三元高斯Copula";
  print "经验相关矩阵:" corr_emp;
quit;

/* 8.4.1 基于相依性测度的Copula参数估计 */
/* 生成模拟数据 */
proc iml;
  call randseed(101);
  n_obs = 100;
  theta_true = 3.0;

  /* 生成Clayton Copula相依数据 */
  u1 = j(n_obs, 1, 0);
  u2 = j(n_obs, 1, 0);
  call randgen(u1, "uniform");
  t = j(n_obs, 1, 0);
  call randgen(t, "uniform");
  u2 = (1 + u1##(-theta_true) # (t##(-theta_true/(1+theta_true)) - 1))##(-1/theta_true);

  /* 转换为伽马分布的边际 */
  loss = quantile("gamma", u1, 2) / 0.001;
  alae = quantile("gamma", u2, 3) / 0.0005;

  /* 计算Kendall秩相关系数 */
  tau_kendall = 0;
  do i = 1 to n_obs;
    do j = i+1 to n_obs;
      if (loss[i] - loss[j]) * (alae[i] - alae[j]) > 0 then
        tau_kendall = tau_kendall + 1;
      else tau_kendall = tau_kendall - 1;
    end;
  end;
  tau_kendall = tau_kendall / (n_obs * (n_obs - 1) / 2);
  print "Kendall秩相关系数:" tau_kendall;

  /* Clayton Copula参数: theta = 2*tau/(1-tau) */
  theta_clayton = 2 * tau_kendall / (1 - tau_kendall);
  print "Clayton Copula参数:" theta_clayton;

  /* Gumbel Copula参数: theta = 1/(1-tau) */
  theta_gumbel = 1 / (1 - tau_kendall);
  print "Gumbel Copula参数:" theta_gumbel;
quit;

/****************************************************************************/
/* 第8章 相依风险与Copula                                                   */
/* 对应教材：section8.tex                                                   */
/* 内容：Copula概念、常见Copula函数、参数估计、尾部相关性、                  */
/*       多元Copula、 vine Copula                                            */
/****************************************************************************/

/****************************************************************************/
/* 8.1 Copula基本概念                                                       */
/****************************************************************************/
proc iml;
  /* Sklar定理：F(x,y) = C(F_X(x), F_Y(y)) */
  /* Copula是将边缘分布连接为联合分布的函数 */

  /* 独立Copula: C(u,v) = u*v */
  u = {0.1, 0.3, 0.5, 0.7, 0.9};
  v = {0.2, 0.4, 0.6, 0.8, 1.0};
  C_indep = u # v;
  print "独立Copula" u v C_indep;
quit;


/****************************************************************************/
/* 8.2 高斯Copula                                                           */
/****************************************************************************/
proc iml;
  /* 高斯Copula: C(u,v;ρ) = Φ_ρ(Φ⁻¹(u), Φ⁻¹(v)) */
  rho = 0.5;
  u = {0.1, 0.3, 0.5, 0.7, 0.9};
  v = {0.2, 0.4, 0.6, 0.8, 1.0};

  /* 转换为标准正态 */
  z1 = quantile('NORMAL', u);
  z2 = quantile('NORMAL', v);

  /* 二元正态CDF */
  C_gaussian = j(nrow(u), 1, 0);
  do i = 1 to nrow(u);
    C_gaussian[i] = probbnrm(z1[i], z2[i], rho);
  end;

  print "高斯Copula (ρ=0.5)" u v C_gaussian;
quit;


/* 8.2.1 高斯Copula模拟 */
proc iml;
  call streaminit(123);
  rho = 0.7;
  n = 1000;

  /* 生成相关正态随机数 */
  z = j(n, 2, 0);
  do i = 1 to n;
    e1 = rand('NORMAL', 0, 1);
    e2 = rand('NORMAL', 0, 1);
    z[i, 1] = e1;
    z[i, 2] = rho * e1 + sqrt(1 - rho**2) * e2;
  end;

  /* 转换为均匀分布 */
  u = cdf('NORMAL', z, 0, 1);

  /* 转换为任意边缘分布（这里用指数分布） */
  x1 = quantile('EXPONENTIAL', u[, 1], 1);
  x2 = quantile('EXPONENTIAL', u[, 2], 1);

  /* 计算相关系数 */
  corr_emp = corr(x1 || x2);
  print "高斯Copula模拟";
  print "经验相关系数:" corr_emp;

  /* 绘图数据 */
  create gauss_copula from x1 [colname={'x1'}];
  append from x1;
  close;
  create gauss_copula2 from x2 [colname={'x2'}];
  append from x2;
  close;
quit;

data gauss_plot;
  merge gauss_copula gauss_copula2;
  i = _n_;
run;

title "高斯Copula模拟散点图";
proc sgplot data=gauss_plot;
  scatter x=x1 y=x2;
run;
title;


/****************************************************************************/
/* 8.3 T Copula                                                             */
/****************************************************************************/
proc iml;
  call streaminit(456);
  rho = 0.5;
  df = 4;
  n = 1000;

  z = j(n, 2, 0);
  do i = 1 to n;
    e1 = rand('NORMAL', 0, 1);
    e2 = rand('NORMAL', 0, 1);
    z[i, 1] = e1;
    z[i, 2] = rho * e1 + sqrt(1 - rho**2) * e2;
    /* T分布 */
    w = rand('CHISQUARE', df) / df;
    z[i, ] = z[i, ] / sqrt(w);
  end;

  /* 转换为均匀分布 */
  u = cdf('T', z, df);

  /* 转换为指数分布 */
  x1 = quantile('EXPONENTIAL', u[, 1], 1);
  x2 = quantile('EXPONENTIAL', u[, 2], 1);

  corr_emp = corr(x1 || x2);
  print "T Copula模拟 (df=4)";
  print "经验相关系数:" corr_emp;
quit;


/****************************************************************************/
/* 8.4 阿基米德Copula                                                       */
/****************************************************************************/
proc iml;
  /* 8.4.1 Gumbel Copula: C(u,v;θ) = exp(-[(-ln u)^θ + (-ln v)^θ]^(1/θ)) */
  theta = 2;
  u = {0.1, 0.3, 0.5, 0.7, 0.9};
  v = {0.2, 0.4, 0.6, 0.8, 1.0};

  C_gumbel = exp(-((-log(u))##theta + (-log(v))##theta)##(1/theta));
  print "Gumbel Copula (θ=2)" u v C_gumbel;

  /* 8.4.2 Clayton Copula: C(u,v;θ) = (u^(-θ) + v^(-θ) - 1)^(-1/θ) */
  theta_c = 2;
  C_clayton = (u##(-theta_c) + v##(-theta_c) - 1)##(-1/theta_c);
  print "Clayton Copula (θ=2)" u v C_clayton;

  /* 8.4.3 Frank Copula */
  theta_f = 5;
  C_frank = -1/theta_f * log(1 +
             (exp(-theta_f * u) - 1) * (exp(-theta_f * v) - 1) /
             (exp(-theta_f) - 1));
  print "Frank Copula (θ=5)" u v C_frank;
quit;


/* 8.4.4 Clayton Copula模拟 */
proc iml;
  call streaminit(789);
  theta = 2;
  n = 1000;

  x1 = j(n, 1, 0);
  x2 = j(n, 1, 0);
  do i = 1 to n;
    u1 = rand('UNIFORM');
    t = rand('UNIFORM');
    /* Clayton Copula的条件分布法 */
    u2 = (1 - u1##(-theta) + u1##(-theta) * t##(-theta/(1+theta)))##(-1/theta);
    /* 转换为指数分布 */
    x1[i] = quantile('EXPONENTIAL', u1, 1);
    x2[i] = quantile('EXPONENTIAL', u2, 1);
  end;

  corr_emp = corr(x1 || x2);
  print "Clayton Copula模拟 (θ=2)";
  print "经验相关系数:" corr_emp;

  create clayton_data from x1 [colname={'x1'}];
  append from x1;
  close;
  create clayton_data2 from x2 [colname={'x2'}];
  append from x2;
  close;
quit;

data clayton_plot;
  merge clayton_data clayton_data2;
run;

title "Clayton Copula模拟散点图";
proc sgplot data=clayton_plot;
  scatter x=x1 y=x2;
run;
title;


/****************************************************************************/
/* 8.5 尾部相关性                                                           */
/****************************************************************************/
proc iml;
  /* 尾部相关性: λ(U) = P(V>u | U>u) as u→1 */

  /* 高斯Copula的尾部相关性 */
  rho_values = {0.0, 0.3, 0.5, 0.7, 0.9};
  print "高斯Copula的尾部相关性（渐近为0）";
  do i = 1 to nrow(rho_values);
    rho = rho_values[i];
    /* 高斯Copula的上下尾相关性都为0（ρ<1时） */
    print "ρ=" rho "λ_upper=0 λ_lower=0";
  end;

  /* T Copula的尾部相关性 */
  print "T Copula的尾部相关性";
  do i = 1 to nrow(rho_values);
    rho = rho_values[i];
    df = 4;
    /* T Copula尾部相关性公式 */
    lambda = 2 * tinv(1 - 0.5, df + 1) * (1 - rho) /
             sqrt((df + 1) * (1 - rho**2));
    print "ρ=" rho "λ=" lambda;
  end;

  /* Clayton Copula的下尾相关性 */
  print "Clayton Copula的下尾相关性";
  theta_values = {1, 2, 5, 10};
  do i = 1 to nrow(theta_values);
    theta = theta_values[i];
    lambda_lower = 2##(-1/theta);
    print "θ=" theta "λ_lower=" lambda_lower;
  end;

  /* Gumbel Copula的上尾相关性 */
  print "Gumbel Copula的上尾相关性";
  do i = 1 to nrow(theta_values);
    theta = theta_values[i];
    lambda_upper = 2 - 2##(1/theta);
    print "θ=" theta "λ_upper=" lambda_upper;
  end;
quit;


/****************************************************************************/
/* 8.6 Copula参数估计                                                       */
/****************************************************************************/
proc iml;
  /* 生成Clayton Copula数据 */
  call streaminit(123);
  theta_true = 2;
  n = 1000;

  x1 = j(n, 1, 0);
  x2 = j(n, 1, 0);
  do i = 1 to n;
    u1 = rand('UNIFORM');
    t = rand('UNIFORM');
    u2 = (1 - u1##(-theta_true) + u1##(-theta_true) * t##(-theta_true/(1+theta_true)))##(-1/theta_true);
    x1[i] = quantile('EXPONENTIAL', u1, 1);
    x2[i] = quantile('EXPONENTIAL', u2, 1);
  end;

  /* 方法1：矩估计（基于Kendall tau） */
  /* Kendall tau = θ / (θ + 2) for Clayton */
  tau = 0;
  do i = 1 to n-1;
    do j = i+1 to n;
      a = sign(x1[i] - x1[j]) * sign(x2[i] - x2[j]);
      tau = tau + a;
    end;
  end;
  tau = tau / (n * (n-1) / 2);
  theta_hat = 2 * tau / (1 - tau);
  print "Clayton Copula参数估计";
  print "Kendall tau =" tau;
  print "θ_hat（矩估计）=" theta_hat;
quit;


/****************************************************************************/
/* 8.7 多元Copula应用                                                       */
/****************************************************************************/
proc iml;
  /* 三元高斯Copula */
  call streaminit(202);
  n = 1000;

  /* 相关矩阵 */
  Sigma = {1.0 0.5 0.3,
           0.5 1.0 0.4,
           0.3 0.4 1.0};

  /* 生成多元正态 */
  z = j(n, 3, 0);
  do i = 1 to n;
    e = rand('NORMAL', 0, 1) || rand('NORMAL', 0, 1) || rand('NORMAL', 0, 1);
    L = root(Sigma);
    z[i, ] = t(L * t(e));
  end;

  /* 转换为均匀分布 */
  u = cdf('NORMAL', z, 0, 1);

  /* 转换为不同边缘分布 */
  x1 = quantile('EXPONENTIAL', u[, 1], 1);      /* 指数分布 */
  x2 = quantile('GAMMA', u[, 2], 2, 1);         /* 伽马分布 */
  x3 = quantile('LOGNORMAL', u[, 3], 0, 1);     /* 对数正态分布 */

  corr_emp = corr(x1 || x2 || x3);
  print "三元高斯Copula";
  print "经验相关矩阵:" corr_emp;
quit;

/******************************************************************************/
/* 8.2.5 应用复合函数生成Copula                                               */
/******************************************************************************/

/* 1. 拉普拉斯变换与Clayton Copula的对应关系 */
data laplace_verify;
    alpha = 2;
    do t = 0.01 to 0.99 by 0.1;
        tau_inv = t**(-alpha) - 1;
        alpha_psi = alpha * (t**(-alpha) - 1) / alpha;
        diff = abs(tau_inv - alpha_psi);
        output;
    end;
run;

proc print data=laplace_verify;
    title "验证 tau^{-1}(t) = alpha * psi(t)";
run;

/* 2. 基于脆弱性模型的Copula模拟 */
data frailty_sim;
    call streaminit(2024);
    alpha = 2;
    do i = 1 to 1000;
        gamma = rand('GAMMA', 1/alpha);
        U1 = rand('UNIFORM');
        U2 = rand('UNIFORM');
        /* 拉普拉斯变换 */
        u1 = (1 + (-log(U1)) / gamma)**(-1/alpha);
        u2 = (1 + (-log(U2)) / gamma)**(-1/alpha);
        /* 转换为指数分布 */
        x1 = -log(1 - u1);
        x2 = -log(1 - u2);
        output;
    end;
run;

proc corr data=frailty_sim kendall;
    var x1 x2;
    title "脆弱性模型模拟 - Kendall tau";
run;


/******************************************************************************/
/* 8.3.1 高斯Copula的模拟                                                     */
/******************************************************************************/

/* 二元高斯Copula模拟 - Cholesky分解法 */
proc iml;
    rho = 0.6;
    n = 2000;
    Sigma = {1 0.6, 0.6 1};
    L = root(Sigma);

    Y = j(n, 2, .);
    do i = 1 to n;
        Y[i,] = randnormal(1, {0 0}, {1 0, 0 1});
    end;
    Z = Y * L`;
    U1 = cdf('NORMAL', Z[,1]);
    U2 = cdf('NORMAL', Z[,2]);

    x1 = quantile('EXPONENTIAL', U1);
    x2 = quantile('EXPONENTIAL', U2);

    /* 计算Kendall tau */
    tau = 0;
    do i = 1 to n-1;
        do j = i+1 to n;
            concordant = (x1[i]-x1[j])*(x2[i]-x2[j]);
            if concordant > 0 then tau = tau + 1;
            else if concordant < 0 then tau = tau - 1;
        end;
    end;
    tau = tau / (n*(n-1)/2);

    tau_theory = 2/constant('PI') * arcsin(rho/2);
    print "二元高斯Copula模拟 (rho=0.6):";
    print "Kendall tau (经验) = " tau;
    print "Kendall tau (理论) = " tau_theory;
quit;

/* 多元高斯Copula模拟 (d=3) */
proc iml;
    n = 2000;
    Sigma = {1.0 0.5 0.3,
             0.5 1.0 0.4,
             0.3 0.4 1.0};
    L = root(Sigma);

    Y = j(n, 3, .);
    do i = 1 to n;
        Y[i,] = randnormal(1, {0 0 0}, {1 0 0, 0 1 0, 0 0 1});
    end;
    Z = Y * L`;
    U1 = cdf('NORMAL', Z[,1]);
    U2 = cdf('NORMAL', Z[,2]);
    U3 = cdf('NORMAL', Z[,3]);

    x1 = quantile('EXPONENTIAL', U1);
    x2 = quantile('GAMMA', 2, U2);
    x3 = quantile('LOGNORMAL', 0, 1, U3);

    corr_mat = corr(x1 || x2 || x3);
    print "三元高斯Copula模拟 - 经验相关矩阵:";
    print corr_mat;
quit;


/******************************************************************************/
/* 8.3.2 阿基米德Copula的模拟                                                 */
/******************************************************************************/

/* 方法1：条件分布法（Clayton Copula） */
data clayton_cond;
    call streaminit(2024);
    alpha = 2;
    do i = 1 to 2000;
        u1 = rand('UNIFORM');
        t = rand('UNIFORM');
        u2 = (1 - u1**(-alpha) + u1**(-alpha) * t**(-alpha/(1+alpha)))**(-1/alpha);
        x1 = -log(1 - u1);
        x2 = -log(1 - u2);
        output;
    end;
run;

proc corr data=clayton_cond kendall;
    var x1 x2;
    title "Clayton Copula模拟 (条件分布法, alpha=2) - Kendall tau";
run;

/* 方法1b：Frank Copula条件分布法 */
data frank_cond;
    call streaminit(2024);
    alpha = 5;
    do i = 1 to 2000;
        u1 = rand('UNIFORM');
        t = rand('UNIFORM');
        denom = exp(-alpha*u1) - t*(exp(-alpha*u1)-1) - exp(-alpha) + t*(exp(-alpha)-1);
        u2 = -1/alpha * log(1 + (1-exp(-alpha))*t / denom);
        x1 = -log(1 - u1);
        x2 = -log(1 - u2);
        output;
    end;
run;

proc corr data=frank_cond kendall;
    var x1 x2;
    title "Frank Copula模拟 (条件分布法, alpha=5) - Kendall tau";
run;

/* 方法2：脆弱性模型法 */
data frailty_clayton;
    call streaminit(2024);
    alpha = 2;
    do i = 1 to 2000;
        gamma = rand('GAMMA', 1/alpha);
        U1 = rand('UNIFORM');
        U2 = rand('UNIFORM');
        u1 = (1 + (-log(U1)) / gamma)**(-1/alpha);
        u2 = (1 + (-log(U2)) / gamma)**(-1/alpha);
        x1 = -log(1 - u1);
        x2 = -log(1 - u2);
        output;
    end;
run;

proc corr data=frailty_clayton kendall;
    var x1 x2;
    title "Clayton Copula模拟 (脆弱性模型法, alpha=2) - Kendall tau";
run;
