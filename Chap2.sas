/* Chap2 SAS代码 */
/* 自动从chap2.html同步生成 */

/* 2.1 短期聚合风险模型：卷积法（例2.1） */
proc iml;
  pn = {0.3, 0.5, 0.2};           /* 索赔次数概率 */
  fx = {0.2, 0.4, 0.2, 0.1, 0.1}; /* 索赔强度概率 */

  /* 卷积法计算累积索赔金额分布 */
  /* S = X1 + X2 + ... + XN */
  /* 先计算N=0,1,2时的分布 */
  max_s = 4 * 4;  /* 最大索赔金额*最大索赔次数 */
  FS = j(max_s+1, 1, 0);

  /* N=0: S=0, P(S=0)=P(N=0)=0.3 */
  FS[1] = pn[1];

  /* N=1: S=X, P(S=k)=P(N=1)*fx[k] */
  do k = 1 to 4;
    FS[k+1] = FS[k+1] + pn[2] * fx[k];
  end;

  /* N=2: S=X1+X2, 卷积 */
  do k1 = 1 to 4;
    do k2 = 1 to 4;
      s = k1 + k2;
      FS[s+1] = FS[s+1] + pn[3] * fx[k1] * fx[k2];
    end;
  end;

  /* 输出累积分布 */
  S_values = (0:max_s)`;
  print S_values FS;
  print "累积损失等于0的概率:" FS[1];
quit;


/****************************************************************************/

/****************************************************************************/
/* 第2章 风险模型                                                           */
/* 对应教材：section2.tex                                                   */
/* 内容：短期聚合风险模型、复合分布、短期个体风险模型、                      */
/*       参数不确定性的影响、近似计算方法                                    */
/****************************************************************************/

/****************************************************************************/
/* 2.1 短期聚合风险模型：卷积法                                             */
/****************************************************************************/
/* 索赔次数概率分布和索赔强度概率分布 */
proc iml;
  pn = {0.3, 0.5, 0.2};           /* 索赔次数概率 */
  fx = {0.2, 0.4, 0.2, 0.1, 0.1}; /* 索赔强度概率 */

  /* 卷积法计算累积索赔金额分布 */
  /* S = X1 + X2 + ... + XN */
  /* 先计算N=0,1,2时的分布 */
  max_s = 4 * 4;  /* 最大索赔金额*最大索赔次数 */
  FS = j(max_s+1, 1, 0);

  /* N=0: S=0, P(S=0)=P(N=0)=0.3 */
  FS[1] = pn[1];

  /* N=1: S=X, P(S=k)=P(N=1)*fx[k] */
  do k = 1 to 4;
    FS[k+1] = FS[k+1] + pn[2] * fx[k];
  end;

  /* N=2: S=X1+X2, 卷积 */
  do k1 = 1 to 4;
    do k2 = 1 to 4;
      s = k1 + k2;
      FS[s+1] = FS[s+1] + pn[3] * fx[k1] * fx[k2];
    end;
  end;

  /* 输出累积分布 */
  S_values = (0:max_s)`;
  print S_values FS;
  print "累积损失等于0的概率:" FS[1];
quit;


/****************************************************************************/
/* 2.2 复合分布的随机模拟                                                   */
/****************************************************************************/

/* 2.2.1 复合泊松分布的随机模拟 */
data compound_poisson;
  call streaminit(123);
  do sim = 1 to 10000;
    N = rand('POISSON', 2.5);
    S = 0;
    do j = 1 to N;
      S = S + rand('GAMMA', 2, 500);  /* shape=2, scale=500 */
    end;
    output;
  end;
run;

title "复合泊松分布模拟";
proc means data=compound_poisson mean var std;
  var S;
run;

proc sgplot data=compound_poisson;
  histogram S / nbins=30 scale=count;
  density S / type=kernel;
  yaxis label="频率";
run;
title;


/* 2.2.2 复合二项分布的随机模拟 */
data compound_binomial;
  call streaminit(123);
  do sim = 1 to 10000;
    N = rand('BINOMIAL', 100, 0.01);
    S = 0;
    do j = 1 to N;
      S = S + rand('GAMMA', 10, 5);  /* shape=10, scale=5 => rate=0.2 */
    end;
    output;
  end;
run;

title "复合二项分布模拟";
proc means data=compound_binomial mean var std;
  var S;
run;
title;


/* 2.2.3 复合负二项分布的随机模拟 */
data compound_negbinom;
  call streaminit(123);
  do sim = 1 to 10000;
    N = rand('NEGBINOMIAL', 0.5, 2.5);  /* p=0.5, r=2.5 */
    S = 0;
    do j = 1 to N;
      S = S + rand('GAMMA', 2, 500);
    end;
    output;
  end;
run;

title "复合负二项分布模拟";
proc means data=compound_negbinom mean var std;
  var S;
run;
title;


/****************************************************************************/
/* 2.3 短期个体风险模型                                                     */
/****************************************************************************/
data individual_risk;
  call streaminit(123);
  do sim = 1 to 10000;
    S = 0;
    /* 50个低风险个体（索赔概率0.1） */
    do i = 1 to 50;
      Ni = rand('BINOMIAL', 1, 0.1);
      Xi = rand('GAMMA', 10, 50);  /* shape=10, scale=50 => rate=0.02 */
      S = S + Ni * Xi;
    end;
    /* 50个高风险个体（索赔概率0.2） */
    do i = 1 to 50;
      Ni = rand('BINOMIAL', 1, 0.2);
      Xi = rand('LOGNORMAL', 5, 1);
      S = S + Ni * Xi;
    end;
    output;
  end;
run;

title "个体风险模型模拟";
proc means data=individual_risk mean var std;
  var S;
run;

proc sgplot data=individual_risk;
  histogram S / nbins=30 scale=count;
  density S / type=kernel;
run;
title;


/****************************************************************************/
/* 2.4 聚合风险模型计算方法                                                 */
/****************************************************************************/

/* 2.4.1 Panjer递推 */
proc iml;
  /* Panjer递推（泊松分布） */
  start Panjer_Poisson(p, lambda);
    if sum(p) > 1 | any(p < 0) then
      print "Error: p is not a density";
    cumul = exp(-lambda * sum(p));
    f = cumul;
    s = 0;
    do until(cumul > 0.99999999);
      s = s + 1;
      m = min(s, nrow(p));
      last = lambda / s * sum((1:m) # p[1:m] # f[(s+1-m):s]);
      f = f // last;
      cumul = cumul + last;
    end;
    return(f);
  finish;

  p = {0.25, 0.5, 0.25};
  lambda = 4;
  f = Panjer_Poisson(p, lambda);
  f_scaled = f * exp(lambda);
  print "Panjer递推结果" f_scaled;
quit;


/* 2.4.2 FFT法（快速傅里叶变换） */
proc iml;
  x = {0, 0.5, 0.4, 0.1} // j(40, 1, 0);
  phi_x = fft(x);
  phi_s = exp(3 * (phi_x - 1));
  fs = fft(phil_s) / nrow(phil_s);
  fs = real(fs);
  Fs = cusum(fs);

  s_vals = (0:43)`;
  print "FFT法计算结果" s_vals fs;
quit;


/****************************************************************************/
/* 2.5 随机模拟求累积损失的分布（例2.7）                                    */
/****************************************************************************/
data claim_sim;
  call streaminit(321);
  d = 250; u = 1000;       /* 免赔额和限额 */
  r = 3; beta = 2;         /* 负二项分布参数 */
  alpha = 100; theta = 0.2; /* 伽马分布参数 */
  do iter = 1 to 10000;
    N = rand('NEGBINOMIAL', 1/(1+beta), r);
    S = 0; w_total = 0;
    do j = 1 to N;
      x = rand('GAMMA', alpha, 1/theta);
      w = min(x, d);
      w_total = w_total + w;
      S = S + x;
    end;
    v = min(w_total, u);
    P = S - v;  /* 保险人的年度累积赔款 */
    output;
  end;
run;

title "例2.7：保险人累积赔款模拟";
proc means data=claim_sim mean std p95;
  var P;
run;

proc sgplot data=claim_sim;
  histogram P / nbins=50 scale=count;
  density P / type=kernel;
run;
title;


/****************************************************************************/
/* 2.6 近似计算方法                                                         */
/****************************************************************************/

/* 2.6.1 Tweedie分布模拟 */
data tweedie_sim;
  call streaminit(11);
  lambda = 1; alpha = 10; beta_rate = 2;
  do i = 1 to 10000;
    N = rand('POISSON', lambda);
    Y = 0;
    do j = 1 to N;
      Y = Y + rand('GAMMA', alpha, 1/beta_rate);
    end;
    output;
  end;
run;

title "Tweedie分布模拟";
proc sgplot data=tweedie_sim;
  histogram Y / nbins=50;
run;
title;


/* 2.6.2 近似计算比较 */
proc iml;
  call streaminit(123);
  n_sim = 10000;
  S = j(n_sim, 1, 0);
  do i = 1 to n_sim;
    N = rand('BINOMIAL', 1000, 0.001);
    S[i] = N;  /* 每次索赔金额为1 */
  end;

  ES = mean(S);
  varS = var(S);

  /* 1. 精确分布：二项分布和泊松分布 */
  p0 = 1 - cdf('BINOMIAL', 3.5, 0.001, 1000);
  p1 = 1 - cdf('POISSON', 3.5, 1);

  /* 2. 正态近似 */
  p2 = 1 - cdf('NORMAL', 3.5, ES, sqrt(varS));

  /* 3. 平移伽马近似 */
  /* S+1 ~ Gamma(4, 2) */
  p4 = 1 - cdf('GAMMA', 4.5, 4, 0.5);

  /* 4. NP近似 */
  mu_np = 1; sigma_np = 1; gamma_np = 1;
  a = -3/gamma_np + sqrt(9/gamma_np**2 + 1 + 6/gamma_np * (3.5 - mu_np) / sigma_np);
  p5 = 1 - cdf('NORMAL', a);

  print "近似计算比较 P(S>3.5)";
  print "二项分布(精确):" p0;
  print "泊松分布(精确):" p1;
  print "正态近似:" p2;
  print "平移伽马近似:" p4;
  print "NP近似:" p5;
quit;

/****************************************************************************/
/* 第2章 风险模型                                                           */
/* 对应教材：section2.tex                                                   */
/* 内容：短期聚合风险模型、复合分布、短期个体风险模型、                      */
/*       参数不确定性的影响、近似计算方法                                    */
/****************************************************************************/

/****************************************************************************/
/* 2.1 短期聚合风险模型：卷积法                                             */
/****************************************************************************/
/* 索赔次数概率分布和索赔强度概率分布 */
proc iml;
  pn = {0.3, 0.5, 0.2};           /* 索赔次数概率 */
  fx = {0.2, 0.4, 0.2, 0.1, 0.1}; /* 索赔强度概率 */

  /* 卷积法计算累积索赔金额分布 */
  /* S = X1 + X2 + ... + XN */
  /* 先计算N=0,1,2时的分布 */
  max_s = 4 * 4;  /* 最大索赔金额*最大索赔次数 */
  FS = j(max_s+1, 1, 0);

  /* N=0: S=0, P(S=0)=P(N=0)=0.3 */
  FS[1] = pn[1];

  /* N=1: S=X, P(S=k)=P(N=1)*fx[k] */
  do k = 1 to 4;
    FS[k+1] = FS[k+1] + pn[2] * fx[k];
  end;

  /* N=2: S=X1+X2, 卷积 */
  do k1 = 1 to 4;
    do k2 = 1 to 4;
      s = k1 + k2;
      FS[s+1] = FS[s+1] + pn[3] * fx[k1] * fx[k2];
    end;
  end;

  /* 输出累积分布 */
  S_values = (0:max_s)`;
  print S_values FS;
  print "累积损失等于0的概率:" FS[1];
quit;


/****************************************************************************/
/* 2.2 复合分布的随机模拟                                                   */
/****************************************************************************/

/* 2.2.1 复合泊松分布的随机模拟 */
data compound_poisson;
  call streaminit(123);
  do sim = 1 to 10000;
    N = rand('POISSON', 2.5);
    S = 0;
    do j = 1 to N;
      S = S + rand('GAMMA', 2, 500);  /* shape=2, scale=500 */
    end;
    output;
  end;
run;

title "复合泊松分布模拟";
proc means data=compound_poisson mean var std;
  var S;
run;

proc sgplot data=compound_poisson;
  histogram S / nbins=30 scale=count;
  density S / type=kernel;
  yaxis label="频率";
run;
title;


/* 2.2.2 复合二项分布的随机模拟 */
data compound_binomial;
  call streaminit(123);
  do sim = 1 to 10000;
    N = rand('BINOMIAL', 100, 0.01);
    S = 0;
    do j = 1 to N;
      S = S + rand('GAMMA', 10, 5);  /* shape=10, scale=5 => rate=0.2 */
    end;
    output;
  end;
run;

title "复合二项分布模拟";
proc means data=compound_binomial mean var std;
  var S;
run;
title;


/* 2.2.3 复合负二项分布的随机模拟 */
data compound_negbinom;
  call streaminit(123);
  do sim = 1 to 10000;
    N = rand('NEGBINOMIAL', 0.5, 2.5);  /* p=0.5, r=2.5 */
    S = 0;
    do j = 1 to N;
      S = S + rand('GAMMA', 2, 500);
    end;
    output;
  end;
run;

title "复合负二项分布模拟";
proc means data=compound_negbinom mean var std;
  var S;
run;
title;


/****************************************************************************/
/* 2.3 短期个体风险模型                                                     */
/****************************************************************************/
data individual_risk;
  call streaminit(123);
  do sim = 1 to 10000;
    S = 0;
    /* 50个低风险个体（索赔概率0.1） */
    do i = 1 to 50;
      Ni = rand('BINOMIAL', 1, 0.1);
      Xi = rand('GAMMA', 10, 50);  /* shape=10, scale=50 => rate=0.02 */
      S = S + Ni * Xi;
    end;
    /* 50个高风险个体（索赔概率0.2） */
    do i = 1 to 50;
      Ni = rand('BINOMIAL', 1, 0.2);
      Xi = rand('LOGNORMAL', 5, 1);
      S = S + Ni * Xi;
    end;
    output;
  end;
run;

title "个体风险模型模拟";
proc means data=individual_risk mean var std;
  var S;
run;

proc sgplot data=individual_risk;
  histogram S / nbins=30 scale=count;
  density S / type=kernel;
run;
title;


/****************************************************************************/
/* 2.4 聚合风险模型计算方法                                                 */
/****************************************************************************/

/* 2.4.1 Panjer递推 */
proc iml;
  /* Panjer递推（泊松分布） */
  start Panjer_Poisson(p, lambda);
    if sum(p) > 1 | any(p < 0) then
      print "Error: p is not a density";
    cumul = exp(-lambda * sum(p));
    f = cumul;
    s = 0;
    do until(cumul > 0.99999999);
      s = s + 1;
      m = min(s, nrow(p));
      last = lambda / s * sum((1:m) # p[1:m] # f[(s+1-m):s]);
      f = f // last;
      cumul = cumul + last;
    end;
    return(f);
  finish;

  p = {0.25, 0.5, 0.25};
  lambda = 4;
  f = Panjer_Poisson(p, lambda);
  f_scaled = f * exp(lambda);
  print "Panjer递推结果" f_scaled;
quit;


/* 2.4.2 FFT法（快速傅里叶变换） */
proc iml;
  x = {0, 0.5, 0.4, 0.1} // j(40, 1, 0);
  phi_x = fft(x);
  phi_s = exp(3 * (phi_x - 1));
  fs = fft(phil_s) / nrow(phil_s);
  fs = real(fs);
  Fs = cusum(fs);

  s_vals = (0:43)`;
  print "FFT法计算结果" s_vals fs;
quit;


/****************************************************************************/
/* 2.5 随机模拟求累积损失的分布（例2.7）                                    */
/****************************************************************************/
data claim_sim;
  call streaminit(321);
  d = 250; u = 1000;       /* 免赔额和限额 */
  r = 3; beta = 2;         /* 负二项分布参数 */
  alpha = 100; theta = 0.2; /* 伽马分布参数 */
  do iter = 1 to 10000;
    N = rand('NEGBINOMIAL', 1/(1+beta), r);
    S = 0; w_total = 0;
    do j = 1 to N;
      x = rand('GAMMA', alpha, 1/theta);
      w = min(x, d);
      w_total = w_total + w;
      S = S + x;
    end;
    v = min(w_total, u);
    P = S - v;  /* 保险人的年度累积赔款 */
    output;
  end;
run;

title "例2.7：保险人累积赔款模拟";
proc means data=claim_sim mean std p95;
  var P;
run;

proc sgplot data=claim_sim;
  histogram P / nbins=50 scale=count;
  density P / type=kernel;
run;
title;


/****************************************************************************/
/* 2.6 近似计算方法                                                         */
/****************************************************************************/

/* 2.6.1 Tweedie分布模拟 */
data tweedie_sim;
  call streaminit(11);
  lambda = 1; alpha = 10; beta_rate = 2;
  do i = 1 to 10000;
    N = rand('POISSON', lambda);
    Y = 0;
    do j = 1 to N;
      Y = Y + rand('GAMMA', alpha, 1/beta_rate);
    end;
    output;
  end;
run;

title "Tweedie分布模拟";
proc sgplot data=tweedie_sim;
  histogram Y / nbins=50;
run;
title;


/* 2.6.2 近似计算比较 */
proc iml;
  call streaminit(123);
  n_sim = 10000;
  S = j(n_sim, 1, 0);
  do i = 1 to n_sim;
    N = rand('BINOMIAL', 1000, 0.001);
    S[i] = N;  /* 每次索赔金额为1 */
  end;

  ES = mean(S);
  varS = var(S);

  /* 1. 精确分布：二项分布和泊松分布 */
  p0 = 1 - cdf('BINOMIAL', 3.5, 0.001, 1000);
  p1 = 1 - cdf('POISSON', 3.5, 1);

  /* 2. 正态近似 */
  p2 = 1 - cdf('NORMAL', 3.5, ES, sqrt(varS));

  /* 3. 平移伽马近似 */
  /* S+1 ~ Gamma(4, 2) */
  p4 = 1 - cdf('GAMMA', 4.5, 4, 0.5);

  /* 4. NP近似 */
  mu_np = 1; sigma_np = 1; gamma_np = 1;
  a = -3/gamma_np + sqrt(9/gamma_np**2 + 1 + 6/gamma_np * (3.5 - mu_np) / sigma_np);
  p5 = 1 - cdf('NORMAL', a);

  print "近似计算比较 P(S>3.5)";
  print "二项分布(精确):" p0;
  print "泊松分布(精确):" p1;
  print "正态近似:" p2;
  print "平移伽马近似:" p4;
  print "NP近似:" p5;
quit;

/****************************************************************************/
/* 第2章 风险模型                                                           */
/* 对应教材：section2.tex                                                   */
/* 内容：短期聚合风险模型、复合分布、短期个体风险模型、                      */
/*       参数不确定性的影响、近似计算方法                                    */
/****************************************************************************/

/****************************************************************************/
/* 2.1 短期聚合风险模型：卷积法                                             */
/****************************************************************************/
/* 索赔次数概率分布和索赔强度概率分布 */
proc iml;
  pn = {0.3, 0.5, 0.2};           /* 索赔次数概率 */
  fx = {0.2, 0.4, 0.2, 0.1, 0.1}; /* 索赔强度概率 */

  /* 卷积法计算累积索赔金额分布 */
  /* S = X1 + X2 + ... + XN */
  /* 先计算N=0,1,2时的分布 */
  max_s = 4 * 4;  /* 最大索赔金额*最大索赔次数 */
  FS = j(max_s+1, 1, 0);

  /* N=0: S=0, P(S=0)=P(N=0)=0.3 */
  FS[1] = pn[1];

  /* N=1: S=X, P(S=k)=P(N=1)*fx[k] */
  do k = 1 to 4;
    FS[k+1] = FS[k+1] + pn[2] * fx[k];
  end;

  /* N=2: S=X1+X2, 卷积 */
  do k1 = 1 to 4;
    do k2 = 1 to 4;
      s = k1 + k2;
      FS[s+1] = FS[s+1] + pn[3] * fx[k1] * fx[k2];
    end;
  end;

  /* 输出累积分布 */
  S_values = (0:max_s)`;
  print S_values FS;
  print "累积损失等于0的概率:" FS[1];
quit;


/****************************************************************************/
/* 2.2 复合分布的随机模拟                                                   */
/****************************************************************************/

/* 2.2.1 复合泊松分布的随机模拟 */
data compound_poisson;
  call streaminit(123);
  do sim = 1 to 10000;
    N = rand('POISSON', 2.5);
    S = 0;
    do j = 1 to N;
      S = S + rand('GAMMA', 2, 500);  /* shape=2, scale=500 */
    end;
    output;
  end;
run;

title "复合泊松分布模拟";
proc means data=compound_poisson mean var std;
  var S;
run;

proc sgplot data=compound_poisson;
  histogram S / nbins=30 scale=count;
  density S / type=kernel;
  yaxis label="频率";
run;
title;


/* 2.2.2 复合二项分布的随机模拟 */
data compound_binomial;
  call streaminit(123);
  do sim = 1 to 10000;
    N = rand('BINOMIAL', 100, 0.01);
    S = 0;
    do j = 1 to N;
      S = S + rand('GAMMA', 10, 5);  /* shape=10, scale=5 => rate=0.2 */
    end;
    output;
  end;
run;

title "复合二项分布模拟";
proc means data=compound_binomial mean var std;
  var S;
run;
title;


/* 2.2.3 复合负二项分布的随机模拟 */
data compound_negbinom;
  call streaminit(123);
  do sim = 1 to 10000;
    N = rand('NEGBINOMIAL', 0.5, 2.5);  /* p=0.5, r=2.5 */
    S = 0;
    do j = 1 to N;
      S = S + rand('GAMMA', 2, 500);
    end;
    output;
  end;
run;

title "复合负二项分布模拟";
proc means data=compound_negbinom mean var std;
  var S;
run;
title;


/****************************************************************************/
/* 2.3 短期个体风险模型                                                     */
/****************************************************************************/
data individual_risk;
  call streaminit(123);
  do sim = 1 to 10000;
    S = 0;
    /* 50个低风险个体（索赔概率0.1） */
    do i = 1 to 50;
      Ni = rand('BINOMIAL', 1, 0.1);
      Xi = rand('GAMMA', 10, 50);  /* shape=10, scale=50 => rate=0.02 */
      S = S + Ni * Xi;
    end;
    /* 50个高风险个体（索赔概率0.2） */
    do i = 1 to 50;
      Ni = rand('BINOMIAL', 1, 0.2);
      Xi = rand('LOGNORMAL', 5, 1);
      S = S + Ni * Xi;
    end;
    output;
  end;
run;

title "个体风险模型模拟";
proc means data=individual_risk mean var std;
  var S;
run;

proc sgplot data=individual_risk;
  histogram S / nbins=30 scale=count;
  density S / type=kernel;
run;
title;


/****************************************************************************/
/* 2.4 聚合风险模型计算方法                                                 */
/****************************************************************************/

/* 2.4.1 Panjer递推 */
proc iml;
  /* Panjer递推（泊松分布） */
  start Panjer_Poisson(p, lambda);
    if sum(p) > 1 | any(p < 0) then
      print "Error: p is not a density";
    cumul = exp(-lambda * sum(p));
    f = cumul;
    s = 0;
    do until(cumul > 0.99999999);
      s = s + 1;
      m = min(s, nrow(p));
      last = lambda / s * sum((1:m) # p[1:m] # f[(s+1-m):s]);
      f = f // last;
      cumul = cumul + last;
    end;
    return(f);
  finish;

  p = {0.25, 0.5, 0.25};
  lambda = 4;
  f = Panjer_Poisson(p, lambda);
  f_scaled = f * exp(lambda);
  print "Panjer递推结果" f_scaled;
quit;


/* 2.4.2 FFT法（快速傅里叶变换） */
proc iml;
  x = {0, 0.5, 0.4, 0.1} // j(40, 1, 0);
  phi_x = fft(x);
  phi_s = exp(3 * (phi_x - 1));
  fs = fft(phil_s) / nrow(phil_s);
  fs = real(fs);
  Fs = cusum(fs);

  s_vals = (0:43)`;
  print "FFT法计算结果" s_vals fs;
quit;


/****************************************************************************/
/* 2.5 随机模拟求累积损失的分布（例2.7）                                    */
/****************************************************************************/
data claim_sim;
  call streaminit(321);
  d = 250; u = 1000;       /* 免赔额和限额 */
  r = 3; beta = 2;         /* 负二项分布参数 */
  alpha = 100; theta = 0.2; /* 伽马分布参数 */
  do iter = 1 to 10000;
    N = rand('NEGBINOMIAL', 1/(1+beta), r);
    S = 0; w_total = 0;
    do j = 1 to N;
      x = rand('GAMMA', alpha, 1/theta);
      w = min(x, d);
      w_total = w_total + w;
      S = S + x;
    end;
    v = min(w_total, u);
    P = S - v;  /* 保险人的年度累积赔款 */
    output;
  end;
run;

title "例2.7：保险人累积赔款模拟";
proc means data=claim_sim mean std p95;
  var P;
run;

proc sgplot data=claim_sim;
  histogram P / nbins=50 scale=count;
  density P / type=kernel;
run;
title;


/****************************************************************************/
/* 2.6 近似计算方法                                                         */
/****************************************************************************/

/* 2.6.1 Tweedie分布模拟 */
data tweedie_sim;
  call streaminit(11);
  lambda = 1; alpha = 10; beta_rate = 2;
  do i = 1 to 10000;
    N = rand('POISSON', lambda);
    Y = 0;
    do j = 1 to N;
      Y = Y + rand('GAMMA', alpha, 1/beta_rate);
    end;
    output;
  end;
run;

title "Tweedie分布模拟";
proc sgplot data=tweedie_sim;
  histogram Y / nbins=50;
run;
title;


/* 2.6.2 近似计算比较 */
proc iml;
  call streaminit(123);
  n_sim = 10000;
  S = j(n_sim, 1, 0);
  do i = 1 to n_sim;
    N = rand('BINOMIAL', 1000, 0.001);
    S[i] = N;  /* 每次索赔金额为1 */
  end;

  ES = mean(S);
  varS = var(S);

  /* 1. 精确分布：二项分布和泊松分布 */
  p0 = 1 - cdf('BINOMIAL', 3.5, 0.001, 1000);
  p1 = 1 - cdf('POISSON', 3.5, 1);

  /* 2. 正态近似 */
  p2 = 1 - cdf('NORMAL', 3.5, ES, sqrt(varS));

  /* 3. 平移伽马近似 */
  /* S+1 ~ Gamma(4, 2) */
  p4 = 1 - cdf('GAMMA', 4.5, 4, 0.5);

  /* 4. NP近似 */
  mu_np = 1; sigma_np = 1; gamma_np = 1;
  a = -3/gamma_np + sqrt(9/gamma_np**2 + 1 + 6/gamma_np * (3.5 - mu_np) / sigma_np);
  p5 = 1 - cdf('NORMAL', a);

  print "近似计算比较 P(S>3.5)";
  print "二项分布(精确):" p0;
  print "泊松分布(精确):" p1;
  print "正态近似:" p2;
  print "平移伽马近似:" p4;
  print "NP近似:" p5;
quit;

/****************************************************************************/
/* 第2章 风险模型                                                           */
/* 对应教材：section2.tex                                                   */
/* 内容：短期聚合风险模型、复合分布、短期个体风险模型、                      */
/*       参数不确定性的影响、近似计算方法                                    */
/****************************************************************************/

/****************************************************************************/
/* 2.1 短期聚合风险模型：卷积法                                             */
/****************************************************************************/
/* 索赔次数概率分布和索赔强度概率分布 */
proc iml;
  pn = {0.3, 0.5, 0.2};           /* 索赔次数概率 */
  fx = {0.2, 0.4, 0.2, 0.1, 0.1}; /* 索赔强度概率 */

  /* 卷积法计算累积索赔金额分布 */
  /* S = X1 + X2 + ... + XN */
  /* 先计算N=0,1,2时的分布 */
  max_s = 4 * 4;  /* 最大索赔金额*最大索赔次数 */
  FS = j(max_s+1, 1, 0);

  /* N=0: S=0, P(S=0)=P(N=0)=0.3 */
  FS[1] = pn[1];

  /* N=1: S=X, P(S=k)=P(N=1)*fx[k] */
  do k = 1 to 4;
    FS[k+1] = FS[k+1] + pn[2] * fx[k];
  end;

  /* N=2: S=X1+X2, 卷积 */
  do k1 = 1 to 4;
    do k2 = 1 to 4;
      s = k1 + k2;
      FS[s+1] = FS[s+1] + pn[3] * fx[k1] * fx[k2];
    end;
  end;

  /* 输出累积分布 */
  S_values = (0:max_s)`;
  print S_values FS;
  print "累积损失等于0的概率:" FS[1];
quit;


/****************************************************************************/
/* 2.2 复合分布的随机模拟                                                   */
/****************************************************************************/

/* 2.2.1 复合泊松分布的随机模拟 */
data compound_poisson;
  call streaminit(123);
  do sim = 1 to 10000;
    N = rand('POISSON', 2.5);
    S = 0;
    do j = 1 to N;
      S = S + rand('GAMMA', 2, 500);  /* shape=2, scale=500 */
    end;
    output;
  end;
run;

title "复合泊松分布模拟";
proc means data=compound_poisson mean var std;
  var S;
run;

proc sgplot data=compound_poisson;
  histogram S / nbins=30 scale=count;
  density S / type=kernel;
  yaxis label="频率";
run;
title;


/* 2.2.2 复合二项分布的随机模拟 */
data compound_binomial;
  call streaminit(123);
  do sim = 1 to 10000;
    N = rand('BINOMIAL', 100, 0.01);
    S = 0;
    do j = 1 to N;
      S = S + rand('GAMMA', 10, 5);  /* shape=10, scale=5 => rate=0.2 */
    end;
    output;
  end;
run;

title "复合二项分布模拟";
proc means data=compound_binomial mean var std;
  var S;
run;
title;


/* 2.2.3 复合负二项分布的随机模拟 */
data compound_negbinom;
  call streaminit(123);
  do sim = 1 to 10000;
    N = rand('NEGBINOMIAL', 0.5, 2.5);  /* p=0.5, r=2.5 */
    S = 0;
    do j = 1 to N;
      S = S + rand('GAMMA', 2, 500);
    end;
    output;
  end;
run;

title "复合负二项分布模拟";
proc means data=compound_negbinom mean var std;
  var S;
run;
title;


/****************************************************************************/
/* 2.3 短期个体风险模型                                                     */
/****************************************************************************/
data individual_risk;
  call streaminit(123);
  do sim = 1 to 10000;
    S = 0;
    /* 50个低风险个体（索赔概率0.1） */
    do i = 1 to 50;
      Ni = rand('BINOMIAL', 1, 0.1);
      Xi = rand('GAMMA', 10, 50);  /* shape=10, scale=50 => rate=0.02 */
      S = S + Ni * Xi;
    end;
    /* 50个高风险个体（索赔概率0.2） */
    do i = 1 to 50;
      Ni = rand('BINOMIAL', 1, 0.2);
      Xi = rand('LOGNORMAL', 5, 1);
      S = S + Ni * Xi;
    end;
    output;
  end;
run;

title "个体风险模型模拟";
proc means data=individual_risk mean var std;
  var S;
run;

proc sgplot data=individual_risk;
  histogram S / nbins=30 scale=count;
  density S / type=kernel;
run;
title;


/****************************************************************************/
/* 2.4 聚合风险模型计算方法                                                 */
/****************************************************************************/

/* 2.4.1 Panjer递推 */
proc iml;
  /* Panjer递推（泊松分布） */
  start Panjer_Poisson(p, lambda);
    if sum(p) > 1 | any(p < 0) then
      print "Error: p is not a density";
    cumul = exp(-lambda * sum(p));
    f = cumul;
    s = 0;
    do until(cumul > 0.99999999);
      s = s + 1;
      m = min(s, nrow(p));
      last = lambda / s * sum((1:m) # p[1:m] # f[(s+1-m):s]);
      f = f // last;
      cumul = cumul + last;
    end;
    return(f);
  finish;

  p = {0.25, 0.5, 0.25};
  lambda = 4;
  f = Panjer_Poisson(p, lambda);
  f_scaled = f * exp(lambda);
  print "Panjer递推结果" f_scaled;
quit;


/* 2.4.2 FFT法（快速傅里叶变换） */
proc iml;
  x = {0, 0.5, 0.4, 0.1} // j(40, 1, 0);
  phi_x = fft(x);
  phi_s = exp(3 * (phi_x - 1));
  fs = fft(phil_s) / nrow(phil_s);
  fs = real(fs);
  Fs = cusum(fs);

  s_vals = (0:43)`;
  print "FFT法计算结果" s_vals fs;
quit;


/****************************************************************************/
/* 2.5 随机模拟求累积损失的分布（例2.7）                                    */
/****************************************************************************/
data claim_sim;
  call streaminit(321);
  d = 250; u = 1000;       /* 免赔额和限额 */
  r = 3; beta = 2;         /* 负二项分布参数 */
  alpha = 100; theta = 0.2; /* 伽马分布参数 */
  do iter = 1 to 10000;
    N = rand('NEGBINOMIAL', 1/(1+beta), r);
    S = 0; w_total = 0;
    do j = 1 to N;
      x = rand('GAMMA', alpha, 1/theta);
      w = min(x, d);
      w_total = w_total + w;
      S = S + x;
    end;
    v = min(w_total, u);
    P = S - v;  /* 保险人的年度累积赔款 */
    output;
  end;
run;

title "例2.7：保险人累积赔款模拟";
proc means data=claim_sim mean std p95;
  var P;
run;

proc sgplot data=claim_sim;
  histogram P / nbins=50 scale=count;
  density P / type=kernel;
run;
title;


/****************************************************************************/
/* 2.6 近似计算方法                                                         */
/****************************************************************************/

/* 2.6.1 Tweedie分布模拟 */
data tweedie_sim;
  call streaminit(11);
  lambda = 1; alpha = 10; beta_rate = 2;
  do i = 1 to 10000;
    N = rand('POISSON', lambda);
    Y = 0;
    do j = 1 to N;
      Y = Y + rand('GAMMA', alpha, 1/beta_rate);
    end;
    output;
  end;
run;

title "Tweedie分布模拟";
proc sgplot data=tweedie_sim;
  histogram Y / nbins=50;
run;
title;


/* 2.6.2 近似计算比较 */
proc iml;
  call streaminit(123);
  n_sim = 10000;
  S = j(n_sim, 1, 0);
  do i = 1 to n_sim;
    N = rand('BINOMIAL', 1000, 0.001);
    S[i] = N;  /* 每次索赔金额为1 */
  end;

  ES = mean(S);
  varS = var(S);

  /* 1. 精确分布：二项分布和泊松分布 */
  p0 = 1 - cdf('BINOMIAL', 3.5, 0.001, 1000);
  p1 = 1 - cdf('POISSON', 3.5, 1);

  /* 2. 正态近似 */
  p2 = 1 - cdf('NORMAL', 3.5, ES, sqrt(varS));

  /* 3. 平移伽马近似 */
  /* S+1 ~ Gamma(4, 2) */
  p4 = 1 - cdf('GAMMA', 4.5, 4, 0.5);

  /* 4. NP近似 */
  mu_np = 1; sigma_np = 1; gamma_np = 1;
  a = -3/gamma_np + sqrt(9/gamma_np**2 + 1 + 6/gamma_np * (3.5 - mu_np) / sigma_np);
  p5 = 1 - cdf('NORMAL', a);

  print "近似计算比较 P(S>3.5)";
  print "二项分布(精确):" p0;
  print "泊松分布(精确):" p1;
  print "正态近似:" p2;
  print "平移伽马近似:" p4;
  print "NP近似:" p5;
quit;

/* 2.5 随机模拟求累积损失的分布（例2.7）                                    */
/****************************************************************************/
data claim_sim;
  call streaminit(321);
  d = 250; u = 1000;       /* 免赔额和限额 */
  r = 3; beta = 2;         /* 负二项分布参数 */
  alpha = 100; theta = 0.2; /* 伽马分布参数 */
  do iter = 1 to 10000;
    N = rand('NEGBINOMIAL', 1/(1+beta), r);
    S = 0; w_total = 0;
    do j = 1 to N;
      x = rand('GAMMA', alpha, 1/theta);
      w = min(x, d);
      w_total = w_total + w;
      S = S + x;
    end;
    v = min(w_total, u);
    P = S - v;  /* 保险人的年度累积赔款 */
    output;
  end;
run;

title "例2.7：保险人累积赔款模拟";
proc means data=claim_sim mean std p95;
  var P;
run;

proc sgplot data=claim_sim;
  histogram P / nbins=50 scale=count;
  density P / type=kernel;
run;
title;


/****************************************************************************/

/****************************************************************************/
/* 第2章 风险模型                                                           */
/* 对应教材：section2.tex                                                   */
/* 内容：短期聚合风险模型、复合分布、短期个体风险模型、                      */
/*       参数不确定性的影响、近似计算方法                                    */
/****************************************************************************/

/****************************************************************************/
/* 2.1 短期聚合风险模型：卷积法                                             */
/****************************************************************************/
/* 索赔次数概率分布和索赔强度概率分布 */
proc iml;
  pn = {0.3, 0.5, 0.2};           /* 索赔次数概率 */
  fx = {0.2, 0.4, 0.2, 0.1, 0.1}; /* 索赔强度概率 */

  /* 卷积法计算累积索赔金额分布 */
  /* S = X1 + X2 + ... + XN */
  /* 先计算N=0,1,2时的分布 */
  max_s = 4 * 4;  /* 最大索赔金额*最大索赔次数 */
  FS = j(max_s+1, 1, 0);

  /* N=0: S=0, P(S=0)=P(N=0)=0.3 */
  FS[1] = pn[1];

  /* N=1: S=X, P(S=k)=P(N=1)*fx[k] */
  do k = 1 to 4;
    FS[k+1] = FS[k+1] + pn[2] * fx[k];
  end;

  /* N=2: S=X1+X2, 卷积 */
  do k1 = 1 to 4;
    do k2 = 1 to 4;
      s = k1 + k2;
      FS[s+1] = FS[s+1] + pn[3] * fx[k1] * fx[k2];
    end;
  end;

  /* 输出累积分布 */
  S_values = (0:max_s)`;
  print S_values FS;
  print "累积损失等于0的概率:" FS[1];
quit;


/****************************************************************************/
/* 2.2 复合分布的随机模拟                                                   */
/****************************************************************************/

/* 2.2.1 复合泊松分布的随机模拟 */
data compound_poisson;
  call streaminit(123);
  do sim = 1 to 10000;
    N = rand('POISSON', 2.5);
    S = 0;
    do j = 1 to N;
      S = S + rand('GAMMA', 2, 500);  /* shape=2, scale=500 */
    end;
    output;
  end;
run;

title "复合泊松分布模拟";
proc means data=compound_poisson mean var std;
  var S;
run;

proc sgplot data=compound_poisson;
  histogram S / nbins=30 scale=count;
  density S / type=kernel;
  yaxis label="频率";
run;
title;


/* 2.2.2 复合二项分布的随机模拟 */
data compound_binomial;
  call streaminit(123);
  do sim = 1 to 10000;
    N = rand('BINOMIAL', 100, 0.01);
    S = 0;
    do j = 1 to N;
      S = S + rand('GAMMA', 10, 5);  /* shape=10, scale=5 => rate=0.2 */
    end;
    output;
  end;
run;

title "复合二项分布模拟";
proc means data=compound_binomial mean var std;
  var S;
run;
title;


/* 2.2.3 复合负二项分布的随机模拟 */
data compound_negbinom;
  call streaminit(123);
  do sim = 1 to 10000;
    N = rand('NEGBINOMIAL', 0.5, 2.5);  /* p=0.5, r=2.5 */
    S = 0;
    do j = 1 to N;
      S = S + rand('GAMMA', 2, 500);
    end;
    output;
  end;
run;

title "复合负二项分布模拟";
proc means data=compound_negbinom mean var std;
  var S;
run;
title;


/****************************************************************************/
/* 2.3 短期个体风险模型                                                     */
/****************************************************************************/
data individual_risk;
  call streaminit(123);
  do sim = 1 to 10000;
    S = 0;
    /* 50个低风险个体（索赔概率0.1） */
    do i = 1 to 50;
      Ni = rand('BINOMIAL', 1, 0.1);
      Xi = rand('GAMMA', 10, 50);  /* shape=10, scale=50 => rate=0.02 */
      S = S + Ni * Xi;
    end;
    /* 50个高风险个体（索赔概率0.2） */
    do i = 1 to 50;
      Ni = rand('BINOMIAL', 1, 0.2);
      Xi = rand('LOGNORMAL', 5, 1);
      S = S + Ni * Xi;
    end;
    output;
  end;
run;

title "个体风险模型模拟";
proc means data=individual_risk mean var std;
  var S;
run;

proc sgplot data=individual_risk;
  histogram S / nbins=30 scale=count;
  density S / type=kernel;
run;
title;


/****************************************************************************/
/* 2.4 聚合风险模型计算方法                                                 */
/****************************************************************************/

/* 2.4.1 Panjer递推 */
proc iml;
  /* Panjer递推（泊松分布） */
  start Panjer_Poisson(p, lambda);
    if sum(p) > 1 | any(p < 0) then
      print "Error: p is not a density";
    cumul = exp(-lambda * sum(p));
    f = cumul;
    s = 0;
    do until(cumul > 0.99999999);
      s = s + 1;
      m = min(s, nrow(p));
      last = lambda / s * sum((1:m) # p[1:m] # f[(s+1-m):s]);
      f = f // last;
      cumul = cumul + last;
    end;
    return(f);
  finish;

  p = {0.25, 0.5, 0.25};
  lambda = 4;
  f = Panjer_Poisson(p, lambda);
  f_scaled = f * exp(lambda);
  print "Panjer递推结果" f_scaled;
quit;


/* 2.4.2 FFT法（快速傅里叶变换） */
proc iml;
  x = {0, 0.5, 0.4, 0.1} // j(40, 1, 0);
  phi_x = fft(x);
  phi_s = exp(3 * (phi_x - 1));
  fs = fft(phil_s) / nrow(phil_s);
  fs = real(fs);
  Fs = cusum(fs);

  s_vals = (0:43)`;
  print "FFT法计算结果" s_vals fs;
quit;


/****************************************************************************/
/* 2.5 随机模拟求累积损失的分布（例2.7）                                    */
/****************************************************************************/
data claim_sim;
  call streaminit(321);
  d = 250; u = 1000;       /* 免赔额和限额 */
  r = 3; beta = 2;         /* 负二项分布参数 */
  alpha = 100; theta = 0.2; /* 伽马分布参数 */
  do iter = 1 to 10000;
    N = rand('NEGBINOMIAL', 1/(1+beta), r);
    S = 0; w_total = 0;
    do j = 1 to N;
      x = rand('GAMMA', alpha, 1/theta);
      w = min(x, d);
      w_total = w_total + w;
      S = S + x;
    end;
    v = min(w_total, u);
    P = S - v;  /* 保险人的年度累积赔款 */
    output;
  end;
run;

title "例2.7：保险人累积赔款模拟";
proc means data=claim_sim mean std p95;
  var P;
run;

proc sgplot data=claim_sim;
  histogram P / nbins=50 scale=count;
  density P / type=kernel;
run;
title;


/****************************************************************************/
/* 2.6 近似计算方法                                                         */
/****************************************************************************/

/* 2.6.1 Tweedie分布模拟 */
data tweedie_sim;
  call streaminit(11);
  lambda = 1; alpha = 10; beta_rate = 2;
  do i = 1 to 10000;
    N = rand('POISSON', lambda);
    Y = 0;
    do j = 1 to N;
      Y = Y + rand('GAMMA', alpha, 1/beta_rate);
    end;
    output;
  end;
run;

title "Tweedie分布模拟";
proc sgplot data=tweedie_sim;
  histogram Y / nbins=50;
run;
title;


/* 2.6.2 近似计算比较 */
proc iml;
  call streaminit(123);
  n_sim = 10000;
  S = j(n_sim, 1, 0);
  do i = 1 to n_sim;
    N = rand('BINOMIAL', 1000, 0.001);
    S[i] = N;  /* 每次索赔金额为1 */
  end;

  ES = mean(S);
  varS = var(S);

  /* 1. 精确分布：二项分布和泊松分布 */
  p0 = 1 - cdf('BINOMIAL', 3.5, 0.001, 1000);
  p1 = 1 - cdf('POISSON', 3.5, 1);

  /* 2. 正态近似 */
  p2 = 1 - cdf('NORMAL', 3.5, ES, sqrt(varS));

  /* 3. 平移伽马近似 */
  /* S+1 ~ Gamma(4, 2) */
  p4 = 1 - cdf('GAMMA', 4.5, 4, 0.5);

  /* 4. NP近似 */
  mu_np = 1; sigma_np = 1; gamma_np = 1;
  a = -3/gamma_np + sqrt(9/gamma_np**2 + 1 + 6/gamma_np * (3.5 - mu_np) / sigma_np);
  p5 = 1 - cdf('NORMAL', a);

  print "近似计算比较 P(S>3.5)";
  print "二项分布(精确):" p0;
  print "泊松分布(精确):" p1;
  print "正态近似:" p2;
  print "平移伽马近似:" p4;
  print "NP近似:" p5;
quit;