/* Chap7 SAS代码 */
/* 自动从chap7.html同步生成 */

/****************************************************************************/
/* 第7章 广义线性模型                                                       */
/* 对应教材：section7.tex                                                   */
/* 内容：指数族分布、连接函数、参数估计、模型诊断、                          */
/*       泊松回归、负二项回归、伽马回归、GAM与GAMLSS                         */
/****************************************************************************/

/****************************************************************************/
/* 7.1 指数族分布                                                           */
/****************************************************************************/
proc iml;
  /* 指数族分布的统一形式 */
  /* f(y;θ,φ) = exp{(yθ - b(θ))/a(φ) + c(y,φ)} */

  /* 正态分布: b(θ)=θ²/2, a(φ)=φ */
  /* 泊松分布: b(θ)=exp(θ), a(φ)=1 */
  /* 二项分布: b(θ)=log(1+exp(θ)), a(φ)=1 */
  /* 伽马分布: b(θ)=-log(-θ), a(φ)=1/ν */

  print "指数族分布的b(θ)函数";
  theta = {-2, -1, 0, 1, 2};
  b_normal = theta##2 / 2;
  b_poisson = exp(theta);
  b_gamma = -log(-theta);
  print theta b_normal b_poisson b_gamma;
quit;


/****************************************************************************/
/* 7.2 泊松回归                                                             */
/****************************************************************************/

/* 7.2.1 生成模拟数据 */
data poisson_data;
  call streaminit(123);
  do i = 1 to 500;
    x1 = rand('NORMAL', 0, 1);
    x2 = rand('BERNOULLI', 0.5);
    /* 真实模型: log(λ) = 1 + 0.5*x1 + 0.3*x2 */
    eta = 1 + 0.5 * x1 + 0.3 * x2;
    lambda = exp(eta);
    y = rand('POISSON', lambda);
    output;
  end;
run;

title "泊松回归数据描述";
proc means data=poisson_data mean std min max;
  var y x1 x2;
run;
title;

/* 7.2.2 泊松回归模型 */
title "泊松回归";
proc genmod data=poisson_data;
  model y = x1 x2 / dist=poisson link=log;
  output out=poisson_out pred=pred resdev=resdev;
run;
title;

/* 7.2.3 模型拟合统计量 */
proc genmod data=poisson_data;
  model y = x1 x2 / dist=poisson link=log type1 type3;
run;


/****************************************************************************/
/* 7.3 负二项回归                                                           */
/****************************************************************************/

/* 7.3.1 生成过散布数据 */
data negbinom_data;
  call streaminit(456);
  do i = 1 to 500;
    x1 = rand('NORMAL', 0, 1);
    x2 = rand('BERNOULLI', 0.5);
    eta = 1 + 0.5 * x1 + 0.3 * x2;
    mu = exp(eta);
    /* 负二项分布: 过散布 */
    k = 2;  /* 散布参数 */
    p = k / (k + mu);
    y = rand('NEGBINOMIAL', p, k);
    output;
  end;
run;

title "负二项回归";
proc genmod data=negbinom_data;
  model y = x1 x2 / dist=negbin link=log;
run;
title;

/* 7.3.2 泊松与负二项回归比较 */
title "泊松回归（过散布数据）";
proc genmod data=negbinom_data;
  model y = x1 x2 / dist=poisson link=log scale=pearson;
run;
title;


/****************************************************************************/
/* 7.4 伽马回归                                                             */
/****************************************************************************/
data gamma_data;
  call streaminit(789);
  do i = 1 to 500;
    x1 = rand('NORMAL', 0, 1);
    x2 = rand('BERNOULLI', 0.5);
    eta = 3 + 0.5 * x1 - 0.3 * x2;
    mu = exp(eta);
    /* 伽马分布 */
    phi = 0.5;  /* 散布参数 */
    shape = 1 / phi;
    scale = mu * phi;
    y = rand('GAMMA', shape, scale);
    output;
  end;
run;

title "伽马回归";
proc genmod data=gamma_data;
  model y = x1 x2 / dist=gamma link=log;
  output out=gamma_out pred=pred resdev=resdev;
run;
title;


/****************************************************************************/
/* 7.5 Logistic回归（二项回归）                                             */
/****************************************************************************/
data logistic_data;
  call streaminit(101);
  do i = 1 to 500;
    x1 = rand('NORMAL', 0, 1);
    x2 = rand('BERNOULLI', 0.5);
    eta = -1 + 0.8 * x1 + 0.5 * x2;
    p = 1 / (1 + exp(-eta));
    y = rand('BERNOULLI', p);
    output;
  end;
run;

title "Logistic回归";
proc genmod data=logistic_data;
  model y = x1 x2 / dist=binomial link=logit;
run;
title;

/* 使用PROC LOGISTIC */
title "PROC LOGISTIC";
proc logistic data=logistic_data;
  model y(event='1') = x1 x2;
run;
title;


/****************************************************************************/
/* 7.6 模型诊断                                                             */
/****************************************************************************/

/* 7.6.1 残差分析 */
title "泊松回归残差分析";
proc genmod data=poisson_data;
  model y = x1 x2 / dist=poisson link=log;
  output out=diag_out pred=pred resraw=resraw resdev=resdev reschi=reschi;
run;
title;

/* 残差图 */
title "Deviance残差 vs 预测值";
proc sgplot data=diag_out;
  scatter x=pred y=resdev;
  refline 0 / axis=y;
run;
title;

title "Deviance残差Q-Q图";
proc univariate data=diag_out;
  var resdev;
  qqplot / normal(mu=est sigma=est);
run;
title;


/* 7.6.2 偏差分析 */
title "偏差分析";
proc genmod data=poisson_data;
  model y = x1 x2 / dist=poisson link=log type1 type3;
run;
title;


/****************************************************************************/
/* 7.7 偏差与模型比较                                                       */
/****************************************************************************/
proc iml;
  /* 饱和模型 vs 拟合模型 */
  /* 偏差 D = 2 * Σ [y_i * log(y_i/μ_i) - (y_i - μ_i)] */

  /* 示例数据 */
  y = {10, 20, 15, 25, 30};
  mu = {12, 18, 17, 23, 28};

  /* 泊松偏差 */
  D_poisson = 2 * sum(y # log(y / mu) - (y - mu));
  print "泊松偏差 D =" D_poisson;

  /* 自由度 */
  df = nrow(y) - 2;  /* 2个参数 */
  p_value = 1 - cdf('CHISQUARE', D_poisson, df);
  print "自由度 =" df "p值 =" p_value;
quit;


/****************************************************************************/
/* 7.8 保险定价应用                                                         */
/****************************************************************************/

/* 7.8.1 车险索赔次数模型 */
data auto_claims;
  call streaminit(202);
  do policy = 1 to 1000;
    age_group = rand('INTEGER', 1, 4);  /* 年龄组 */
    vehicle_age = rand('INTEGER', 1, 3);  /* 车龄 */
    gender = rand('BERNOULLI', 0.5);  /* 性别 */

    /* 真实模型 */
    eta = -1 + 0.3 * (age_group=1) - 0.2 * (age_group=4)
          + 0.4 * (vehicle_age=1) + 0.1 * gender;
    lambda = exp(eta);
    n_claims = rand('POISSON', lambda);
    output;
  end;
run;

title "车险索赔次数泊松回归";
proc genmod data=auto_claims;
  class age_group vehicle_age gender;
  model n_claims = age_group vehicle_age gender / dist=poisson link=log type3;
  estimate '年轻司机' age_group 1 0 0 0 / exp;
  estimate '老旧车辆' vehicle_age 1 0 0 / exp;
run;
title;


/* 7.8.2 车险索赔金额模型 */
data auto_severity;
  call streaminit(303);
  do policy = 1 to 500;
    age_group = rand('INTEGER', 1, 4);
    vehicle_age = rand('INTEGER', 1, 3);

    eta = 8 + 0.2 * (age_group=1) - 0.1 * (age_group=4)
          + 0.3 * (vehicle_age=1);
    mu = exp(eta);
    phi = 0.5;
    shape = 1 / phi;
    scale = mu * phi;
    claim_amount = rand('GAMMA', shape, scale);
    output;
  end;
run;

title "车险索赔金额伽马回归";
proc genmod data=auto_severity;
  class age_group vehicle_age;
  model claim_amount = age_group vehicle_age / dist=gamma link=log type3;
run;
title;


/****************************************************************************/
/* 7.9 GAM（广义可加模型）                                                  */
/****************************************************************************/
/* 使用PROC GAM */
data gam_data;
  call streaminit(404);
  do i = 1 to 500;
    x = rand('UNIFORM', 0, 10);
    eta = 1 + 0.5 * sin(x) + 0.3 * x;
    lambda = exp(eta);
    y = rand('POISSON', lambda);
    output;
  end;
run;

title "广义可加模型（GAM）";
proc gam data=gam_data;
  model y = spline(x, df=4) / dist=poisson;
  output out=gam_out pred=pred;
run;
title;

title "GAM拟合结果";
proc sgplot data=gam_out;
  scatter x=x y=y;
  series x=x y=pred;
run;
title;

/****************************************************************************/
/* 第7章 广义线性模型                                                       */
/* 对应教材：section7.tex                                                   */
/* 内容：指数族分布、连接函数、参数估计、模型诊断、                          */
/*       泊松回归、负二项回归、伽马回归、GAM与GAMLSS                         */
/****************************************************************************/

/****************************************************************************/
/* 7.1 指数族分布                                                           */
/****************************************************************************/
proc iml;
  /* 指数族分布的统一形式 */
  /* f(y;θ,φ) = exp{(yθ - b(θ))/a(φ) + c(y,φ)} */

  /* 正态分布: b(θ)=θ²/2, a(φ)=φ */
  /* 泊松分布: b(θ)=exp(θ), a(φ)=1 */
  /* 二项分布: b(θ)=log(1+exp(θ)), a(φ)=1 */
  /* 伽马分布: b(θ)=-log(-θ), a(φ)=1/ν */

  print "指数族分布的b(θ)函数";
  theta = {-2, -1, 0, 1, 2};
  b_normal = theta##2 / 2;
  b_poisson = exp(theta);
  b_gamma = -log(-theta);
  print theta b_normal b_poisson b_gamma;
quit;


/****************************************************************************/
/* 7.2 泊松回归                                                             */
/****************************************************************************/

/* 7.2.1 生成模拟数据 */
data poisson_data;
  call streaminit(123);
  do i = 1 to 500;
    x1 = rand('NORMAL', 0, 1);
    x2 = rand('BERNOULLI', 0.5);
    /* 真实模型: log(λ) = 1 + 0.5*x1 + 0.3*x2 */
    eta = 1 + 0.5 * x1 + 0.3 * x2;
    lambda = exp(eta);
    y = rand('POISSON', lambda);
    output;
  end;
run;

title "泊松回归数据描述";
proc means data=poisson_data mean std min max;
  var y x1 x2;
run;
title;

/* 7.2.2 泊松回归模型 */
title "泊松回归";
proc genmod data=poisson_data;
  model y = x1 x2 / dist=poisson link=log;
  output out=poisson_out pred=pred resdev=resdev;
run;
title;

/* 7.2.3 模型拟合统计量 */
proc genmod data=poisson_data;
  model y = x1 x2 / dist=poisson link=log type1 type3;
run;


/****************************************************************************/
/* 7.3 负二项回归                                                           */
/****************************************************************************/

/* 7.3.1 生成过散布数据 */
data negbinom_data;
  call streaminit(456);
  do i = 1 to 500;
    x1 = rand('NORMAL', 0, 1);
    x2 = rand('BERNOULLI', 0.5);
    eta = 1 + 0.5 * x1 + 0.3 * x2;
    mu = exp(eta);
    /* 负二项分布: 过散布 */
    k = 2;  /* 散布参数 */
    p = k / (k + mu);
    y = rand('NEGBINOMIAL', p, k);
    output;
  end;
run;

title "负二项回归";
proc genmod data=negbinom_data;
  model y = x1 x2 / dist=negbin link=log;
run;
title;

/* 7.3.2 泊松与负二项回归比较 */
title "泊松回归（过散布数据）";
proc genmod data=negbinom_data;
  model y = x1 x2 / dist=poisson link=log scale=pearson;
run;
title;


/****************************************************************************/
/* 7.4 伽马回归                                                             */
/****************************************************************************/
data gamma_data;
  call streaminit(789);
  do i = 1 to 500;
    x1 = rand('NORMAL', 0, 1);
    x2 = rand('BERNOULLI', 0.5);
    eta = 3 + 0.5 * x1 - 0.3 * x2;
    mu = exp(eta);
    /* 伽马分布 */
    phi = 0.5;  /* 散布参数 */
    shape = 1 / phi;
    scale = mu * phi;
    y = rand('GAMMA', shape, scale);
    output;
  end;
run;

title "伽马回归";
proc genmod data=gamma_data;
  model y = x1 x2 / dist=gamma link=log;
  output out=gamma_out pred=pred resdev=resdev;
run;
title;


/****************************************************************************/
/* 7.5 Logistic回归（二项回归）                                             */
/****************************************************************************/
data logistic_data;
  call streaminit(101);
  do i = 1 to 500;
    x1 = rand('NORMAL', 0, 1);
    x2 = rand('BERNOULLI', 0.5);
    eta = -1 + 0.8 * x1 + 0.5 * x2;
    p = 1 / (1 + exp(-eta));
    y = rand('BERNOULLI', p);
    output;
  end;
run;

title "Logistic回归";
proc genmod data=logistic_data;
  model y = x1 x2 / dist=binomial link=logit;
run;
title;

/* 使用PROC LOGISTIC */
title "PROC LOGISTIC";
proc logistic data=logistic_data;
  model y(event='1') = x1 x2;
run;
title;


/****************************************************************************/
/* 7.6 模型诊断                                                             */
/****************************************************************************/

/* 7.6.1 残差分析 */
title "泊松回归残差分析";
proc genmod data=poisson_data;
  model y = x1 x2 / dist=poisson link=log;
  output out=diag_out pred=pred resraw=resraw resdev=resdev reschi=reschi;
run;
title;

/* 残差图 */
title "Deviance残差 vs 预测值";
proc sgplot data=diag_out;
  scatter x=pred y=resdev;
  refline 0 / axis=y;
run;
title;

title "Deviance残差Q-Q图";
proc univariate data=diag_out;
  var resdev;
  qqplot / normal(mu=est sigma=est);
run;
title;


/* 7.6.2 偏差分析 */
title "偏差分析";
proc genmod data=poisson_data;
  model y = x1 x2 / dist=poisson link=log type1 type3;
run;
title;


/****************************************************************************/
/* 7.7 偏差与模型比较                                                       */
/****************************************************************************/
proc iml;
  /* 饱和模型 vs 拟合模型 */
  /* 偏差 D = 2 * Σ [y_i * log(y_i/μ_i) - (y_i - μ_i)] */

  /* 示例数据 */
  y = {10, 20, 15, 25, 30};
  mu = {12, 18, 17, 23, 28};

  /* 泊松偏差 */
  D_poisson = 2 * sum(y # log(y / mu) - (y - mu));
  print "泊松偏差 D =" D_poisson;

  /* 自由度 */
  df = nrow(y) - 2;  /* 2个参数 */
  p_value = 1 - cdf('CHISQUARE', D_poisson, df);
  print "自由度 =" df "p值 =" p_value;
quit;


/****************************************************************************/
/* 7.8 保险定价应用                                                         */
/****************************************************************************/

/* 7.8.1 车险索赔次数模型 */
data auto_claims;
  call streaminit(202);
  do policy = 1 to 1000;
    age_group = rand('INTEGER', 1, 4);  /* 年龄组 */
    vehicle_age = rand('INTEGER', 1, 3);  /* 车龄 */
    gender = rand('BERNOULLI', 0.5);  /* 性别 */

    /* 真实模型 */
    eta = -1 + 0.3 * (age_group=1) - 0.2 * (age_group=4)
          + 0.4 * (vehicle_age=1) + 0.1 * gender;
    lambda = exp(eta);
    n_claims = rand('POISSON', lambda);
    output;
  end;
run;

title "车险索赔次数泊松回归";
proc genmod data=auto_claims;
  class age_group vehicle_age gender;
  model n_claims = age_group vehicle_age gender / dist=poisson link=log type3;
  estimate '年轻司机' age_group 1 0 0 0 / exp;
  estimate '老旧车辆' vehicle_age 1 0 0 / exp;
run;
title;


/* 7.8.2 车险索赔金额模型 */
data auto_severity;
  call streaminit(303);
  do policy = 1 to 500;
    age_group = rand('INTEGER', 1, 4);
    vehicle_age = rand('INTEGER', 1, 3);

    eta = 8 + 0.2 * (age_group=1) - 0.1 * (age_group=4)
          + 0.3 * (vehicle_age=1);
    mu = exp(eta);
    phi = 0.5;
    shape = 1 / phi;
    scale = mu * phi;
    claim_amount = rand('GAMMA', shape, scale);
    output;
  end;
run;

title "车险索赔金额伽马回归";
proc genmod data=auto_severity;
  class age_group vehicle_age;
  model claim_amount = age_group vehicle_age / dist=gamma link=log type3;
run;
title;


/****************************************************************************/
/* 7.9 GAM（广义可加模型）                                                  */
/****************************************************************************/
/* 使用PROC GAM */
data gam_data;
  call streaminit(404);
  do i = 1 to 500;
    x = rand('UNIFORM', 0, 10);
    eta = 1 + 0.5 * sin(x) + 0.3 * x;
    lambda = exp(eta);
    y = rand('POISSON', lambda);
    output;
  end;
run;

title "广义可加模型（GAM）";
proc gam data=gam_data;
  model y = spline(x, df=4) / dist=poisson;
  output out=gam_out pred=pred;
run;
title;

title "GAM拟合结果";
proc sgplot data=gam_out;
  scatter x=x y=y;
  series x=x y=pred;
run;
title;

/****************************************************************************/
/* 第7章 广义线性模型                                                       */
/* 对应教材：section7.tex                                                   */
/* 内容：指数族分布、连接函数、参数估计、模型诊断、                          */
/*       泊松回归、负二项回归、伽马回归、GAM与GAMLSS                         */
/****************************************************************************/

/****************************************************************************/
/* 7.1 指数族分布                                                           */
/****************************************************************************/
proc iml;
  /* 指数族分布的统一形式 */
  /* f(y;θ,φ) = exp{(yθ - b(θ))/a(φ) + c(y,φ)} */

  /* 正态分布: b(θ)=θ²/2, a(φ)=φ */
  /* 泊松分布: b(θ)=exp(θ), a(φ)=1 */
  /* 二项分布: b(θ)=log(1+exp(θ)), a(φ)=1 */
  /* 伽马分布: b(θ)=-log(-θ), a(φ)=1/ν */

  print "指数族分布的b(θ)函数";
  theta = {-2, -1, 0, 1, 2};
  b_normal = theta##2 / 2;
  b_poisson = exp(theta);
  b_gamma = -log(-theta);
  print theta b_normal b_poisson b_gamma;
quit;


/****************************************************************************/
/* 7.2 泊松回归                                                             */
/****************************************************************************/

/* 7.2.1 生成模拟数据 */
data poisson_data;
  call streaminit(123);
  do i = 1 to 500;
    x1 = rand('NORMAL', 0, 1);
    x2 = rand('BERNOULLI', 0.5);
    /* 真实模型: log(λ) = 1 + 0.5*x1 + 0.3*x2 */
    eta = 1 + 0.5 * x1 + 0.3 * x2;
    lambda = exp(eta);
    y = rand('POISSON', lambda);
    output;
  end;
run;

title "泊松回归数据描述";
proc means data=poisson_data mean std min max;
  var y x1 x2;
run;
title;

/* 7.2.2 泊松回归模型 */
title "泊松回归";
proc genmod data=poisson_data;
  model y = x1 x2 / dist=poisson link=log;
  output out=poisson_out pred=pred resdev=resdev;
run;
title;

/* 7.2.3 模型拟合统计量 */
proc genmod data=poisson_data;
  model y = x1 x2 / dist=poisson link=log type1 type3;
run;


/****************************************************************************/
/* 7.3 负二项回归                                                           */
/****************************************************************************/

/* 7.3.1 生成过散布数据 */
data negbinom_data;
  call streaminit(456);
  do i = 1 to 500;
    x1 = rand('NORMAL', 0, 1);
    x2 = rand('BERNOULLI', 0.5);
    eta = 1 + 0.5 * x1 + 0.3 * x2;
    mu = exp(eta);
    /* 负二项分布: 过散布 */
    k = 2;  /* 散布参数 */
    p = k / (k + mu);
    y = rand('NEGBINOMIAL', p, k);
    output;
  end;
run;

title "负二项回归";
proc genmod data=negbinom_data;
  model y = x1 x2 / dist=negbin link=log;
run;
title;

/* 7.3.2 泊松与负二项回归比较 */
title "泊松回归（过散布数据）";
proc genmod data=negbinom_data;
  model y = x1 x2 / dist=poisson link=log scale=pearson;
run;
title;


/****************************************************************************/
/* 7.4 伽马回归                                                             */
/****************************************************************************/
data gamma_data;
  call streaminit(789);
  do i = 1 to 500;
    x1 = rand('NORMAL', 0, 1);
    x2 = rand('BERNOULLI', 0.5);
    eta = 3 + 0.5 * x1 - 0.3 * x2;
    mu = exp(eta);
    /* 伽马分布 */
    phi = 0.5;  /* 散布参数 */
    shape = 1 / phi;
    scale = mu * phi;
    y = rand('GAMMA', shape, scale);
    output;
  end;
run;

title "伽马回归";
proc genmod data=gamma_data;
  model y = x1 x2 / dist=gamma link=log;
  output out=gamma_out pred=pred resdev=resdev;
run;
title;


/****************************************************************************/
/* 7.5 Logistic回归（二项回归）                                             */
/****************************************************************************/
data logistic_data;
  call streaminit(101);
  do i = 1 to 500;
    x1 = rand('NORMAL', 0, 1);
    x2 = rand('BERNOULLI', 0.5);
    eta = -1 + 0.8 * x1 + 0.5 * x2;
    p = 1 / (1 + exp(-eta));
    y = rand('BERNOULLI', p);
    output;
  end;
run;

title "Logistic回归";
proc genmod data=logistic_data;
  model y = x1 x2 / dist=binomial link=logit;
run;
title;

/* 使用PROC LOGISTIC */
title "PROC LOGISTIC";
proc logistic data=logistic_data;
  model y(event='1') = x1 x2;
run;
title;


/****************************************************************************/
/* 7.6 模型诊断                                                             */
/****************************************************************************/

/* 7.6.1 残差分析 */
title "泊松回归残差分析";
proc genmod data=poisson_data;
  model y = x1 x2 / dist=poisson link=log;
  output out=diag_out pred=pred resraw=resraw resdev=resdev reschi=reschi;
run;
title;

/* 残差图 */
title "Deviance残差 vs 预测值";
proc sgplot data=diag_out;
  scatter x=pred y=resdev;
  refline 0 / axis=y;
run;
title;

title "Deviance残差Q-Q图";
proc univariate data=diag_out;
  var resdev;
  qqplot / normal(mu=est sigma=est);
run;
title;


/* 7.6.2 偏差分析 */
title "偏差分析";
proc genmod data=poisson_data;
  model y = x1 x2 / dist=poisson link=log type1 type3;
run;
title;


/****************************************************************************/
/* 7.7 偏差与模型比较                                                       */
/****************************************************************************/
proc iml;
  /* 饱和模型 vs 拟合模型 */
  /* 偏差 D = 2 * Σ [y_i * log(y_i/μ_i) - (y_i - μ_i)] */

  /* 示例数据 */
  y = {10, 20, 15, 25, 30};
  mu = {12, 18, 17, 23, 28};

  /* 泊松偏差 */
  D_poisson = 2 * sum(y # log(y / mu) - (y - mu));
  print "泊松偏差 D =" D_poisson;

  /* 自由度 */
  df = nrow(y) - 2;  /* 2个参数 */
  p_value = 1 - cdf('CHISQUARE', D_poisson, df);
  print "自由度 =" df "p值 =" p_value;
quit;


/****************************************************************************/
/* 7.8 保险定价应用                                                         */
/****************************************************************************/

/* 7.8.1 车险索赔次数模型 */
data auto_claims;
  call streaminit(202);
  do policy = 1 to 1000;
    age_group = rand('INTEGER', 1, 4);  /* 年龄组 */
    vehicle_age = rand('INTEGER', 1, 3);  /* 车龄 */
    gender = rand('BERNOULLI', 0.5);  /* 性别 */

    /* 真实模型 */
    eta = -1 + 0.3 * (age_group=1) - 0.2 * (age_group=4)
          + 0.4 * (vehicle_age=1) + 0.1 * gender;
    lambda = exp(eta);
    n_claims = rand('POISSON', lambda);
    output;
  end;
run;

title "车险索赔次数泊松回归";
proc genmod data=auto_claims;
  class age_group vehicle_age gender;
  model n_claims = age_group vehicle_age gender / dist=poisson link=log type3;
  estimate '年轻司机' age_group 1 0 0 0 / exp;
  estimate '老旧车辆' vehicle_age 1 0 0 / exp;
run;
title;


/* 7.8.2 车险索赔金额模型 */
data auto_severity;
  call streaminit(303);
  do policy = 1 to 500;
    age_group = rand('INTEGER', 1, 4);
    vehicle_age = rand('INTEGER', 1, 3);

    eta = 8 + 0.2 * (age_group=1) - 0.1 * (age_group=4)
          + 0.3 * (vehicle_age=1);
    mu = exp(eta);
    phi = 0.5;
    shape = 1 / phi;
    scale = mu * phi;
    claim_amount = rand('GAMMA', shape, scale);
    output;
  end;
run;

title "车险索赔金额伽马回归";
proc genmod data=auto_severity;
  class age_group vehicle_age;
  model claim_amount = age_group vehicle_age / dist=gamma link=log type3;
run;
title;


/****************************************************************************/
/* 7.9 GAM（广义可加模型）                                                  */
/****************************************************************************/
/* 使用PROC GAM */
data gam_data;
  call streaminit(404);
  do i = 1 to 500;
    x = rand('UNIFORM', 0, 10);
    eta = 1 + 0.5 * sin(x) + 0.3 * x;
    lambda = exp(eta);
    y = rand('POISSON', lambda);
    output;
  end;
run;

title "广义可加模型（GAM）";
proc gam data=gam_data;
  model y = spline(x, df=4) / dist=poisson;
  output out=gam_out pred=pred;
run;
title;

title "GAM拟合结果";
proc sgplot data=gam_out;
  scatter x=x y=y;
  series x=x y=pred;
run;
title;

/****************************************************************************/
/* 第7章 广义线性模型                                                       */
/* 对应教材：section7.tex                                                   */
/* 内容：指数族分布、连接函数、参数估计、模型诊断、                          */
/*       泊松回归、负二项回归、伽马回归、GAM与GAMLSS                         */
/****************************************************************************/

/****************************************************************************/
/* 7.1 指数族分布                                                           */
/****************************************************************************/
proc iml;
  /* 指数族分布的统一形式 */
  /* f(y;θ,φ) = exp{(yθ - b(θ))/a(φ) + c(y,φ)} */

  /* 正态分布: b(θ)=θ²/2, a(φ)=φ */
  /* 泊松分布: b(θ)=exp(θ), a(φ)=1 */
  /* 二项分布: b(θ)=log(1+exp(θ)), a(φ)=1 */
  /* 伽马分布: b(θ)=-log(-θ), a(φ)=1/ν */

  print "指数族分布的b(θ)函数";
  theta = {-2, -1, 0, 1, 2};
  b_normal = theta##2 / 2;
  b_poisson = exp(theta);
  b_gamma = -log(-theta);
  print theta b_normal b_poisson b_gamma;
quit;


/****************************************************************************/
/* 7.2 泊松回归                                                             */
/****************************************************************************/

/* 7.2.1 生成模拟数据 */
data poisson_data;
  call streaminit(123);
  do i = 1 to 500;
    x1 = rand('NORMAL', 0, 1);
    x2 = rand('BERNOULLI', 0.5);
    /* 真实模型: log(λ) = 1 + 0.5*x1 + 0.3*x2 */
    eta = 1 + 0.5 * x1 + 0.3 * x2;
    lambda = exp(eta);
    y = rand('POISSON', lambda);
    output;
  end;
run;

title "泊松回归数据描述";
proc means data=poisson_data mean std min max;
  var y x1 x2;
run;
title;

/* 7.2.2 泊松回归模型 */
title "泊松回归";
proc genmod data=poisson_data;
  model y = x1 x2 / dist=poisson link=log;
  output out=poisson_out pred=pred resdev=resdev;
run;
title;

/* 7.2.3 模型拟合统计量 */
proc genmod data=poisson_data;
  model y = x1 x2 / dist=poisson link=log type1 type3;
run;


/****************************************************************************/
/* 7.3 负二项回归                                                           */
/****************************************************************************/

/* 7.3.1 生成过散布数据 */
data negbinom_data;
  call streaminit(456);
  do i = 1 to 500;
    x1 = rand('NORMAL', 0, 1);
    x2 = rand('BERNOULLI', 0.5);
    eta = 1 + 0.5 * x1 + 0.3 * x2;
    mu = exp(eta);
    /* 负二项分布: 过散布 */
    k = 2;  /* 散布参数 */
    p = k / (k + mu);
    y = rand('NEGBINOMIAL', p, k);
    output;
  end;
run;

title "负二项回归";
proc genmod data=negbinom_data;
  model y = x1 x2 / dist=negbin link=log;
run;
title;

/* 7.3.2 泊松与负二项回归比较 */
title "泊松回归（过散布数据）";
proc genmod data=negbinom_data;
  model y = x1 x2 / dist=poisson link=log scale=pearson;
run;
title;


/****************************************************************************/
/* 7.4 伽马回归                                                             */
/****************************************************************************/
data gamma_data;
  call streaminit(789);
  do i = 1 to 500;
    x1 = rand('NORMAL', 0, 1);
    x2 = rand('BERNOULLI', 0.5);
    eta = 3 + 0.5 * x1 - 0.3 * x2;
    mu = exp(eta);
    /* 伽马分布 */
    phi = 0.5;  /* 散布参数 */
    shape = 1 / phi;
    scale = mu * phi;
    y = rand('GAMMA', shape, scale);
    output;
  end;
run;

title "伽马回归";
proc genmod data=gamma_data;
  model y = x1 x2 / dist=gamma link=log;
  output out=gamma_out pred=pred resdev=resdev;
run;
title;


/****************************************************************************/
/* 7.5 Logistic回归（二项回归）                                             */
/****************************************************************************/
data logistic_data;
  call streaminit(101);
  do i = 1 to 500;
    x1 = rand('NORMAL', 0, 1);
    x2 = rand('BERNOULLI', 0.5);
    eta = -1 + 0.8 * x1 + 0.5 * x2;
    p = 1 / (1 + exp(-eta));
    y = rand('BERNOULLI', p);
    output;
  end;
run;

title "Logistic回归";
proc genmod data=logistic_data;
  model y = x1 x2 / dist=binomial link=logit;
run;
title;

/* 使用PROC LOGISTIC */
title "PROC LOGISTIC";
proc logistic data=logistic_data;
  model y(event='1') = x1 x2;
run;
title;


/****************************************************************************/
/* 7.6 模型诊断                                                             */
/****************************************************************************/

/* 7.6.1 残差分析 */
title "泊松回归残差分析";
proc genmod data=poisson_data;
  model y = x1 x2 / dist=poisson link=log;
  output out=diag_out pred=pred resraw=resraw resdev=resdev reschi=reschi;
run;
title;

/* 残差图 */
title "Deviance残差 vs 预测值";
proc sgplot data=diag_out;
  scatter x=pred y=resdev;
  refline 0 / axis=y;
run;
title;

title "Deviance残差Q-Q图";
proc univariate data=diag_out;
  var resdev;
  qqplot / normal(mu=est sigma=est);
run;
title;


/* 7.6.2 偏差分析 */
title "偏差分析";
proc genmod data=poisson_data;
  model y = x1 x2 / dist=poisson link=log type1 type3;
run;
title;


/****************************************************************************/
/* 7.7 偏差与模型比较                                                       */
/****************************************************************************/
proc iml;
  /* 饱和模型 vs 拟合模型 */
  /* 偏差 D = 2 * Σ [y_i * log(y_i/μ_i) - (y_i - μ_i)] */

  /* 示例数据 */
  y = {10, 20, 15, 25, 30};
  mu = {12, 18, 17, 23, 28};

  /* 泊松偏差 */
  D_poisson = 2 * sum(y # log(y / mu) - (y - mu));
  print "泊松偏差 D =" D_poisson;

  /* 自由度 */
  df = nrow(y) - 2;  /* 2个参数 */
  p_value = 1 - cdf('CHISQUARE', D_poisson, df);
  print "自由度 =" df "p值 =" p_value;
quit;


/****************************************************************************/
/* 7.8 保险定价应用                                                         */
/****************************************************************************/

/* 7.8.1 车险索赔次数模型 */
data auto_claims;
  call streaminit(202);
  do policy = 1 to 1000;
    age_group = rand('INTEGER', 1, 4);  /* 年龄组 */
    vehicle_age = rand('INTEGER', 1, 3);  /* 车龄 */
    gender = rand('BERNOULLI', 0.5);  /* 性别 */

    /* 真实模型 */
    eta = -1 + 0.3 * (age_group=1) - 0.2 * (age_group=4)
          + 0.4 * (vehicle_age=1) + 0.1 * gender;
    lambda = exp(eta);
    n_claims = rand('POISSON', lambda);
    output;
  end;
run;

title "车险索赔次数泊松回归";
proc genmod data=auto_claims;
  class age_group vehicle_age gender;
  model n_claims = age_group vehicle_age gender / dist=poisson link=log type3;
  estimate '年轻司机' age_group 1 0 0 0 / exp;
  estimate '老旧车辆' vehicle_age 1 0 0 / exp;
run;
title;


/* 7.8.2 车险索赔金额模型 */
data auto_severity;
  call streaminit(303);
  do policy = 1 to 500;
    age_group = rand('INTEGER', 1, 4);
    vehicle_age = rand('INTEGER', 1, 3);

    eta = 8 + 0.2 * (age_group=1) - 0.1 * (age_group=4)
          + 0.3 * (vehicle_age=1);
    mu = exp(eta);
    phi = 0.5;
    shape = 1 / phi;
    scale = mu * phi;
    claim_amount = rand('GAMMA', shape, scale);
    output;
  end;
run;

title "车险索赔金额伽马回归";
proc genmod data=auto_severity;
  class age_group vehicle_age;
  model claim_amount = age_group vehicle_age / dist=gamma link=log type3;
run;
title;


/****************************************************************************/
/* 7.9 GAM（广义可加模型）                                                  */
/****************************************************************************/
/* 使用PROC GAM */
data gam_data;
  call streaminit(404);
  do i = 1 to 500;
    x = rand('UNIFORM', 0, 10);
    eta = 1 + 0.5 * sin(x) + 0.3 * x;
    lambda = exp(eta);
    y = rand('POISSON', lambda);
    output;
  end;
run;

title "广义可加模型（GAM）";
proc gam data=gam_data;
  model y = spline(x, df=4) / dist=poisson;
  output out=gam_out pred=pred;
run;
title;

title "GAM拟合结果";
proc sgplot data=gam_out;
  scatter x=x y=y;
  series x=x y=pred;
run;
title;

/****************************************************************************/
/* 第7章 广义线性模型                                                       */
/* 对应教材：section7.tex                                                   */
/* 内容：指数族分布、连接函数、参数估计、模型诊断、                          */
/*       泊松回归、负二项回归、伽马回归、GAM与GAMLSS                         */
/****************************************************************************/

/****************************************************************************/
/* 7.1 指数族分布                                                           */
/****************************************************************************/
proc iml;
  /* 指数族分布的统一形式 */
  /* f(y;θ,φ) = exp{(yθ - b(θ))/a(φ) + c(y,φ)} */

  /* 正态分布: b(θ)=θ²/2, a(φ)=φ */
  /* 泊松分布: b(θ)=exp(θ), a(φ)=1 */
  /* 二项分布: b(θ)=log(1+exp(θ)), a(φ)=1 */
  /* 伽马分布: b(θ)=-log(-θ), a(φ)=1/ν */

  print "指数族分布的b(θ)函数";
  theta = {-2, -1, 0, 1, 2};
  b_normal = theta##2 / 2;
  b_poisson = exp(theta);
  b_gamma = -log(-theta);
  print theta b_normal b_poisson b_gamma;
quit;


/****************************************************************************/
/* 7.2 泊松回归                                                             */
/****************************************************************************/

/* 7.2.1 生成模拟数据 */
data poisson_data;
  call streaminit(123);
  do i = 1 to 500;
    x1 = rand('NORMAL', 0, 1);
    x2 = rand('BERNOULLI', 0.5);
    /* 真实模型: log(λ) = 1 + 0.5*x1 + 0.3*x2 */
    eta = 1 + 0.5 * x1 + 0.3 * x2;
    lambda = exp(eta);
    y = rand('POISSON', lambda);
    output;
  end;
run;

title "泊松回归数据描述";
proc means data=poisson_data mean std min max;
  var y x1 x2;
run;
title;

/* 7.2.2 泊松回归模型 */
title "泊松回归";
proc genmod data=poisson_data;
  model y = x1 x2 / dist=poisson link=log;
  output out=poisson_out pred=pred resdev=resdev;
run;
title;

/* 7.2.3 模型拟合统计量 */
proc genmod data=poisson_data;
  model y = x1 x2 / dist=poisson link=log type1 type3;
run;


/****************************************************************************/
/* 7.3 负二项回归                                                           */
/****************************************************************************/

/* 7.3.1 生成过散布数据 */
data negbinom_data;
  call streaminit(456);
  do i = 1 to 500;
    x1 = rand('NORMAL', 0, 1);
    x2 = rand('BERNOULLI', 0.5);
    eta = 1 + 0.5 * x1 + 0.3 * x2;
    mu = exp(eta);
    /* 负二项分布: 过散布 */
    k = 2;  /* 散布参数 */
    p = k / (k + mu);
    y = rand('NEGBINOMIAL', p, k);
    output;
  end;
run;

title "负二项回归";
proc genmod data=negbinom_data;
  model y = x1 x2 / dist=negbin link=log;
run;
title;

/* 7.3.2 泊松与负二项回归比较 */
title "泊松回归（过散布数据）";
proc genmod data=negbinom_data;
  model y = x1 x2 / dist=poisson link=log scale=pearson;
run;
title;


/****************************************************************************/
/* 7.4 伽马回归                                                             */
/****************************************************************************/
data gamma_data;
  call streaminit(789);
  do i = 1 to 500;
    x1 = rand('NORMAL', 0, 1);
    x2 = rand('BERNOULLI', 0.5);
    eta = 3 + 0.5 * x1 - 0.3 * x2;
    mu = exp(eta);
    /* 伽马分布 */
    phi = 0.5;  /* 散布参数 */
    shape = 1 / phi;
    scale = mu * phi;
    y = rand('GAMMA', shape, scale);
    output;
  end;
run;

title "伽马回归";
proc genmod data=gamma_data;
  model y = x1 x2 / dist=gamma link=log;
  output out=gamma_out pred=pred resdev=resdev;
run;
title;


/****************************************************************************/
/* 7.5 Logistic回归（二项回归）                                             */
/****************************************************************************/
data logistic_data;
  call streaminit(101);
  do i = 1 to 500;
    x1 = rand('NORMAL', 0, 1);
    x2 = rand('BERNOULLI', 0.5);
    eta = -1 + 0.8 * x1 + 0.5 * x2;
    p = 1 / (1 + exp(-eta));
    y = rand('BERNOULLI', p);
    output;
  end;
run;

title "Logistic回归";
proc genmod data=logistic_data;
  model y = x1 x2 / dist=binomial link=logit;
run;
title;

/* 使用PROC LOGISTIC */
title "PROC LOGISTIC";
proc logistic data=logistic_data;
  model y(event='1') = x1 x2;
run;
title;


/****************************************************************************/
/* 7.6 模型诊断                                                             */
/****************************************************************************/

/* 7.6.1 残差分析 */
title "泊松回归残差分析";
proc genmod data=poisson_data;
  model y = x1 x2 / dist=poisson link=log;
  output out=diag_out pred=pred resraw=resraw resdev=resdev reschi=reschi;
run;
title;

/* 残差图 */
title "Deviance残差 vs 预测值";
proc sgplot data=diag_out;
  scatter x=pred y=resdev;
  refline 0 / axis=y;
run;
title;

title "Deviance残差Q-Q图";
proc univariate data=diag_out;
  var resdev;
  qqplot / normal(mu=est sigma=est);
run;
title;


/* 7.6.2 偏差分析 */
title "偏差分析";
proc genmod data=poisson_data;
  model y = x1 x2 / dist=poisson link=log type1 type3;
run;
title;


/****************************************************************************/
/* 7.7 偏差与模型比较                                                       */
/****************************************************************************/
proc iml;
  /* 饱和模型 vs 拟合模型 */
  /* 偏差 D = 2 * Σ [y_i * log(y_i/μ_i) - (y_i - μ_i)] */

  /* 示例数据 */
  y = {10, 20, 15, 25, 30};
  mu = {12, 18, 17, 23, 28};

  /* 泊松偏差 */
  D_poisson = 2 * sum(y # log(y / mu) - (y - mu));
  print "泊松偏差 D =" D_poisson;

  /* 自由度 */
  df = nrow(y) - 2;  /* 2个参数 */
  p_value = 1 - cdf('CHISQUARE', D_poisson, df);
  print "自由度 =" df "p值 =" p_value;
quit;


/****************************************************************************/
/* 7.8 保险定价应用                                                         */
/****************************************************************************/

/* 7.8.1 车险索赔次数模型 */
data auto_claims;
  call streaminit(202);
  do policy = 1 to 1000;
    age_group = rand('INTEGER', 1, 4);  /* 年龄组 */
    vehicle_age = rand('INTEGER', 1, 3);  /* 车龄 */
    gender = rand('BERNOULLI', 0.5);  /* 性别 */

    /* 真实模型 */
    eta = -1 + 0.3 * (age_group=1) - 0.2 * (age_group=4)
          + 0.4 * (vehicle_age=1) + 0.1 * gender;
    lambda = exp(eta);
    n_claims = rand('POISSON', lambda);
    output;
  end;
run;

title "车险索赔次数泊松回归";
proc genmod data=auto_claims;
  class age_group vehicle_age gender;
  model n_claims = age_group vehicle_age gender / dist=poisson link=log type3;
  estimate '年轻司机' age_group 1 0 0 0 / exp;
  estimate '老旧车辆' vehicle_age 1 0 0 / exp;
run;
title;


/* 7.8.2 车险索赔金额模型 */
data auto_severity;
  call streaminit(303);
  do policy = 1 to 500;
    age_group = rand('INTEGER', 1, 4);
    vehicle_age = rand('INTEGER', 1, 3);

    eta = 8 + 0.2 * (age_group=1) - 0.1 * (age_group=4)
          + 0.3 * (vehicle_age=1);
    mu = exp(eta);
    phi = 0.5;
    shape = 1 / phi;
    scale = mu * phi;
    claim_amount = rand('GAMMA', shape, scale);
    output;
  end;
run;

title "车险索赔金额伽马回归";
proc genmod data=auto_severity;
  class age_group vehicle_age;
  model claim_amount = age_group vehicle_age / dist=gamma link=log type3;
run;
title;


/****************************************************************************/
/* 7.9 GAM（广义可加模型）                                                  */
/****************************************************************************/
/* 使用PROC GAM */
data gam_data;
  call streaminit(404);
  do i = 1 to 500;
    x = rand('UNIFORM', 0, 10);
    eta = 1 + 0.5 * sin(x) + 0.3 * x;
    lambda = exp(eta);
    y = rand('POISSON', lambda);
    output;
  end;
run;

title "广义可加模型（GAM）";
proc gam data=gam_data;
  model y = spline(x, df=4) / dist=poisson;
  output out=gam_out pred=pred;
run;
title;

title "GAM拟合结果";
proc sgplot data=gam_out;
  scatter x=x y=y;
  series x=x y=pred;
run;
title;

/****************************************************************************/
/* 第7章 广义线性模型                                                       */
/* 对应教材：section7.tex                                                   */
/* 内容：指数族分布、连接函数、参数估计、模型诊断、                          */
/*       泊松回归、负二项回归、伽马回归、GAM与GAMLSS                         */
/****************************************************************************/

/****************************************************************************/
/* 7.1 指数族分布                                                           */
/****************************************************************************/
proc iml;
  /* 指数族分布的统一形式 */
  /* f(y;θ,φ) = exp{(yθ - b(θ))/a(φ) + c(y,φ)} */

  /* 正态分布: b(θ)=θ²/2, a(φ)=φ */
  /* 泊松分布: b(θ)=exp(θ), a(φ)=1 */
  /* 二项分布: b(θ)=log(1+exp(θ)), a(φ)=1 */
  /* 伽马分布: b(θ)=-log(-θ), a(φ)=1/ν */

  print "指数族分布的b(θ)函数";
  theta = {-2, -1, 0, 1, 2};
  b_normal = theta##2 / 2;
  b_poisson = exp(theta);
  b_gamma = -log(-theta);
  print theta b_normal b_poisson b_gamma;
quit;


/****************************************************************************/
/* 7.2 泊松回归                                                             */
/****************************************************************************/

/* 7.2.1 生成模拟数据 */
data poisson_data;
  call streaminit(123);
  do i = 1 to 500;
    x1 = rand('NORMAL', 0, 1);
    x2 = rand('BERNOULLI', 0.5);
    /* 真实模型: log(λ) = 1 + 0.5*x1 + 0.3*x2 */
    eta = 1 + 0.5 * x1 + 0.3 * x2;
    lambda = exp(eta);
    y = rand('POISSON', lambda);
    output;
  end;
run;

title "泊松回归数据描述";
proc means data=poisson_data mean std min max;
  var y x1 x2;
run;
title;

/* 7.2.2 泊松回归模型 */
title "泊松回归";
proc genmod data=poisson_data;
  model y = x1 x2 / dist=poisson link=log;
  output out=poisson_out pred=pred resdev=resdev;
run;
title;

/* 7.2.3 模型拟合统计量 */
proc genmod data=poisson_data;
  model y = x1 x2 / dist=poisson link=log type1 type3;
run;


/****************************************************************************/
/* 7.3 负二项回归                                                           */
/****************************************************************************/

/* 7.3.1 生成过散布数据 */
data negbinom_data;
  call streaminit(456);
  do i = 1 to 500;
    x1 = rand('NORMAL', 0, 1);
    x2 = rand('BERNOULLI', 0.5);
    eta = 1 + 0.5 * x1 + 0.3 * x2;
    mu = exp(eta);
    /* 负二项分布: 过散布 */
    k = 2;  /* 散布参数 */
    p = k / (k + mu);
    y = rand('NEGBINOMIAL', p, k);
    output;
  end;
run;

title "负二项回归";
proc genmod data=negbinom_data;
  model y = x1 x2 / dist=negbin link=log;
run;
title;

/* 7.3.2 泊松与负二项回归比较 */
title "泊松回归（过散布数据）";
proc genmod data=negbinom_data;
  model y = x1 x2 / dist=poisson link=log scale=pearson;
run;
title;


/****************************************************************************/
/* 7.4 伽马回归                                                             */
/****************************************************************************/
data gamma_data;
  call streaminit(789);
  do i = 1 to 500;
    x1 = rand('NORMAL', 0, 1);
    x2 = rand('BERNOULLI', 0.5);
    eta = 3 + 0.5 * x1 - 0.3 * x2;
    mu = exp(eta);
    /* 伽马分布 */
    phi = 0.5;  /* 散布参数 */
    shape = 1 / phi;
    scale = mu * phi;
    y = rand('GAMMA', shape, scale);
    output;
  end;
run;

title "伽马回归";
proc genmod data=gamma_data;
  model y = x1 x2 / dist=gamma link=log;
  output out=gamma_out pred=pred resdev=resdev;
run;
title;


/****************************************************************************/
/* 7.5 Logistic回归（二项回归）                                             */
/****************************************************************************/
data logistic_data;
  call streaminit(101);
  do i = 1 to 500;
    x1 = rand('NORMAL', 0, 1);
    x2 = rand('BERNOULLI', 0.5);
    eta = -1 + 0.8 * x1 + 0.5 * x2;
    p = 1 / (1 + exp(-eta));
    y = rand('BERNOULLI', p);
    output;
  end;
run;

title "Logistic回归";
proc genmod data=logistic_data;
  model y = x1 x2 / dist=binomial link=logit;
run;
title;

/* 使用PROC LOGISTIC */
title "PROC LOGISTIC";
proc logistic data=logistic_data;
  model y(event='1') = x1 x2;
run;
title;


/****************************************************************************/
/* 7.6 模型诊断                                                             */
/****************************************************************************/

/* 7.6.1 残差分析 */
title "泊松回归残差分析";
proc genmod data=poisson_data;
  model y = x1 x2 / dist=poisson link=log;
  output out=diag_out pred=pred resraw=resraw resdev=resdev reschi=reschi;
run;
title;

/* 残差图 */
title "Deviance残差 vs 预测值";
proc sgplot data=diag_out;
  scatter x=pred y=resdev;
  refline 0 / axis=y;
run;
title;

title "Deviance残差Q-Q图";
proc univariate data=diag_out;
  var resdev;
  qqplot / normal(mu=est sigma=est);
run;
title;


/* 7.6.2 偏差分析 */
title "偏差分析";
proc genmod data=poisson_data;
  model y = x1 x2 / dist=poisson link=log type1 type3;
run;
title;


/****************************************************************************/
/* 7.7 偏差与模型比较                                                       */
/****************************************************************************/
proc iml;
  /* 饱和模型 vs 拟合模型 */
  /* 偏差 D = 2 * Σ [y_i * log(y_i/μ_i) - (y_i - μ_i)] */

  /* 示例数据 */
  y = {10, 20, 15, 25, 30};
  mu = {12, 18, 17, 23, 28};

  /* 泊松偏差 */
  D_poisson = 2 * sum(y # log(y / mu) - (y - mu));
  print "泊松偏差 D =" D_poisson;

  /* 自由度 */
  df = nrow(y) - 2;  /* 2个参数 */
  p_value = 1 - cdf('CHISQUARE', D_poisson, df);
  print "自由度 =" df "p值 =" p_value;
quit;


/****************************************************************************/
/* 7.8 保险定价应用                                                         */
/****************************************************************************/

/* 7.8.1 车险索赔次数模型 */
data auto_claims;
  call streaminit(202);
  do policy = 1 to 1000;
    age_group = rand('INTEGER', 1, 4);  /* 年龄组 */
    vehicle_age = rand('INTEGER', 1, 3);  /* 车龄 */
    gender = rand('BERNOULLI', 0.5);  /* 性别 */

    /* 真实模型 */
    eta = -1 + 0.3 * (age_group=1) - 0.2 * (age_group=4)
          + 0.4 * (vehicle_age=1) + 0.1 * gender;
    lambda = exp(eta);
    n_claims = rand('POISSON', lambda);
    output;
  end;
run;

title "车险索赔次数泊松回归";
proc genmod data=auto_claims;
  class age_group vehicle_age gender;
  model n_claims = age_group vehicle_age gender / dist=poisson link=log type3;
  estimate '年轻司机' age_group 1 0 0 0 / exp;
  estimate '老旧车辆' vehicle_age 1 0 0 / exp;
run;
title;


/* 7.8.2 车险索赔金额模型 */
data auto_severity;
  call streaminit(303);
  do policy = 1 to 500;
    age_group = rand('INTEGER', 1, 4);
    vehicle_age = rand('INTEGER', 1, 3);

    eta = 8 + 0.2 * (age_group=1) - 0.1 * (age_group=4)
          + 0.3 * (vehicle_age=1);
    mu = exp(eta);
    phi = 0.5;
    shape = 1 / phi;
    scale = mu * phi;
    claim_amount = rand('GAMMA', shape, scale);
    output;
  end;
run;

title "车险索赔金额伽马回归";
proc genmod data=auto_severity;
  class age_group vehicle_age;
  model claim_amount = age_group vehicle_age / dist=gamma link=log type3;
run;
title;


/****************************************************************************/
/* 7.9 GAM（广义可加模型）                                                  */
/****************************************************************************/
/* 使用PROC GAM */
data gam_data;
  call streaminit(404);
  do i = 1 to 500;
    x = rand('UNIFORM', 0, 10);
    eta = 1 + 0.5 * sin(x) + 0.3 * x;
    lambda = exp(eta);
    y = rand('POISSON', lambda);
    output;
  end;
run;

title "广义可加模型（GAM）";
proc gam data=gam_data;
  model y = spline(x, df=4) / dist=poisson;
  output out=gam_out pred=pred;
run;
title;

title "GAM拟合结果";
proc sgplot data=gam_out;
  scatter x=x y=y;
  series x=x y=pred;
run;
title;

/****************************************************************************/
/* 第7章 广义线性模型                                                       */
/* 对应教材：section7.tex                                                   */
/* 内容：指数族分布、连接函数、参数估计、模型诊断、                          */
/*       泊松回归、负二项回归、伽马回归、GAM与GAMLSS                         */
/****************************************************************************/

/****************************************************************************/
/* 7.1 指数族分布                                                           */
/****************************************************************************/
proc iml;
  /* 指数族分布的统一形式 */
  /* f(y;θ,φ) = exp{(yθ - b(θ))/a(φ) + c(y,φ)} */

  /* 正态分布: b(θ)=θ²/2, a(φ)=φ */
  /* 泊松分布: b(θ)=exp(θ), a(φ)=1 */
  /* 二项分布: b(θ)=log(1+exp(θ)), a(φ)=1 */
  /* 伽马分布: b(θ)=-log(-θ), a(φ)=1/ν */

  print "指数族分布的b(θ)函数";
  theta = {-2, -1, 0, 1, 2};
  b_normal = theta##2 / 2;
  b_poisson = exp(theta);
  b_gamma = -log(-theta);
  print theta b_normal b_poisson b_gamma;
quit;


/****************************************************************************/
/* 7.2 泊松回归                                                             */
/****************************************************************************/

/* 7.2.1 生成模拟数据 */
data poisson_data;
  call streaminit(123);
  do i = 1 to 500;
    x1 = rand('NORMAL', 0, 1);
    x2 = rand('BERNOULLI', 0.5);
    /* 真实模型: log(λ) = 1 + 0.5*x1 + 0.3*x2 */
    eta = 1 + 0.5 * x1 + 0.3 * x2;
    lambda = exp(eta);
    y = rand('POISSON', lambda);
    output;
  end;
run;

title "泊松回归数据描述";
proc means data=poisson_data mean std min max;
  var y x1 x2;
run;
title;

/* 7.2.2 泊松回归模型 */
title "泊松回归";
proc genmod data=poisson_data;
  model y = x1 x2 / dist=poisson link=log;
  output out=poisson_out pred=pred resdev=resdev;
run;
title;

/* 7.2.3 模型拟合统计量 */
proc genmod data=poisson_data;
  model y = x1 x2 / dist=poisson link=log type1 type3;
run;


/****************************************************************************/
/* 7.3 负二项回归                                                           */
/****************************************************************************/

/* 7.3.1 生成过散布数据 */
data negbinom_data;
  call streaminit(456);
  do i = 1 to 500;
    x1 = rand('NORMAL', 0, 1);
    x2 = rand('BERNOULLI', 0.5);
    eta = 1 + 0.5 * x1 + 0.3 * x2;
    mu = exp(eta);
    /* 负二项分布: 过散布 */
    k = 2;  /* 散布参数 */
    p = k / (k + mu);
    y = rand('NEGBINOMIAL', p, k);
    output;
  end;
run;

title "负二项回归";
proc genmod data=negbinom_data;
  model y = x1 x2 / dist=negbin link=log;
run;
title;

/* 7.3.2 泊松与负二项回归比较 */
title "泊松回归（过散布数据）";
proc genmod data=negbinom_data;
  model y = x1 x2 / dist=poisson link=log scale=pearson;
run;
title;


/****************************************************************************/
/* 7.4 伽马回归                                                             */
/****************************************************************************/
data gamma_data;
  call streaminit(789);
  do i = 1 to 500;
    x1 = rand('NORMAL', 0, 1);
    x2 = rand('BERNOULLI', 0.5);
    eta = 3 + 0.5 * x1 - 0.3 * x2;
    mu = exp(eta);
    /* 伽马分布 */
    phi = 0.5;  /* 散布参数 */
    shape = 1 / phi;
    scale = mu * phi;
    y = rand('GAMMA', shape, scale);
    output;
  end;
run;

title "伽马回归";
proc genmod data=gamma_data;
  model y = x1 x2 / dist=gamma link=log;
  output out=gamma_out pred=pred resdev=resdev;
run;
title;


/****************************************************************************/
/* 7.5 Logistic回归（二项回归）                                             */
/****************************************************************************/
data logistic_data;
  call streaminit(101);
  do i = 1 to 500;
    x1 = rand('NORMAL', 0, 1);
    x2 = rand('BERNOULLI', 0.5);
    eta = -1 + 0.8 * x1 + 0.5 * x2;
    p = 1 / (1 + exp(-eta));
    y = rand('BERNOULLI', p);
    output;
  end;
run;

title "Logistic回归";
proc genmod data=logistic_data;
  model y = x1 x2 / dist=binomial link=logit;
run;
title;

/* 使用PROC LOGISTIC */
title "PROC LOGISTIC";
proc logistic data=logistic_data;
  model y(event='1') = x1 x2;
run;
title;


/****************************************************************************/
/* 7.6 模型诊断                                                             */
/****************************************************************************/

/* 7.6.1 残差分析 */
title "泊松回归残差分析";
proc genmod data=poisson_data;
  model y = x1 x2 / dist=poisson link=log;
  output out=diag_out pred=pred resraw=resraw resdev=resdev reschi=reschi;
run;
title;

/* 残差图 */
title "Deviance残差 vs 预测值";
proc sgplot data=diag_out;
  scatter x=pred y=resdev;
  refline 0 / axis=y;
run;
title;

title "Deviance残差Q-Q图";
proc univariate data=diag_out;
  var resdev;
  qqplot / normal(mu=est sigma=est);
run;
title;


/* 7.6.2 偏差分析 */
title "偏差分析";
proc genmod data=poisson_data;
  model y = x1 x2 / dist=poisson link=log type1 type3;
run;
title;


/****************************************************************************/
/* 7.7 偏差与模型比较                                                       */
/****************************************************************************/
proc iml;
  /* 饱和模型 vs 拟合模型 */
  /* 偏差 D = 2 * Σ [y_i * log(y_i/μ_i) - (y_i - μ_i)] */

  /* 示例数据 */
  y = {10, 20, 15, 25, 30};
  mu = {12, 18, 17, 23, 28};

  /* 泊松偏差 */
  D_poisson = 2 * sum(y # log(y / mu) - (y - mu));
  print "泊松偏差 D =" D_poisson;

  /* 自由度 */
  df = nrow(y) - 2;  /* 2个参数 */
  p_value = 1 - cdf('CHISQUARE', D_poisson, df);
  print "自由度 =" df "p值 =" p_value;
quit;


/****************************************************************************/
/* 7.8 保险定价应用                                                         */
/****************************************************************************/

/* 7.8.1 车险索赔次数模型 */
data auto_claims;
  call streaminit(202);
  do policy = 1 to 1000;
    age_group = rand('INTEGER', 1, 4);  /* 年龄组 */
    vehicle_age = rand('INTEGER', 1, 3);  /* 车龄 */
    gender = rand('BERNOULLI', 0.5);  /* 性别 */

    /* 真实模型 */
    eta = -1 + 0.3 * (age_group=1) - 0.2 * (age_group=4)
          + 0.4 * (vehicle_age=1) + 0.1 * gender;
    lambda = exp(eta);
    n_claims = rand('POISSON', lambda);
    output;
  end;
run;

title "车险索赔次数泊松回归";
proc genmod data=auto_claims;
  class age_group vehicle_age gender;
  model n_claims = age_group vehicle_age gender / dist=poisson link=log type3;
  estimate '年轻司机' age_group 1 0 0 0 / exp;
  estimate '老旧车辆' vehicle_age 1 0 0 / exp;
run;
title;


/* 7.8.2 车险索赔金额模型 */
data auto_severity;
  call streaminit(303);
  do policy = 1 to 500;
    age_group = rand('INTEGER', 1, 4);
    vehicle_age = rand('INTEGER', 1, 3);

    eta = 8 + 0.2 * (age_group=1) - 0.1 * (age_group=4)
          + 0.3 * (vehicle_age=1);
    mu = exp(eta);
    phi = 0.5;
    shape = 1 / phi;
    scale = mu * phi;
    claim_amount = rand('GAMMA', shape, scale);
    output;
  end;
run;

title "车险索赔金额伽马回归";
proc genmod data=auto_severity;
  class age_group vehicle_age;
  model claim_amount = age_group vehicle_age / dist=gamma link=log type3;
run;
title;


/****************************************************************************/
/* 7.9 GAM（广义可加模型）                                                  */
/****************************************************************************/
/* 使用PROC GAM */
data gam_data;
  call streaminit(404);
  do i = 1 to 500;
    x = rand('UNIFORM', 0, 10);
    eta = 1 + 0.5 * sin(x) + 0.3 * x;
    lambda = exp(eta);
    y = rand('POISSON', lambda);
    output;
  end;
run;

title "广义可加模型（GAM）";
proc gam data=gam_data;
  model y = spline(x, df=4) / dist=poisson;
  output out=gam_out pred=pred;
run;
title;

title "GAM拟合结果";
proc sgplot data=gam_out;
  scatter x=x y=y;
  series x=x y=pred;
run;
title;

/****************************************************************************/
/* 第7章 广义线性模型                                                       */
/* 对应教材：section7.tex                                                   */
/* 内容：指数族分布、连接函数、参数估计、模型诊断、                          */
/*       泊松回归、负二项回归、伽马回归、GAM与GAMLSS                         */
/****************************************************************************/

/****************************************************************************/
/* 7.1 指数族分布                                                           */
/****************************************************************************/
proc iml;
  /* 指数族分布的统一形式 */
  /* f(y;θ,φ) = exp{(yθ - b(θ))/a(φ) + c(y,φ)} */

  /* 正态分布: b(θ)=θ²/2, a(φ)=φ */
  /* 泊松分布: b(θ)=exp(θ), a(φ)=1 */
  /* 二项分布: b(θ)=log(1+exp(θ)), a(φ)=1 */
  /* 伽马分布: b(θ)=-log(-θ), a(φ)=1/ν */

  print "指数族分布的b(θ)函数";
  theta = {-2, -1, 0, 1, 2};
  b_normal = theta##2 / 2;
  b_poisson = exp(theta);
  b_gamma = -log(-theta);
  print theta b_normal b_poisson b_gamma;
quit;


/****************************************************************************/
/* 7.2 泊松回归                                                             */
/****************************************************************************/

/* 7.2.1 生成模拟数据 */
data poisson_data;
  call streaminit(123);
  do i = 1 to 500;
    x1 = rand('NORMAL', 0, 1);
    x2 = rand('BERNOULLI', 0.5);
    /* 真实模型: log(λ) = 1 + 0.5*x1 + 0.3*x2 */
    eta = 1 + 0.5 * x1 + 0.3 * x2;
    lambda = exp(eta);
    y = rand('POISSON', lambda);
    output;
  end;
run;

title "泊松回归数据描述";
proc means data=poisson_data mean std min max;
  var y x1 x2;
run;
title;

/* 7.2.2 泊松回归模型 */
title "泊松回归";
proc genmod data=poisson_data;
  model y = x1 x2 / dist=poisson link=log;
  output out=poisson_out pred=pred resdev=resdev;
run;
title;

/* 7.2.3 模型拟合统计量 */
proc genmod data=poisson_data;
  model y = x1 x2 / dist=poisson link=log type1 type3;
run;


/****************************************************************************/
/* 7.3 负二项回归                                                           */
/****************************************************************************/

/* 7.3.1 生成过散布数据 */
data negbinom_data;
  call streaminit(456);
  do i = 1 to 500;
    x1 = rand('NORMAL', 0, 1);
    x2 = rand('BERNOULLI', 0.5);
    eta = 1 + 0.5 * x1 + 0.3 * x2;
    mu = exp(eta);
    /* 负二项分布: 过散布 */
    k = 2;  /* 散布参数 */
    p = k / (k + mu);
    y = rand('NEGBINOMIAL', p, k);
    output;
  end;
run;

title "负二项回归";
proc genmod data=negbinom_data;
  model y = x1 x2 / dist=negbin link=log;
run;
title;

/* 7.3.2 泊松与负二项回归比较 */
title "泊松回归（过散布数据）";
proc genmod data=negbinom_data;
  model y = x1 x2 / dist=poisson link=log scale=pearson;
run;
title;


/****************************************************************************/
/* 7.4 伽马回归                                                             */
/****************************************************************************/
data gamma_data;
  call streaminit(789);
  do i = 1 to 500;
    x1 = rand('NORMAL', 0, 1);
    x2 = rand('BERNOULLI', 0.5);
    eta = 3 + 0.5 * x1 - 0.3 * x2;
    mu = exp(eta);
    /* 伽马分布 */
    phi = 0.5;  /* 散布参数 */
    shape = 1 / phi;
    scale = mu * phi;
    y = rand('GAMMA', shape, scale);
    output;
  end;
run;

title "伽马回归";
proc genmod data=gamma_data;
  model y = x1 x2 / dist=gamma link=log;
  output out=gamma_out pred=pred resdev=resdev;
run;
title;


/****************************************************************************/
/* 7.5 Logistic回归（二项回归）                                             */
/****************************************************************************/
data logistic_data;
  call streaminit(101);
  do i = 1 to 500;
    x1 = rand('NORMAL', 0, 1);
    x2 = rand('BERNOULLI', 0.5);
    eta = -1 + 0.8 * x1 + 0.5 * x2;
    p = 1 / (1 + exp(-eta));
    y = rand('BERNOULLI', p);
    output;
  end;
run;

title "Logistic回归";
proc genmod data=logistic_data;
  model y = x1 x2 / dist=binomial link=logit;
run;
title;

/* 使用PROC LOGISTIC */
title "PROC LOGISTIC";
proc logistic data=logistic_data;
  model y(event='1') = x1 x2;
run;
title;


/****************************************************************************/
/* 7.6 模型诊断                                                             */
/****************************************************************************/

/* 7.6.1 残差分析 */
title "泊松回归残差分析";
proc genmod data=poisson_data;
  model y = x1 x2 / dist=poisson link=log;
  output out=diag_out pred=pred resraw=resraw resdev=resdev reschi=reschi;
run;
title;

/* 残差图 */
title "Deviance残差 vs 预测值";
proc sgplot data=diag_out;
  scatter x=pred y=resdev;
  refline 0 / axis=y;
run;
title;

title "Deviance残差Q-Q图";
proc univariate data=diag_out;
  var resdev;
  qqplot / normal(mu=est sigma=est);
run;
title;


/* 7.6.2 偏差分析 */
title "偏差分析";
proc genmod data=poisson_data;
  model y = x1 x2 / dist=poisson link=log type1 type3;
run;
title;


/****************************************************************************/
/* 7.7 偏差与模型比较                                                       */
/****************************************************************************/
proc iml;
  /* 饱和模型 vs 拟合模型 */
  /* 偏差 D = 2 * Σ [y_i * log(y_i/μ_i) - (y_i - μ_i)] */

  /* 示例数据 */
  y = {10, 20, 15, 25, 30};
  mu = {12, 18, 17, 23, 28};

  /* 泊松偏差 */
  D_poisson = 2 * sum(y # log(y / mu) - (y - mu));
  print "泊松偏差 D =" D_poisson;

  /* 自由度 */
  df = nrow(y) - 2;  /* 2个参数 */
  p_value = 1 - cdf('CHISQUARE', D_poisson, df);
  print "自由度 =" df "p值 =" p_value;
quit;


/****************************************************************************/
/* 7.8 保险定价应用                                                         */
/****************************************************************************/

/* 7.8.1 车险索赔次数模型 */
data auto_claims;
  call streaminit(202);
  do policy = 1 to 1000;
    age_group = rand('INTEGER', 1, 4);  /* 年龄组 */
    vehicle_age = rand('INTEGER', 1, 3);  /* 车龄 */
    gender = rand('BERNOULLI', 0.5);  /* 性别 */

    /* 真实模型 */
    eta = -1 + 0.3 * (age_group=1) - 0.2 * (age_group=4)
          + 0.4 * (vehicle_age=1) + 0.1 * gender;
    lambda = exp(eta);
    n_claims = rand('POISSON', lambda);
    output;
  end;
run;

title "车险索赔次数泊松回归";
proc genmod data=auto_claims;
  class age_group vehicle_age gender;
  model n_claims = age_group vehicle_age gender / dist=poisson link=log type3;
  estimate '年轻司机' age_group 1 0 0 0 / exp;
  estimate '老旧车辆' vehicle_age 1 0 0 / exp;
run;
title;


/* 7.8.2 车险索赔金额模型 */
data auto_severity;
  call streaminit(303);
  do policy = 1 to 500;
    age_group = rand('INTEGER', 1, 4);
    vehicle_age = rand('INTEGER', 1, 3);

    eta = 8 + 0.2 * (age_group=1) - 0.1 * (age_group=4)
          + 0.3 * (vehicle_age=1);
    mu = exp(eta);
    phi = 0.5;
    shape = 1 / phi;
    scale = mu * phi;
    claim_amount = rand('GAMMA', shape, scale);
    output;
  end;
run;

title "车险索赔金额伽马回归";
proc genmod data=auto_severity;
  class age_group vehicle_age;
  model claim_amount = age_group vehicle_age / dist=gamma link=log type3;
run;
title;


/****************************************************************************/
/* 7.9 GAM（广义可加模型）                                                  */
/****************************************************************************/
/* 使用PROC GAM */
data gam_data;
  call streaminit(404);
  do i = 1 to 500;
    x = rand('UNIFORM', 0, 10);
    eta = 1 + 0.5 * sin(x) + 0.3 * x;
    lambda = exp(eta);
    y = rand('POISSON', lambda);
    output;
  end;
run;

title "广义可加模型（GAM）";
proc gam data=gam_data;
  model y = spline(x, df=4) / dist=poisson;
  output out=gam_out pred=pred;
run;
title;

title "GAM拟合结果";
proc sgplot data=gam_out;
  scatter x=x y=y;
  series x=x y=pred;
run;
title;