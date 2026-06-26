/* Chap3 SAS代码 */
/* 自动从chap3.html同步生成 */

/* 3.1.1 三家工厂次品问题（例3.1） */
proc iml;
  P_B = {0.6, 0.3, 0.1};              /* 各工厂份额 */
  P_A_given_B = {0.10, 0.05, 0.15};   /* 各工厂次品率 */
  P_A = sum(P_A_given_B # P_B);       /* 全概率公式 */
  P_B3_given_A = P_A_given_B[3] * P_B[3] / P_A;
  print "例3.1: 三家工厂次品问题";
  print "P(A) =" P_A;
  print "P(B3|A) =" P_B3_given_A;
quit;

/* 3.1.2 S/M/L三类索赔（例3.2） */
proc iml;
  theta = {100, 1000, 2500};
  P_prior = {0.80, 0.15, 0.05};
  x = 5000;
  P_A_given_B = 2 * theta##2 / x##3;
  P_A = sum(P_A_given_B # P_prior);
  P_post = P_A_given_B # P_prior / P_A;
  print "例3.2: S/M/L三类索赔";
  print "后验概率:" P_post;
quit;

/* 3.1.3 二项分布/贝塔共轭先验（例3.3） */
proc iml;
  alpha = 2; beta = 3;
  x = {1, 0, 1, 1, 0, 1, 1, 1, 0, 1};
  n = nrow(x);
  alpha_post = sum(x) + alpha;
  beta_post = n - sum(x) + beta;
  E_theta_post = alpha_post / (alpha_post + beta_post);
  E_prior = alpha / (alpha + beta);
  print "例3.3: 二项/贝塔共轭先验";
  print "先验期望:" E_prior "后验期望:" E_theta_post "样本比例:" (sum(x)/n);
quit;


/****************************************************************************/

/* 3.2.1 泊松/伽马共轭（例3.4） */
proc iml;
  alpha = 100; lambda = 1;
  x = {144, 144, 174, 148, 151, 156, 168, 147, 140, 161};
  n = nrow(x);
  alpha_post = sum(x) + alpha;
  lambda_post = n + lambda;
  E_mu_post = alpha_post / lambda_post;
  print "例3.4: 泊松/伽马共轭 后验期望 =" E_mu_post;
quit;

/* 3.2.2 正态/正态共轭（例3.5） */
proc iml;
  sigma1 = 50; mu0 = 300; sigma2 = 20;
  n = 10; xbar = 270;
  mu_post = (sigma1**2 * mu0 + n * sigma2**2 * xbar) /
            (sigma1**2 + n * sigma2**2);
  sigma2_post = sigma1**2 * sigma2**2 / (sigma1**2 + n * sigma2**2);
  print "例3.5: 正态/正态共轭";
  print "后验均值 =" mu_post "后验标准差 =" sqrt(sigma2_post);
quit;

/* 3.2.3 贝塔后验的二次损失贝叶斯估计（例3.6） */
proc iml;
  alpha = 2; beta = 3;
  sum_x = 7; n = 10;
  theta_hat = (sum_x + alpha) / (n + alpha + beta);
  print "例3.6: 二次损失贝叶斯估计 theta_hat =" theta_hat;
quit;


/****************************************************************************/

/* 3.3.1 三种损失函数下的贝叶斯估计（例3.7） */
proc iml;
  /* 后验分布Ga(7, 13) */
  mu_hat1 = 7 / 13;                                    /* 二次损失：后验期望 */
  mu_hat2 = quantile('GAMMA', 0.5, 7, 1/13);          /* 绝对损失：后验中位数 */
  mu_hat3 = (7 - 1) / 13;                              /* 0/1损失：后验众数 */
  print "例3.7: 贝叶斯估计（不同损失函数）";
  print "二次损失(后验期望):" mu_hat1;
  print "绝对损失(后验中位数):" mu_hat2;
  print "0/1损失(后验众数):" mu_hat3;
quit;


/****************************************************************************/

/* 3.4.2 逆伽马先验下的信度估计（例3.8） */
proc iml;
  theta = 40; alpha = 1.5; sum_x = 9826; n = 100;
  mu_hat = (sum_x + theta) / (n + alpha - 1);
  Z = n / (n + alpha - 1);
  print "例3.8: 逆伽马先验 mu_hat =" mu_hat "Z =" Z;
quit;


/****************************************************************************/

/* 3.5.1 泊松/伽马模型：信度因子随时间变化（例3.9） */
proc iml;
  x = {144, 144, 174, 148, 151, 156, 168, 147, 140, 161};
  n = nrow(x);

  /* 先验Ga(100, 1) */
  alpha1 = 100; beta1 = 1;
  Z1 = j(n, 1, 0); E_mu1 = j(n, 1, 0);
  do k = 1 to n;
    Z1[k] = k / (k + beta1);
    E_mu1[k] = (sum(x[1:k]) + alpha1) / (k + beta1);
  end;

  /* 先验Ga(500, 5) */
  alpha2 = 500; beta2 = 5;
  Z2 = j(n, 1, 0); E_mu2 = j(n, 1, 0);
  do k = 1 to n;
    Z2[k] = k / (k + beta2);
    E_mu2[k] = (sum(x[1:k]) + alpha2) / (k + beta2);
  end;

  years = (1:n)`;
  print "例3.9: 信度因子变化";
  print "Ga(100,1)" years Z1 E_mu1;
  print "Ga(500,5)" years Z2 E_mu2;
quit;

/* 3.5.2 伯努利/贝塔模型（例3.10） */
proc iml;
  a = 2; b = 3;
  x = {1, 0, 1, 1, 0, 1, 1, 1, 0, 1};
  n = nrow(x);
  a_post = sum(x) + a;
  b_post = n - sum(x) + b;
  E_p_post = a_post / (a_post + b_post);
  Z = n / (n + a + b);
  print "例3.10: 伯努利/贝塔模型";
  print "后验期望:" E_p_post "信度因子Z:" Z;
quit;

/* 3.5.3 正态/正态模型（例3.11） */
proc iml;
  sigma1 = 50; mu0 = 300; sigma2 = 20;
  n = 10; xbar = 270;

  /* 先验概率P(mu < 270) */
  P_prior = cdf('NORMAL', 270, mu0, sigma2);
  print "例3.11: 先验概率 P(mu<270) =" P_prior;

  /* 后验分布 */
  mu_post = (sigma1**2 * mu0 + n * sigma2**2 * xbar) /
            (sigma1**2 + n * sigma2**2);
  sigma2_post = sigma1**2 * sigma2**2 / (sigma1**2 + n * sigma2**2);
  sigma_post = sqrt(sigma2_post);

  /* 后验概率P(mu < 270 | x) */
  P_post = cdf('NORMAL', 270, mu_post, sigma_post);
  print "后验概率 P(mu<270|x) =" P_post;

  /* 信度因子 */
  Z = n * sigma2**2 / (sigma1**2 + n * sigma2**2);
  print "信度因子 Z =" Z;
quit;

/* 3.5.4 EBCT模型1示例：四个国家火灾保单（例3.12） */
proc iml;
  Y = {48 53 42 50 59,
       64 71 64 73 70,
       85 54 76 65 90,
       44 52 69 55 71};

  N = nrow(Y); n = ncol(Y);
  X_bar_i = Y[, :];       /* 行均值 */
  X_bar = mean(Y);        /* 总体均值 */

  E_m = X_bar;

  /* 各行样本方差 */
  s2_i = j(N, 1, 0);
  do i = 1 to N;
    s2_i[i] = var(Y[i, ]);
  end;
  E_s2 = mean(s2_i);

  Var_m = var(X_bar_i) - E_s2 / n;
  Z = n / (n + E_s2 / Var_m);
  credibility_premium = Z * X_bar_i + (1 - Z) * E_m;

  print "例3.12: 四个国家EBCT1";
  print "E[m(theta)] =" E_m;
  print "E[s^2(theta)] =" E_s2;
  print "Var[m(theta)] =" Var_m;
  print "Z =" Z;
  print "信度保费:" credibility_premium;
quit;


/****************************************************************************/

/* 3.6.1 EBCT模型1示例：五个车队（例3.13） */
proc iml;
  Y = {1250  980 1800 2040 1000 1180,
       1700 3080 1700 2820 5760 3480,
       2050 3560 2800 1600 4200 2650,
       4690 4370 4800 9070 3770 5250,
       7150 3480 5010 4810 8740 7260};

  N = nrow(Y); n = ncol(Y);
  X_bar_i = Y[, :];       /* 行均值 */
  X_bar = mean(Y);        /* 总体均值 */

  E_m = X_bar;
  /* 各行样本方差 */
  E_s2 = j(N, 1, 0);
  do i = 1 to N;
    E_s2[i] = var(Y[i, ]);
  end;
  E_s2 = mean(E_s2);

  Var_m = var(X_bar_i) - E_s2 / n;
  Z = n / (n + E_s2 / Var_m);
  credibility_premium = Z * X_bar_i + (1 - Z) * E_m;

  print "例3.13: EBCT模型1（Bühlmann信度）";
  print "E[m(theta)] =" E_m;
  print "E[s^2(theta)] =" E_s2;
  print "Var[m(theta)] =" Var_m;
  print "Z =" Z;
  print "信度保费:" credibility_premium;
quit;


/****************************************************************************/

/* 3.7.1 EBCT模型2示例：五个车队（例3.14） */
proc iml;
  Y = {1250  980 1800 2040 1000 1180,
       1700 3080 1700 2820 5760 3480,
       2050 3560 2800 1600 4200 2650,
       4690 4370 4800 9070 3770 5250,
       7150 3480 5010 4810 8740 7260};

  P = {5 5 4 6 5 5,
       11 13 10 12 15 14,
       3 4 4 3 3 2,
       9 9 8 8 9 10,
       7 7 8 8 9 10};

  N = nrow(Y); n = ncol(Y);
  X = Y / P;
  P_bar_i = P[, +];
  P_bar = sum(P);
  X_bar_i = (P # X)[, +] / P_bar_i;
  X_bar = sum(P # X) / P_bar;

  E_m = X_bar;
  E_s2 = sum((P # (X - X_bar_i)##2)[, +]) / (N * (n - 1));
  P_star = sum(P_bar_i # (1 - P_bar_i / P_bar)) / (N * n - 1);
  Var_m = (sum((P # (X - X_bar)##2)[, +]) / (N * n - 1) - E_s2) / P_star;

  Z_i = P_bar_i / (P_bar_i + E_s2 / Var_m);
  premium_unit = Z_i # X_bar_i + (1 - Z_i) * E_m;

  P_2021 = {5, 14, 2, 10, 10};
  credibility_premium = premium_unit # P_2021;

  print "例3.14: EBCT模型2（Bühlmann-Straub信度）";
  print "E[m(theta)] =" E_m;
  print "E[s^2(theta)] =" E_s2;
  print "Var[m(theta)] =" Var_m;
  print "Z_i =" Z_i;
  print "单位风险信度保费:" premium_unit;
  print "2021年信度保费:" credibility_premium;
quit;

/* 3.7.2 EBCT模型2示例：三个保险公司（例3.15） */
proc iml;
  Y = {14.2 15.8 22.7 19.0,
       58.6 63.1 81.0 64.2,
       123  132  161  133};

  P = {163  189  252  199,
       4435 4761 5576 4581,
       16184 17443 20102 18000};

  N = nrow(Y); n = ncol(Y);
  X = Y / P;
  P_bar_i = P[, +];
  P_bar = sum(P);
  X_bar_i = (P # X)[, +] / P_bar_i;
  X_bar = sum(P # X) / P_bar;

  E_m = X_bar;
  E_s2 = sum((P # (X - X_bar_i)##2)[, +]) / (N * (n - 1));
  P_star = sum(P_bar_i # (1 - P_bar_i / P_bar)) / (N * n - 1);
  Var_m = (sum((P # (X - X_bar)##2)[, +]) / (N * n - 1) - E_s2) / P_star;

  Z_B = P_bar_i[2] / (P_bar_i[2] + E_s2 / Var_m);
  cred_B_unit = Z_B * X_bar_i[2] + (1 - Z_B) * E_m;
  cred_B = cred_B_unit * 4800;

  print "例3.15: 保险公司B";
  print "Z_B =" Z_B;
  print "单位风险信度保费 =" cred_B_unit;
  print "下年保费(4800) =" cred_B;
quit;

/* 3.7.3 EBCT模型2假设条件示例（例3.16） */
proc iml;
  call randseed(123);
  P = {100, 150, 120};
  m_theta = 5;
  s2_theta = 25;

  /* 模拟各年索赔 */
  Y = j(1, 3, 0);
  do j = 1 to 3;
    do k = 1 to P[j];
      Y[j] = Y[j] + rand('NORMAL', m_theta, sqrt(s2_theta));
    end;
  end;
  X = Y / P;

  print "例3.16: E(X_j)均值 =" (mean(X)) "理论值 =" m_theta;

  /* 模拟验证 P_j * Var(X_j) */
  n_sim = 1000;
  X_sim = j(n_sim, 3, 0);
  do s = 1 to n_sim;
    do j = 1 to 3;
      Y_sim = 0;
      do k = 1 to P[j];
        Y_sim = Y_sim + rand('NORMAL', m_theta, sqrt(s2_theta));
      end;
      X_sim[s, j] = Y_sim / P[j];
    end;
  end;
  P_var_X = var(X_sim) # P`;
  print "P_j*Var(X_j)均值 =" (mean(P_var_X)) "理论值 =" s2_theta;
quit;