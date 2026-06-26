/* Chap1 SAS代码 */
/* 自动从chap1.html同步生成 */

data binomial;
  do k = 0 to 9;
    p1 = pdf('BINOMIAL', k, 0.1, 9);
    p2 = pdf('BINOMIAL', k, 0.2, 9);
    p3 = pdf('BINOMIAL', k, 0.3, 9);
    p4 = pdf('BINOMIAL', k, 0.5, 9);
    cdf1 = cdf('BINOMIAL', k, 0.1, 9);
    cdf2 = cdf('BINOMIAL', k, 0.2, 9);
    cdf3 = cdf('BINOMIAL', k, 0.3, 9);
    cdf4 = cdf('BINOMIAL', k, 0.5, 9);
    output;
  end;
run;

title "二项分布概率质量函数";
proc sgplot data=binomial;
  series x=k y=p1 / legendlabel="p=0.1";
  series x=k y=p2 / legendlabel="p=0.2";
  series x=k y=p3 / legendlabel="p=0.3";
  series x=k y=p4 / legendlabel="p=0.5";
  yaxis label="p(k)";
run;
title;

data poisson;
  do k = 0 to 10;
    p1 = pdf('POISSON', k, 1);
    p2 = pdf('POISSON', k, 2);
    p3 = pdf('POISSON', k, 3);
    p4 = pdf('POISSON', k, 5);
    output;
  end;
run;

title "泊松分布概率质量函数";
proc sgplot data=poisson;
  series x=k y=p1 / legendlabel="lambda=1";
  series x=k y=p2 / legendlabel="lambda=2";
  series x=k y=p3 / legendlabel="lambda=3";
  series x=k y=p4 / legendlabel="lambda=5";
  yaxis label="p(k)";
run;
title;

data negbinom;
  do k = 0 to 10;
    p1 = pdf('NEGBINOMIAL', k, 1/(1+0.1), 2);
    p2 = pdf('NEGBINOMIAL', k, 1/(1+0.2), 2);
    p3 = pdf('NEGBINOMIAL', k, 1/(1+0.3), 2);
    p4 = pdf('NEGBINOMIAL', k, 1/(1+0.5), 2);
    output;
  end;
run;

title "负二项分布概率质量函数";
proc sgplot data=negbinom;
  series x=k y=p1 / legendlabel="beta=0.1";
  series x=k y=p2 / legendlabel="beta=0.2";
  series x=k y=p3 / legendlabel="beta=0.3";
  series x=k y=p4 / legendlabel="beta=0.5";
  yaxis label="p(k)";
run;
title;

data exponential;
  do x = 0 to 5 by 0.01;
    f1 = pdf('EXPONENTIAL', x, 0.5);
    f2 = pdf('EXPONENTIAL', x, 1);
    f3 = pdf('EXPONENTIAL', x, 2);
    f4 = pdf('EXPONENTIAL', x, 5);
    output;
  end;
run;

title "指数分布概率密度函数";
proc sgplot data=exponential;
  series x=x y=f1 / legendlabel="rate=0.5";
  series x=x y=f2 / legendlabel="rate=1";
  series x=x y=f3 / legendlabel="rate=2";
  series x=x y=f4 / legendlabel="rate=5";
  yaxis label="f(x)";
run;
title;

data gamma_dist;
  do x = 0 to 4 by 0.001;
    f1 = pdf('GAMMA', x, 1);
    f2 = pdf('GAMMA', x, 2);
    f3 = pdf('GAMMA', x, 3);
    f4 = pdf('GAMMA', x, 0.5);
    output;
  end;
run;

title "伽马分布（改变形状参数）";
proc sgplot data=gamma_dist;
  series x=x y=f1 / legendlabel="shape=1";
  series x=x y=f2 / legendlabel="shape=2";
  series x=x y=f3 / legendlabel="shape=3";
  series x=x y=f4 / legendlabel="shape=0.5";
  yaxis label="f(x)";
run;
title;

/* SAS没有内置帕累托PDF，使用DATA步自定义 */
data pareto;
  do x = 0 to 3 by 0.01;
    f1 = 1 * 3**1 / (3 + x)**(1 + 1);
    f2 = 2 * 3**2 / (3 + x)**(2 + 1);
    f3 = 5 * 3**5 / (3 + x)**(5 + 1);
    f4 = 0.5 * 3**0.5 / (3 + x)**(0.5 + 1);
    output;
  end;
run;

title "帕累托分布（改变形状参数）";
proc sgplot data=pareto;
  series x=x y=f1 / legendlabel="alpha=1";
  series x=x y=f2 / legendlabel="alpha=2";
  series x=x y=f3 / legendlabel="alpha=5";
  series x=x y=f4 / legendlabel="alpha=0.5";
  yaxis label="f(x)";
run;
title;

data lognormal;
  do x = 0 to 7 by 0.01;
    f1 = pdf('LOGNORMAL', x, 1, 0.5);
    f2 = pdf('LOGNORMAL', x, 1, 1);
    f3 = pdf('LOGNORMAL', x, 1, 3);
    f4 = pdf('LOGNORMAL', x, 1, 10);
    output;
  end;
run;

title "对数正态分布";
proc sgplot data=lognormal;
  series x=x y=f1 / legendlabel="sdlog=0.5";
  series x=x y=f2 / legendlabel="sdlog=1";
  series x=x y=f3 / legendlabel="sdlog=3";
  series x=x y=f4 / legendlabel="sdlog=10";
  yaxis label="f(x)";
run;
title;

data weibull;
  do x = 0 to 3 by 0.01;
    f1 = pdf('WEIBULL', x, 1, 1);
    f2 = pdf('WEIBULL', x, 2, 1);
    f3 = pdf('WEIBULL', x, 3, 1);
    f4 = pdf('WEIBULL', x, 0.5, 1);
    output;
  end;
run;

title "威布尔分布（改变形状参数）";
proc sgplot data=weibull;
  series x=x y=f1 / legendlabel="shape=1";
  series x=x y=f2 / legendlabel="shape=2";
  series x=x y=f3 / legendlabel="shape=3";
  series x=x y=f4 / legendlabel="shape=0.5";
  yaxis label="f(x)";
run;
title;

/* 例1.4：混合分布的概率密度函数 */
data mixture;
  do x = 0 to 1 by 0.001;
    f_mix = 0.3 * pdf('LOGNORMAL', x, 1, 2) + 0.7 * pdf('LOGNORMAL', x, 3, 4);
    f1 = pdf('LOGNORMAL', x, 1, 2);
    f2 = pdf('LOGNORMAL', x, 3, 4);
    output;
  end;
run;

title "混合分布的概率密度函数";
proc sgplot data=mixture;
  series x=x y=f_mix / legendlabel="混合分布" lineattrs=(thickness=2 color=red);
  series x=x y=f1 / legendlabel="LN(1,2)";
  series x=x y=f2 / legendlabel="LN(3,4)";
  yaxis label="f(x)" min=0 max=2;
run;
title;