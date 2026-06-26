/* Chap6 SAS代码 */
/* 自动从chap6.html同步生成 */

/****************************************************************************/
/* 第6章 时间序列分析                                                       */
/* 对应教材：section6.tex                                                   */
/* 内容：平稳性检验、AR/MA/ARMA/ARIMA模型、模型识别、参数估计、              */
/*       诊断检验、外推预测、多元时间序列、非线性时间序列                    */
/****************************************************************************/

/****************************************************************************/
/* 6.1 时间序列数据生成与可视化                                             */
/****************************************************************************/

/* 6.1.1 生成模拟时间序列 */
data ts_data;
  call streaminit(123);
  /* AR(1)过程 */
  y_ar1 = 0;
  /* 白噪声 */
  y_wn = 0;
  /* 随机游走 */
  y_rw = 0;
  do t = 1 to 200;
    e = rand('NORMAL', 0, 1);
    y_ar1 = 0.7 * y_ar1 + e;
    y_wn = e;
    y_rw = y_rw + e;
    output;
  end;
run;

title "AR(1)过程";
proc sgplot data=ts_data;
  series x=t y=y_ar1;
run;
title;

title "白噪声过程";
proc sgplot data=ts_data;
  series x=t y=y_wn;
run;
title;

title "随机游走过程";
proc sgplot data=ts_data;
  series x=t y=y_rw;
run;
title;


/****************************************************************************/
/* 6.2 平稳性检验                                                           */
/****************************************************************************/

/* 6.2.1 自相关函数和偏自相关函数 */
title "AR(1)过程的自相关函数";
proc arima data=ts_data;
  identify var=y_ar1 nlag=20;
run;
title;


/* 6.2.2 ADF检验 */
proc autoreg data=ts_data;
  model y_rw = / stationarity=(adf=2);
run;


/****************************************************************************/
/* 6.3 AR模型                                                               */
/****************************************************************************/

/* 6.3.1 AR(1)模型估计 */
title "AR(1)模型估计";
proc arima data=ts_data;
  identify var=y_ar1 nlag=20;
  estimate p=1 method=ml;
run;
title;


/* 6.3.2 AR(2)模型 */
data ts_ar2;
  call streaminit(456);
  y = 0; y_lag1 = 0; y_lag2 = 0;
  do t = 1 to 300;
    e = rand('NORMAL', 0, 1);
    y = 0.3 * y_lag1 + 0.4 * y_lag2 + e;
    y_lag2 = y_lag1;
    y_lag1 = y;
    output;
  end;
run;

title "AR(2)模型估计";
proc arima data=ts_ar2;
  identify var=y nlag=20;
  estimate p=2 method=ml;
run;
title;


/****************************************************************************/
/* 6.4 MA模型                                                               */
/****************************************************************************/
data ts_ma1;
  call streaminit(789);
  e_lag = 0;
  do t = 1 to 300;
    e = rand('NORMAL', 0, 1);
    y = e + 0.6 * e_lag;
    e_lag = e;
    output;
  end;
run;

title "MA(1)模型估计";
proc arima data=ts_ma1;
  identify var=y nlag=20;
  estimate q=1 method=ml;
run;
title;


/****************************************************************************/
/* 6.5 ARMA模型                                                             */
/****************************************************************************/
data ts_arma11;
  call streaminit(101);
  y_lag = 0; e_lag = 0;
  do t = 1 to 500;
    e = rand('NORMAL', 0, 1);
    y = 0.5 * y_lag + e + 0.3 * e_lag;
    y_lag = y;
    e_lag = e;
    output;
  end;
run;

title "ARMA(1,1)模型估计";
proc arima data=ts_arma11;
  identify var=y nlag=20;
  estimate p=1 q=1 method=ml;
run;
title;


/****************************************************************************/
/* 6.6 ARIMA模型                                                            */
/****************************************************************************/

/* 6.6.1 非平稳序列的差分 */
data ts_arima;
  call streaminit(202);
  y = 0; y_lag = 0;
  do t = 1 to 200;
    e = rand('NORMAL', 0, 1);
    y = 0.5 + y_lag + e;  /* 带漂移的随机游走 */
    y_lag = y;
    output;
  end;
run;

title "带漂移的随机游走";
proc sgplot data=ts_arima;
  series x=t y=y;
run;
title;

/* 差分 */
proc arima data=ts_arima;
  identify var=y(1) nlag=20;
  estimate p=0 q=0 method=ml;
run;


/****************************************************************************/
/* 6.7 模型诊断检验                                                         */
/****************************************************************************/
title "AR(1)模型残差诊断";
proc arima data=ts_data;
  identify var=y_ar1 nlag=20;
  estimate p=1 method=ml;
  forecast out=forecast_res lead=10;
run;
title;

/* 残差白噪声检验 */
proc arima data=ts_data;
  identify var=y_ar1 nlag=20;
  estimate p=1 method=ml;
  outlier;
run;


/****************************************************************************/
/* 6.8 外推预测                                                             */
/****************************************************************************/
title "AR(1)模型预测";
proc arima data=ts_data;
  identify var=y_ar1 nlag=20;
  estimate p=1 method=ml;
  forecast out=forecast_ar1 lead=20 alpha=0.05;
run;
title;

/* 绘制预测结果 */
data forecast_plot;
  merge ts_data forecast_ar1;
  by t;
run;

title "AR(1)模型预测与置信区间";
proc sgplot data=forecast_plot;
  series x=t y=y_ar1;
  series x=t y=FORECAST;
  series x=t y=L95;
  series x=t y=U95;
  where t > 180;
run;
title;


/****************************************************************************/
/* 6.9 季节性时间序列                                                       */
/****************************************************************************/
data ts_seasonal;
  call streaminit(303);
  do t = 1 to 240;
    seasonal = 10 * sin(2 * constant('PI') * t / 12);
    trend = 0.5 * t;
    e = rand('NORMAL', 0, 2);
    y = trend + seasonal + 50 + e;
    output;
  end;
run;

title "季节性时间序列";
proc sgplot data=ts_seasonal;
  series x=t y=y;
run;
title;

/* 季节差分 */
proc arima data=ts_seasonal;
  identify var=y(1,12) nlag=24;
  estimate p=1 q=1 method=ml;
  forecast out=forecast_seasonal lead=12;
run;


/****************************************************************************/
/* 6.10 指数平滑预测                                                        */
/****************************************************************************/
title "简单指数平滑";
proc forecast data=ts_data method=expo trend=1 lead=20
              out=forecast_expo outfull;
  var y_ar1;
run;
title;

title "Holt-Winters方法";
proc forecast data=ts_seasonal method=winters trend=1
              seasons=12 lead=12
              out=forecast_hw outfull;
  var y;
run;
title;


/****************************************************************************/
/* 6.11 非线性时间序列：ARCH/GARCH模型                                      */
/****************************************************************************/
data ts_arch;
  call streaminit(404);
  y = 0; sigma2 = 1;
  do t = 1 to 500;
    e = rand('NORMAL', 0, sqrt(sigma2));
    y = e;
    sigma2 = 0.1 + 0.8 * e**2;  /* ARCH(1) */
    output;
  end;
run;

title "ARCH(1)过程";
proc sgplot data=ts_arch;
  series x=t y=y;
run;
title;

/* GARCH模型估计 */
proc autoreg data=ts_arch;
  model y = / garch=(q=1, p=1);
  output out=garch_out cev=cev;
run;


/****************************************************************************/
/* 6.12 多元时间序列                                                        */
/****************************************************************************/
data ts_bivariate;
  call streaminit(505);
  y1 = 0; y2 = 0;
  do t = 1 to 200;
    e1 = rand('NORMAL', 0, 1);
    e2 = rand('NORMAL', 0, 1);
    y1 = 0.5 * y1 + e1;
    y2 = 0.3 * y2 + 0.4 * y1 + e2;  /* y2受y1影响 */
    output;
  end;
run;

title "二元时间序列";
proc sgplot data=ts_bivariate;
  series x=t y=y1 / legendlabel="y1";
  series x=t y=y2 / legendlabel="y2";
run;
title;

/* VAR模型 */
proc varmax data=ts_bivariate;
  model y1 y2 / p=1;
run;

/* ARIMA模型拟合与预测 */
/* 生成模拟数据 */
data arima_data;
  call streaminit(123);
  xt = 0;
  do t = 1 to 100;
    e = rand('normal', 0, 1);
    if t > 1 then xt = 0.5 + xt + e;
    output;
  end;
run;

/* 一阶差分 */
data arima_diff;
  set arima_data;
  xt1 = dif(xt);
run;

/* ADF检验（使用PROC ARIMA的平稳性检验） */
proc arima data=arima_data;
  identify var=xt(1) stationarity=(adf=0);
run;

/* 拟合ARIMA(2,1,2)模型 */
proc arima data=arima_data;
  identify var=xt(1);
  estimate p=2 q=2;
  forecast out=forecast_arima lead=10;
run;

/* 绘制预测图 */
data plot_data;
  set arima_data(keep=t xt rename=(xt=value))
      forecast_arima(keep=t FORECAST L95 U95 rename=(FORECAST=value));
run;

proc sgplot data=plot_data;
  series x=t y=value / legendlabel="观测值/预测值";
  band x=t upper=U95 lower=L95 / legendlabel="95%置信区间" transparency=0.5;
  xaxis label="t";
  yaxis label="xt";
  title "ARIMA(2,1,2)模型预测";
run;
title;

/****************************************************************************/
/* 第6章 时间序列分析                                                       */
/* 对应教材：section6.tex                                                   */
/* 内容：平稳性检验、AR/MA/ARMA/ARIMA模型、模型识别、参数估计、              */
/*       诊断检验、外推预测、多元时间序列、非线性时间序列                    */
/****************************************************************************/

/****************************************************************************/
/* 6.1 时间序列数据生成与可视化                                             */
/****************************************************************************/

/* 6.1.1 生成模拟时间序列 */
data ts_data;
  call streaminit(123);
  /* AR(1)过程 */
  y_ar1 = 0;
  /* 白噪声 */
  y_wn = 0;
  /* 随机游走 */
  y_rw = 0;
  do t = 1 to 200;
    e = rand('NORMAL', 0, 1);
    y_ar1 = 0.7 * y_ar1 + e;
    y_wn = e;
    y_rw = y_rw + e;
    output;
  end;
run;

title "AR(1)过程";
proc sgplot data=ts_data;
  series x=t y=y_ar1;
run;
title;

title "白噪声过程";
proc sgplot data=ts_data;
  series x=t y=y_wn;
run;
title;

title "随机游走过程";
proc sgplot data=ts_data;
  series x=t y=y_rw;
run;
title;


/****************************************************************************/
/* 6.2 平稳性检验                                                           */
/****************************************************************************/

/* 6.2.1 自相关函数和偏自相关函数 */
title "AR(1)过程的自相关函数";
proc arima data=ts_data;
  identify var=y_ar1 nlag=20;
run;
title;


/* 6.2.2 ADF检验 */
proc autoreg data=ts_data;
  model y_rw = / stationarity=(adf=2);
run;


/****************************************************************************/
/* 6.3 AR模型                                                               */
/****************************************************************************/

/* 6.3.1 AR(1)模型估计 */
title "AR(1)模型估计";
proc arima data=ts_data;
  identify var=y_ar1 nlag=20;
  estimate p=1 method=ml;
run;
title;


/* 6.3.2 AR(2)模型 */
data ts_ar2;
  call streaminit(456);
  y = 0; y_lag1 = 0; y_lag2 = 0;
  do t = 1 to 300;
    e = rand('NORMAL', 0, 1);
    y = 0.3 * y_lag1 + 0.4 * y_lag2 + e;
    y_lag2 = y_lag1;
    y_lag1 = y;
    output;
  end;
run;

title "AR(2)模型估计";
proc arima data=ts_ar2;
  identify var=y nlag=20;
  estimate p=2 method=ml;
run;
title;


/****************************************************************************/
/* 6.4 MA模型                                                               */
/****************************************************************************/
data ts_ma1;
  call streaminit(789);
  e_lag = 0;
  do t = 1 to 300;
    e = rand('NORMAL', 0, 1);
    y = e + 0.6 * e_lag;
    e_lag = e;
    output;
  end;
run;

title "MA(1)模型估计";
proc arima data=ts_ma1;
  identify var=y nlag=20;
  estimate q=1 method=ml;
run;
title;


/****************************************************************************/
/* 6.5 ARMA模型                                                             */
/****************************************************************************/
data ts_arma11;
  call streaminit(101);
  y_lag = 0; e_lag = 0;
  do t = 1 to 500;
    e = rand('NORMAL', 0, 1);
    y = 0.5 * y_lag + e + 0.3 * e_lag;
    y_lag = y;
    e_lag = e;
    output;
  end;
run;

title "ARMA(1,1)模型估计";
proc arima data=ts_arma11;
  identify var=y nlag=20;
  estimate p=1 q=1 method=ml;
run;
title;


/****************************************************************************/
/* 6.6 ARIMA模型                                                            */
/****************************************************************************/

/* 6.6.1 非平稳序列的差分 */
data ts_arima;
  call streaminit(202);
  y = 0; y_lag = 0;
  do t = 1 to 200;
    e = rand('NORMAL', 0, 1);
    y = 0.5 + y_lag + e;  /* 带漂移的随机游走 */
    y_lag = y;
    output;
  end;
run;

title "带漂移的随机游走";
proc sgplot data=ts_arima;
  series x=t y=y;
run;
title;

/* 差分 */
proc arima data=ts_arima;
  identify var=y(1) nlag=20;
  estimate p=0 q=0 method=ml;
run;


/****************************************************************************/
/* 6.7 模型诊断检验                                                         */
/****************************************************************************/
title "AR(1)模型残差诊断";
proc arima data=ts_data;
  identify var=y_ar1 nlag=20;
  estimate p=1 method=ml;
  forecast out=forecast_res lead=10;
run;
title;

/* 残差白噪声检验 */
proc arima data=ts_data;
  identify var=y_ar1 nlag=20;
  estimate p=1 method=ml;
  outlier;
run;


/****************************************************************************/
/* 6.8 外推预测                                                             */
/****************************************************************************/
title "AR(1)模型预测";
proc arima data=ts_data;
  identify var=y_ar1 nlag=20;
  estimate p=1 method=ml;
  forecast out=forecast_ar1 lead=20 alpha=0.05;
run;
title;

/* 绘制预测结果 */
data forecast_plot;
  merge ts_data forecast_ar1;
  by t;
run;

title "AR(1)模型预测与置信区间";
proc sgplot data=forecast_plot;
  series x=t y=y_ar1;
  series x=t y=FORECAST;
  series x=t y=L95;
  series x=t y=U95;
  where t > 180;
run;
title;


/****************************************************************************/
/* 6.9 季节性时间序列                                                       */
/****************************************************************************/
data ts_seasonal;
  call streaminit(303);
  do t = 1 to 240;
    seasonal = 10 * sin(2 * constant('PI') * t / 12);
    trend = 0.5 * t;
    e = rand('NORMAL', 0, 2);
    y = trend + seasonal + 50 + e;
    output;
  end;
run;

title "季节性时间序列";
proc sgplot data=ts_seasonal;
  series x=t y=y;
run;
title;

/* 季节差分 */
proc arima data=ts_seasonal;
  identify var=y(1,12) nlag=24;
  estimate p=1 q=1 method=ml;
  forecast out=forecast_seasonal lead=12;
run;


/****************************************************************************/
/* 6.10 指数平滑预测                                                        */
/****************************************************************************/
title "简单指数平滑";
proc forecast data=ts_data method=expo trend=1 lead=20
              out=forecast_expo outfull;
  var y_ar1;
run;
title;

title "Holt-Winters方法";
proc forecast data=ts_seasonal method=winters trend=1
              seasons=12 lead=12
              out=forecast_hw outfull;
  var y;
run;
title;


/****************************************************************************/
/* 6.11 非线性时间序列：ARCH/GARCH模型                                      */
/****************************************************************************/
data ts_arch;
  call streaminit(404);
  y = 0; sigma2 = 1;
  do t = 1 to 500;
    e = rand('NORMAL', 0, sqrt(sigma2));
    y = e;
    sigma2 = 0.1 + 0.8 * e**2;  /* ARCH(1) */
    output;
  end;
run;

title "ARCH(1)过程";
proc sgplot data=ts_arch;
  series x=t y=y;
run;
title;

/* GARCH模型估计 */
proc autoreg data=ts_arch;
  model y = / garch=(q=1, p=1);
  output out=garch_out cev=cev;
run;


/****************************************************************************/
/* 6.12 多元时间序列                                                        */
/****************************************************************************/
data ts_bivariate;
  call streaminit(505);
  y1 = 0; y2 = 0;
  do t = 1 to 200;
    e1 = rand('NORMAL', 0, 1);
    e2 = rand('NORMAL', 0, 1);
    y1 = 0.5 * y1 + e1;
    y2 = 0.3 * y2 + 0.4 * y1 + e2;  /* y2受y1影响 */
    output;
  end;
run;

title "二元时间序列";
proc sgplot data=ts_bivariate;
  series x=t y=y1 / legendlabel="y1";
  series x=t y=y2 / legendlabel="y2";
run;
title;

/* VAR模型 */
proc varmax data=ts_bivariate;
  model y1 y2 / p=1;
run;