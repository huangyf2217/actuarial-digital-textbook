/* Chap10 SAS代码 */
/* 自动从chap10.html同步生成 */

if n_claims > 0 then do;
      eta_sev = 8 + 0.01 * (age - 40) + 0.05 * vehicle_age;
      mu = exp(eta_sev);
      claim_amount = rand('GAMMA', 2, mu/2) * n_claims;
    end;
    else claim_amount = 0;

    output;
  end;
run;

title "车险数据描述";
proc means data=auto_insurance mean std min max;
  var age vehicle_age n_claims claim_amount;
run;
title;

/* 随机森林预测索赔次数 */
title "随机森林预测索赔次数";
proc hpforest data=auto_insurance seed=123
              maxtrees=100 vars_to_try=3;
  model n_claims = age gender vehicle_age region no_claim_years;
run;
title;

/* 梯度提升树预测索赔金额 */
title "梯度提升树预测索赔金额";
proc gradboost data=auto_insurance seed=123;
  model claim_amount = age gender vehicle_age region no_claim_years;
  partition fraction(validate=0.3);
run;
title;

/****************************************************************************/
/* 第10章 机器学习                                                          */
/* 对应教材：section10.tex                                                  */
/* 内容：无监督学习（PCA、因子分析、聚类）、监督学习                         */
/*       （正则化回归、决策树、随机森林、梯度提升树、神经网络）、SHAP值      */
/****************************************************************************/

/****************************************************************************/
/* 10.1 数据生成                                                           */
/****************************************************************************/
data ml_data;
  call streaminit(123);
  do i = 1 to 1000;
    x1 = rand('NORMAL', 0, 1);
    x2 = rand('NORMAL', 0, 1);
    x3 = rand('NORMAL', 0, 1);
    x4 = rand('NORMAL', 0, 1);
    x5 = rand('NORMAL', 0, 1);
    /* 分类目标 */
    eta = 0.5 + 1.2 * x1 - 0.8 * x2 + 0.5 * x3;
    p = 1 / (1 + exp(-eta));
    y_class = (rand('UNIFORM') < p);
    /* 回归目标 */
    y_reg = 2 + 1.5 * x1 - 0.8 * x2 + 0.5 * x3 +
            0.3 * x4 + rand('NORMAL', 0, 0.5);
    output;
  end;
run;

title "数据描述";
proc means data=ml_data mean std min max;
  var x1 x2 x3 x4 x5 y_reg;
run;
title;


/****************************************************************************/
/* 10.2 无监督学习：主成分分析（PCA）                                       */
/****************************************************************************/
title "主成分分析";
proc princomp data=ml_data out=pca_out std;
  var x1 x2 x3 x4 x5;
run;
title;

title "PCA碎石图";
proc sgplot data=pca_out;
  /* 需要先生成特征值数据 */
run;
title;

/* 使用PROC FACTOR */
title "因子分析";
proc factor data=ml_data method=principal nfact=2
            rotate=varimax scree;
  var x1 x2 x3 x4 x5;
run;
title;


/****************************************************************************/
/* 10.3 无监督学习：聚类分析                                               */
/****************************************************************************/

/* 10.3.1 K-means聚类 */
title "K-means聚类";
proc fastclus data=ml_data maxclusters=3 maxiter=100
              out=cluster_out;
  var x1 x2 x3 x4 x5;
run;
title;

title "聚类结果";
proc sgplot data=cluster_out;
  scatter x=x1 y=x2 / group=CLUSTER;
run;
title;

/* 10.3.2 层次聚类 */
title "层次聚类";
proc cluster data=ml_data method=ward std
             outtree=tree_out;
  var x1 x2 x3 x4 x5;
run;
title;

title "聚类树状图";
proc tree data=tree_out;
run;
title;


/****************************************************************************/
/* 10.4 监督学习：正则化回归                                               */
/****************************************************************************/

/* 10.4.1 岭回归 */
title "岭回归";
proc reg data=ml_data outest=ridge_out ridge=0.5;
  model y_reg = x1 x2 x3 x4 x5;
run;
title;

/* 10.4.2 LASSO回归 */
title "LASSO回归";
proc glmselect data=ml_data
               method=lasso(choose=cvex) cvmethod=split(10);
  model y_reg = x1 x2 x3 x4 x5 / selection=lasso;
run;
title;

/* 10.4.3 Elastic Net */
title "Elastic Net";
proc glmselect data=ml_data
               method=elasticnet(choose=cvex) cvmethod=split(10);
  model y_reg = x1 x2 x3 x4 x5 / selection=elasticnet;
run;
title;


/****************************************************************************/
/* 10.5 监督学习：决策树                                                   */
/****************************************************************************/

/* 10.5.1 分类树 */
title "分类决策树";
proc hpsplit data=ml_data seed=123;
  class y_class;
  model y_class(event='1') = x1 x2 x3 x4 x5;
  grow entropy;
  prune costcomplexity;
  partition fraction(validate=0.3);
run;
title;

/* 10.5.2 回归树 */
title "回归决策树";
proc hpsplit data=ml_data seed=123;
  model y_reg = x1 x2 x3 x4 x5;
  grow variance;
  prune costcomplexity;
  partition fraction(validate=0.3);
run;
title;


/****************************************************************************/
/* 10.6 监督学习：随机森林                                                 */
/****************************************************************************/
title "随机森林（分类）";
proc hpforest data=ml_data seed=123
              maxtrees=100 vars_to_try=2;
  class y_class;
  model y_class(event='1') = x1 x2 x3 x4 x5;
run;
title;

title "随机森林（回归）";
proc hpforest data=ml_data seed=123
              maxtrees=100 vars_to_try=2;
  model y_reg = x1 x2 x3 x4 x5;
run;
title;


/****************************************************************************/
/* 10.7 监督学习：梯度提升树                                               */
/****************************************************************************/
title "梯度提升树（分类）";
proc gradboost data=ml_data seed=123;
  class y_class;
  model y_class(event='1') = x1 x2 x3 x4 x5;
  partition fraction(validate=0.3);
  autotune objective=validationmce;
run;
title;

title "梯度提升树（回归）";
proc gradboost data=ml_data seed=123;
  model y_reg = x1 x2 x3 x4 x5;
  partition fraction(validate=0.3);
  autotune objective=validationase;
run;
title;


/****************************************************************************/
/* 10.8 监督学习：神经网络                                                 */
/****************************************************************************/
title "神经网络（分类）";
proc nnet data=ml_data seed=123;
  class y_class;
  model y_class(event='1') = x1 x2 x3 x4 x5;
  hiddenunits 5 3;
  partition fraction(validate=0.3);
run;
title;

title "神经网络（回归）";
proc nnet data=ml_data seed=123;
  model y_reg = x1 x2 x3 x4 x5;
  hiddenunits 5 3;
  partition fraction(validate=0.3);
run;
title;


/****************************************************************************/
/* 10.9 模型比较                                                           */
/****************************************************************************/
/* 划分训练集和测试集 */
data ml_split;
  set ml_data;
  if rand('UNIFORM', 123) < 0.7 then role = 'train';
  else role = 'test';
run;

/* Logistic回归 */
title "Logistic回归（基线模型）";
proc logistic data=ml_split;
  where role='train';
  model y_class(event='1') = x1 x2 x3 x4 x5;
  score data=ml_split out=logit_scored;
run;
title;

/* 计算AUC */
proc logistic data=logit_scored;
  class y_class;
  model y_class(event='1') = x1 x2 x3 x4 x5 / roc;
run;


/****************************************************************************/
/* 10.10 模型解释：变量重要性                                             */
/****************************************************************************/
title "随机森林变量重要性";
proc hpforest data=ml_data seed=123
              maxtrees=100 vars_to_try=2;
  model y_reg = x1 x2 x3 x4 x5 / importance;
run;
title;

title "梯度提升树变量重要性";
proc gradboost data=ml_data seed=123;
  model y_reg = x1 x2 x3 x4 x5;
  partition fraction(validate=0.3);
  assess importance;
run;
title;


/****************************************************************************/
/* 10.11 保险应用：车险定价                                               */
/****************************************************************************/
data auto_insurance;
  call streaminit(456);
  do policy = 1 to 2000;
    age = rand('UNIFORM', 18, 70);
    gender = rand('BERNOULLI', 0.5);
    vehicle_age = rand('UNIFORM', 0, 15);
    region = rand('INTEGER', 1, 4);
    no_claim_years = rand('INTEGER', 0, 10);

    /* 索赔频率 */
    eta_freq = -1 + 0.02 * (age - 40) - 0.1 * gender +
               0.1 * vehicle_age + 0.3 * (region=1) - 0.1 * no_claim_years;
    lambda = exp(eta_freq);
    n_claims = rand('POISSON', lambda);

    /* 索赔金额 */
    if n_claims > 0 then do;
      eta_sev = 8 + 0.01 * (age - 40) + 0.05 * vehicle_age;
      mu = exp(eta_sev);
      claim_amount = rand('GAMMA', 2, mu/2) * n_claims;
    end;
    else claim_amount = 0;

    output;
  end;
run;

title "车险数据描述";
proc means data=auto_insurance mean std min max;
  var age vehicle_age n_claims claim_amount;
run;
title;

/* 随机森林预测索赔次数 */
title "随机森林预测索赔次数";
proc hpforest data=auto_insurance seed=123
              maxtrees=100 vars_to_try=3;
  model n_claims = age gender vehicle_age region no_claim_years;
run;
title;

/* 梯度提升树预测索赔金额 */
title "梯度提升树预测索赔金额";
proc gradboost data=auto_insurance seed=123;
  model claim_amount = age gender vehicle_age region no_claim_years;
  partition fraction(validate=0.3);
run;
title;

if n_claims > 0 then do;
      eta_sev = 8 + 0.01 * (age - 40) + 0.05 * vehicle_age;
      mu = exp(eta_sev);
      claim_amount = rand('GAMMA', 2, mu/2) * n_claims;
    end;
    else claim_amount = 0;

    output;
  end;
run;

title "车险数据描述";
proc means data=auto_insurance mean std min max;
  var age vehicle_age n_claims claim_amount;
run;
title;

/* 随机森林预测索赔次数 */
title "随机森林预测索赔次数";
proc hpforest data=auto_insurance seed=123
              maxtrees=100 vars_to_try=3;
  model n_claims = age gender vehicle_age region no_claim_years;
run;
title;

/* 梯度提升树预测索赔金额 */
title "梯度提升树预测索赔金额";
proc gradboost data=auto_insurance seed=123;
  model claim_amount = age gender vehicle_age region no_claim_years;
  partition fraction(validate=0.3);
run;
title;

/****************************************************************************/
/* 第10章 机器学习                                                          */
/* 对应教材：section10.tex                                                  */
/* 内容：无监督学习（PCA、因子分析、聚类）、监督学习                         */
/*       （正则化回归、决策树、随机森林、梯度提升树、神经网络）、SHAP值      */
/****************************************************************************/

/****************************************************************************/
/* 10.1 数据生成                                                           */
/****************************************************************************/
data ml_data;
  call streaminit(123);
  do i = 1 to 1000;
    x1 = rand('NORMAL', 0, 1);
    x2 = rand('NORMAL', 0, 1);
    x3 = rand('NORMAL', 0, 1);
    x4 = rand('NORMAL', 0, 1);
    x5 = rand('NORMAL', 0, 1);
    /* 分类目标 */
    eta = 0.5 + 1.2 * x1 - 0.8 * x2 + 0.5 * x3;
    p = 1 / (1 + exp(-eta));
    y_class = (rand('UNIFORM') < p);
    /* 回归目标 */
    y_reg = 2 + 1.5 * x1 - 0.8 * x2 + 0.5 * x3 +
            0.3 * x4 + rand('NORMAL', 0, 0.5);
    output;
  end;
run;

title "数据描述";
proc means data=ml_data mean std min max;
  var x1 x2 x3 x4 x5 y_reg;
run;
title;


/****************************************************************************/
/* 10.2 无监督学习：主成分分析（PCA）                                       */
/****************************************************************************/
title "主成分分析";
proc princomp data=ml_data out=pca_out std;
  var x1 x2 x3 x4 x5;
run;
title;

title "PCA碎石图";
proc sgplot data=pca_out;
  /* 需要先生成特征值数据 */
run;
title;

/* 使用PROC FACTOR */
title "因子分析";
proc factor data=ml_data method=principal nfact=2
            rotate=varimax scree;
  var x1 x2 x3 x4 x5;
run;
title;


/****************************************************************************/
/* 10.3 无监督学习：聚类分析                                               */
/****************************************************************************/

/* 10.3.1 K-means聚类 */
title "K-means聚类";
proc fastclus data=ml_data maxclusters=3 maxiter=100
              out=cluster_out;
  var x1 x2 x3 x4 x5;
run;
title;

title "聚类结果";
proc sgplot data=cluster_out;
  scatter x=x1 y=x2 / group=CLUSTER;
run;
title;

/* 10.3.2 层次聚类 */
title "层次聚类";
proc cluster data=ml_data method=ward std
             outtree=tree_out;
  var x1 x2 x3 x4 x5;
run;
title;

title "聚类树状图";
proc tree data=tree_out;
run;
title;


/****************************************************************************/
/* 10.4 监督学习：正则化回归                                               */
/****************************************************************************/

/* 10.4.1 岭回归 */
title "岭回归";
proc reg data=ml_data outest=ridge_out ridge=0.5;
  model y_reg = x1 x2 x3 x4 x5;
run;
title;

/* 10.4.2 LASSO回归 */
title "LASSO回归";
proc glmselect data=ml_data
               method=lasso(choose=cvex) cvmethod=split(10);
  model y_reg = x1 x2 x3 x4 x5 / selection=lasso;
run;
title;

/* 10.4.3 Elastic Net */
title "Elastic Net";
proc glmselect data=ml_data
               method=elasticnet(choose=cvex) cvmethod=split(10);
  model y_reg = x1 x2 x3 x4 x5 / selection=elasticnet;
run;
title;


/****************************************************************************/
/* 10.5 监督学习：决策树                                                   */
/****************************************************************************/

/* 10.5.1 分类树 */
title "分类决策树";
proc hpsplit data=ml_data seed=123;
  class y_class;
  model y_class(event='1') = x1 x2 x3 x4 x5;
  grow entropy;
  prune costcomplexity;
  partition fraction(validate=0.3);
run;
title;

/* 10.5.2 回归树 */
title "回归决策树";
proc hpsplit data=ml_data seed=123;
  model y_reg = x1 x2 x3 x4 x5;
  grow variance;
  prune costcomplexity;
  partition fraction(validate=0.3);
run;
title;


/****************************************************************************/
/* 10.6 监督学习：随机森林                                                 */
/****************************************************************************/
title "随机森林（分类）";
proc hpforest data=ml_data seed=123
              maxtrees=100 vars_to_try=2;
  class y_class;
  model y_class(event='1') = x1 x2 x3 x4 x5;
run;
title;

title "随机森林（回归）";
proc hpforest data=ml_data seed=123
              maxtrees=100 vars_to_try=2;
  model y_reg = x1 x2 x3 x4 x5;
run;
title;


/****************************************************************************/
/* 10.7 监督学习：梯度提升树                                               */
/****************************************************************************/
title "梯度提升树（分类）";
proc gradboost data=ml_data seed=123;
  class y_class;
  model y_class(event='1') = x1 x2 x3 x4 x5;
  partition fraction(validate=0.3);
  autotune objective=validationmce;
run;
title;

title "梯度提升树（回归）";
proc gradboost data=ml_data seed=123;
  model y_reg = x1 x2 x3 x4 x5;
  partition fraction(validate=0.3);
  autotune objective=validationase;
run;
title;


/****************************************************************************/
/* 10.8 监督学习：神经网络                                                 */
/****************************************************************************/
title "神经网络（分类）";
proc nnet data=ml_data seed=123;
  class y_class;
  model y_class(event='1') = x1 x2 x3 x4 x5;
  hiddenunits 5 3;
  partition fraction(validate=0.3);
run;
title;

title "神经网络（回归）";
proc nnet data=ml_data seed=123;
  model y_reg = x1 x2 x3 x4 x5;
  hiddenunits 5 3;
  partition fraction(validate=0.3);
run;
title;


/****************************************************************************/
/* 10.9 模型比较                                                           */
/****************************************************************************/
/* 划分训练集和测试集 */
data ml_split;
  set ml_data;
  if rand('UNIFORM', 123) < 0.7 then role = 'train';
  else role = 'test';
run;

/* Logistic回归 */
title "Logistic回归（基线模型）";
proc logistic data=ml_split;
  where role='train';
  model y_class(event='1') = x1 x2 x3 x4 x5;
  score data=ml_split out=logit_scored;
run;
title;

/* 计算AUC */
proc logistic data=logit_scored;
  class y_class;
  model y_class(event='1') = x1 x2 x3 x4 x5 / roc;
run;


/****************************************************************************/
/* 10.10 模型解释：变量重要性                                             */
/****************************************************************************/
title "随机森林变量重要性";
proc hpforest data=ml_data seed=123
              maxtrees=100 vars_to_try=2;
  model y_reg = x1 x2 x3 x4 x5 / importance;
run;
title;

title "梯度提升树变量重要性";
proc gradboost data=ml_data seed=123;
  model y_reg = x1 x2 x3 x4 x5;
  partition fraction(validate=0.3);
  assess importance;
run;
title;


/****************************************************************************/
/* 10.11 保险应用：车险定价                                               */
/****************************************************************************/
data auto_insurance;
  call streaminit(456);
  do policy = 1 to 2000;
    age = rand('UNIFORM', 18, 70);
    gender = rand('BERNOULLI', 0.5);
    vehicle_age = rand('UNIFORM', 0, 15);
    region = rand('INTEGER', 1, 4);
    no_claim_years = rand('INTEGER', 0, 10);

    /* 索赔频率 */
    eta_freq = -1 + 0.02 * (age - 40) - 0.1 * gender +
               0.1 * vehicle_age + 0.3 * (region=1) - 0.1 * no_claim_years;
    lambda = exp(eta_freq);
    n_claims = rand('POISSON', lambda);

    /* 索赔金额 */
    if n_claims > 0 then do;
      eta_sev = 8 + 0.01 * (age - 40) + 0.05 * vehicle_age;
      mu = exp(eta_sev);
      claim_amount = rand('GAMMA', 2, mu/2) * n_claims;
    end;
    else claim_amount = 0;

    output;
  end;
run;

title "车险数据描述";
proc means data=auto_insurance mean std min max;
  var age vehicle_age n_claims claim_amount;
run;
title;

/* 随机森林预测索赔次数 */
title "随机森林预测索赔次数";
proc hpforest data=auto_insurance seed=123
              maxtrees=100 vars_to_try=3;
  model n_claims = age gender vehicle_age region no_claim_years;
run;
title;

/* 梯度提升树预测索赔金额 */
title "梯度提升树预测索赔金额";
proc gradboost data=auto_insurance seed=123;
  model claim_amount = age gender vehicle_age region no_claim_years;
  partition fraction(validate=0.3);
run;
title;

if n_claims > 0 then do;
      eta_sev = 8 + 0.01 * (age - 40) + 0.05 * vehicle_age;
      mu = exp(eta_sev);
      claim_amount = rand('GAMMA', 2, mu/2) * n_claims;
    end;
    else claim_amount = 0;

    output;
  end;
run;

title "车险数据描述";
proc means data=auto_insurance mean std min max;
  var age vehicle_age n_claims claim_amount;
run;
title;

/* 随机森林预测索赔次数 */
title "随机森林预测索赔次数";
proc hpforest data=auto_insurance seed=123
              maxtrees=100 vars_to_try=3;
  model n_claims = age gender vehicle_age region no_claim_years;
run;
title;

/* 梯度提升树预测索赔金额 */
title "梯度提升树预测索赔金额";
proc gradboost data=auto_insurance seed=123;
  model claim_amount = age gender vehicle_age region no_claim_years;
  partition fraction(validate=0.3);
run;
title;

/****************************************************************************/
/* 第10章 机器学习                                                          */
/* 对应教材：section10.tex                                                  */
/* 内容：无监督学习（PCA、因子分析、聚类）、监督学习                         */
/*       （正则化回归、决策树、随机森林、梯度提升树、神经网络）、SHAP值      */
/****************************************************************************/

/****************************************************************************/
/* 10.1 数据生成                                                           */
/****************************************************************************/
data ml_data;
  call streaminit(123);
  do i = 1 to 1000;
    x1 = rand('NORMAL', 0, 1);
    x2 = rand('NORMAL', 0, 1);
    x3 = rand('NORMAL', 0, 1);
    x4 = rand('NORMAL', 0, 1);
    x5 = rand('NORMAL', 0, 1);
    /* 分类目标 */
    eta = 0.5 + 1.2 * x1 - 0.8 * x2 + 0.5 * x3;
    p = 1 / (1 + exp(-eta));
    y_class = (rand('UNIFORM') < p);
    /* 回归目标 */
    y_reg = 2 + 1.5 * x1 - 0.8 * x2 + 0.5 * x3 +
            0.3 * x4 + rand('NORMAL', 0, 0.5);
    output;
  end;
run;

title "数据描述";
proc means data=ml_data mean std min max;
  var x1 x2 x3 x4 x5 y_reg;
run;
title;


/****************************************************************************/
/* 10.2 无监督学习：主成分分析（PCA）                                       */
/****************************************************************************/
title "主成分分析";
proc princomp data=ml_data out=pca_out std;
  var x1 x2 x3 x4 x5;
run;
title;

title "PCA碎石图";
proc sgplot data=pca_out;
  /* 需要先生成特征值数据 */
run;
title;

/* 使用PROC FACTOR */
title "因子分析";
proc factor data=ml_data method=principal nfact=2
            rotate=varimax scree;
  var x1 x2 x3 x4 x5;
run;
title;


/****************************************************************************/
/* 10.3 无监督学习：聚类分析                                               */
/****************************************************************************/

/* 10.3.1 K-means聚类 */
title "K-means聚类";
proc fastclus data=ml_data maxclusters=3 maxiter=100
              out=cluster_out;
  var x1 x2 x3 x4 x5;
run;
title;

title "聚类结果";
proc sgplot data=cluster_out;
  scatter x=x1 y=x2 / group=CLUSTER;
run;
title;

/* 10.3.2 层次聚类 */
title "层次聚类";
proc cluster data=ml_data method=ward std
             outtree=tree_out;
  var x1 x2 x3 x4 x5;
run;
title;

title "聚类树状图";
proc tree data=tree_out;
run;
title;


/****************************************************************************/
/* 10.4 监督学习：正则化回归                                               */
/****************************************************************************/

/* 10.4.1 岭回归 */
title "岭回归";
proc reg data=ml_data outest=ridge_out ridge=0.5;
  model y_reg = x1 x2 x3 x4 x5;
run;
title;

/* 10.4.2 LASSO回归 */
title "LASSO回归";
proc glmselect data=ml_data
               method=lasso(choose=cvex) cvmethod=split(10);
  model y_reg = x1 x2 x3 x4 x5 / selection=lasso;
run;
title;

/* 10.4.3 Elastic Net */
title "Elastic Net";
proc glmselect data=ml_data
               method=elasticnet(choose=cvex) cvmethod=split(10);
  model y_reg = x1 x2 x3 x4 x5 / selection=elasticnet;
run;
title;


/****************************************************************************/
/* 10.5 监督学习：决策树                                                   */
/****************************************************************************/

/* 10.5.1 分类树 */
title "分类决策树";
proc hpsplit data=ml_data seed=123;
  class y_class;
  model y_class(event='1') = x1 x2 x3 x4 x5;
  grow entropy;
  prune costcomplexity;
  partition fraction(validate=0.3);
run;
title;

/* 10.5.2 回归树 */
title "回归决策树";
proc hpsplit data=ml_data seed=123;
  model y_reg = x1 x2 x3 x4 x5;
  grow variance;
  prune costcomplexity;
  partition fraction(validate=0.3);
run;
title;


/****************************************************************************/
/* 10.6 监督学习：随机森林                                                 */
/****************************************************************************/
title "随机森林（分类）";
proc hpforest data=ml_data seed=123
              maxtrees=100 vars_to_try=2;
  class y_class;
  model y_class(event='1') = x1 x2 x3 x4 x5;
run;
title;

title "随机森林（回归）";
proc hpforest data=ml_data seed=123
              maxtrees=100 vars_to_try=2;
  model y_reg = x1 x2 x3 x4 x5;
run;
title;


/****************************************************************************/
/* 10.7 监督学习：梯度提升树                                               */
/****************************************************************************/
title "梯度提升树（分类）";
proc gradboost data=ml_data seed=123;
  class y_class;
  model y_class(event='1') = x1 x2 x3 x4 x5;
  partition fraction(validate=0.3);
  autotune objective=validationmce;
run;
title;

title "梯度提升树（回归）";
proc gradboost data=ml_data seed=123;
  model y_reg = x1 x2 x3 x4 x5;
  partition fraction(validate=0.3);
  autotune objective=validationase;
run;
title;


/****************************************************************************/
/* 10.8 监督学习：神经网络                                                 */
/****************************************************************************/
title "神经网络（分类）";
proc nnet data=ml_data seed=123;
  class y_class;
  model y_class(event='1') = x1 x2 x3 x4 x5;
  hiddenunits 5 3;
  partition fraction(validate=0.3);
run;
title;

title "神经网络（回归）";
proc nnet data=ml_data seed=123;
  model y_reg = x1 x2 x3 x4 x5;
  hiddenunits 5 3;
  partition fraction(validate=0.3);
run;
title;


/****************************************************************************/
/* 10.9 模型比较                                                           */
/****************************************************************************/
/* 划分训练集和测试集 */
data ml_split;
  set ml_data;
  if rand('UNIFORM', 123) < 0.7 then role = 'train';
  else role = 'test';
run;

/* Logistic回归 */
title "Logistic回归（基线模型）";
proc logistic data=ml_split;
  where role='train';
  model y_class(event='1') = x1 x2 x3 x4 x5;
  score data=ml_split out=logit_scored;
run;
title;

/* 计算AUC */
proc logistic data=logit_scored;
  class y_class;
  model y_class(event='1') = x1 x2 x3 x4 x5 / roc;
run;


/****************************************************************************/
/* 10.10 模型解释：变量重要性                                             */
/****************************************************************************/
title "随机森林变量重要性";
proc hpforest data=ml_data seed=123
              maxtrees=100 vars_to_try=2;
  model y_reg = x1 x2 x3 x4 x5 / importance;
run;
title;

title "梯度提升树变量重要性";
proc gradboost data=ml_data seed=123;
  model y_reg = x1 x2 x3 x4 x5;
  partition fraction(validate=0.3);
  assess importance;
run;
title;


/****************************************************************************/
/* 10.11 保险应用：车险定价                                               */
/****************************************************************************/
data auto_insurance;
  call streaminit(456);
  do policy = 1 to 2000;
    age = rand('UNIFORM', 18, 70);
    gender = rand('BERNOULLI', 0.5);
    vehicle_age = rand('UNIFORM', 0, 15);
    region = rand('INTEGER', 1, 4);
    no_claim_years = rand('INTEGER', 0, 10);

    /* 索赔频率 */
    eta_freq = -1 + 0.02 * (age - 40) - 0.1 * gender +
               0.1 * vehicle_age + 0.3 * (region=1) - 0.1 * no_claim_years;
    lambda = exp(eta_freq);
    n_claims = rand('POISSON', lambda);

    /* 索赔金额 */
    if n_claims > 0 then do;
      eta_sev = 8 + 0.01 * (age - 40) + 0.05 * vehicle_age;
      mu = exp(eta_sev);
      claim_amount = rand('GAMMA', 2, mu/2) * n_claims;
    end;
    else claim_amount = 0;

    output;
  end;
run;

title "车险数据描述";
proc means data=auto_insurance mean std min max;
  var age vehicle_age n_claims claim_amount;
run;
title;

/* 随机森林预测索赔次数 */
title "随机森林预测索赔次数";
proc hpforest data=auto_insurance seed=123
              maxtrees=100 vars_to_try=3;
  model n_claims = age gender vehicle_age region no_claim_years;
run;
title;

/* 梯度提升树预测索赔金额 */
title "梯度提升树预测索赔金额";
proc gradboost data=auto_insurance seed=123;
  model claim_amount = age gender vehicle_age region no_claim_years;
  partition fraction(validate=0.3);
run;
title;

/****************************************************************************/
/* 第10章 机器学习                                                          */
/* 对应教材：section10.tex                                                  */
/* 内容：无监督学习（PCA、因子分析、聚类）、监督学习                         */
/*       （正则化回归、决策树、随机森林、梯度提升树、神经网络）、SHAP值      */
/****************************************************************************/

/****************************************************************************/
/* 10.1 数据生成                                                           */
/****************************************************************************/
data ml_data;
  call streaminit(123);
  do i = 1 to 1000;
    x1 = rand('NORMAL', 0, 1);
    x2 = rand('NORMAL', 0, 1);
    x3 = rand('NORMAL', 0, 1);
    x4 = rand('NORMAL', 0, 1);
    x5 = rand('NORMAL', 0, 1);
    /* 分类目标 */
    eta = 0.5 + 1.2 * x1 - 0.8 * x2 + 0.5 * x3;
    p = 1 / (1 + exp(-eta));
    y_class = (rand('UNIFORM') < p);
    /* 回归目标 */
    y_reg = 2 + 1.5 * x1 - 0.8 * x2 + 0.5 * x3 +
            0.3 * x4 + rand('NORMAL', 0, 0.5);
    output;
  end;
run;

title "数据描述";
proc means data=ml_data mean std min max;
  var x1 x2 x3 x4 x5 y_reg;
run;
title;


/****************************************************************************/
/* 10.2 无监督学习：主成分分析（PCA）                                       */
/****************************************************************************/
title "主成分分析";
proc princomp data=ml_data out=pca_out std;
  var x1 x2 x3 x4 x5;
run;
title;

title "PCA碎石图";
proc sgplot data=pca_out;
  /* 需要先生成特征值数据 */
run;
title;

/* 使用PROC FACTOR */
title "因子分析";
proc factor data=ml_data method=principal nfact=2
            rotate=varimax scree;
  var x1 x2 x3 x4 x5;
run;
title;


/****************************************************************************/
/* 10.3 无监督学习：聚类分析                                               */
/****************************************************************************/

/* 10.3.1 K-means聚类 */
title "K-means聚类";
proc fastclus data=ml_data maxclusters=3 maxiter=100
              out=cluster_out;
  var x1 x2 x3 x4 x5;
run;
title;

title "聚类结果";
proc sgplot data=cluster_out;
  scatter x=x1 y=x2 / group=CLUSTER;
run;
title;

/* 10.3.2 层次聚类 */
title "层次聚类";
proc cluster data=ml_data method=ward std
             outtree=tree_out;
  var x1 x2 x3 x4 x5;
run;
title;

title "聚类树状图";
proc tree data=tree_out;
run;
title;


/****************************************************************************/
/* 10.4 监督学习：正则化回归                                               */
/****************************************************************************/

/* 10.4.1 岭回归 */
title "岭回归";
proc reg data=ml_data outest=ridge_out ridge=0.5;
  model y_reg = x1 x2 x3 x4 x5;
run;
title;

/* 10.4.2 LASSO回归 */
title "LASSO回归";
proc glmselect data=ml_data
               method=lasso(choose=cvex) cvmethod=split(10);
  model y_reg = x1 x2 x3 x4 x5 / selection=lasso;
run;
title;

/* 10.4.3 Elastic Net */
title "Elastic Net";
proc glmselect data=ml_data
               method=elasticnet(choose=cvex) cvmethod=split(10);
  model y_reg = x1 x2 x3 x4 x5 / selection=elasticnet;
run;
title;


/****************************************************************************/
/* 10.5 监督学习：决策树                                                   */
/****************************************************************************/

/* 10.5.1 分类树 */
title "分类决策树";
proc hpsplit data=ml_data seed=123;
  class y_class;
  model y_class(event='1') = x1 x2 x3 x4 x5;
  grow entropy;
  prune costcomplexity;
  partition fraction(validate=0.3);
run;
title;

/* 10.5.2 回归树 */
title "回归决策树";
proc hpsplit data=ml_data seed=123;
  model y_reg = x1 x2 x3 x4 x5;
  grow variance;
  prune costcomplexity;
  partition fraction(validate=0.3);
run;
title;


/****************************************************************************/
/* 10.6 监督学习：随机森林                                                 */
/****************************************************************************/
title "随机森林（分类）";
proc hpforest data=ml_data seed=123
              maxtrees=100 vars_to_try=2;
  class y_class;
  model y_class(event='1') = x1 x2 x3 x4 x5;
run;
title;

title "随机森林（回归）";
proc hpforest data=ml_data seed=123
              maxtrees=100 vars_to_try=2;
  model y_reg = x1 x2 x3 x4 x5;
run;
title;


/****************************************************************************/
/* 10.7 监督学习：梯度提升树                                               */
/****************************************************************************/
title "梯度提升树（分类）";
proc gradboost data=ml_data seed=123;
  class y_class;
  model y_class(event='1') = x1 x2 x3 x4 x5;
  partition fraction(validate=0.3);
  autotune objective=validationmce;
run;
title;

title "梯度提升树（回归）";
proc gradboost data=ml_data seed=123;
  model y_reg = x1 x2 x3 x4 x5;
  partition fraction(validate=0.3);
  autotune objective=validationase;
run;
title;


/****************************************************************************/
/* 10.8 监督学习：神经网络                                                 */
/****************************************************************************/
title "神经网络（分类）";
proc nnet data=ml_data seed=123;
  class y_class;
  model y_class(event='1') = x1 x2 x3 x4 x5;
  hiddenunits 5 3;
  partition fraction(validate=0.3);
run;
title;

title "神经网络（回归）";
proc nnet data=ml_data seed=123;
  model y_reg = x1 x2 x3 x4 x5;
  hiddenunits 5 3;
  partition fraction(validate=0.3);
run;
title;


/****************************************************************************/
/* 10.9 模型比较                                                           */
/****************************************************************************/
/* 划分训练集和测试集 */
data ml_split;
  set ml_data;
  if rand('UNIFORM', 123) < 0.7 then role = 'train';
  else role = 'test';
run;

/* Logistic回归 */
title "Logistic回归（基线模型）";
proc logistic data=ml_split;
  where role='train';
  model y_class(event='1') = x1 x2 x3 x4 x5;
  score data=ml_split out=logit_scored;
run;
title;

/* 计算AUC */
proc logistic data=logit_scored;
  class y_class;
  model y_class(event='1') = x1 x2 x3 x4 x5 / roc;
run;


/****************************************************************************/
/* 10.10 模型解释：变量重要性                                             */
/****************************************************************************/
title "随机森林变量重要性";
proc hpforest data=ml_data seed=123
              maxtrees=100 vars_to_try=2;
  model y_reg = x1 x2 x3 x4 x5 / importance;
run;
title;

title "梯度提升树变量重要性";
proc gradboost data=ml_data seed=123;
  model y_reg = x1 x2 x3 x4 x5;
  partition fraction(validate=0.3);
  assess importance;
run;
title;


/****************************************************************************/
/* 10.11 保险应用：车险定价                                               */
/****************************************************************************/
data auto_insurance;
  call streaminit(456);
  do policy = 1 to 2000;
    age = rand('UNIFORM', 18, 70);
    gender = rand('BERNOULLI', 0.5);
    vehicle_age = rand('UNIFORM', 0, 15);
    region = rand('INTEGER', 1, 4);
    no_claim_years = rand('INTEGER', 0, 10);

    /* 索赔频率 */
    eta_freq = -1 + 0.02 * (age - 40) - 0.1 * gender +
               0.1 * vehicle_age + 0.3 * (region=1) - 0.1 * no_claim_years;
    lambda = exp(eta_freq);
    n_claims = rand('POISSON', lambda);

    /* 索赔金额 */
    if n_claims > 0 then do;
      eta_sev = 8 + 0.01 * (age - 40) + 0.05 * vehicle_age;
      mu = exp(eta_sev);
      claim_amount = rand('GAMMA', 2, mu/2) * n_claims;
    end;
    else claim_amount = 0;

    output;
  end;
run;

title "车险数据描述";
proc means data=auto_insurance mean std min max;
  var age vehicle_age n_claims claim_amount;
run;
title;

/* 随机森林预测索赔次数 */
title "随机森林预测索赔次数";
proc hpforest data=auto_insurance seed=123
              maxtrees=100 vars_to_try=3;
  model n_claims = age gender vehicle_age region no_claim_years;
run;
title;

/* 梯度提升树预测索赔金额 */
title "梯度提升树预测索赔金额";
proc gradboost data=auto_insurance seed=123;
  model claim_amount = age gender vehicle_age region no_claim_years;
  partition fraction(validate=0.3);
run;
title;

/****************************************************************************/
/* 第10章 机器学习                                                          */
/* 对应教材：section10.tex                                                  */
/* 内容：无监督学习（PCA、因子分析、聚类）、监督学习                         */
/*       （正则化回归、决策树、随机森林、梯度提升树、神经网络）、SHAP值      */
/****************************************************************************/

/****************************************************************************/
/* 10.1 数据生成                                                           */
/****************************************************************************/
data ml_data;
  call streaminit(123);
  do i = 1 to 1000;
    x1 = rand('NORMAL', 0, 1);
    x2 = rand('NORMAL', 0, 1);
    x3 = rand('NORMAL', 0, 1);
    x4 = rand('NORMAL', 0, 1);
    x5 = rand('NORMAL', 0, 1);
    /* 分类目标 */
    eta = 0.5 + 1.2 * x1 - 0.8 * x2 + 0.5 * x3;
    p = 1 / (1 + exp(-eta));
    y_class = (rand('UNIFORM') < p);
    /* 回归目标 */
    y_reg = 2 + 1.5 * x1 - 0.8 * x2 + 0.5 * x3 +
            0.3 * x4 + rand('NORMAL', 0, 0.5);
    output;
  end;
run;

title "数据描述";
proc means data=ml_data mean std min max;
  var x1 x2 x3 x4 x5 y_reg;
run;
title;


/****************************************************************************/
/* 10.2 无监督学习：主成分分析（PCA）                                       */
/****************************************************************************/
title "主成分分析";
proc princomp data=ml_data out=pca_out std;
  var x1 x2 x3 x4 x5;
run;
title;

title "PCA碎石图";
proc sgplot data=pca_out;
  /* 需要先生成特征值数据 */
run;
title;

/* 使用PROC FACTOR */
title "因子分析";
proc factor data=ml_data method=principal nfact=2
            rotate=varimax scree;
  var x1 x2 x3 x4 x5;
run;
title;


/****************************************************************************/
/* 10.3 无监督学习：聚类分析                                               */
/****************************************************************************/

/* 10.3.1 K-means聚类 */
title "K-means聚类";
proc fastclus data=ml_data maxclusters=3 maxiter=100
              out=cluster_out;
  var x1 x2 x3 x4 x5;
run;
title;

title "聚类结果";
proc sgplot data=cluster_out;
  scatter x=x1 y=x2 / group=CLUSTER;
run;
title;

/* 10.3.2 层次聚类 */
title "层次聚类";
proc cluster data=ml_data method=ward std
             outtree=tree_out;
  var x1 x2 x3 x4 x5;
run;
title;

title "聚类树状图";
proc tree data=tree_out;
run;
title;


/****************************************************************************/
/* 10.4 监督学习：正则化回归                                               */
/****************************************************************************/

/* 10.4.1 岭回归 */
title "岭回归";
proc reg data=ml_data outest=ridge_out ridge=0.5;
  model y_reg = x1 x2 x3 x4 x5;
run;
title;

/* 10.4.2 LASSO回归 */
title "LASSO回归";
proc glmselect data=ml_data
               method=lasso(choose=cvex) cvmethod=split(10);
  model y_reg = x1 x2 x3 x4 x5 / selection=lasso;
run;
title;

/* 10.4.3 Elastic Net */
title "Elastic Net";
proc glmselect data=ml_data
               method=elasticnet(choose=cvex) cvmethod=split(10);
  model y_reg = x1 x2 x3 x4 x5 / selection=elasticnet;
run;
title;


/****************************************************************************/
/* 10.5 监督学习：决策树                                                   */
/****************************************************************************/

/* 10.5.1 分类树 */
title "分类决策树";
proc hpsplit data=ml_data seed=123;
  class y_class;
  model y_class(event='1') = x1 x2 x3 x4 x5;
  grow entropy;
  prune costcomplexity;
  partition fraction(validate=0.3);
run;
title;

/* 10.5.2 回归树 */
title "回归决策树";
proc hpsplit data=ml_data seed=123;
  model y_reg = x1 x2 x3 x4 x5;
  grow variance;
  prune costcomplexity;
  partition fraction(validate=0.3);
run;
title;


/****************************************************************************/
/* 10.6 监督学习：随机森林                                                 */
/****************************************************************************/
title "随机森林（分类）";
proc hpforest data=ml_data seed=123
              maxtrees=100 vars_to_try=2;
  class y_class;
  model y_class(event='1') = x1 x2 x3 x4 x5;
run;
title;

title "随机森林（回归）";
proc hpforest data=ml_data seed=123
              maxtrees=100 vars_to_try=2;
  model y_reg = x1 x2 x3 x4 x5;
run;
title;


/****************************************************************************/
/* 10.7 监督学习：梯度提升树                                               */
/****************************************************************************/
title "梯度提升树（分类）";
proc gradboost data=ml_data seed=123;
  class y_class;
  model y_class(event='1') = x1 x2 x3 x4 x5;
  partition fraction(validate=0.3);
  autotune objective=validationmce;
run;
title;

title "梯度提升树（回归）";
proc gradboost data=ml_data seed=123;
  model y_reg = x1 x2 x3 x4 x5;
  partition fraction(validate=0.3);
  autotune objective=validationase;
run;
title;


/****************************************************************************/
/* 10.8 监督学习：神经网络                                                 */
/****************************************************************************/
title "神经网络（分类）";
proc nnet data=ml_data seed=123;
  class y_class;
  model y_class(event='1') = x1 x2 x3 x4 x5;
  hiddenunits 5 3;
  partition fraction(validate=0.3);
run;
title;

title "神经网络（回归）";
proc nnet data=ml_data seed=123;
  model y_reg = x1 x2 x3 x4 x5;
  hiddenunits 5 3;
  partition fraction(validate=0.3);
run;
title;


/****************************************************************************/
/* 10.9 模型比较                                                           */
/****************************************************************************/
/* 划分训练集和测试集 */
data ml_split;
  set ml_data;
  if rand('UNIFORM', 123) < 0.7 then role = 'train';
  else role = 'test';
run;

/* Logistic回归 */
title "Logistic回归（基线模型）";
proc logistic data=ml_split;
  where role='train';
  model y_class(event='1') = x1 x2 x3 x4 x5;
  score data=ml_split out=logit_scored;
run;
title;

/* 计算AUC */
proc logistic data=logit_scored;
  class y_class;
  model y_class(event='1') = x1 x2 x3 x4 x5 / roc;
run;


/****************************************************************************/
/* 10.10 模型解释：变量重要性                                             */
/****************************************************************************/
title "随机森林变量重要性";
proc hpforest data=ml_data seed=123
              maxtrees=100 vars_to_try=2;
  model y_reg = x1 x2 x3 x4 x5 / importance;
run;
title;

title "梯度提升树变量重要性";
proc gradboost data=ml_data seed=123;
  model y_reg = x1 x2 x3 x4 x5;
  partition fraction(validate=0.3);
  assess importance;
run;
title;


/****************************************************************************/
/* 10.11 保险应用：车险定价                                               */
/****************************************************************************/
data auto_insurance;
  call streaminit(456);
  do policy = 1 to 2000;
    age = rand('UNIFORM', 18, 70);
    gender = rand('BERNOULLI', 0.5);
    vehicle_age = rand('UNIFORM', 0, 15);
    region = rand('INTEGER', 1, 4);
    no_claim_years = rand('INTEGER', 0, 10);

    /* 索赔频率 */
    eta_freq = -1 + 0.02 * (age - 40) - 0.1 * gender +
               0.1 * vehicle_age + 0.3 * (region=1) - 0.1 * no_claim_years;
    lambda = exp(eta_freq);
    n_claims = rand('POISSON', lambda);

    /* 索赔金额 */
    if n_claims > 0 then do;
      eta_sev = 8 + 0.01 * (age - 40) + 0.05 * vehicle_age;
      mu = exp(eta_sev);
      claim_amount = rand('GAMMA', 2, mu/2) * n_claims;
    end;
    else claim_amount = 0;

    output;
  end;
run;

title "车险数据描述";
proc means data=auto_insurance mean std min max;
  var age vehicle_age n_claims claim_amount;
run;
title;

/* 随机森林预测索赔次数 */
title "随机森林预测索赔次数";
proc hpforest data=auto_insurance seed=123
              maxtrees=100 vars_to_try=3;
  model n_claims = age gender vehicle_age region no_claim_years;
run;
title;

/* 梯度提升树预测索赔金额 */
title "梯度提升树预测索赔金额";
proc gradboost data=auto_insurance seed=123;
  model claim_amount = age gender vehicle_age region no_claim_years;
  partition fraction(validate=0.3);
run;
title;

if n_claims > 0 then do;
      eta_sev = 8 + 0.01 * (age - 40) + 0.05 * vehicle_age;
      mu = exp(eta_sev);
      claim_amount = rand('GAMMA', 2, mu/2) * n_claims;
    end;
    else claim_amount = 0;

    output;
  end;
run;

title "车险数据描述";
proc means data=auto_insurance mean std min max;
  var age vehicle_age n_claims claim_amount;
run;
title;

/* 随机森林预测索赔次数 */
title "随机森林预测索赔次数";
proc hpforest data=auto_insurance seed=123
              maxtrees=100 vars_to_try=3;
  model n_claims = age gender vehicle_age region no_claim_years;
run;
title;

/* 梯度提升树预测索赔金额 */
title "梯度提升树预测索赔金额";
proc gradboost data=auto_insurance seed=123;
  model claim_amount = age gender vehicle_age region no_claim_years;
  partition fraction(validate=0.3);
run;
title;

/****************************************************************************/
/* 第10章 机器学习                                                          */
/* 对应教材：section10.tex                                                  */
/* 内容：无监督学习（PCA、因子分析、聚类）、监督学习                         */
/*       （正则化回归、决策树、随机森林、梯度提升树、神经网络）、SHAP值      */
/****************************************************************************/

/****************************************************************************/
/* 10.1 数据生成                                                           */
/****************************************************************************/
data ml_data;
  call streaminit(123);
  do i = 1 to 1000;
    x1 = rand('NORMAL', 0, 1);
    x2 = rand('NORMAL', 0, 1);
    x3 = rand('NORMAL', 0, 1);
    x4 = rand('NORMAL', 0, 1);
    x5 = rand('NORMAL', 0, 1);
    /* 分类目标 */
    eta = 0.5 + 1.2 * x1 - 0.8 * x2 + 0.5 * x3;
    p = 1 / (1 + exp(-eta));
    y_class = (rand('UNIFORM') < p);
    /* 回归目标 */
    y_reg = 2 + 1.5 * x1 - 0.8 * x2 + 0.5 * x3 +
            0.3 * x4 + rand('NORMAL', 0, 0.5);
    output;
  end;
run;

title "数据描述";
proc means data=ml_data mean std min max;
  var x1 x2 x3 x4 x5 y_reg;
run;
title;


/****************************************************************************/
/* 10.2 无监督学习：主成分分析（PCA）                                       */
/****************************************************************************/
title "主成分分析";
proc princomp data=ml_data out=pca_out std;
  var x1 x2 x3 x4 x5;
run;
title;

title "PCA碎石图";
proc sgplot data=pca_out;
  /* 需要先生成特征值数据 */
run;
title;

/* 使用PROC FACTOR */
title "因子分析";
proc factor data=ml_data method=principal nfact=2
            rotate=varimax scree;
  var x1 x2 x3 x4 x5;
run;
title;


/****************************************************************************/
/* 10.3 无监督学习：聚类分析                                               */
/****************************************************************************/

/* 10.3.1 K-means聚类 */
title "K-means聚类";
proc fastclus data=ml_data maxclusters=3 maxiter=100
              out=cluster_out;
  var x1 x2 x3 x4 x5;
run;
title;

title "聚类结果";
proc sgplot data=cluster_out;
  scatter x=x1 y=x2 / group=CLUSTER;
run;
title;

/* 10.3.2 层次聚类 */
title "层次聚类";
proc cluster data=ml_data method=ward std
             outtree=tree_out;
  var x1 x2 x3 x4 x5;
run;
title;

title "聚类树状图";
proc tree data=tree_out;
run;
title;


/****************************************************************************/
/* 10.4 监督学习：正则化回归                                               */
/****************************************************************************/

/* 10.4.1 岭回归 */
title "岭回归";
proc reg data=ml_data outest=ridge_out ridge=0.5;
  model y_reg = x1 x2 x3 x4 x5;
run;
title;

/* 10.4.2 LASSO回归 */
title "LASSO回归";
proc glmselect data=ml_data
               method=lasso(choose=cvex) cvmethod=split(10);
  model y_reg = x1 x2 x3 x4 x5 / selection=lasso;
run;
title;

/* 10.4.3 Elastic Net */
title "Elastic Net";
proc glmselect data=ml_data
               method=elasticnet(choose=cvex) cvmethod=split(10);
  model y_reg = x1 x2 x3 x4 x5 / selection=elasticnet;
run;
title;


/****************************************************************************/
/* 10.5 监督学习：决策树                                                   */
/****************************************************************************/

/* 10.5.1 分类树 */
title "分类决策树";
proc hpsplit data=ml_data seed=123;
  class y_class;
  model y_class(event='1') = x1 x2 x3 x4 x5;
  grow entropy;
  prune costcomplexity;
  partition fraction(validate=0.3);
run;
title;

/* 10.5.2 回归树 */
title "回归决策树";
proc hpsplit data=ml_data seed=123;
  model y_reg = x1 x2 x3 x4 x5;
  grow variance;
  prune costcomplexity;
  partition fraction(validate=0.3);
run;
title;


/****************************************************************************/
/* 10.6 监督学习：随机森林                                                 */
/****************************************************************************/
title "随机森林（分类）";
proc hpforest data=ml_data seed=123
              maxtrees=100 vars_to_try=2;
  class y_class;
  model y_class(event='1') = x1 x2 x3 x4 x5;
run;
title;

title "随机森林（回归）";
proc hpforest data=ml_data seed=123
              maxtrees=100 vars_to_try=2;
  model y_reg = x1 x2 x3 x4 x5;
run;
title;


/****************************************************************************/
/* 10.7 监督学习：梯度提升树                                               */
/****************************************************************************/
title "梯度提升树（分类）";
proc gradboost data=ml_data seed=123;
  class y_class;
  model y_class(event='1') = x1 x2 x3 x4 x5;
  partition fraction(validate=0.3);
  autotune objective=validationmce;
run;
title;

title "梯度提升树（回归）";
proc gradboost data=ml_data seed=123;
  model y_reg = x1 x2 x3 x4 x5;
  partition fraction(validate=0.3);
  autotune objective=validationase;
run;
title;


/****************************************************************************/
/* 10.8 监督学习：神经网络                                                 */
/****************************************************************************/
title "神经网络（分类）";
proc nnet data=ml_data seed=123;
  class y_class;
  model y_class(event='1') = x1 x2 x3 x4 x5;
  hiddenunits 5 3;
  partition fraction(validate=0.3);
run;
title;

title "神经网络（回归）";
proc nnet data=ml_data seed=123;
  model y_reg = x1 x2 x3 x4 x5;
  hiddenunits 5 3;
  partition fraction(validate=0.3);
run;
title;


/****************************************************************************/
/* 10.9 模型比较                                                           */
/****************************************************************************/
/* 划分训练集和测试集 */
data ml_split;
  set ml_data;
  if rand('UNIFORM', 123) < 0.7 then role = 'train';
  else role = 'test';
run;

/* Logistic回归 */
title "Logistic回归（基线模型）";
proc logistic data=ml_split;
  where role='train';
  model y_class(event='1') = x1 x2 x3 x4 x5;
  score data=ml_split out=logit_scored;
run;
title;

/* 计算AUC */
proc logistic data=logit_scored;
  class y_class;
  model y_class(event='1') = x1 x2 x3 x4 x5 / roc;
run;


/****************************************************************************/
/* 10.10 模型解释：变量重要性                                             */
/****************************************************************************/
title "随机森林变量重要性";
proc hpforest data=ml_data seed=123
              maxtrees=100 vars_to_try=2;
  model y_reg = x1 x2 x3 x4 x5 / importance;
run;
title;

title "梯度提升树变量重要性";
proc gradboost data=ml_data seed=123;
  model y_reg = x1 x2 x3 x4 x5;
  partition fraction(validate=0.3);
  assess importance;
run;
title;


/****************************************************************************/
/* 10.11 保险应用：车险定价                                               */
/****************************************************************************/
data auto_insurance;
  call streaminit(456);
  do policy = 1 to 2000;
    age = rand('UNIFORM', 18, 70);
    gender = rand('BERNOULLI', 0.5);
    vehicle_age = rand('UNIFORM', 0, 15);
    region = rand('INTEGER', 1, 4);
    no_claim_years = rand('INTEGER', 0, 10);

    /* 索赔频率 */
    eta_freq = -1 + 0.02 * (age - 40) - 0.1 * gender +
               0.1 * vehicle_age + 0.3 * (region=1) - 0.1 * no_claim_years;
    lambda = exp(eta_freq);
    n_claims = rand('POISSON', lambda);

    /* 索赔金额 */
    if n_claims > 0 then do;
      eta_sev = 8 + 0.01 * (age - 40) + 0.05 * vehicle_age;
      mu = exp(eta_sev);
      claim_amount = rand('GAMMA', 2, mu/2) * n_claims;
    end;
    else claim_amount = 0;

    output;
  end;
run;

title "车险数据描述";
proc means data=auto_insurance mean std min max;
  var age vehicle_age n_claims claim_amount;
run;
title;

/* 随机森林预测索赔次数 */
title "随机森林预测索赔次数";
proc hpforest data=auto_insurance seed=123
              maxtrees=100 vars_to_try=3;
  model n_claims = age gender vehicle_age region no_claim_years;
run;
title;

/* 梯度提升树预测索赔金额 */
title "梯度提升树预测索赔金额";
proc gradboost data=auto_insurance seed=123;
  model claim_amount = age gender vehicle_age region no_claim_years;
  partition fraction(validate=0.3);
run;
title;

if n_claims > 0 then do;
      eta_sev = 8 + 0.01 * (age - 40) + 0.05 * vehicle_age;
      mu = exp(eta_sev);
      claim_amount = rand('GAMMA', 2, mu/2) * n_claims;
    end;
    else claim_amount = 0;

    output;
  end;
run;

title "车险数据描述";
proc means data=auto_insurance mean std min max;
  var age vehicle_age n_claims claim_amount;
run;
title;

/* 随机森林预测索赔次数 */
title "随机森林预测索赔次数";
proc hpforest data=auto_insurance seed=123
              maxtrees=100 vars_to_try=3;
  model n_claims = age gender vehicle_age region no_claim_years;
run;
title;

/* 梯度提升树预测索赔金额 */
title "梯度提升树预测索赔金额";
proc gradboost data=auto_insurance seed=123;
  model claim_amount = age gender vehicle_age region no_claim_years;
  partition fraction(validate=0.3);
run;
title;

/****************************************************************************/
/* 第10章 机器学习                                                          */
/* 对应教材：section10.tex                                                  */
/* 内容：无监督学习（PCA、因子分析、聚类）、监督学习                         */
/*       （正则化回归、决策树、随机森林、梯度提升树、神经网络）、SHAP值      */
/****************************************************************************/

/****************************************************************************/
/* 10.1 数据生成                                                           */
/****************************************************************************/
data ml_data;
  call streaminit(123);
  do i = 1 to 1000;
    x1 = rand('NORMAL', 0, 1);
    x2 = rand('NORMAL', 0, 1);
    x3 = rand('NORMAL', 0, 1);
    x4 = rand('NORMAL', 0, 1);
    x5 = rand('NORMAL', 0, 1);
    /* 分类目标 */
    eta = 0.5 + 1.2 * x1 - 0.8 * x2 + 0.5 * x3;
    p = 1 / (1 + exp(-eta));
    y_class = (rand('UNIFORM') < p);
    /* 回归目标 */
    y_reg = 2 + 1.5 * x1 - 0.8 * x2 + 0.5 * x3 +
            0.3 * x4 + rand('NORMAL', 0, 0.5);
    output;
  end;
run;

title "数据描述";
proc means data=ml_data mean std min max;
  var x1 x2 x3 x4 x5 y_reg;
run;
title;


/****************************************************************************/
/* 10.2 无监督学习：主成分分析（PCA）                                       */
/****************************************************************************/
title "主成分分析";
proc princomp data=ml_data out=pca_out std;
  var x1 x2 x3 x4 x5;
run;
title;

title "PCA碎石图";
proc sgplot data=pca_out;
  /* 需要先生成特征值数据 */
run;
title;

/* 使用PROC FACTOR */
title "因子分析";
proc factor data=ml_data method=principal nfact=2
            rotate=varimax scree;
  var x1 x2 x3 x4 x5;
run;
title;


/****************************************************************************/
/* 10.3 无监督学习：聚类分析                                               */
/****************************************************************************/

/* 10.3.1 K-means聚类 */
title "K-means聚类";
proc fastclus data=ml_data maxclusters=3 maxiter=100
              out=cluster_out;
  var x1 x2 x3 x4 x5;
run;
title;

title "聚类结果";
proc sgplot data=cluster_out;
  scatter x=x1 y=x2 / group=CLUSTER;
run;
title;

/* 10.3.2 层次聚类 */
title "层次聚类";
proc cluster data=ml_data method=ward std
             outtree=tree_out;
  var x1 x2 x3 x4 x5;
run;
title;

title "聚类树状图";
proc tree data=tree_out;
run;
title;


/****************************************************************************/
/* 10.4 监督学习：正则化回归                                               */
/****************************************************************************/

/* 10.4.1 岭回归 */
title "岭回归";
proc reg data=ml_data outest=ridge_out ridge=0.5;
  model y_reg = x1 x2 x3 x4 x5;
run;
title;

/* 10.4.2 LASSO回归 */
title "LASSO回归";
proc glmselect data=ml_data
               method=lasso(choose=cvex) cvmethod=split(10);
  model y_reg = x1 x2 x3 x4 x5 / selection=lasso;
run;
title;

/* 10.4.3 Elastic Net */
title "Elastic Net";
proc glmselect data=ml_data
               method=elasticnet(choose=cvex) cvmethod=split(10);
  model y_reg = x1 x2 x3 x4 x5 / selection=elasticnet;
run;
title;


/****************************************************************************/
/* 10.5 监督学习：决策树                                                   */
/****************************************************************************/

/* 10.5.1 分类树 */
title "分类决策树";
proc hpsplit data=ml_data seed=123;
  class y_class;
  model y_class(event='1') = x1 x2 x3 x4 x5;
  grow entropy;
  prune costcomplexity;
  partition fraction(validate=0.3);
run;
title;

/* 10.5.2 回归树 */
title "回归决策树";
proc hpsplit data=ml_data seed=123;
  model y_reg = x1 x2 x3 x4 x5;
  grow variance;
  prune costcomplexity;
  partition fraction(validate=0.3);
run;
title;


/****************************************************************************/
/* 10.6 监督学习：随机森林                                                 */
/****************************************************************************/
title "随机森林（分类）";
proc hpforest data=ml_data seed=123
              maxtrees=100 vars_to_try=2;
  class y_class;
  model y_class(event='1') = x1 x2 x3 x4 x5;
run;
title;

title "随机森林（回归）";
proc hpforest data=ml_data seed=123
              maxtrees=100 vars_to_try=2;
  model y_reg = x1 x2 x3 x4 x5;
run;
title;


/****************************************************************************/
/* 10.7 监督学习：梯度提升树                                               */
/****************************************************************************/
title "梯度提升树（分类）";
proc gradboost data=ml_data seed=123;
  class y_class;
  model y_class(event='1') = x1 x2 x3 x4 x5;
  partition fraction(validate=0.3);
  autotune objective=validationmce;
run;
title;

title "梯度提升树（回归）";
proc gradboost data=ml_data seed=123;
  model y_reg = x1 x2 x3 x4 x5;
  partition fraction(validate=0.3);
  autotune objective=validationase;
run;
title;


/****************************************************************************/
/* 10.8 监督学习：神经网络                                                 */
/****************************************************************************/
title "神经网络（分类）";
proc nnet data=ml_data seed=123;
  class y_class;
  model y_class(event='1') = x1 x2 x3 x4 x5;
  hiddenunits 5 3;
  partition fraction(validate=0.3);
run;
title;

title "神经网络（回归）";
proc nnet data=ml_data seed=123;
  model y_reg = x1 x2 x3 x4 x5;
  hiddenunits 5 3;
  partition fraction(validate=0.3);
run;
title;


/****************************************************************************/
/* 10.9 模型比较                                                           */
/****************************************************************************/
/* 划分训练集和测试集 */
data ml_split;
  set ml_data;
  if rand('UNIFORM', 123) < 0.7 then role = 'train';
  else role = 'test';
run;

/* Logistic回归 */
title "Logistic回归（基线模型）";
proc logistic data=ml_split;
  where role='train';
  model y_class(event='1') = x1 x2 x3 x4 x5;
  score data=ml_split out=logit_scored;
run;
title;

/* 计算AUC */
proc logistic data=logit_scored;
  class y_class;
  model y_class(event='1') = x1 x2 x3 x4 x5 / roc;
run;


/****************************************************************************/
/* 10.10 模型解释：变量重要性                                             */
/****************************************************************************/
title "随机森林变量重要性";
proc hpforest data=ml_data seed=123
              maxtrees=100 vars_to_try=2;
  model y_reg = x1 x2 x3 x4 x5 / importance;
run;
title;

title "梯度提升树变量重要性";
proc gradboost data=ml_data seed=123;
  model y_reg = x1 x2 x3 x4 x5;
  partition fraction(validate=0.3);
  assess importance;
run;
title;


/****************************************************************************/
/* 10.11 保险应用：车险定价                                               */
/****************************************************************************/
data auto_insurance;
  call streaminit(456);
  do policy = 1 to 2000;
    age = rand('UNIFORM', 18, 70);
    gender = rand('BERNOULLI', 0.5);
    vehicle_age = rand('UNIFORM', 0, 15);
    region = rand('INTEGER', 1, 4);
    no_claim_years = rand('INTEGER', 0, 10);

    /* 索赔频率 */
    eta_freq = -1 + 0.02 * (age - 40) - 0.1 * gender +
               0.1 * vehicle_age + 0.3 * (region=1) - 0.1 * no_claim_years;
    lambda = exp(eta_freq);
    n_claims = rand('POISSON', lambda);

    /* 索赔金额 */
    if n_claims > 0 then do;
      eta_sev = 8 + 0.01 * (age - 40) + 0.05 * vehicle_age;
      mu = exp(eta_sev);
      claim_amount = rand('GAMMA', 2, mu/2) * n_claims;
    end;
    else claim_amount = 0;

    output;
  end;
run;

title "车险数据描述";
proc means data=auto_insurance mean std min max;
  var age vehicle_age n_claims claim_amount;
run;
title;

/* 随机森林预测索赔次数 */
title "随机森林预测索赔次数";
proc hpforest data=auto_insurance seed=123
              maxtrees=100 vars_to_try=3;
  model n_claims = age gender vehicle_age region no_claim_years;
run;
title;

/* 梯度提升树预测索赔金额 */
title "梯度提升树预测索赔金额";
proc gradboost data=auto_insurance seed=123;
  model claim_amount = age gender vehicle_age region no_claim_years;
  partition fraction(validate=0.3);
run;
title;

if n_claims > 0 then do;
      eta_sev = 8 + 0.01 * (age - 40) + 0.05 * vehicle_age;
      mu = exp(eta_sev);
      claim_amount = rand('GAMMA', 2, mu/2) * n_claims;
    end;
    else claim_amount = 0;

    output;
  end;
run;

title "车险数据描述";
proc means data=auto_insurance mean std min max;
  var age vehicle_age n_claims claim_amount;
run;
title;

/* 随机森林预测索赔次数 */
title "随机森林预测索赔次数";
proc hpforest data=auto_insurance seed=123
              maxtrees=100 vars_to_try=3;
  model n_claims = age gender vehicle_age region no_claim_years;
run;
title;

/* 梯度提升树预测索赔金额 */
title "梯度提升树预测索赔金额";
proc gradboost data=auto_insurance seed=123;
  model claim_amount = age gender vehicle_age region no_claim_years;
  partition fraction(validate=0.3);
run;
title;

/****************************************************************************/
/* 第10章 机器学习                                                          */
/* 对应教材：section10.tex                                                  */
/* 内容：无监督学习（PCA、因子分析、聚类）、监督学习                         */
/*       （正则化回归、决策树、随机森林、梯度提升树、神经网络）、SHAP值      */
/****************************************************************************/

/****************************************************************************/
/* 10.1 数据生成                                                           */
/****************************************************************************/
data ml_data;
  call streaminit(123);
  do i = 1 to 1000;
    x1 = rand('NORMAL', 0, 1);
    x2 = rand('NORMAL', 0, 1);
    x3 = rand('NORMAL', 0, 1);
    x4 = rand('NORMAL', 0, 1);
    x5 = rand('NORMAL', 0, 1);
    /* 分类目标 */
    eta = 0.5 + 1.2 * x1 - 0.8 * x2 + 0.5 * x3;
    p = 1 / (1 + exp(-eta));
    y_class = (rand('UNIFORM') < p);
    /* 回归目标 */
    y_reg = 2 + 1.5 * x1 - 0.8 * x2 + 0.5 * x3 +
            0.3 * x4 + rand('NORMAL', 0, 0.5);
    output;
  end;
run;

title "数据描述";
proc means data=ml_data mean std min max;
  var x1 x2 x3 x4 x5 y_reg;
run;
title;


/****************************************************************************/
/* 10.2 无监督学习：主成分分析（PCA）                                       */
/****************************************************************************/
title "主成分分析";
proc princomp data=ml_data out=pca_out std;
  var x1 x2 x3 x4 x5;
run;
title;

title "PCA碎石图";
proc sgplot data=pca_out;
  /* 需要先生成特征值数据 */
run;
title;

/* 使用PROC FACTOR */
title "因子分析";
proc factor data=ml_data method=principal nfact=2
            rotate=varimax scree;
  var x1 x2 x3 x4 x5;
run;
title;


/****************************************************************************/
/* 10.3 无监督学习：聚类分析                                               */
/****************************************************************************/

/* 10.3.1 K-means聚类 */
title "K-means聚类";
proc fastclus data=ml_data maxclusters=3 maxiter=100
              out=cluster_out;
  var x1 x2 x3 x4 x5;
run;
title;

title "聚类结果";
proc sgplot data=cluster_out;
  scatter x=x1 y=x2 / group=CLUSTER;
run;
title;

/* 10.3.2 层次聚类 */
title "层次聚类";
proc cluster data=ml_data method=ward std
             outtree=tree_out;
  var x1 x2 x3 x4 x5;
run;
title;

title "聚类树状图";
proc tree data=tree_out;
run;
title;


/****************************************************************************/
/* 10.4 监督学习：正则化回归                                               */
/****************************************************************************/

/* 10.4.1 岭回归 */
title "岭回归";
proc reg data=ml_data outest=ridge_out ridge=0.5;
  model y_reg = x1 x2 x3 x4 x5;
run;
title;

/* 10.4.2 LASSO回归 */
title "LASSO回归";
proc glmselect data=ml_data
               method=lasso(choose=cvex) cvmethod=split(10);
  model y_reg = x1 x2 x3 x4 x5 / selection=lasso;
run;
title;

/* 10.4.3 Elastic Net */
title "Elastic Net";
proc glmselect data=ml_data
               method=elasticnet(choose=cvex) cvmethod=split(10);
  model y_reg = x1 x2 x3 x4 x5 / selection=elasticnet;
run;
title;


/****************************************************************************/
/* 10.5 监督学习：决策树                                                   */
/****************************************************************************/

/* 10.5.1 分类树 */
title "分类决策树";
proc hpsplit data=ml_data seed=123;
  class y_class;
  model y_class(event='1') = x1 x2 x3 x4 x5;
  grow entropy;
  prune costcomplexity;
  partition fraction(validate=0.3);
run;
title;

/* 10.5.2 回归树 */
title "回归决策树";
proc hpsplit data=ml_data seed=123;
  model y_reg = x1 x2 x3 x4 x5;
  grow variance;
  prune costcomplexity;
  partition fraction(validate=0.3);
run;
title;


/****************************************************************************/
/* 10.6 监督学习：随机森林                                                 */
/****************************************************************************/
title "随机森林（分类）";
proc hpforest data=ml_data seed=123
              maxtrees=100 vars_to_try=2;
  class y_class;
  model y_class(event='1') = x1 x2 x3 x4 x5;
run;
title;

title "随机森林（回归）";
proc hpforest data=ml_data seed=123
              maxtrees=100 vars_to_try=2;
  model y_reg = x1 x2 x3 x4 x5;
run;
title;


/****************************************************************************/
/* 10.7 监督学习：梯度提升树                                               */
/****************************************************************************/
title "梯度提升树（分类）";
proc gradboost data=ml_data seed=123;
  class y_class;
  model y_class(event='1') = x1 x2 x3 x4 x5;
  partition fraction(validate=0.3);
  autotune objective=validationmce;
run;
title;

title "梯度提升树（回归）";
proc gradboost data=ml_data seed=123;
  model y_reg = x1 x2 x3 x4 x5;
  partition fraction(validate=0.3);
  autotune objective=validationase;
run;
title;


/****************************************************************************/
/* 10.8 监督学习：神经网络                                                 */
/****************************************************************************/
title "神经网络（分类）";
proc nnet data=ml_data seed=123;
  class y_class;
  model y_class(event='1') = x1 x2 x3 x4 x5;
  hiddenunits 5 3;
  partition fraction(validate=0.3);
run;
title;

title "神经网络（回归）";
proc nnet data=ml_data seed=123;
  model y_reg = x1 x2 x3 x4 x5;
  hiddenunits 5 3;
  partition fraction(validate=0.3);
run;
title;


/****************************************************************************/
/* 10.9 模型比较                                                           */
/****************************************************************************/
/* 划分训练集和测试集 */
data ml_split;
  set ml_data;
  if rand('UNIFORM', 123) < 0.7 then role = 'train';
  else role = 'test';
run;

/* Logistic回归 */
title "Logistic回归（基线模型）";
proc logistic data=ml_split;
  where role='train';
  model y_class(event='1') = x1 x2 x3 x4 x5;
  score data=ml_split out=logit_scored;
run;
title;

/* 计算AUC */
proc logistic data=logit_scored;
  class y_class;
  model y_class(event='1') = x1 x2 x3 x4 x5 / roc;
run;


/****************************************************************************/
/* 10.10 模型解释：变量重要性                                             */
/****************************************************************************/
title "随机森林变量重要性";
proc hpforest data=ml_data seed=123
              maxtrees=100 vars_to_try=2;
  model y_reg = x1 x2 x3 x4 x5 / importance;
run;
title;

title "梯度提升树变量重要性";
proc gradboost data=ml_data seed=123;
  model y_reg = x1 x2 x3 x4 x5;
  partition fraction(validate=0.3);
  assess importance;
run;
title;


/****************************************************************************/
/* 10.11 保险应用：车险定价                                               */
/****************************************************************************/
data auto_insurance;
  call streaminit(456);
  do policy = 1 to 2000;
    age = rand('UNIFORM', 18, 70);
    gender = rand('BERNOULLI', 0.5);
    vehicle_age = rand('UNIFORM', 0, 15);
    region = rand('INTEGER', 1, 4);
    no_claim_years = rand('INTEGER', 0, 10);

    /* 索赔频率 */
    eta_freq = -1 + 0.02 * (age - 40) - 0.1 * gender +
               0.1 * vehicle_age + 0.3 * (region=1) - 0.1 * no_claim_years;
    lambda = exp(eta_freq);
    n_claims = rand('POISSON', lambda);

    /* 索赔金额 */
    if n_claims > 0 then do;
      eta_sev = 8 + 0.01 * (age - 40) + 0.05 * vehicle_age;
      mu = exp(eta_sev);
      claim_amount = rand('GAMMA', 2, mu/2) * n_claims;
    end;
    else claim_amount = 0;

    output;
  end;
run;

title "车险数据描述";
proc means data=auto_insurance mean std min max;
  var age vehicle_age n_claims claim_amount;
run;
title;

/* 随机森林预测索赔次数 */
title "随机森林预测索赔次数";
proc hpforest data=auto_insurance seed=123
              maxtrees=100 vars_to_try=3;
  model n_claims = age gender vehicle_age region no_claim_years;
run;
title;

/* 梯度提升树预测索赔金额 */
title "梯度提升树预测索赔金额";
proc gradboost data=auto_insurance seed=123;
  model claim_amount = age gender vehicle_age region no_claim_years;
  partition fraction(validate=0.3);
run;
title;

if n_claims > 0 then do;
      eta_sev = 8 + 0.01 * (age - 40) + 0.05 * vehicle_age;
      mu = exp(eta_sev);
      claim_amount = rand('GAMMA', 2, mu/2) * n_claims;
    end;
    else claim_amount = 0;

    output;
  end;
run;

title "车险数据描述";
proc means data=auto_insurance mean std min max;
  var age vehicle_age n_claims claim_amount;
run;
title;

/* 随机森林预测索赔次数 */
title "随机森林预测索赔次数";
proc hpforest data=auto_insurance seed=123
              maxtrees=100 vars_to_try=3;
  model n_claims = age gender vehicle_age region no_claim_years;
run;
title;

/* 梯度提升树预测索赔金额 */
title "梯度提升树预测索赔金额";
proc gradboost data=auto_insurance seed=123;
  model claim_amount = age gender vehicle_age region no_claim_years;
  partition fraction(validate=0.3);
run;
title;

/****************************************************************************/
/* 第10章 机器学习                                                          */
/* 对应教材：section10.tex                                                  */
/* 内容：无监督学习（PCA、因子分析、聚类）、监督学习                         */
/*       （正则化回归、决策树、随机森林、梯度提升树、神经网络）、SHAP值      */
/****************************************************************************/

/****************************************************************************/
/* 10.1 数据生成                                                           */
/****************************************************************************/
data ml_data;
  call streaminit(123);
  do i = 1 to 1000;
    x1 = rand('NORMAL', 0, 1);
    x2 = rand('NORMAL', 0, 1);
    x3 = rand('NORMAL', 0, 1);
    x4 = rand('NORMAL', 0, 1);
    x5 = rand('NORMAL', 0, 1);
    /* 分类目标 */
    eta = 0.5 + 1.2 * x1 - 0.8 * x2 + 0.5 * x3;
    p = 1 / (1 + exp(-eta));
    y_class = (rand('UNIFORM') < p);
    /* 回归目标 */
    y_reg = 2 + 1.5 * x1 - 0.8 * x2 + 0.5 * x3 +
            0.3 * x4 + rand('NORMAL', 0, 0.5);
    output;
  end;
run;

title "数据描述";
proc means data=ml_data mean std min max;
  var x1 x2 x3 x4 x5 y_reg;
run;
title;


/****************************************************************************/
/* 10.2 无监督学习：主成分分析（PCA）                                       */
/****************************************************************************/
title "主成分分析";
proc princomp data=ml_data out=pca_out std;
  var x1 x2 x3 x4 x5;
run;
title;

title "PCA碎石图";
proc sgplot data=pca_out;
  /* 需要先生成特征值数据 */
run;
title;

/* 使用PROC FACTOR */
title "因子分析";
proc factor data=ml_data method=principal nfact=2
            rotate=varimax scree;
  var x1 x2 x3 x4 x5;
run;
title;


/****************************************************************************/
/* 10.3 无监督学习：聚类分析                                               */
/****************************************************************************/

/* 10.3.1 K-means聚类 */
title "K-means聚类";
proc fastclus data=ml_data maxclusters=3 maxiter=100
              out=cluster_out;
  var x1 x2 x3 x4 x5;
run;
title;

title "聚类结果";
proc sgplot data=cluster_out;
  scatter x=x1 y=x2 / group=CLUSTER;
run;
title;

/* 10.3.2 层次聚类 */
title "层次聚类";
proc cluster data=ml_data method=ward std
             outtree=tree_out;
  var x1 x2 x3 x4 x5;
run;
title;

title "聚类树状图";
proc tree data=tree_out;
run;
title;


/****************************************************************************/
/* 10.4 监督学习：正则化回归                                               */
/****************************************************************************/

/* 10.4.1 岭回归 */
title "岭回归";
proc reg data=ml_data outest=ridge_out ridge=0.5;
  model y_reg = x1 x2 x3 x4 x5;
run;
title;

/* 10.4.2 LASSO回归 */
title "LASSO回归";
proc glmselect data=ml_data
               method=lasso(choose=cvex) cvmethod=split(10);
  model y_reg = x1 x2 x3 x4 x5 / selection=lasso;
run;
title;

/* 10.4.3 Elastic Net */
title "Elastic Net";
proc glmselect data=ml_data
               method=elasticnet(choose=cvex) cvmethod=split(10);
  model y_reg = x1 x2 x3 x4 x5 / selection=elasticnet;
run;
title;


/****************************************************************************/
/* 10.5 监督学习：决策树                                                   */
/****************************************************************************/

/* 10.5.1 分类树 */
title "分类决策树";
proc hpsplit data=ml_data seed=123;
  class y_class;
  model y_class(event='1') = x1 x2 x3 x4 x5;
  grow entropy;
  prune costcomplexity;
  partition fraction(validate=0.3);
run;
title;

/* 10.5.2 回归树 */
title "回归决策树";
proc hpsplit data=ml_data seed=123;
  model y_reg = x1 x2 x3 x4 x5;
  grow variance;
  prune costcomplexity;
  partition fraction(validate=0.3);
run;
title;


/****************************************************************************/
/* 10.6 监督学习：随机森林                                                 */
/****************************************************************************/
title "随机森林（分类）";
proc hpforest data=ml_data seed=123
              maxtrees=100 vars_to_try=2;
  class y_class;
  model y_class(event='1') = x1 x2 x3 x4 x5;
run;
title;

title "随机森林（回归）";
proc hpforest data=ml_data seed=123
              maxtrees=100 vars_to_try=2;
  model y_reg = x1 x2 x3 x4 x5;
run;
title;


/****************************************************************************/
/* 10.7 监督学习：梯度提升树                                               */
/****************************************************************************/
title "梯度提升树（分类）";
proc gradboost data=ml_data seed=123;
  class y_class;
  model y_class(event='1') = x1 x2 x3 x4 x5;
  partition fraction(validate=0.3);
  autotune objective=validationmce;
run;
title;

title "梯度提升树（回归）";
proc gradboost data=ml_data seed=123;
  model y_reg = x1 x2 x3 x4 x5;
  partition fraction(validate=0.3);
  autotune objective=validationase;
run;
title;


/****************************************************************************/
/* 10.8 监督学习：神经网络                                                 */
/****************************************************************************/
title "神经网络（分类）";
proc nnet data=ml_data seed=123;
  class y_class;
  model y_class(event='1') = x1 x2 x3 x4 x5;
  hiddenunits 5 3;
  partition fraction(validate=0.3);
run;
title;

title "神经网络（回归）";
proc nnet data=ml_data seed=123;
  model y_reg = x1 x2 x3 x4 x5;
  hiddenunits 5 3;
  partition fraction(validate=0.3);
run;
title;


/****************************************************************************/
/* 10.9 模型比较                                                           */
/****************************************************************************/
/* 划分训练集和测试集 */
data ml_split;
  set ml_data;
  if rand('UNIFORM', 123) < 0.7 then role = 'train';
  else role = 'test';
run;

/* Logistic回归 */
title "Logistic回归（基线模型）";
proc logistic data=ml_split;
  where role='train';
  model y_class(event='1') = x1 x2 x3 x4 x5;
  score data=ml_split out=logit_scored;
run;
title;

/* 计算AUC */
proc logistic data=logit_scored;
  class y_class;
  model y_class(event='1') = x1 x2 x3 x4 x5 / roc;
run;


/****************************************************************************/
/* 10.10 模型解释：变量重要性                                             */
/****************************************************************************/
title "随机森林变量重要性";
proc hpforest data=ml_data seed=123
              maxtrees=100 vars_to_try=2;
  model y_reg = x1 x2 x3 x4 x5 / importance;
run;
title;

title "梯度提升树变量重要性";
proc gradboost data=ml_data seed=123;
  model y_reg = x1 x2 x3 x4 x5;
  partition fraction(validate=0.3);
  assess importance;
run;
title;


/****************************************************************************/
/* 10.11 保险应用：车险定价                                               */
/****************************************************************************/
data auto_insurance;
  call streaminit(456);
  do policy = 1 to 2000;
    age = rand('UNIFORM', 18, 70);
    gender = rand('BERNOULLI', 0.5);
    vehicle_age = rand('UNIFORM', 0, 15);
    region = rand('INTEGER', 1, 4);
    no_claim_years = rand('INTEGER', 0, 10);

    /* 索赔频率 */
    eta_freq = -1 + 0.02 * (age - 40) - 0.1 * gender +
               0.1 * vehicle_age + 0.3 * (region=1) - 0.1 * no_claim_years;
    lambda = exp(eta_freq);
    n_claims = rand('POISSON', lambda);

    /* 索赔金额 */
    if n_claims > 0 then do;
      eta_sev = 8 + 0.01 * (age - 40) + 0.05 * vehicle_age;
      mu = exp(eta_sev);
      claim_amount = rand('GAMMA', 2, mu/2) * n_claims;
    end;
    else claim_amount = 0;

    output;
  end;
run;

title "车险数据描述";
proc means data=auto_insurance mean std min max;
  var age vehicle_age n_claims claim_amount;
run;
title;

/* 随机森林预测索赔次数 */
title "随机森林预测索赔次数";
proc hpforest data=auto_insurance seed=123
              maxtrees=100 vars_to_try=3;
  model n_claims = age gender vehicle_age region no_claim_years;
run;
title;

/* 梯度提升树预测索赔金额 */
title "梯度提升树预测索赔金额";
proc gradboost data=auto_insurance seed=123;
  model claim_amount = age gender vehicle_age region no_claim_years;
  partition fraction(validate=0.3);
run;
title;

if n_claims > 0 then do;
      eta_sev = 8 + 0.01 * (age - 40) + 0.05 * vehicle_age;
      mu = exp(eta_sev);
      claim_amount = rand('GAMMA', 2, mu/2) * n_claims;
    end;
    else claim_amount = 0;

    output;
  end;
run;

title "车险数据描述";
proc means data=auto_insurance mean std min max;
  var age vehicle_age n_claims claim_amount;
run;
title;

/* 随机森林预测索赔次数 */
title "随机森林预测索赔次数";
proc hpforest data=auto_insurance seed=123
              maxtrees=100 vars_to_try=3;
  model n_claims = age gender vehicle_age region no_claim_years;
run;
title;

/* 梯度提升树预测索赔金额 */
title "梯度提升树预测索赔金额";
proc gradboost data=auto_insurance seed=123;
  model claim_amount = age gender vehicle_age region no_claim_years;
  partition fraction(validate=0.3);
run;
title;

/****************************************************************************/
/* 第10章 机器学习                                                          */
/* 对应教材：section10.tex                                                  */
/* 内容：无监督学习（PCA、因子分析、聚类）、监督学习                         */
/*       （正则化回归、决策树、随机森林、梯度提升树、神经网络）、SHAP值      */
/****************************************************************************/

/****************************************************************************/
/* 10.1 数据生成                                                           */
/****************************************************************************/
data ml_data;
  call streaminit(123);
  do i = 1 to 1000;
    x1 = rand('NORMAL', 0, 1);
    x2 = rand('NORMAL', 0, 1);
    x3 = rand('NORMAL', 0, 1);
    x4 = rand('NORMAL', 0, 1);
    x5 = rand('NORMAL', 0, 1);
    /* 分类目标 */
    eta = 0.5 + 1.2 * x1 - 0.8 * x2 + 0.5 * x3;
    p = 1 / (1 + exp(-eta));
    y_class = (rand('UNIFORM') < p);
    /* 回归目标 */
    y_reg = 2 + 1.5 * x1 - 0.8 * x2 + 0.5 * x3 +
            0.3 * x4 + rand('NORMAL', 0, 0.5);
    output;
  end;
run;

title "数据描述";
proc means data=ml_data mean std min max;
  var x1 x2 x3 x4 x5 y_reg;
run;
title;


/****************************************************************************/
/* 10.2 无监督学习：主成分分析（PCA）                                       */
/****************************************************************************/
title "主成分分析";
proc princomp data=ml_data out=pca_out std;
  var x1 x2 x3 x4 x5;
run;
title;

title "PCA碎石图";
proc sgplot data=pca_out;
  /* 需要先生成特征值数据 */
run;
title;

/* 使用PROC FACTOR */
title "因子分析";
proc factor data=ml_data method=principal nfact=2
            rotate=varimax scree;
  var x1 x2 x3 x4 x5;
run;
title;


/****************************************************************************/
/* 10.3 无监督学习：聚类分析                                               */
/****************************************************************************/

/* 10.3.1 K-means聚类 */
title "K-means聚类";
proc fastclus data=ml_data maxclusters=3 maxiter=100
              out=cluster_out;
  var x1 x2 x3 x4 x5;
run;
title;

title "聚类结果";
proc sgplot data=cluster_out;
  scatter x=x1 y=x2 / group=CLUSTER;
run;
title;

/* 10.3.2 层次聚类 */
title "层次聚类";
proc cluster data=ml_data method=ward std
             outtree=tree_out;
  var x1 x2 x3 x4 x5;
run;
title;

title "聚类树状图";
proc tree data=tree_out;
run;
title;


/****************************************************************************/
/* 10.4 监督学习：正则化回归                                               */
/****************************************************************************/

/* 10.4.1 岭回归 */
title "岭回归";
proc reg data=ml_data outest=ridge_out ridge=0.5;
  model y_reg = x1 x2 x3 x4 x5;
run;
title;

/* 10.4.2 LASSO回归 */
title "LASSO回归";
proc glmselect data=ml_data
               method=lasso(choose=cvex) cvmethod=split(10);
  model y_reg = x1 x2 x3 x4 x5 / selection=lasso;
run;
title;

/* 10.4.3 Elastic Net */
title "Elastic Net";
proc glmselect data=ml_data
               method=elasticnet(choose=cvex) cvmethod=split(10);
  model y_reg = x1 x2 x3 x4 x5 / selection=elasticnet;
run;
title;


/****************************************************************************/
/* 10.5 监督学习：决策树                                                   */
/****************************************************************************/

/* 10.5.1 分类树 */
title "分类决策树";
proc hpsplit data=ml_data seed=123;
  class y_class;
  model y_class(event='1') = x1 x2 x3 x4 x5;
  grow entropy;
  prune costcomplexity;
  partition fraction(validate=0.3);
run;
title;

/* 10.5.2 回归树 */
title "回归决策树";
proc hpsplit data=ml_data seed=123;
  model y_reg = x1 x2 x3 x4 x5;
  grow variance;
  prune costcomplexity;
  partition fraction(validate=0.3);
run;
title;


/****************************************************************************/
/* 10.6 监督学习：随机森林                                                 */
/****************************************************************************/
title "随机森林（分类）";
proc hpforest data=ml_data seed=123
              maxtrees=100 vars_to_try=2;
  class y_class;
  model y_class(event='1') = x1 x2 x3 x4 x5;
run;
title;

title "随机森林（回归）";
proc hpforest data=ml_data seed=123
              maxtrees=100 vars_to_try=2;
  model y_reg = x1 x2 x3 x4 x5;
run;
title;


/****************************************************************************/
/* 10.7 监督学习：梯度提升树                                               */
/****************************************************************************/
title "梯度提升树（分类）";
proc gradboost data=ml_data seed=123;
  class y_class;
  model y_class(event='1') = x1 x2 x3 x4 x5;
  partition fraction(validate=0.3);
  autotune objective=validationmce;
run;
title;

title "梯度提升树（回归）";
proc gradboost data=ml_data seed=123;
  model y_reg = x1 x2 x3 x4 x5;
  partition fraction(validate=0.3);
  autotune objective=validationase;
run;
title;


/****************************************************************************/
/* 10.8 监督学习：神经网络                                                 */
/****************************************************************************/
title "神经网络（分类）";
proc nnet data=ml_data seed=123;
  class y_class;
  model y_class(event='1') = x1 x2 x3 x4 x5;
  hiddenunits 5 3;
  partition fraction(validate=0.3);
run;
title;

title "神经网络（回归）";
proc nnet data=ml_data seed=123;
  model y_reg = x1 x2 x3 x4 x5;
  hiddenunits 5 3;
  partition fraction(validate=0.3);
run;
title;


/****************************************************************************/
/* 10.9 模型比较                                                           */
/****************************************************************************/
/* 划分训练集和测试集 */
data ml_split;
  set ml_data;
  if rand('UNIFORM', 123) < 0.7 then role = 'train';
  else role = 'test';
run;

/* Logistic回归 */
title "Logistic回归（基线模型）";
proc logistic data=ml_split;
  where role='train';
  model y_class(event='1') = x1 x2 x3 x4 x5;
  score data=ml_split out=logit_scored;
run;
title;

/* 计算AUC */
proc logistic data=logit_scored;
  class y_class;
  model y_class(event='1') = x1 x2 x3 x4 x5 / roc;
run;


/****************************************************************************/
/* 10.10 模型解释：变量重要性                                             */
/****************************************************************************/
title "随机森林变量重要性";
proc hpforest data=ml_data seed=123
              maxtrees=100 vars_to_try=2;
  model y_reg = x1 x2 x3 x4 x5 / importance;
run;
title;

title "梯度提升树变量重要性";
proc gradboost data=ml_data seed=123;
  model y_reg = x1 x2 x3 x4 x5;
  partition fraction(validate=0.3);
  assess importance;
run;
title;


/****************************************************************************/
/* 10.11 保险应用：车险定价                                               */
/****************************************************************************/
data auto_insurance;
  call streaminit(456);
  do policy = 1 to 2000;
    age = rand('UNIFORM', 18, 70);
    gender = rand('BERNOULLI', 0.5);
    vehicle_age = rand('UNIFORM', 0, 15);
    region = rand('INTEGER', 1, 4);
    no_claim_years = rand('INTEGER', 0, 10);

    /* 索赔频率 */
    eta_freq = -1 + 0.02 * (age - 40) - 0.1 * gender +
               0.1 * vehicle_age + 0.3 * (region=1) - 0.1 * no_claim_years;
    lambda = exp(eta_freq);
    n_claims = rand('POISSON', lambda);

    /* 索赔金额 */
    if n_claims > 0 then do;
      eta_sev = 8 + 0.01 * (age - 40) + 0.05 * vehicle_age;
      mu = exp(eta_sev);
      claim_amount = rand('GAMMA', 2, mu/2) * n_claims;
    end;
    else claim_amount = 0;

    output;
  end;
run;

title "车险数据描述";
proc means data=auto_insurance mean std min max;
  var age vehicle_age n_claims claim_amount;
run;
title;

/* 随机森林预测索赔次数 */
title "随机森林预测索赔次数";
proc hpforest data=auto_insurance seed=123
              maxtrees=100 vars_to_try=3;
  model n_claims = age gender vehicle_age region no_claim_years;
run;
title;

/* 梯度提升树预测索赔金额 */
title "梯度提升树预测索赔金额";
proc gradboost data=auto_insurance seed=123;
  model claim_amount = age gender vehicle_age region no_claim_years;
  partition fraction(validate=0.3);
run;
title;

/****************************************************************************/
/* 第10章 机器学习                                                          */
/* 对应教材：section10.tex                                                  */
/* 内容：无监督学习（PCA、因子分析、聚类）、监督学习                         */
/*       （正则化回归、决策树、随机森林、梯度提升树、神经网络）、SHAP值      */
/****************************************************************************/

/****************************************************************************/
/* 10.1 数据生成                                                           */
/****************************************************************************/
data ml_data;
  call streaminit(123);
  do i = 1 to 1000;
    x1 = rand('NORMAL', 0, 1);
    x2 = rand('NORMAL', 0, 1);
    x3 = rand('NORMAL', 0, 1);
    x4 = rand('NORMAL', 0, 1);
    x5 = rand('NORMAL', 0, 1);
    /* 分类目标 */
    eta = 0.5 + 1.2 * x1 - 0.8 * x2 + 0.5 * x3;
    p = 1 / (1 + exp(-eta));
    y_class = (rand('UNIFORM') < p);
    /* 回归目标 */
    y_reg = 2 + 1.5 * x1 - 0.8 * x2 + 0.5 * x3 +
            0.3 * x4 + rand('NORMAL', 0, 0.5);
    output;
  end;
run;

title "数据描述";
proc means data=ml_data mean std min max;
  var x1 x2 x3 x4 x5 y_reg;
run;
title;


/****************************************************************************/
/* 10.2 无监督学习：主成分分析（PCA）                                       */
/****************************************************************************/
title "主成分分析";
proc princomp data=ml_data out=pca_out std;
  var x1 x2 x3 x4 x5;
run;
title;

title "PCA碎石图";
proc sgplot data=pca_out;
  /* 需要先生成特征值数据 */
run;
title;

/* 使用PROC FACTOR */
title "因子分析";
proc factor data=ml_data method=principal nfact=2
            rotate=varimax scree;
  var x1 x2 x3 x4 x5;
run;
title;


/****************************************************************************/
/* 10.3 无监督学习：聚类分析                                               */
/****************************************************************************/

/* 10.3.1 K-means聚类 */
title "K-means聚类";
proc fastclus data=ml_data maxclusters=3 maxiter=100
              out=cluster_out;
  var x1 x2 x3 x4 x5;
run;
title;

title "聚类结果";
proc sgplot data=cluster_out;
  scatter x=x1 y=x2 / group=CLUSTER;
run;
title;

/* 10.3.2 层次聚类 */
title "层次聚类";
proc cluster data=ml_data method=ward std
             outtree=tree_out;
  var x1 x2 x3 x4 x5;
run;
title;

title "聚类树状图";
proc tree data=tree_out;
run;
title;


/****************************************************************************/
/* 10.4 监督学习：正则化回归                                               */
/****************************************************************************/

/* 10.4.1 岭回归 */
title "岭回归";
proc reg data=ml_data outest=ridge_out ridge=0.5;
  model y_reg = x1 x2 x3 x4 x5;
run;
title;

/* 10.4.2 LASSO回归 */
title "LASSO回归";
proc glmselect data=ml_data
               method=lasso(choose=cvex) cvmethod=split(10);
  model y_reg = x1 x2 x3 x4 x5 / selection=lasso;
run;
title;

/* 10.4.3 Elastic Net */
title "Elastic Net";
proc glmselect data=ml_data
               method=elasticnet(choose=cvex) cvmethod=split(10);
  model y_reg = x1 x2 x3 x4 x5 / selection=elasticnet;
run;
title;


/****************************************************************************/
/* 10.5 监督学习：决策树                                                   */
/****************************************************************************/

/* 10.5.1 分类树 */
title "分类决策树";
proc hpsplit data=ml_data seed=123;
  class y_class;
  model y_class(event='1') = x1 x2 x3 x4 x5;
  grow entropy;
  prune costcomplexity;
  partition fraction(validate=0.3);
run;
title;

/* 10.5.2 回归树 */
title "回归决策树";
proc hpsplit data=ml_data seed=123;
  model y_reg = x1 x2 x3 x4 x5;
  grow variance;
  prune costcomplexity;
  partition fraction(validate=0.3);
run;
title;


/****************************************************************************/
/* 10.6 监督学习：随机森林                                                 */
/****************************************************************************/
title "随机森林（分类）";
proc hpforest data=ml_data seed=123
              maxtrees=100 vars_to_try=2;
  class y_class;
  model y_class(event='1') = x1 x2 x3 x4 x5;
run;
title;

title "随机森林（回归）";
proc hpforest data=ml_data seed=123
              maxtrees=100 vars_to_try=2;
  model y_reg = x1 x2 x3 x4 x5;
run;
title;


/****************************************************************************/
/* 10.7 监督学习：梯度提升树                                               */
/****************************************************************************/
title "梯度提升树（分类）";
proc gradboost data=ml_data seed=123;
  class y_class;
  model y_class(event='1') = x1 x2 x3 x4 x5;
  partition fraction(validate=0.3);
  autotune objective=validationmce;
run;
title;

title "梯度提升树（回归）";
proc gradboost data=ml_data seed=123;
  model y_reg = x1 x2 x3 x4 x5;
  partition fraction(validate=0.3);
  autotune objective=validationase;
run;
title;


/****************************************************************************/
/* 10.8 监督学习：神经网络                                                 */
/****************************************************************************/
title "神经网络（分类）";
proc nnet data=ml_data seed=123;
  class y_class;
  model y_class(event='1') = x1 x2 x3 x4 x5;
  hiddenunits 5 3;
  partition fraction(validate=0.3);
run;
title;

title "神经网络（回归）";
proc nnet data=ml_data seed=123;
  model y_reg = x1 x2 x3 x4 x5;
  hiddenunits 5 3;
  partition fraction(validate=0.3);
run;
title;


/****************************************************************************/
/* 10.9 模型比较                                                           */
/****************************************************************************/
/* 划分训练集和测试集 */
data ml_split;
  set ml_data;
  if rand('UNIFORM', 123) < 0.7 then role = 'train';
  else role = 'test';
run;

/* Logistic回归 */
title "Logistic回归（基线模型）";
proc logistic data=ml_split;
  where role='train';
  model y_class(event='1') = x1 x2 x3 x4 x5;
  score data=ml_split out=logit_scored;
run;
title;

/* 计算AUC */
proc logistic data=logit_scored;
  class y_class;
  model y_class(event='1') = x1 x2 x3 x4 x5 / roc;
run;


/****************************************************************************/
/* 10.10 模型解释：变量重要性                                             */
/****************************************************************************/
title "随机森林变量重要性";
proc hpforest data=ml_data seed=123
              maxtrees=100 vars_to_try=2;
  model y_reg = x1 x2 x3 x4 x5 / importance;
run;
title;

title "梯度提升树变量重要性";
proc gradboost data=ml_data seed=123;
  model y_reg = x1 x2 x3 x4 x5;
  partition fraction(validate=0.3);
  assess importance;
run;
title;


/****************************************************************************/
/* 10.11 保险应用：车险定价                                               */
/****************************************************************************/
data auto_insurance;
  call streaminit(456);
  do policy = 1 to 2000;
    age = rand('UNIFORM', 18, 70);
    gender = rand('BERNOULLI', 0.5);
    vehicle_age = rand('UNIFORM', 0, 15);
    region = rand('INTEGER', 1, 4);
    no_claim_years = rand('INTEGER', 0, 10);

    /* 索赔频率 */
    eta_freq = -1 + 0.02 * (age - 40) - 0.1 * gender +
               0.1 * vehicle_age + 0.3 * (region=1) - 0.1 * no_claim_years;
    lambda = exp(eta_freq);
    n_claims = rand('POISSON', lambda);

    /* 索赔金额 */
    if n_claims > 0 then do;
      eta_sev = 8 + 0.01 * (age - 40) + 0.05 * vehicle_age;
      mu = exp(eta_sev);
      claim_amount = rand('GAMMA', 2, mu/2) * n_claims;
    end;
    else claim_amount = 0;

    output;
  end;
run;

title "车险数据描述";
proc means data=auto_insurance mean std min max;
  var age vehicle_age n_claims claim_amount;
run;
title;

/* 随机森林预测索赔次数 */
title "随机森林预测索赔次数";
proc hpforest data=auto_insurance seed=123
              maxtrees=100 vars_to_try=3;
  model n_claims = age gender vehicle_age region no_claim_years;
run;
title;

/* 梯度提升树预测索赔金额 */
title "梯度提升树预测索赔金额";
proc gradboost data=auto_insurance seed=123;
  model claim_amount = age gender vehicle_age region no_claim_years;
  partition fraction(validate=0.3);
run;
title;

if n_claims > 0 then do;
      eta_sev = 8 + 0.01 * (age - 40) + 0.05 * vehicle_age;
      mu = exp(eta_sev);
      claim_amount = rand('GAMMA', 2, mu/2) * n_claims;
    end;
    else claim_amount = 0;

    output;
  end;
run;

title "车险数据描述";
proc means data=auto_insurance mean std min max;
  var age vehicle_age n_claims claim_amount;
run;
title;

/* 随机森林预测索赔次数 */
title "随机森林预测索赔次数";
proc hpforest data=auto_insurance seed=123
              maxtrees=100 vars_to_try=3;
  model n_claims = age gender vehicle_age region no_claim_years;
run;
title;

/* 梯度提升树预测索赔金额 */
title "梯度提升树预测索赔金额";
proc gradboost data=auto_insurance seed=123;
  model claim_amount = age gender vehicle_age region no_claim_years;
  partition fraction(validate=0.3);
run;
title;