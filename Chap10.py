# Chap10 Python代码
# 自动从chap10.html同步生成

# 导入所需的Python模块
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

np.random.seed(123)
n = 1000
X = np.random.normal(0, 1, (n, 5))
df = pd.DataFrame(X, columns=['x1', 'x2', 'x3', 'x4', 'x5'])

# 分类目标
eta = 0.5 + 1.2 * X[:, 0] - 0.8 * X[:, 1] + 0.5 * X[:, 2]
p = 1 / (1 + np.exp(-eta))
y_class = (np.random.uniform(0, 1, n) < p).astype(int)

# 回归目标
y_reg = 2 + 1.5 * X[:, 0] - 0.8 * X[:, 1] + 0.5 * X[:, 2] + \
        0.3 * X[:, 3] + np.random.normal(0, 0.5, n)

df['y_class'] = y_class
df['y_reg'] = y_reg

print("数据描述:")
print(df.describe())


##############################################################################
# 10.2 无监督学习：主成分分析（PCA）
##############################################################################
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

print(f"\nPCA解释方差比: {pca.explained_variance_ratio_}")
print(f"累积解释方差: {np.cumsum(pca.explained_variance_ratio_)}")

##############################################################################
# 第10章 机器学习
# 对应教材：section10.tex
# 内容：无监督学习（PCA、因子分析、聚类）、监督学习
#       （正则化回归、决策树、随机森林、梯度提升树、神经网络）、SHAP值
##############################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


##############################################################################
# 10.1 数据生成
##############################################################################
np.random.seed(123)
n = 1000
X = np.random.normal(0, 1, (n, 5))
df = pd.DataFrame(X, columns=['x1', 'x2', 'x3', 'x4', 'x5'])

# 分类目标
eta = 0.5 + 1.2 * X[:, 0] - 0.8 * X[:, 1] + 0.5 * X[:, 2]
p = 1 / (1 + np.exp(-eta))
y_class = (np.random.uniform(0, 1, n) < p).astype(int)

# 回归目标
y_reg = 2 + 1.5 * X[:, 0] - 0.8 * X[:, 1] + 0.5 * X[:, 2] + \
        0.3 * X[:, 3] + np.random.normal(0, 0.5, n)

df['y_class'] = y_class
df['y_reg'] = y_reg

print("数据描述:")
print(df.describe())


##############################################################################
# 10.2 无监督学习：主成分分析（PCA）
##############################################################################
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

print(f"\nPCA解释方差比: {pca.explained_variance_ratio_}")
print(f"累积解释方差: {np.cumsum(pca.explained_variance_ratio_)}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(range(1, 6), pca.explained_variance_ratio_, alpha=0.7)
axes[0].set_xlabel('主成分'); axes[0].set_ylabel('解释方差比')
axes[0].set_title('PCA碎石图')
axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y_class, alpha=0.5)
axes[1].set_xlabel('PC1'); axes[1].set_ylabel('PC2')
axes[1].set_title('PCA投影')
plt.tight_layout(); plt.show()


## 因子分析
fa = FactorAnalysis(n_components=2, rotation='varimax')
X_fa = fa.fit_transform(X_scaled)
print(f"\n因子分析载荷矩阵:\n{fa.components_.T}")


##############################################################################
# 10.3 无监督学习：聚类分析
##############################################################################

## 10.3.1 K-means聚类
kmeans = KMeans(n_clusters=3, random_state=123, n_init=10)
labels_km = kmeans.fit_predict(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(X[:, 0], X[:, 1], c=labels_km, alpha=0.5)
axes[0].set_title('K-means聚类')
axes[0].set_xlabel('x1'); axes[0].set_ylabel('x2')

## 10.3.2 层次聚类
hc = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels_hc = hc.fit_predict(X_scaled)
axes[1].scatter(X[:, 0], X[:, 1], c=labels_hc, alpha=0.5)
axes[1].set_title('层次聚类')
axes[1].set_xlabel('x1'); axes[1].set_ylabel('x2')
plt.tight_layout(); plt.show()

# 肘部法则
inertias = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=123, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(1, 11), inertias, 'bo-')
ax.set_xlabel('聚类数 k'); ax.set_ylabel('惯性')
ax.set_title('肘部法则')
plt.tight_layout(); plt.show()


##############################################################################
# 10.4 监督学习：正则化回归
##############################################################################
X_train, X_test, y_train, y_test = train_test_split(
    X, y_reg, test_size=0.3, random_state=123)

## 10.4.1 岭回归
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
print(f"\n岭回归: R²={ridge.score(X_test, y_test):.4f}")
print(f"  系数: {ridge.coef_}")

## 10.4.2 LASSO回归
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
print(f"\nLASSO回归: R²={lasso.score(X_test, y_test):.4f}")
print(f"  系数: {lasso.coef_}")

## 10.4.3 Elastic Net
enet = ElasticNet(alpha=0.1, l1_ratio=0.5)
enet.fit(X_train, y_train)
print(f"\nElastic Net: R²={enet.score(X_test, y_test):.4f}")
print(f"  系数: {enet.coef_}")


##############################################################################
# 10.5 监督学习：决策树
##############################################################################
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_class, test_size=0.3, random_state=123)

## 10.5.1 分类树
dt_clf = DecisionTreeClassifier(max_depth=5, random_state=123)
dt_clf.fit(X_train_c, y_train_c)
print(f"\n分类决策树: 准确率={accuracy_score(y_test_c, dt_clf.predict(X_test_c)):.4f}")

## 10.5.2 回归树
dt_reg = DecisionTreeRegressor(max_depth=5, random_state=123)
dt_reg.fit(X_train, y_train)
print(f"回归决策树: R²={dt_reg.score(X_test, y_test):.4f}")


##############################################################################
# 10.6 监督学习：随机森林
##############################################################################
## 10.6.1 分类随机森林
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=5,
                                 random_state=123)
rf_clf.fit(X_train_c, y_train_c)
y_pred_rf = rf_clf.predict_proba(X_test_c)[:, 1]
print(f"\n随机森林（分类）: AUC={roc_auc_score(y_test_c, y_pred_rf):.4f}")

## 10.6.2 回归随机森林
rf_reg = RandomForestRegressor(n_estimators=100, max_depth=5,
                                random_state=123)
rf_reg.fit(X_train, y_train)
print(f"随机森林（回归）: R²={rf_reg.score(X_test, y_test):.4f}")

## 变量重要性
importances = rf_reg.feature_importances_
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(range(5), importances)
ax.set_xticks(range(5))
ax.set_xticklabels(['x1', 'x2', 'x3', 'x4', 'x5'])
ax.set_ylabel('重要性')
ax.set_title('随机森林变量重要性')
plt.tight_layout(); plt.show()


##############################################################################
# 10.7 监督学习：梯度提升树
##############################################################################
## 10.7.1 分类GBDT
gb_clf = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                     random_state=123)
gb_clf.fit(X_train_c, y_train_c)
y_pred_gb = gb_clf.predict_proba(X_test_c)[:, 1]
print(f"\n梯度提升树（分类）: AUC={roc_auc_score(y_test_c, y_pred_gb):.4f}")

## 10.7.2 回归GBDT
gb_reg = GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                    random_state=123)
gb_reg.fit(X_train, y_train)
print(f"梯度提升树（回归）: R²={gb_reg.score(X_test, y_test):.4f}")


##############################################################################
# 10.8 监督学习：神经网络
##############################################################################
## 10.8.1 分类神经网络
nn_clf = MLPClassifier(hidden_layer_sizes=(5, 3), max_iter=500,
                        random_state=123)
nn_clf.fit(X_train_c, y_train_c)
y_pred_nn = nn_clf.predict_proba(X_test_c)[:, 1]
print(f"\n神经网络（分类）: AUC={roc_auc_score(y_test_c, y_pred_nn):.4f}")

## 10.8.2 回归神经网络
nn_reg = MLPRegressor(hidden_layer_sizes=(5, 3), max_iter=500,
                       random_state=123)
nn_reg.fit(X_train, y_train)
print(f"神经网络（回归）: R²={nn_reg.score(X_test, y_test):.4f}")


##############################################################################
# 10.9 模型比较
##############################################################################
models = {
    'Logistic回归': LogisticRegression(random_state=123),
    '决策树': DecisionTreeClassifier(max_depth=5, random_state=123),
    '随机森林': RandomForestClassifier(n_estimators=100, random_state=123),
    '梯度提升树': GradientBoostingClassifier(n_estimators=100, random_state=123),
    '神经网络': MLPClassifier(hidden_layer_sizes=(5, 3), max_iter=500,
                               random_state=123)
}

print(f"\n分类模型比较（AUC）:")
for name, model in models.items():
    model.fit(X_train_c, y_train_c)
    y_pred = model.predict_proba(X_test_c)[:, 1]
    auc = roc_auc_score(y_test_c, y_pred)
    print(f"  {name}: AUC={auc:.4f}")


##############################################################################
# 10.10 模型解释：SHAP值
##############################################################################
try:
    import shap
    explainer = shap.TreeExplainer(rf_reg)
    shap_values = explainer.shap_values(X_test)

    fig, ax = plt.subplots(figsize=(8, 5))
    shap.summary_plot(shap_values, X_test, feature_names=['x1','x2','x3','x4','x5'],
                      show=False)
    plt.title('SHAP值变量重要性')
    plt.tight_layout(); plt.show()
except ImportError:
    print("\nSHAP库未安装，跳过SHAP分析")


##############################################################################
# 10.11 保险应用：车险定价
##############################################################################
np.random.seed(456)
n_auto = 2000
age = np.random.uniform(18, 70, n_auto)
gender = np.random.binomial(1, 0.5, n_auto)
vehicle_age = np.random.uniform(0, 15, n_auto)
region = np.random.randint(1, 5, n_auto)
no_claim_years = np.random.randint(0, 11, n_auto)

eta_freq = -1 + 0.02 * (age - 40) - 0.1 * gender + \
           0.1 * vehicle_age + 0.3 * (region == 1) - 0.1 * no_claim_years
lambda_auto = np.exp(eta_freq)
n_claims = np.random.poisson(lambda_auto)

claim_amount = np.zeros(n_auto)
for i in range(n_auto):
    if n_claims[i] > 0:
        eta_sev = 8 + 0.01 * (age[i] - 40) + 0.05 * vehicle_age[i]
        mu_sev = np.exp(eta_sev)
        claim_amount[i] = np.random.gamma(2, mu_sev/2) * n_claims[i]

df_auto = pd.DataFrame({
    'age': age, 'gender': gender, 'vehicle_age': vehicle_age,
    'region': region, 'no_claim_years': no_claim_years,
    'n_claims': n_claims, 'claim_amount': claim_amount
})

X_auto = df_auto[['age', 'gender', 'vehicle_age', 'region', 'no_claim_years']].values
y_freq = df_auto['n_claims'].values
y_sev = df_auto['claim_amount'].values

X_train_a, X_test_a, y_train_f, y_test_f = train_test_split(
    X_auto, y_freq, test_size=0.3, random_state=123)

# 随机森林预测索赔次数
rf_freq = RandomForestRegressor(n_estimators=100, random_state=123)
rf_freq.fit(X_train_a, y_train_f)
print(f"\n车险索赔次数预测: R²={rf_freq.score(X_test_a, y_test_f):.4f}")

# 梯度提升树预测索赔金额
y_sev_pos = y_sev[y_sev > 0]
X_sev_pos = X_auto[y_sev > 0]
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_sev_pos, y_sev_pos, test_size=0.3, random_state=123)
gb_sev = GradientBoostingRegressor(n_estimators=100, random_state=123)
gb_sev.fit(X_train_s, y_train_s)
print(f"车险索赔金额预测: R²={gb_sev.score(X_test_s, y_test_s):.4f}")

# 导入所需的Python模块
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

np.random.seed(123)
n = 1000
X = np.random.normal(0, 1, (n, 5))
df = pd.DataFrame(X, columns=['x1', 'x2', 'x3', 'x4', 'x5'])

# 分类目标
eta = 0.5 + 1.2 * X[:, 0] - 0.8 * X[:, 1] + 0.5 * X[:, 2]
p = 1 / (1 + np.exp(-eta))
y_class = (np.random.uniform(0, 1, n) < p).astype(int)

# 回归目标
y_reg = 2 + 1.5 * X[:, 0] - 0.8 * X[:, 1] + 0.5 * X[:, 2] + \
        0.3 * X[:, 3] + np.random.normal(0, 0.5, n)

df['y_class'] = y_class
df['y_reg'] = y_reg

print("数据描述:")
print(df.describe())


##############################################################################
# 10.2 无监督学习：主成分分析（PCA）
##############################################################################
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

print(f"\nPCA解释方差比: {pca.explained_variance_ratio_}")
print(f"累积解释方差: {np.cumsum(pca.explained_variance_ratio_)}")

##############################################################################
# 第10章 机器学习
# 对应教材：section10.tex
# 内容：无监督学习（PCA、因子分析、聚类）、监督学习
#       （正则化回归、决策树、随机森林、梯度提升树、神经网络）、SHAP值
##############################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


##############################################################################
# 10.1 数据生成
##############################################################################
np.random.seed(123)
n = 1000
X = np.random.normal(0, 1, (n, 5))
df = pd.DataFrame(X, columns=['x1', 'x2', 'x3', 'x4', 'x5'])

# 分类目标
eta = 0.5 + 1.2 * X[:, 0] - 0.8 * X[:, 1] + 0.5 * X[:, 2]
p = 1 / (1 + np.exp(-eta))
y_class = (np.random.uniform(0, 1, n) < p).astype(int)

# 回归目标
y_reg = 2 + 1.5 * X[:, 0] - 0.8 * X[:, 1] + 0.5 * X[:, 2] + \
        0.3 * X[:, 3] + np.random.normal(0, 0.5, n)

df['y_class'] = y_class
df['y_reg'] = y_reg

print("数据描述:")
print(df.describe())


##############################################################################
# 10.2 无监督学习：主成分分析（PCA）
##############################################################################
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

print(f"\nPCA解释方差比: {pca.explained_variance_ratio_}")
print(f"累积解释方差: {np.cumsum(pca.explained_variance_ratio_)}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(range(1, 6), pca.explained_variance_ratio_, alpha=0.7)
axes[0].set_xlabel('主成分'); axes[0].set_ylabel('解释方差比')
axes[0].set_title('PCA碎石图')
axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y_class, alpha=0.5)
axes[1].set_xlabel('PC1'); axes[1].set_ylabel('PC2')
axes[1].set_title('PCA投影')
plt.tight_layout(); plt.show()


## 因子分析
fa = FactorAnalysis(n_components=2, rotation='varimax')
X_fa = fa.fit_transform(X_scaled)
print(f"\n因子分析载荷矩阵:\n{fa.components_.T}")


##############################################################################
# 10.3 无监督学习：聚类分析
##############################################################################

## 10.3.1 K-means聚类
kmeans = KMeans(n_clusters=3, random_state=123, n_init=10)
labels_km = kmeans.fit_predict(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(X[:, 0], X[:, 1], c=labels_km, alpha=0.5)
axes[0].set_title('K-means聚类')
axes[0].set_xlabel('x1'); axes[0].set_ylabel('x2')

## 10.3.2 层次聚类
hc = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels_hc = hc.fit_predict(X_scaled)
axes[1].scatter(X[:, 0], X[:, 1], c=labels_hc, alpha=0.5)
axes[1].set_title('层次聚类')
axes[1].set_xlabel('x1'); axes[1].set_ylabel('x2')
plt.tight_layout(); plt.show()

# 肘部法则
inertias = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=123, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(1, 11), inertias, 'bo-')
ax.set_xlabel('聚类数 k'); ax.set_ylabel('惯性')
ax.set_title('肘部法则')
plt.tight_layout(); plt.show()


##############################################################################
# 10.4 监督学习：正则化回归
##############################################################################
X_train, X_test, y_train, y_test = train_test_split(
    X, y_reg, test_size=0.3, random_state=123)

## 10.4.1 岭回归
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
print(f"\n岭回归: R²={ridge.score(X_test, y_test):.4f}")
print(f"  系数: {ridge.coef_}")

## 10.4.2 LASSO回归
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
print(f"\nLASSO回归: R²={lasso.score(X_test, y_test):.4f}")
print(f"  系数: {lasso.coef_}")

## 10.4.3 Elastic Net
enet = ElasticNet(alpha=0.1, l1_ratio=0.5)
enet.fit(X_train, y_train)
print(f"\nElastic Net: R²={enet.score(X_test, y_test):.4f}")
print(f"  系数: {enet.coef_}")


##############################################################################
# 10.5 监督学习：决策树
##############################################################################
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_class, test_size=0.3, random_state=123)

## 10.5.1 分类树
dt_clf = DecisionTreeClassifier(max_depth=5, random_state=123)
dt_clf.fit(X_train_c, y_train_c)
print(f"\n分类决策树: 准确率={accuracy_score(y_test_c, dt_clf.predict(X_test_c)):.4f}")

## 10.5.2 回归树
dt_reg = DecisionTreeRegressor(max_depth=5, random_state=123)
dt_reg.fit(X_train, y_train)
print(f"回归决策树: R²={dt_reg.score(X_test, y_test):.4f}")


##############################################################################
# 10.6 监督学习：随机森林
##############################################################################
## 10.6.1 分类随机森林
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=5,
                                 random_state=123)
rf_clf.fit(X_train_c, y_train_c)
y_pred_rf = rf_clf.predict_proba(X_test_c)[:, 1]
print(f"\n随机森林（分类）: AUC={roc_auc_score(y_test_c, y_pred_rf):.4f}")

## 10.6.2 回归随机森林
rf_reg = RandomForestRegressor(n_estimators=100, max_depth=5,
                                random_state=123)
rf_reg.fit(X_train, y_train)
print(f"随机森林（回归）: R²={rf_reg.score(X_test, y_test):.4f}")

## 变量重要性
importances = rf_reg.feature_importances_
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(range(5), importances)
ax.set_xticks(range(5))
ax.set_xticklabels(['x1', 'x2', 'x3', 'x4', 'x5'])
ax.set_ylabel('重要性')
ax.set_title('随机森林变量重要性')
plt.tight_layout(); plt.show()


##############################################################################
# 10.7 监督学习：梯度提升树
##############################################################################
## 10.7.1 分类GBDT
gb_clf = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                     random_state=123)
gb_clf.fit(X_train_c, y_train_c)
y_pred_gb = gb_clf.predict_proba(X_test_c)[:, 1]
print(f"\n梯度提升树（分类）: AUC={roc_auc_score(y_test_c, y_pred_gb):.4f}")

## 10.7.2 回归GBDT
gb_reg = GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                    random_state=123)
gb_reg.fit(X_train, y_train)
print(f"梯度提升树（回归）: R²={gb_reg.score(X_test, y_test):.4f}")


##############################################################################
# 10.8 监督学习：神经网络
##############################################################################
## 10.8.1 分类神经网络
nn_clf = MLPClassifier(hidden_layer_sizes=(5, 3), max_iter=500,
                        random_state=123)
nn_clf.fit(X_train_c, y_train_c)
y_pred_nn = nn_clf.predict_proba(X_test_c)[:, 1]
print(f"\n神经网络（分类）: AUC={roc_auc_score(y_test_c, y_pred_nn):.4f}")

## 10.8.2 回归神经网络
nn_reg = MLPRegressor(hidden_layer_sizes=(5, 3), max_iter=500,
                       random_state=123)
nn_reg.fit(X_train, y_train)
print(f"神经网络（回归）: R²={nn_reg.score(X_test, y_test):.4f}")


##############################################################################
# 10.9 模型比较
##############################################################################
models = {
    'Logistic回归': LogisticRegression(random_state=123),
    '决策树': DecisionTreeClassifier(max_depth=5, random_state=123),
    '随机森林': RandomForestClassifier(n_estimators=100, random_state=123),
    '梯度提升树': GradientBoostingClassifier(n_estimators=100, random_state=123),
    '神经网络': MLPClassifier(hidden_layer_sizes=(5, 3), max_iter=500,
                               random_state=123)
}

print(f"\n分类模型比较（AUC）:")
for name, model in models.items():
    model.fit(X_train_c, y_train_c)
    y_pred = model.predict_proba(X_test_c)[:, 1]
    auc = roc_auc_score(y_test_c, y_pred)
    print(f"  {name}: AUC={auc:.4f}")


##############################################################################
# 10.10 模型解释：SHAP值
##############################################################################
try:
    import shap
    explainer = shap.TreeExplainer(rf_reg)
    shap_values = explainer.shap_values(X_test)

    fig, ax = plt.subplots(figsize=(8, 5))
    shap.summary_plot(shap_values, X_test, feature_names=['x1','x2','x3','x4','x5'],
                      show=False)
    plt.title('SHAP值变量重要性')
    plt.tight_layout(); plt.show()
except ImportError:
    print("\nSHAP库未安装，跳过SHAP分析")


##############################################################################
# 10.11 保险应用：车险定价
##############################################################################
np.random.seed(456)
n_auto = 2000
age = np.random.uniform(18, 70, n_auto)
gender = np.random.binomial(1, 0.5, n_auto)
vehicle_age = np.random.uniform(0, 15, n_auto)
region = np.random.randint(1, 5, n_auto)
no_claim_years = np.random.randint(0, 11, n_auto)

eta_freq = -1 + 0.02 * (age - 40) - 0.1 * gender + \
           0.1 * vehicle_age + 0.3 * (region == 1) - 0.1 * no_claim_years
lambda_auto = np.exp(eta_freq)
n_claims = np.random.poisson(lambda_auto)

claim_amount = np.zeros(n_auto)
for i in range(n_auto):
    if n_claims[i] > 0:
        eta_sev = 8 + 0.01 * (age[i] - 40) + 0.05 * vehicle_age[i]
        mu_sev = np.exp(eta_sev)
        claim_amount[i] = np.random.gamma(2, mu_sev/2) * n_claims[i]

df_auto = pd.DataFrame({
    'age': age, 'gender': gender, 'vehicle_age': vehicle_age,
    'region': region, 'no_claim_years': no_claim_years,
    'n_claims': n_claims, 'claim_amount': claim_amount
})

X_auto = df_auto[['age', 'gender', 'vehicle_age', 'region', 'no_claim_years']].values
y_freq = df_auto['n_claims'].values
y_sev = df_auto['claim_amount'].values

X_train_a, X_test_a, y_train_f, y_test_f = train_test_split(
    X_auto, y_freq, test_size=0.3, random_state=123)

# 随机森林预测索赔次数
rf_freq = RandomForestRegressor(n_estimators=100, random_state=123)
rf_freq.fit(X_train_a, y_train_f)
print(f"\n车险索赔次数预测: R²={rf_freq.score(X_test_a, y_test_f):.4f}")

# 梯度提升树预测索赔金额
y_sev_pos = y_sev[y_sev > 0]
X_sev_pos = X_auto[y_sev > 0]
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_sev_pos, y_sev_pos, test_size=0.3, random_state=123)
gb_sev = GradientBoostingRegressor(n_estimators=100, random_state=123)
gb_sev.fit(X_train_s, y_train_s)
print(f"车险索赔金额预测: R²={gb_sev.score(X_test_s, y_test_s):.4f}")

# 导入所需的Python模块
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

np.random.seed(123)
n = 1000
X = np.random.normal(0, 1, (n, 5))
df = pd.DataFrame(X, columns=['x1', 'x2', 'x3', 'x4', 'x5'])

# 分类目标
eta = 0.5 + 1.2 * X[:, 0] - 0.8 * X[:, 1] + 0.5 * X[:, 2]
p = 1 / (1 + np.exp(-eta))
y_class = (np.random.uniform(0, 1, n) < p).astype(int)

# 回归目标
y_reg = 2 + 1.5 * X[:, 0] - 0.8 * X[:, 1] + 0.5 * X[:, 2] + \
        0.3 * X[:, 3] + np.random.normal(0, 0.5, n)

df['y_class'] = y_class
df['y_reg'] = y_reg

print("数据描述:")
print(df.describe())


##############################################################################
# 10.2 无监督学习：主成分分析（PCA）
##############################################################################
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

print(f"\nPCA解释方差比: {pca.explained_variance_ratio_}")
print(f"累积解释方差: {np.cumsum(pca.explained_variance_ratio_)}")

##############################################################################
# 第10章 机器学习
# 对应教材：section10.tex
# 内容：无监督学习（PCA、因子分析、聚类）、监督学习
#       （正则化回归、决策树、随机森林、梯度提升树、神经网络）、SHAP值
##############################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


##############################################################################
# 10.1 数据生成
##############################################################################
np.random.seed(123)
n = 1000
X = np.random.normal(0, 1, (n, 5))
df = pd.DataFrame(X, columns=['x1', 'x2', 'x3', 'x4', 'x5'])

# 分类目标
eta = 0.5 + 1.2 * X[:, 0] - 0.8 * X[:, 1] + 0.5 * X[:, 2]
p = 1 / (1 + np.exp(-eta))
y_class = (np.random.uniform(0, 1, n) < p).astype(int)

# 回归目标
y_reg = 2 + 1.5 * X[:, 0] - 0.8 * X[:, 1] + 0.5 * X[:, 2] + \
        0.3 * X[:, 3] + np.random.normal(0, 0.5, n)

df['y_class'] = y_class
df['y_reg'] = y_reg

print("数据描述:")
print(df.describe())


##############################################################################
# 10.2 无监督学习：主成分分析（PCA）
##############################################################################
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

print(f"\nPCA解释方差比: {pca.explained_variance_ratio_}")
print(f"累积解释方差: {np.cumsum(pca.explained_variance_ratio_)}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(range(1, 6), pca.explained_variance_ratio_, alpha=0.7)
axes[0].set_xlabel('主成分'); axes[0].set_ylabel('解释方差比')
axes[0].set_title('PCA碎石图')
axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y_class, alpha=0.5)
axes[1].set_xlabel('PC1'); axes[1].set_ylabel('PC2')
axes[1].set_title('PCA投影')
plt.tight_layout(); plt.show()


## 因子分析
fa = FactorAnalysis(n_components=2, rotation='varimax')
X_fa = fa.fit_transform(X_scaled)
print(f"\n因子分析载荷矩阵:\n{fa.components_.T}")


##############################################################################
# 10.3 无监督学习：聚类分析
##############################################################################

## 10.3.1 K-means聚类
kmeans = KMeans(n_clusters=3, random_state=123, n_init=10)
labels_km = kmeans.fit_predict(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(X[:, 0], X[:, 1], c=labels_km, alpha=0.5)
axes[0].set_title('K-means聚类')
axes[0].set_xlabel('x1'); axes[0].set_ylabel('x2')

## 10.3.2 层次聚类
hc = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels_hc = hc.fit_predict(X_scaled)
axes[1].scatter(X[:, 0], X[:, 1], c=labels_hc, alpha=0.5)
axes[1].set_title('层次聚类')
axes[1].set_xlabel('x1'); axes[1].set_ylabel('x2')
plt.tight_layout(); plt.show()

# 肘部法则
inertias = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=123, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(1, 11), inertias, 'bo-')
ax.set_xlabel('聚类数 k'); ax.set_ylabel('惯性')
ax.set_title('肘部法则')
plt.tight_layout(); plt.show()


##############################################################################
# 10.4 监督学习：正则化回归
##############################################################################
X_train, X_test, y_train, y_test = train_test_split(
    X, y_reg, test_size=0.3, random_state=123)

## 10.4.1 岭回归
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
print(f"\n岭回归: R²={ridge.score(X_test, y_test):.4f}")
print(f"  系数: {ridge.coef_}")

## 10.4.2 LASSO回归
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
print(f"\nLASSO回归: R²={lasso.score(X_test, y_test):.4f}")
print(f"  系数: {lasso.coef_}")

## 10.4.3 Elastic Net
enet = ElasticNet(alpha=0.1, l1_ratio=0.5)
enet.fit(X_train, y_train)
print(f"\nElastic Net: R²={enet.score(X_test, y_test):.4f}")
print(f"  系数: {enet.coef_}")


##############################################################################
# 10.5 监督学习：决策树
##############################################################################
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_class, test_size=0.3, random_state=123)

## 10.5.1 分类树
dt_clf = DecisionTreeClassifier(max_depth=5, random_state=123)
dt_clf.fit(X_train_c, y_train_c)
print(f"\n分类决策树: 准确率={accuracy_score(y_test_c, dt_clf.predict(X_test_c)):.4f}")

## 10.5.2 回归树
dt_reg = DecisionTreeRegressor(max_depth=5, random_state=123)
dt_reg.fit(X_train, y_train)
print(f"回归决策树: R²={dt_reg.score(X_test, y_test):.4f}")


##############################################################################
# 10.6 监督学习：随机森林
##############################################################################
## 10.6.1 分类随机森林
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=5,
                                 random_state=123)
rf_clf.fit(X_train_c, y_train_c)
y_pred_rf = rf_clf.predict_proba(X_test_c)[:, 1]
print(f"\n随机森林（分类）: AUC={roc_auc_score(y_test_c, y_pred_rf):.4f}")

## 10.6.2 回归随机森林
rf_reg = RandomForestRegressor(n_estimators=100, max_depth=5,
                                random_state=123)
rf_reg.fit(X_train, y_train)
print(f"随机森林（回归）: R²={rf_reg.score(X_test, y_test):.4f}")

## 变量重要性
importances = rf_reg.feature_importances_
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(range(5), importances)
ax.set_xticks(range(5))
ax.set_xticklabels(['x1', 'x2', 'x3', 'x4', 'x5'])
ax.set_ylabel('重要性')
ax.set_title('随机森林变量重要性')
plt.tight_layout(); plt.show()


##############################################################################
# 10.7 监督学习：梯度提升树
##############################################################################
## 10.7.1 分类GBDT
gb_clf = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                     random_state=123)
gb_clf.fit(X_train_c, y_train_c)
y_pred_gb = gb_clf.predict_proba(X_test_c)[:, 1]
print(f"\n梯度提升树（分类）: AUC={roc_auc_score(y_test_c, y_pred_gb):.4f}")

## 10.7.2 回归GBDT
gb_reg = GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                    random_state=123)
gb_reg.fit(X_train, y_train)
print(f"梯度提升树（回归）: R²={gb_reg.score(X_test, y_test):.4f}")


##############################################################################
# 10.8 监督学习：神经网络
##############################################################################
## 10.8.1 分类神经网络
nn_clf = MLPClassifier(hidden_layer_sizes=(5, 3), max_iter=500,
                        random_state=123)
nn_clf.fit(X_train_c, y_train_c)
y_pred_nn = nn_clf.predict_proba(X_test_c)[:, 1]
print(f"\n神经网络（分类）: AUC={roc_auc_score(y_test_c, y_pred_nn):.4f}")

## 10.8.2 回归神经网络
nn_reg = MLPRegressor(hidden_layer_sizes=(5, 3), max_iter=500,
                       random_state=123)
nn_reg.fit(X_train, y_train)
print(f"神经网络（回归）: R²={nn_reg.score(X_test, y_test):.4f}")


##############################################################################
# 10.9 模型比较
##############################################################################
models = {
    'Logistic回归': LogisticRegression(random_state=123),
    '决策树': DecisionTreeClassifier(max_depth=5, random_state=123),
    '随机森林': RandomForestClassifier(n_estimators=100, random_state=123),
    '梯度提升树': GradientBoostingClassifier(n_estimators=100, random_state=123),
    '神经网络': MLPClassifier(hidden_layer_sizes=(5, 3), max_iter=500,
                               random_state=123)
}

print(f"\n分类模型比较（AUC）:")
for name, model in models.items():
    model.fit(X_train_c, y_train_c)
    y_pred = model.predict_proba(X_test_c)[:, 1]
    auc = roc_auc_score(y_test_c, y_pred)
    print(f"  {name}: AUC={auc:.4f}")


##############################################################################
# 10.10 模型解释：SHAP值
##############################################################################
try:
    import shap
    explainer = shap.TreeExplainer(rf_reg)
    shap_values = explainer.shap_values(X_test)

    fig, ax = plt.subplots(figsize=(8, 5))
    shap.summary_plot(shap_values, X_test, feature_names=['x1','x2','x3','x4','x5'],
                      show=False)
    plt.title('SHAP值变量重要性')
    plt.tight_layout(); plt.show()
except ImportError:
    print("\nSHAP库未安装，跳过SHAP分析")


##############################################################################
# 10.11 保险应用：车险定价
##############################################################################
np.random.seed(456)
n_auto = 2000
age = np.random.uniform(18, 70, n_auto)
gender = np.random.binomial(1, 0.5, n_auto)
vehicle_age = np.random.uniform(0, 15, n_auto)
region = np.random.randint(1, 5, n_auto)
no_claim_years = np.random.randint(0, 11, n_auto)

eta_freq = -1 + 0.02 * (age - 40) - 0.1 * gender + \
           0.1 * vehicle_age + 0.3 * (region == 1) - 0.1 * no_claim_years
lambda_auto = np.exp(eta_freq)
n_claims = np.random.poisson(lambda_auto)

claim_amount = np.zeros(n_auto)
for i in range(n_auto):
    if n_claims[i] > 0:
        eta_sev = 8 + 0.01 * (age[i] - 40) + 0.05 * vehicle_age[i]
        mu_sev = np.exp(eta_sev)
        claim_amount[i] = np.random.gamma(2, mu_sev/2) * n_claims[i]

df_auto = pd.DataFrame({
    'age': age, 'gender': gender, 'vehicle_age': vehicle_age,
    'region': region, 'no_claim_years': no_claim_years,
    'n_claims': n_claims, 'claim_amount': claim_amount
})

X_auto = df_auto[['age', 'gender', 'vehicle_age', 'region', 'no_claim_years']].values
y_freq = df_auto['n_claims'].values
y_sev = df_auto['claim_amount'].values

X_train_a, X_test_a, y_train_f, y_test_f = train_test_split(
    X_auto, y_freq, test_size=0.3, random_state=123)

# 随机森林预测索赔次数
rf_freq = RandomForestRegressor(n_estimators=100, random_state=123)
rf_freq.fit(X_train_a, y_train_f)
print(f"\n车险索赔次数预测: R²={rf_freq.score(X_test_a, y_test_f):.4f}")

# 梯度提升树预测索赔金额
y_sev_pos = y_sev[y_sev > 0]
X_sev_pos = X_auto[y_sev > 0]
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_sev_pos, y_sev_pos, test_size=0.3, random_state=123)
gb_sev = GradientBoostingRegressor(n_estimators=100, random_state=123)
gb_sev.fit(X_train_s, y_train_s)
print(f"车险索赔金额预测: R²={gb_sev.score(X_test_s, y_test_s):.4f}")

##############################################################################
# 第10章 机器学习
# 对应教材：section10.tex
# 内容：无监督学习（PCA、因子分析、聚类）、监督学习
#       （正则化回归、决策树、随机森林、梯度提升树、神经网络）、SHAP值
##############################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


##############################################################################
# 10.1 数据生成
##############################################################################
np.random.seed(123)
n = 1000
X = np.random.normal(0, 1, (n, 5))
df = pd.DataFrame(X, columns=['x1', 'x2', 'x3', 'x4', 'x5'])

# 分类目标
eta = 0.5 + 1.2 * X[:, 0] - 0.8 * X[:, 1] + 0.5 * X[:, 2]
p = 1 / (1 + np.exp(-eta))
y_class = (np.random.uniform(0, 1, n) < p).astype(int)

# 回归目标
y_reg = 2 + 1.5 * X[:, 0] - 0.8 * X[:, 1] + 0.5 * X[:, 2] + \
        0.3 * X[:, 3] + np.random.normal(0, 0.5, n)

df['y_class'] = y_class
df['y_reg'] = y_reg

print("数据描述:")
print(df.describe())


##############################################################################
# 10.2 无监督学习：主成分分析（PCA）
##############################################################################
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

print(f"\nPCA解释方差比: {pca.explained_variance_ratio_}")
print(f"累积解释方差: {np.cumsum(pca.explained_variance_ratio_)}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(range(1, 6), pca.explained_variance_ratio_, alpha=0.7)
axes[0].set_xlabel('主成分'); axes[0].set_ylabel('解释方差比')
axes[0].set_title('PCA碎石图')
axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y_class, alpha=0.5)
axes[1].set_xlabel('PC1'); axes[1].set_ylabel('PC2')
axes[1].set_title('PCA投影')
plt.tight_layout(); plt.show()


## 因子分析
fa = FactorAnalysis(n_components=2, rotation='varimax')
X_fa = fa.fit_transform(X_scaled)
print(f"\n因子分析载荷矩阵:\n{fa.components_.T}")


##############################################################################
# 10.3 无监督学习：聚类分析
##############################################################################

## 10.3.1 K-means聚类
kmeans = KMeans(n_clusters=3, random_state=123, n_init=10)
labels_km = kmeans.fit_predict(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(X[:, 0], X[:, 1], c=labels_km, alpha=0.5)
axes[0].set_title('K-means聚类')
axes[0].set_xlabel('x1'); axes[0].set_ylabel('x2')

## 10.3.2 层次聚类
hc = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels_hc = hc.fit_predict(X_scaled)
axes[1].scatter(X[:, 0], X[:, 1], c=labels_hc, alpha=0.5)
axes[1].set_title('层次聚类')
axes[1].set_xlabel('x1'); axes[1].set_ylabel('x2')
plt.tight_layout(); plt.show()

# 肘部法则
inertias = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=123, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(1, 11), inertias, 'bo-')
ax.set_xlabel('聚类数 k'); ax.set_ylabel('惯性')
ax.set_title('肘部法则')
plt.tight_layout(); plt.show()


##############################################################################
# 10.4 监督学习：正则化回归
##############################################################################
X_train, X_test, y_train, y_test = train_test_split(
    X, y_reg, test_size=0.3, random_state=123)

## 10.4.1 岭回归
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
print(f"\n岭回归: R²={ridge.score(X_test, y_test):.4f}")
print(f"  系数: {ridge.coef_}")

## 10.4.2 LASSO回归
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
print(f"\nLASSO回归: R²={lasso.score(X_test, y_test):.4f}")
print(f"  系数: {lasso.coef_}")

## 10.4.3 Elastic Net
enet = ElasticNet(alpha=0.1, l1_ratio=0.5)
enet.fit(X_train, y_train)
print(f"\nElastic Net: R²={enet.score(X_test, y_test):.4f}")
print(f"  系数: {enet.coef_}")


##############################################################################
# 10.5 监督学习：决策树
##############################################################################
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_class, test_size=0.3, random_state=123)

## 10.5.1 分类树
dt_clf = DecisionTreeClassifier(max_depth=5, random_state=123)
dt_clf.fit(X_train_c, y_train_c)
print(f"\n分类决策树: 准确率={accuracy_score(y_test_c, dt_clf.predict(X_test_c)):.4f}")

## 10.5.2 回归树
dt_reg = DecisionTreeRegressor(max_depth=5, random_state=123)
dt_reg.fit(X_train, y_train)
print(f"回归决策树: R²={dt_reg.score(X_test, y_test):.4f}")


##############################################################################
# 10.6 监督学习：随机森林
##############################################################################
## 10.6.1 分类随机森林
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=5,
                                 random_state=123)
rf_clf.fit(X_train_c, y_train_c)
y_pred_rf = rf_clf.predict_proba(X_test_c)[:, 1]
print(f"\n随机森林（分类）: AUC={roc_auc_score(y_test_c, y_pred_rf):.4f}")

## 10.6.2 回归随机森林
rf_reg = RandomForestRegressor(n_estimators=100, max_depth=5,
                                random_state=123)
rf_reg.fit(X_train, y_train)
print(f"随机森林（回归）: R²={rf_reg.score(X_test, y_test):.4f}")

## 变量重要性
importances = rf_reg.feature_importances_
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(range(5), importances)
ax.set_xticks(range(5))
ax.set_xticklabels(['x1', 'x2', 'x3', 'x4', 'x5'])
ax.set_ylabel('重要性')
ax.set_title('随机森林变量重要性')
plt.tight_layout(); plt.show()


##############################################################################
# 10.7 监督学习：梯度提升树
##############################################################################
## 10.7.1 分类GBDT
gb_clf = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                     random_state=123)
gb_clf.fit(X_train_c, y_train_c)
y_pred_gb = gb_clf.predict_proba(X_test_c)[:, 1]
print(f"\n梯度提升树（分类）: AUC={roc_auc_score(y_test_c, y_pred_gb):.4f}")

## 10.7.2 回归GBDT
gb_reg = GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                    random_state=123)
gb_reg.fit(X_train, y_train)
print(f"梯度提升树（回归）: R²={gb_reg.score(X_test, y_test):.4f}")


##############################################################################
# 10.8 监督学习：神经网络
##############################################################################
## 10.8.1 分类神经网络
nn_clf = MLPClassifier(hidden_layer_sizes=(5, 3), max_iter=500,
                        random_state=123)
nn_clf.fit(X_train_c, y_train_c)
y_pred_nn = nn_clf.predict_proba(X_test_c)[:, 1]
print(f"\n神经网络（分类）: AUC={roc_auc_score(y_test_c, y_pred_nn):.4f}")

## 10.8.2 回归神经网络
nn_reg = MLPRegressor(hidden_layer_sizes=(5, 3), max_iter=500,
                       random_state=123)
nn_reg.fit(X_train, y_train)
print(f"神经网络（回归）: R²={nn_reg.score(X_test, y_test):.4f}")


##############################################################################
# 10.9 模型比较
##############################################################################
models = {
    'Logistic回归': LogisticRegression(random_state=123),
    '决策树': DecisionTreeClassifier(max_depth=5, random_state=123),
    '随机森林': RandomForestClassifier(n_estimators=100, random_state=123),
    '梯度提升树': GradientBoostingClassifier(n_estimators=100, random_state=123),
    '神经网络': MLPClassifier(hidden_layer_sizes=(5, 3), max_iter=500,
                               random_state=123)
}

print(f"\n分类模型比较（AUC）:")
for name, model in models.items():
    model.fit(X_train_c, y_train_c)
    y_pred = model.predict_proba(X_test_c)[:, 1]
    auc = roc_auc_score(y_test_c, y_pred)
    print(f"  {name}: AUC={auc:.4f}")


##############################################################################
# 10.10 模型解释：SHAP值
##############################################################################
try:
    import shap
    explainer = shap.TreeExplainer(rf_reg)
    shap_values = explainer.shap_values(X_test)

    fig, ax = plt.subplots(figsize=(8, 5))
    shap.summary_plot(shap_values, X_test, feature_names=['x1','x2','x3','x4','x5'],
                      show=False)
    plt.title('SHAP值变量重要性')
    plt.tight_layout(); plt.show()
except ImportError:
    print("\nSHAP库未安装，跳过SHAP分析")


##############################################################################
# 10.11 保险应用：车险定价
##############################################################################
np.random.seed(456)
n_auto = 2000
age = np.random.uniform(18, 70, n_auto)
gender = np.random.binomial(1, 0.5, n_auto)
vehicle_age = np.random.uniform(0, 15, n_auto)
region = np.random.randint(1, 5, n_auto)
no_claim_years = np.random.randint(0, 11, n_auto)

eta_freq = -1 + 0.02 * (age - 40) - 0.1 * gender + \
           0.1 * vehicle_age + 0.3 * (region == 1) - 0.1 * no_claim_years
lambda_auto = np.exp(eta_freq)
n_claims = np.random.poisson(lambda_auto)

claim_amount = np.zeros(n_auto)
for i in range(n_auto):
    if n_claims[i] > 0:
        eta_sev = 8 + 0.01 * (age[i] - 40) + 0.05 * vehicle_age[i]
        mu_sev = np.exp(eta_sev)
        claim_amount[i] = np.random.gamma(2, mu_sev/2) * n_claims[i]

df_auto = pd.DataFrame({
    'age': age, 'gender': gender, 'vehicle_age': vehicle_age,
    'region': region, 'no_claim_years': no_claim_years,
    'n_claims': n_claims, 'claim_amount': claim_amount
})

X_auto = df_auto[['age', 'gender', 'vehicle_age', 'region', 'no_claim_years']].values
y_freq = df_auto['n_claims'].values
y_sev = df_auto['claim_amount'].values

X_train_a, X_test_a, y_train_f, y_test_f = train_test_split(
    X_auto, y_freq, test_size=0.3, random_state=123)

# 随机森林预测索赔次数
rf_freq = RandomForestRegressor(n_estimators=100, random_state=123)
rf_freq.fit(X_train_a, y_train_f)
print(f"\n车险索赔次数预测: R²={rf_freq.score(X_test_a, y_test_f):.4f}")

# 梯度提升树预测索赔金额
y_sev_pos = y_sev[y_sev > 0]
X_sev_pos = X_auto[y_sev > 0]
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_sev_pos, y_sev_pos, test_size=0.3, random_state=123)
gb_sev = GradientBoostingRegressor(n_estimators=100, random_state=123)
gb_sev.fit(X_train_s, y_train_s)
print(f"车险索赔金额预测: R²={gb_sev.score(X_test_s, y_test_s):.4f}")

##############################################################################
# 第10章 机器学习
# 对应教材：section10.tex
# 内容：无监督学习（PCA、因子分析、聚类）、监督学习
#       （正则化回归、决策树、随机森林、梯度提升树、神经网络）、SHAP值
##############################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


##############################################################################
# 10.1 数据生成
##############################################################################
np.random.seed(123)
n = 1000
X = np.random.normal(0, 1, (n, 5))
df = pd.DataFrame(X, columns=['x1', 'x2', 'x3', 'x4', 'x5'])

# 分类目标
eta = 0.5 + 1.2 * X[:, 0] - 0.8 * X[:, 1] + 0.5 * X[:, 2]
p = 1 / (1 + np.exp(-eta))
y_class = (np.random.uniform(0, 1, n) < p).astype(int)

# 回归目标
y_reg = 2 + 1.5 * X[:, 0] - 0.8 * X[:, 1] + 0.5 * X[:, 2] + \
        0.3 * X[:, 3] + np.random.normal(0, 0.5, n)

df['y_class'] = y_class
df['y_reg'] = y_reg

print("数据描述:")
print(df.describe())


##############################################################################
# 10.2 无监督学习：主成分分析（PCA）
##############################################################################
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

print(f"\nPCA解释方差比: {pca.explained_variance_ratio_}")
print(f"累积解释方差: {np.cumsum(pca.explained_variance_ratio_)}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(range(1, 6), pca.explained_variance_ratio_, alpha=0.7)
axes[0].set_xlabel('主成分'); axes[0].set_ylabel('解释方差比')
axes[0].set_title('PCA碎石图')
axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y_class, alpha=0.5)
axes[1].set_xlabel('PC1'); axes[1].set_ylabel('PC2')
axes[1].set_title('PCA投影')
plt.tight_layout(); plt.show()


## 因子分析
fa = FactorAnalysis(n_components=2, rotation='varimax')
X_fa = fa.fit_transform(X_scaled)
print(f"\n因子分析载荷矩阵:\n{fa.components_.T}")


##############################################################################
# 10.3 无监督学习：聚类分析
##############################################################################

## 10.3.1 K-means聚类
kmeans = KMeans(n_clusters=3, random_state=123, n_init=10)
labels_km = kmeans.fit_predict(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(X[:, 0], X[:, 1], c=labels_km, alpha=0.5)
axes[0].set_title('K-means聚类')
axes[0].set_xlabel('x1'); axes[0].set_ylabel('x2')

## 10.3.2 层次聚类
hc = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels_hc = hc.fit_predict(X_scaled)
axes[1].scatter(X[:, 0], X[:, 1], c=labels_hc, alpha=0.5)
axes[1].set_title('层次聚类')
axes[1].set_xlabel('x1'); axes[1].set_ylabel('x2')
plt.tight_layout(); plt.show()

# 肘部法则
inertias = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=123, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(1, 11), inertias, 'bo-')
ax.set_xlabel('聚类数 k'); ax.set_ylabel('惯性')
ax.set_title('肘部法则')
plt.tight_layout(); plt.show()


##############################################################################
# 10.4 监督学习：正则化回归
##############################################################################
X_train, X_test, y_train, y_test = train_test_split(
    X, y_reg, test_size=0.3, random_state=123)

## 10.4.1 岭回归
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
print(f"\n岭回归: R²={ridge.score(X_test, y_test):.4f}")
print(f"  系数: {ridge.coef_}")

## 10.4.2 LASSO回归
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
print(f"\nLASSO回归: R²={lasso.score(X_test, y_test):.4f}")
print(f"  系数: {lasso.coef_}")

## 10.4.3 Elastic Net
enet = ElasticNet(alpha=0.1, l1_ratio=0.5)
enet.fit(X_train, y_train)
print(f"\nElastic Net: R²={enet.score(X_test, y_test):.4f}")
print(f"  系数: {enet.coef_}")


##############################################################################
# 10.5 监督学习：决策树
##############################################################################
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_class, test_size=0.3, random_state=123)

## 10.5.1 分类树
dt_clf = DecisionTreeClassifier(max_depth=5, random_state=123)
dt_clf.fit(X_train_c, y_train_c)
print(f"\n分类决策树: 准确率={accuracy_score(y_test_c, dt_clf.predict(X_test_c)):.4f}")

## 10.5.2 回归树
dt_reg = DecisionTreeRegressor(max_depth=5, random_state=123)
dt_reg.fit(X_train, y_train)
print(f"回归决策树: R²={dt_reg.score(X_test, y_test):.4f}")


##############################################################################
# 10.6 监督学习：随机森林
##############################################################################
## 10.6.1 分类随机森林
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=5,
                                 random_state=123)
rf_clf.fit(X_train_c, y_train_c)
y_pred_rf = rf_clf.predict_proba(X_test_c)[:, 1]
print(f"\n随机森林（分类）: AUC={roc_auc_score(y_test_c, y_pred_rf):.4f}")

## 10.6.2 回归随机森林
rf_reg = RandomForestRegressor(n_estimators=100, max_depth=5,
                                random_state=123)
rf_reg.fit(X_train, y_train)
print(f"随机森林（回归）: R²={rf_reg.score(X_test, y_test):.4f}")

## 变量重要性
importances = rf_reg.feature_importances_
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(range(5), importances)
ax.set_xticks(range(5))
ax.set_xticklabels(['x1', 'x2', 'x3', 'x4', 'x5'])
ax.set_ylabel('重要性')
ax.set_title('随机森林变量重要性')
plt.tight_layout(); plt.show()


##############################################################################
# 10.7 监督学习：梯度提升树
##############################################################################
## 10.7.1 分类GBDT
gb_clf = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                     random_state=123)
gb_clf.fit(X_train_c, y_train_c)
y_pred_gb = gb_clf.predict_proba(X_test_c)[:, 1]
print(f"\n梯度提升树（分类）: AUC={roc_auc_score(y_test_c, y_pred_gb):.4f}")

## 10.7.2 回归GBDT
gb_reg = GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                    random_state=123)
gb_reg.fit(X_train, y_train)
print(f"梯度提升树（回归）: R²={gb_reg.score(X_test, y_test):.4f}")


##############################################################################
# 10.8 监督学习：神经网络
##############################################################################
## 10.8.1 分类神经网络
nn_clf = MLPClassifier(hidden_layer_sizes=(5, 3), max_iter=500,
                        random_state=123)
nn_clf.fit(X_train_c, y_train_c)
y_pred_nn = nn_clf.predict_proba(X_test_c)[:, 1]
print(f"\n神经网络（分类）: AUC={roc_auc_score(y_test_c, y_pred_nn):.4f}")

## 10.8.2 回归神经网络
nn_reg = MLPRegressor(hidden_layer_sizes=(5, 3), max_iter=500,
                       random_state=123)
nn_reg.fit(X_train, y_train)
print(f"神经网络（回归）: R²={nn_reg.score(X_test, y_test):.4f}")


##############################################################################
# 10.9 模型比较
##############################################################################
models = {
    'Logistic回归': LogisticRegression(random_state=123),
    '决策树': DecisionTreeClassifier(max_depth=5, random_state=123),
    '随机森林': RandomForestClassifier(n_estimators=100, random_state=123),
    '梯度提升树': GradientBoostingClassifier(n_estimators=100, random_state=123),
    '神经网络': MLPClassifier(hidden_layer_sizes=(5, 3), max_iter=500,
                               random_state=123)
}

print(f"\n分类模型比较（AUC）:")
for name, model in models.items():
    model.fit(X_train_c, y_train_c)
    y_pred = model.predict_proba(X_test_c)[:, 1]
    auc = roc_auc_score(y_test_c, y_pred)
    print(f"  {name}: AUC={auc:.4f}")


##############################################################################
# 10.10 模型解释：SHAP值
##############################################################################
try:
    import shap
    explainer = shap.TreeExplainer(rf_reg)
    shap_values = explainer.shap_values(X_test)

    fig, ax = plt.subplots(figsize=(8, 5))
    shap.summary_plot(shap_values, X_test, feature_names=['x1','x2','x3','x4','x5'],
                      show=False)
    plt.title('SHAP值变量重要性')
    plt.tight_layout(); plt.show()
except ImportError:
    print("\nSHAP库未安装，跳过SHAP分析")


##############################################################################
# 10.11 保险应用：车险定价
##############################################################################
np.random.seed(456)
n_auto = 2000
age = np.random.uniform(18, 70, n_auto)
gender = np.random.binomial(1, 0.5, n_auto)
vehicle_age = np.random.uniform(0, 15, n_auto)
region = np.random.randint(1, 5, n_auto)
no_claim_years = np.random.randint(0, 11, n_auto)

eta_freq = -1 + 0.02 * (age - 40) - 0.1 * gender + \
           0.1 * vehicle_age + 0.3 * (region == 1) - 0.1 * no_claim_years
lambda_auto = np.exp(eta_freq)
n_claims = np.random.poisson(lambda_auto)

claim_amount = np.zeros(n_auto)
for i in range(n_auto):
    if n_claims[i] > 0:
        eta_sev = 8 + 0.01 * (age[i] - 40) + 0.05 * vehicle_age[i]
        mu_sev = np.exp(eta_sev)
        claim_amount[i] = np.random.gamma(2, mu_sev/2) * n_claims[i]

df_auto = pd.DataFrame({
    'age': age, 'gender': gender, 'vehicle_age': vehicle_age,
    'region': region, 'no_claim_years': no_claim_years,
    'n_claims': n_claims, 'claim_amount': claim_amount
})

X_auto = df_auto[['age', 'gender', 'vehicle_age', 'region', 'no_claim_years']].values
y_freq = df_auto['n_claims'].values
y_sev = df_auto['claim_amount'].values

X_train_a, X_test_a, y_train_f, y_test_f = train_test_split(
    X_auto, y_freq, test_size=0.3, random_state=123)

# 随机森林预测索赔次数
rf_freq = RandomForestRegressor(n_estimators=100, random_state=123)
rf_freq.fit(X_train_a, y_train_f)
print(f"\n车险索赔次数预测: R²={rf_freq.score(X_test_a, y_test_f):.4f}")

# 梯度提升树预测索赔金额
y_sev_pos = y_sev[y_sev > 0]
X_sev_pos = X_auto[y_sev > 0]
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_sev_pos, y_sev_pos, test_size=0.3, random_state=123)
gb_sev = GradientBoostingRegressor(n_estimators=100, random_state=123)
gb_sev.fit(X_train_s, y_train_s)
print(f"车险索赔金额预测: R²={gb_sev.score(X_test_s, y_test_s):.4f}")

# 导入所需的Python模块
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

np.random.seed(123)
n = 1000
X = np.random.normal(0, 1, (n, 5))
df = pd.DataFrame(X, columns=['x1', 'x2', 'x3', 'x4', 'x5'])

# 分类目标
eta = 0.5 + 1.2 * X[:, 0] - 0.8 * X[:, 1] + 0.5 * X[:, 2]
p = 1 / (1 + np.exp(-eta))
y_class = (np.random.uniform(0, 1, n) < p).astype(int)

# 回归目标
y_reg = 2 + 1.5 * X[:, 0] - 0.8 * X[:, 1] + 0.5 * X[:, 2] + \
        0.3 * X[:, 3] + np.random.normal(0, 0.5, n)

df['y_class'] = y_class
df['y_reg'] = y_reg

print("数据描述:")
print(df.describe())


##############################################################################
# 10.2 无监督学习：主成分分析（PCA）
##############################################################################
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

print(f"\nPCA解释方差比: {pca.explained_variance_ratio_}")
print(f"累积解释方差: {np.cumsum(pca.explained_variance_ratio_)}")

##############################################################################
# 第10章 机器学习
# 对应教材：section10.tex
# 内容：无监督学习（PCA、因子分析、聚类）、监督学习
#       （正则化回归、决策树、随机森林、梯度提升树、神经网络）、SHAP值
##############################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


##############################################################################
# 10.1 数据生成
##############################################################################
np.random.seed(123)
n = 1000
X = np.random.normal(0, 1, (n, 5))
df = pd.DataFrame(X, columns=['x1', 'x2', 'x3', 'x4', 'x5'])

# 分类目标
eta = 0.5 + 1.2 * X[:, 0] - 0.8 * X[:, 1] + 0.5 * X[:, 2]
p = 1 / (1 + np.exp(-eta))
y_class = (np.random.uniform(0, 1, n) < p).astype(int)

# 回归目标
y_reg = 2 + 1.5 * X[:, 0] - 0.8 * X[:, 1] + 0.5 * X[:, 2] + \
        0.3 * X[:, 3] + np.random.normal(0, 0.5, n)

df['y_class'] = y_class
df['y_reg'] = y_reg

print("数据描述:")
print(df.describe())


##############################################################################
# 10.2 无监督学习：主成分分析（PCA）
##############################################################################
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

print(f"\nPCA解释方差比: {pca.explained_variance_ratio_}")
print(f"累积解释方差: {np.cumsum(pca.explained_variance_ratio_)}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(range(1, 6), pca.explained_variance_ratio_, alpha=0.7)
axes[0].set_xlabel('主成分'); axes[0].set_ylabel('解释方差比')
axes[0].set_title('PCA碎石图')
axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y_class, alpha=0.5)
axes[1].set_xlabel('PC1'); axes[1].set_ylabel('PC2')
axes[1].set_title('PCA投影')
plt.tight_layout(); plt.show()


## 因子分析
fa = FactorAnalysis(n_components=2, rotation='varimax')
X_fa = fa.fit_transform(X_scaled)
print(f"\n因子分析载荷矩阵:\n{fa.components_.T}")


##############################################################################
# 10.3 无监督学习：聚类分析
##############################################################################

## 10.3.1 K-means聚类
kmeans = KMeans(n_clusters=3, random_state=123, n_init=10)
labels_km = kmeans.fit_predict(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(X[:, 0], X[:, 1], c=labels_km, alpha=0.5)
axes[0].set_title('K-means聚类')
axes[0].set_xlabel('x1'); axes[0].set_ylabel('x2')

## 10.3.2 层次聚类
hc = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels_hc = hc.fit_predict(X_scaled)
axes[1].scatter(X[:, 0], X[:, 1], c=labels_hc, alpha=0.5)
axes[1].set_title('层次聚类')
axes[1].set_xlabel('x1'); axes[1].set_ylabel('x2')
plt.tight_layout(); plt.show()

# 肘部法则
inertias = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=123, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(1, 11), inertias, 'bo-')
ax.set_xlabel('聚类数 k'); ax.set_ylabel('惯性')
ax.set_title('肘部法则')
plt.tight_layout(); plt.show()


##############################################################################
# 10.4 监督学习：正则化回归
##############################################################################
X_train, X_test, y_train, y_test = train_test_split(
    X, y_reg, test_size=0.3, random_state=123)

## 10.4.1 岭回归
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
print(f"\n岭回归: R²={ridge.score(X_test, y_test):.4f}")
print(f"  系数: {ridge.coef_}")

## 10.4.2 LASSO回归
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
print(f"\nLASSO回归: R²={lasso.score(X_test, y_test):.4f}")
print(f"  系数: {lasso.coef_}")

## 10.4.3 Elastic Net
enet = ElasticNet(alpha=0.1, l1_ratio=0.5)
enet.fit(X_train, y_train)
print(f"\nElastic Net: R²={enet.score(X_test, y_test):.4f}")
print(f"  系数: {enet.coef_}")


##############################################################################
# 10.5 监督学习：决策树
##############################################################################
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_class, test_size=0.3, random_state=123)

## 10.5.1 分类树
dt_clf = DecisionTreeClassifier(max_depth=5, random_state=123)
dt_clf.fit(X_train_c, y_train_c)
print(f"\n分类决策树: 准确率={accuracy_score(y_test_c, dt_clf.predict(X_test_c)):.4f}")

## 10.5.2 回归树
dt_reg = DecisionTreeRegressor(max_depth=5, random_state=123)
dt_reg.fit(X_train, y_train)
print(f"回归决策树: R²={dt_reg.score(X_test, y_test):.4f}")


##############################################################################
# 10.6 监督学习：随机森林
##############################################################################
## 10.6.1 分类随机森林
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=5,
                                 random_state=123)
rf_clf.fit(X_train_c, y_train_c)
y_pred_rf = rf_clf.predict_proba(X_test_c)[:, 1]
print(f"\n随机森林（分类）: AUC={roc_auc_score(y_test_c, y_pred_rf):.4f}")

## 10.6.2 回归随机森林
rf_reg = RandomForestRegressor(n_estimators=100, max_depth=5,
                                random_state=123)
rf_reg.fit(X_train, y_train)
print(f"随机森林（回归）: R²={rf_reg.score(X_test, y_test):.4f}")

## 变量重要性
importances = rf_reg.feature_importances_
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(range(5), importances)
ax.set_xticks(range(5))
ax.set_xticklabels(['x1', 'x2', 'x3', 'x4', 'x5'])
ax.set_ylabel('重要性')
ax.set_title('随机森林变量重要性')
plt.tight_layout(); plt.show()


##############################################################################
# 10.7 监督学习：梯度提升树
##############################################################################
## 10.7.1 分类GBDT
gb_clf = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                     random_state=123)
gb_clf.fit(X_train_c, y_train_c)
y_pred_gb = gb_clf.predict_proba(X_test_c)[:, 1]
print(f"\n梯度提升树（分类）: AUC={roc_auc_score(y_test_c, y_pred_gb):.4f}")

## 10.7.2 回归GBDT
gb_reg = GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                    random_state=123)
gb_reg.fit(X_train, y_train)
print(f"梯度提升树（回归）: R²={gb_reg.score(X_test, y_test):.4f}")


##############################################################################
# 10.8 监督学习：神经网络
##############################################################################
## 10.8.1 分类神经网络
nn_clf = MLPClassifier(hidden_layer_sizes=(5, 3), max_iter=500,
                        random_state=123)
nn_clf.fit(X_train_c, y_train_c)
y_pred_nn = nn_clf.predict_proba(X_test_c)[:, 1]
print(f"\n神经网络（分类）: AUC={roc_auc_score(y_test_c, y_pred_nn):.4f}")

## 10.8.2 回归神经网络
nn_reg = MLPRegressor(hidden_layer_sizes=(5, 3), max_iter=500,
                       random_state=123)
nn_reg.fit(X_train, y_train)
print(f"神经网络（回归）: R²={nn_reg.score(X_test, y_test):.4f}")


##############################################################################
# 10.9 模型比较
##############################################################################
models = {
    'Logistic回归': LogisticRegression(random_state=123),
    '决策树': DecisionTreeClassifier(max_depth=5, random_state=123),
    '随机森林': RandomForestClassifier(n_estimators=100, random_state=123),
    '梯度提升树': GradientBoostingClassifier(n_estimators=100, random_state=123),
    '神经网络': MLPClassifier(hidden_layer_sizes=(5, 3), max_iter=500,
                               random_state=123)
}

print(f"\n分类模型比较（AUC）:")
for name, model in models.items():
    model.fit(X_train_c, y_train_c)
    y_pred = model.predict_proba(X_test_c)[:, 1]
    auc = roc_auc_score(y_test_c, y_pred)
    print(f"  {name}: AUC={auc:.4f}")


##############################################################################
# 10.10 模型解释：SHAP值
##############################################################################
try:
    import shap
    explainer = shap.TreeExplainer(rf_reg)
    shap_values = explainer.shap_values(X_test)

    fig, ax = plt.subplots(figsize=(8, 5))
    shap.summary_plot(shap_values, X_test, feature_names=['x1','x2','x3','x4','x5'],
                      show=False)
    plt.title('SHAP值变量重要性')
    plt.tight_layout(); plt.show()
except ImportError:
    print("\nSHAP库未安装，跳过SHAP分析")


##############################################################################
# 10.11 保险应用：车险定价
##############################################################################
np.random.seed(456)
n_auto = 2000
age = np.random.uniform(18, 70, n_auto)
gender = np.random.binomial(1, 0.5, n_auto)
vehicle_age = np.random.uniform(0, 15, n_auto)
region = np.random.randint(1, 5, n_auto)
no_claim_years = np.random.randint(0, 11, n_auto)

eta_freq = -1 + 0.02 * (age - 40) - 0.1 * gender + \
           0.1 * vehicle_age + 0.3 * (region == 1) - 0.1 * no_claim_years
lambda_auto = np.exp(eta_freq)
n_claims = np.random.poisson(lambda_auto)

claim_amount = np.zeros(n_auto)
for i in range(n_auto):
    if n_claims[i] > 0:
        eta_sev = 8 + 0.01 * (age[i] - 40) + 0.05 * vehicle_age[i]
        mu_sev = np.exp(eta_sev)
        claim_amount[i] = np.random.gamma(2, mu_sev/2) * n_claims[i]

df_auto = pd.DataFrame({
    'age': age, 'gender': gender, 'vehicle_age': vehicle_age,
    'region': region, 'no_claim_years': no_claim_years,
    'n_claims': n_claims, 'claim_amount': claim_amount
})

X_auto = df_auto[['age', 'gender', 'vehicle_age', 'region', 'no_claim_years']].values
y_freq = df_auto['n_claims'].values
y_sev = df_auto['claim_amount'].values

X_train_a, X_test_a, y_train_f, y_test_f = train_test_split(
    X_auto, y_freq, test_size=0.3, random_state=123)

# 随机森林预测索赔次数
rf_freq = RandomForestRegressor(n_estimators=100, random_state=123)
rf_freq.fit(X_train_a, y_train_f)
print(f"\n车险索赔次数预测: R²={rf_freq.score(X_test_a, y_test_f):.4f}")

# 梯度提升树预测索赔金额
y_sev_pos = y_sev[y_sev > 0]
X_sev_pos = X_auto[y_sev > 0]
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_sev_pos, y_sev_pos, test_size=0.3, random_state=123)
gb_sev = GradientBoostingRegressor(n_estimators=100, random_state=123)
gb_sev.fit(X_train_s, y_train_s)
print(f"车险索赔金额预测: R²={gb_sev.score(X_test_s, y_test_s):.4f}")

# 导入所需的Python模块
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

np.random.seed(123)
n = 1000
X = np.random.normal(0, 1, (n, 5))
df = pd.DataFrame(X, columns=['x1', 'x2', 'x3', 'x4', 'x5'])

# 分类目标
eta = 0.5 + 1.2 * X[:, 0] - 0.8 * X[:, 1] + 0.5 * X[:, 2]
p = 1 / (1 + np.exp(-eta))
y_class = (np.random.uniform(0, 1, n) < p).astype(int)

# 回归目标
y_reg = 2 + 1.5 * X[:, 0] - 0.8 * X[:, 1] + 0.5 * X[:, 2] + \
        0.3 * X[:, 3] + np.random.normal(0, 0.5, n)

df['y_class'] = y_class
df['y_reg'] = y_reg

print("数据描述:")
print(df.describe())


##############################################################################
# 10.2 无监督学习：主成分分析（PCA）
##############################################################################
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

print(f"\nPCA解释方差比: {pca.explained_variance_ratio_}")
print(f"累积解释方差: {np.cumsum(pca.explained_variance_ratio_)}")

##############################################################################
# 第10章 机器学习
# 对应教材：section10.tex
# 内容：无监督学习（PCA、因子分析、聚类）、监督学习
#       （正则化回归、决策树、随机森林、梯度提升树、神经网络）、SHAP值
##############################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


##############################################################################
# 10.1 数据生成
##############################################################################
np.random.seed(123)
n = 1000
X = np.random.normal(0, 1, (n, 5))
df = pd.DataFrame(X, columns=['x1', 'x2', 'x3', 'x4', 'x5'])

# 分类目标
eta = 0.5 + 1.2 * X[:, 0] - 0.8 * X[:, 1] + 0.5 * X[:, 2]
p = 1 / (1 + np.exp(-eta))
y_class = (np.random.uniform(0, 1, n) < p).astype(int)

# 回归目标
y_reg = 2 + 1.5 * X[:, 0] - 0.8 * X[:, 1] + 0.5 * X[:, 2] + \
        0.3 * X[:, 3] + np.random.normal(0, 0.5, n)

df['y_class'] = y_class
df['y_reg'] = y_reg

print("数据描述:")
print(df.describe())


##############################################################################
# 10.2 无监督学习：主成分分析（PCA）
##############################################################################
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

print(f"\nPCA解释方差比: {pca.explained_variance_ratio_}")
print(f"累积解释方差: {np.cumsum(pca.explained_variance_ratio_)}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(range(1, 6), pca.explained_variance_ratio_, alpha=0.7)
axes[0].set_xlabel('主成分'); axes[0].set_ylabel('解释方差比')
axes[0].set_title('PCA碎石图')
axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y_class, alpha=0.5)
axes[1].set_xlabel('PC1'); axes[1].set_ylabel('PC2')
axes[1].set_title('PCA投影')
plt.tight_layout(); plt.show()


## 因子分析
fa = FactorAnalysis(n_components=2, rotation='varimax')
X_fa = fa.fit_transform(X_scaled)
print(f"\n因子分析载荷矩阵:\n{fa.components_.T}")


##############################################################################
# 10.3 无监督学习：聚类分析
##############################################################################

## 10.3.1 K-means聚类
kmeans = KMeans(n_clusters=3, random_state=123, n_init=10)
labels_km = kmeans.fit_predict(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(X[:, 0], X[:, 1], c=labels_km, alpha=0.5)
axes[0].set_title('K-means聚类')
axes[0].set_xlabel('x1'); axes[0].set_ylabel('x2')

## 10.3.2 层次聚类
hc = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels_hc = hc.fit_predict(X_scaled)
axes[1].scatter(X[:, 0], X[:, 1], c=labels_hc, alpha=0.5)
axes[1].set_title('层次聚类')
axes[1].set_xlabel('x1'); axes[1].set_ylabel('x2')
plt.tight_layout(); plt.show()

# 肘部法则
inertias = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=123, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(1, 11), inertias, 'bo-')
ax.set_xlabel('聚类数 k'); ax.set_ylabel('惯性')
ax.set_title('肘部法则')
plt.tight_layout(); plt.show()


##############################################################################
# 10.4 监督学习：正则化回归
##############################################################################
X_train, X_test, y_train, y_test = train_test_split(
    X, y_reg, test_size=0.3, random_state=123)

## 10.4.1 岭回归
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
print(f"\n岭回归: R²={ridge.score(X_test, y_test):.4f}")
print(f"  系数: {ridge.coef_}")

## 10.4.2 LASSO回归
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
print(f"\nLASSO回归: R²={lasso.score(X_test, y_test):.4f}")
print(f"  系数: {lasso.coef_}")

## 10.4.3 Elastic Net
enet = ElasticNet(alpha=0.1, l1_ratio=0.5)
enet.fit(X_train, y_train)
print(f"\nElastic Net: R²={enet.score(X_test, y_test):.4f}")
print(f"  系数: {enet.coef_}")


##############################################################################
# 10.5 监督学习：决策树
##############################################################################
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_class, test_size=0.3, random_state=123)

## 10.5.1 分类树
dt_clf = DecisionTreeClassifier(max_depth=5, random_state=123)
dt_clf.fit(X_train_c, y_train_c)
print(f"\n分类决策树: 准确率={accuracy_score(y_test_c, dt_clf.predict(X_test_c)):.4f}")

## 10.5.2 回归树
dt_reg = DecisionTreeRegressor(max_depth=5, random_state=123)
dt_reg.fit(X_train, y_train)
print(f"回归决策树: R²={dt_reg.score(X_test, y_test):.4f}")


##############################################################################
# 10.6 监督学习：随机森林
##############################################################################
## 10.6.1 分类随机森林
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=5,
                                 random_state=123)
rf_clf.fit(X_train_c, y_train_c)
y_pred_rf = rf_clf.predict_proba(X_test_c)[:, 1]
print(f"\n随机森林（分类）: AUC={roc_auc_score(y_test_c, y_pred_rf):.4f}")

## 10.6.2 回归随机森林
rf_reg = RandomForestRegressor(n_estimators=100, max_depth=5,
                                random_state=123)
rf_reg.fit(X_train, y_train)
print(f"随机森林（回归）: R²={rf_reg.score(X_test, y_test):.4f}")

## 变量重要性
importances = rf_reg.feature_importances_
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(range(5), importances)
ax.set_xticks(range(5))
ax.set_xticklabels(['x1', 'x2', 'x3', 'x4', 'x5'])
ax.set_ylabel('重要性')
ax.set_title('随机森林变量重要性')
plt.tight_layout(); plt.show()


##############################################################################
# 10.7 监督学习：梯度提升树
##############################################################################
## 10.7.1 分类GBDT
gb_clf = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                     random_state=123)
gb_clf.fit(X_train_c, y_train_c)
y_pred_gb = gb_clf.predict_proba(X_test_c)[:, 1]
print(f"\n梯度提升树（分类）: AUC={roc_auc_score(y_test_c, y_pred_gb):.4f}")

## 10.7.2 回归GBDT
gb_reg = GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                    random_state=123)
gb_reg.fit(X_train, y_train)
print(f"梯度提升树（回归）: R²={gb_reg.score(X_test, y_test):.4f}")


##############################################################################
# 10.8 监督学习：神经网络
##############################################################################
## 10.8.1 分类神经网络
nn_clf = MLPClassifier(hidden_layer_sizes=(5, 3), max_iter=500,
                        random_state=123)
nn_clf.fit(X_train_c, y_train_c)
y_pred_nn = nn_clf.predict_proba(X_test_c)[:, 1]
print(f"\n神经网络（分类）: AUC={roc_auc_score(y_test_c, y_pred_nn):.4f}")

## 10.8.2 回归神经网络
nn_reg = MLPRegressor(hidden_layer_sizes=(5, 3), max_iter=500,
                       random_state=123)
nn_reg.fit(X_train, y_train)
print(f"神经网络（回归）: R²={nn_reg.score(X_test, y_test):.4f}")


##############################################################################
# 10.9 模型比较
##############################################################################
models = {
    'Logistic回归': LogisticRegression(random_state=123),
    '决策树': DecisionTreeClassifier(max_depth=5, random_state=123),
    '随机森林': RandomForestClassifier(n_estimators=100, random_state=123),
    '梯度提升树': GradientBoostingClassifier(n_estimators=100, random_state=123),
    '神经网络': MLPClassifier(hidden_layer_sizes=(5, 3), max_iter=500,
                               random_state=123)
}

print(f"\n分类模型比较（AUC）:")
for name, model in models.items():
    model.fit(X_train_c, y_train_c)
    y_pred = model.predict_proba(X_test_c)[:, 1]
    auc = roc_auc_score(y_test_c, y_pred)
    print(f"  {name}: AUC={auc:.4f}")


##############################################################################
# 10.10 模型解释：SHAP值
##############################################################################
try:
    import shap
    explainer = shap.TreeExplainer(rf_reg)
    shap_values = explainer.shap_values(X_test)

    fig, ax = plt.subplots(figsize=(8, 5))
    shap.summary_plot(shap_values, X_test, feature_names=['x1','x2','x3','x4','x5'],
                      show=False)
    plt.title('SHAP值变量重要性')
    plt.tight_layout(); plt.show()
except ImportError:
    print("\nSHAP库未安装，跳过SHAP分析")


##############################################################################
# 10.11 保险应用：车险定价
##############################################################################
np.random.seed(456)
n_auto = 2000
age = np.random.uniform(18, 70, n_auto)
gender = np.random.binomial(1, 0.5, n_auto)
vehicle_age = np.random.uniform(0, 15, n_auto)
region = np.random.randint(1, 5, n_auto)
no_claim_years = np.random.randint(0, 11, n_auto)

eta_freq = -1 + 0.02 * (age - 40) - 0.1 * gender + \
           0.1 * vehicle_age + 0.3 * (region == 1) - 0.1 * no_claim_years
lambda_auto = np.exp(eta_freq)
n_claims = np.random.poisson(lambda_auto)

claim_amount = np.zeros(n_auto)
for i in range(n_auto):
    if n_claims[i] > 0:
        eta_sev = 8 + 0.01 * (age[i] - 40) + 0.05 * vehicle_age[i]
        mu_sev = np.exp(eta_sev)
        claim_amount[i] = np.random.gamma(2, mu_sev/2) * n_claims[i]

df_auto = pd.DataFrame({
    'age': age, 'gender': gender, 'vehicle_age': vehicle_age,
    'region': region, 'no_claim_years': no_claim_years,
    'n_claims': n_claims, 'claim_amount': claim_amount
})

X_auto = df_auto[['age', 'gender', 'vehicle_age', 'region', 'no_claim_years']].values
y_freq = df_auto['n_claims'].values
y_sev = df_auto['claim_amount'].values

X_train_a, X_test_a, y_train_f, y_test_f = train_test_split(
    X_auto, y_freq, test_size=0.3, random_state=123)

# 随机森林预测索赔次数
rf_freq = RandomForestRegressor(n_estimators=100, random_state=123)
rf_freq.fit(X_train_a, y_train_f)
print(f"\n车险索赔次数预测: R²={rf_freq.score(X_test_a, y_test_f):.4f}")

# 梯度提升树预测索赔金额
y_sev_pos = y_sev[y_sev > 0]
X_sev_pos = X_auto[y_sev > 0]
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_sev_pos, y_sev_pos, test_size=0.3, random_state=123)
gb_sev = GradientBoostingRegressor(n_estimators=100, random_state=123)
gb_sev.fit(X_train_s, y_train_s)
print(f"车险索赔金额预测: R²={gb_sev.score(X_test_s, y_test_s):.4f}")

# 导入所需的Python模块
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

np.random.seed(123)
n = 1000
X = np.random.normal(0, 1, (n, 5))
df = pd.DataFrame(X, columns=['x1', 'x2', 'x3', 'x4', 'x5'])

# 分类目标
eta = 0.5 + 1.2 * X[:, 0] - 0.8 * X[:, 1] + 0.5 * X[:, 2]
p = 1 / (1 + np.exp(-eta))
y_class = (np.random.uniform(0, 1, n) < p).astype(int)

# 回归目标
y_reg = 2 + 1.5 * X[:, 0] - 0.8 * X[:, 1] + 0.5 * X[:, 2] + \
        0.3 * X[:, 3] + np.random.normal(0, 0.5, n)

df['y_class'] = y_class
df['y_reg'] = y_reg

print("数据描述:")
print(df.describe())


##############################################################################
# 10.2 无监督学习：主成分分析（PCA）
##############################################################################
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

print(f"\nPCA解释方差比: {pca.explained_variance_ratio_}")
print(f"累积解释方差: {np.cumsum(pca.explained_variance_ratio_)}")

##############################################################################
# 第10章 机器学习
# 对应教材：section10.tex
# 内容：无监督学习（PCA、因子分析、聚类）、监督学习
#       （正则化回归、决策树、随机森林、梯度提升树、神经网络）、SHAP值
##############################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


##############################################################################
# 10.1 数据生成
##############################################################################
np.random.seed(123)
n = 1000
X = np.random.normal(0, 1, (n, 5))
df = pd.DataFrame(X, columns=['x1', 'x2', 'x3', 'x4', 'x5'])

# 分类目标
eta = 0.5 + 1.2 * X[:, 0] - 0.8 * X[:, 1] + 0.5 * X[:, 2]
p = 1 / (1 + np.exp(-eta))
y_class = (np.random.uniform(0, 1, n) < p).astype(int)

# 回归目标
y_reg = 2 + 1.5 * X[:, 0] - 0.8 * X[:, 1] + 0.5 * X[:, 2] + \
        0.3 * X[:, 3] + np.random.normal(0, 0.5, n)

df['y_class'] = y_class
df['y_reg'] = y_reg

print("数据描述:")
print(df.describe())


##############################################################################
# 10.2 无监督学习：主成分分析（PCA）
##############################################################################
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

print(f"\nPCA解释方差比: {pca.explained_variance_ratio_}")
print(f"累积解释方差: {np.cumsum(pca.explained_variance_ratio_)}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(range(1, 6), pca.explained_variance_ratio_, alpha=0.7)
axes[0].set_xlabel('主成分'); axes[0].set_ylabel('解释方差比')
axes[0].set_title('PCA碎石图')
axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y_class, alpha=0.5)
axes[1].set_xlabel('PC1'); axes[1].set_ylabel('PC2')
axes[1].set_title('PCA投影')
plt.tight_layout(); plt.show()


## 因子分析
fa = FactorAnalysis(n_components=2, rotation='varimax')
X_fa = fa.fit_transform(X_scaled)
print(f"\n因子分析载荷矩阵:\n{fa.components_.T}")


##############################################################################
# 10.3 无监督学习：聚类分析
##############################################################################

## 10.3.1 K-means聚类
kmeans = KMeans(n_clusters=3, random_state=123, n_init=10)
labels_km = kmeans.fit_predict(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(X[:, 0], X[:, 1], c=labels_km, alpha=0.5)
axes[0].set_title('K-means聚类')
axes[0].set_xlabel('x1'); axes[0].set_ylabel('x2')

## 10.3.2 层次聚类
hc = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels_hc = hc.fit_predict(X_scaled)
axes[1].scatter(X[:, 0], X[:, 1], c=labels_hc, alpha=0.5)
axes[1].set_title('层次聚类')
axes[1].set_xlabel('x1'); axes[1].set_ylabel('x2')
plt.tight_layout(); plt.show()

# 肘部法则
inertias = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=123, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(1, 11), inertias, 'bo-')
ax.set_xlabel('聚类数 k'); ax.set_ylabel('惯性')
ax.set_title('肘部法则')
plt.tight_layout(); plt.show()


##############################################################################
# 10.4 监督学习：正则化回归
##############################################################################
X_train, X_test, y_train, y_test = train_test_split(
    X, y_reg, test_size=0.3, random_state=123)

## 10.4.1 岭回归
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
print(f"\n岭回归: R²={ridge.score(X_test, y_test):.4f}")
print(f"  系数: {ridge.coef_}")

## 10.4.2 LASSO回归
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
print(f"\nLASSO回归: R²={lasso.score(X_test, y_test):.4f}")
print(f"  系数: {lasso.coef_}")

## 10.4.3 Elastic Net
enet = ElasticNet(alpha=0.1, l1_ratio=0.5)
enet.fit(X_train, y_train)
print(f"\nElastic Net: R²={enet.score(X_test, y_test):.4f}")
print(f"  系数: {enet.coef_}")


##############################################################################
# 10.5 监督学习：决策树
##############################################################################
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_class, test_size=0.3, random_state=123)

## 10.5.1 分类树
dt_clf = DecisionTreeClassifier(max_depth=5, random_state=123)
dt_clf.fit(X_train_c, y_train_c)
print(f"\n分类决策树: 准确率={accuracy_score(y_test_c, dt_clf.predict(X_test_c)):.4f}")

## 10.5.2 回归树
dt_reg = DecisionTreeRegressor(max_depth=5, random_state=123)
dt_reg.fit(X_train, y_train)
print(f"回归决策树: R²={dt_reg.score(X_test, y_test):.4f}")


##############################################################################
# 10.6 监督学习：随机森林
##############################################################################
## 10.6.1 分类随机森林
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=5,
                                 random_state=123)
rf_clf.fit(X_train_c, y_train_c)
y_pred_rf = rf_clf.predict_proba(X_test_c)[:, 1]
print(f"\n随机森林（分类）: AUC={roc_auc_score(y_test_c, y_pred_rf):.4f}")

## 10.6.2 回归随机森林
rf_reg = RandomForestRegressor(n_estimators=100, max_depth=5,
                                random_state=123)
rf_reg.fit(X_train, y_train)
print(f"随机森林（回归）: R²={rf_reg.score(X_test, y_test):.4f}")

## 变量重要性
importances = rf_reg.feature_importances_
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(range(5), importances)
ax.set_xticks(range(5))
ax.set_xticklabels(['x1', 'x2', 'x3', 'x4', 'x5'])
ax.set_ylabel('重要性')
ax.set_title('随机森林变量重要性')
plt.tight_layout(); plt.show()


##############################################################################
# 10.7 监督学习：梯度提升树
##############################################################################
## 10.7.1 分类GBDT
gb_clf = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                     random_state=123)
gb_clf.fit(X_train_c, y_train_c)
y_pred_gb = gb_clf.predict_proba(X_test_c)[:, 1]
print(f"\n梯度提升树（分类）: AUC={roc_auc_score(y_test_c, y_pred_gb):.4f}")

## 10.7.2 回归GBDT
gb_reg = GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                    random_state=123)
gb_reg.fit(X_train, y_train)
print(f"梯度提升树（回归）: R²={gb_reg.score(X_test, y_test):.4f}")


##############################################################################
# 10.8 监督学习：神经网络
##############################################################################
## 10.8.1 分类神经网络
nn_clf = MLPClassifier(hidden_layer_sizes=(5, 3), max_iter=500,
                        random_state=123)
nn_clf.fit(X_train_c, y_train_c)
y_pred_nn = nn_clf.predict_proba(X_test_c)[:, 1]
print(f"\n神经网络（分类）: AUC={roc_auc_score(y_test_c, y_pred_nn):.4f}")

## 10.8.2 回归神经网络
nn_reg = MLPRegressor(hidden_layer_sizes=(5, 3), max_iter=500,
                       random_state=123)
nn_reg.fit(X_train, y_train)
print(f"神经网络（回归）: R²={nn_reg.score(X_test, y_test):.4f}")


##############################################################################
# 10.9 模型比较
##############################################################################
models = {
    'Logistic回归': LogisticRegression(random_state=123),
    '决策树': DecisionTreeClassifier(max_depth=5, random_state=123),
    '随机森林': RandomForestClassifier(n_estimators=100, random_state=123),
    '梯度提升树': GradientBoostingClassifier(n_estimators=100, random_state=123),
    '神经网络': MLPClassifier(hidden_layer_sizes=(5, 3), max_iter=500,
                               random_state=123)
}

print(f"\n分类模型比较（AUC）:")
for name, model in models.items():
    model.fit(X_train_c, y_train_c)
    y_pred = model.predict_proba(X_test_c)[:, 1]
    auc = roc_auc_score(y_test_c, y_pred)
    print(f"  {name}: AUC={auc:.4f}")


##############################################################################
# 10.10 模型解释：SHAP值
##############################################################################
try:
    import shap
    explainer = shap.TreeExplainer(rf_reg)
    shap_values = explainer.shap_values(X_test)

    fig, ax = plt.subplots(figsize=(8, 5))
    shap.summary_plot(shap_values, X_test, feature_names=['x1','x2','x3','x4','x5'],
                      show=False)
    plt.title('SHAP值变量重要性')
    plt.tight_layout(); plt.show()
except ImportError:
    print("\nSHAP库未安装，跳过SHAP分析")


##############################################################################
# 10.11 保险应用：车险定价
##############################################################################
np.random.seed(456)
n_auto = 2000
age = np.random.uniform(18, 70, n_auto)
gender = np.random.binomial(1, 0.5, n_auto)
vehicle_age = np.random.uniform(0, 15, n_auto)
region = np.random.randint(1, 5, n_auto)
no_claim_years = np.random.randint(0, 11, n_auto)

eta_freq = -1 + 0.02 * (age - 40) - 0.1 * gender + \
           0.1 * vehicle_age + 0.3 * (region == 1) - 0.1 * no_claim_years
lambda_auto = np.exp(eta_freq)
n_claims = np.random.poisson(lambda_auto)

claim_amount = np.zeros(n_auto)
for i in range(n_auto):
    if n_claims[i] > 0:
        eta_sev = 8 + 0.01 * (age[i] - 40) + 0.05 * vehicle_age[i]
        mu_sev = np.exp(eta_sev)
        claim_amount[i] = np.random.gamma(2, mu_sev/2) * n_claims[i]

df_auto = pd.DataFrame({
    'age': age, 'gender': gender, 'vehicle_age': vehicle_age,
    'region': region, 'no_claim_years': no_claim_years,
    'n_claims': n_claims, 'claim_amount': claim_amount
})

X_auto = df_auto[['age', 'gender', 'vehicle_age', 'region', 'no_claim_years']].values
y_freq = df_auto['n_claims'].values
y_sev = df_auto['claim_amount'].values

X_train_a, X_test_a, y_train_f, y_test_f = train_test_split(
    X_auto, y_freq, test_size=0.3, random_state=123)

# 随机森林预测索赔次数
rf_freq = RandomForestRegressor(n_estimators=100, random_state=123)
rf_freq.fit(X_train_a, y_train_f)
print(f"\n车险索赔次数预测: R²={rf_freq.score(X_test_a, y_test_f):.4f}")

# 梯度提升树预测索赔金额
y_sev_pos = y_sev[y_sev > 0]
X_sev_pos = X_auto[y_sev > 0]
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_sev_pos, y_sev_pos, test_size=0.3, random_state=123)
gb_sev = GradientBoostingRegressor(n_estimators=100, random_state=123)
gb_sev.fit(X_train_s, y_train_s)
print(f"车险索赔金额预测: R²={gb_sev.score(X_test_s, y_test_s):.4f}")

# 导入所需的Python模块
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

np.random.seed(123)
n = 1000
X = np.random.normal(0, 1, (n, 5))
df = pd.DataFrame(X, columns=['x1', 'x2', 'x3', 'x4', 'x5'])

# 分类目标
eta = 0.5 + 1.2 * X[:, 0] - 0.8 * X[:, 1] + 0.5 * X[:, 2]
p = 1 / (1 + np.exp(-eta))
y_class = (np.random.uniform(0, 1, n) < p).astype(int)

# 回归目标
y_reg = 2 + 1.5 * X[:, 0] - 0.8 * X[:, 1] + 0.5 * X[:, 2] + \
        0.3 * X[:, 3] + np.random.normal(0, 0.5, n)

df['y_class'] = y_class
df['y_reg'] = y_reg

print("数据描述:")
print(df.describe())


##############################################################################
# 10.2 无监督学习：主成分分析（PCA）
##############################################################################
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

print(f"\nPCA解释方差比: {pca.explained_variance_ratio_}")
print(f"累积解释方差: {np.cumsum(pca.explained_variance_ratio_)}")

##############################################################################
# 第10章 机器学习
# 对应教材：section10.tex
# 内容：无监督学习（PCA、因子分析、聚类）、监督学习
#       （正则化回归、决策树、随机森林、梯度提升树、神经网络）、SHAP值
##############################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


##############################################################################
# 10.1 数据生成
##############################################################################
np.random.seed(123)
n = 1000
X = np.random.normal(0, 1, (n, 5))
df = pd.DataFrame(X, columns=['x1', 'x2', 'x3', 'x4', 'x5'])

# 分类目标
eta = 0.5 + 1.2 * X[:, 0] - 0.8 * X[:, 1] + 0.5 * X[:, 2]
p = 1 / (1 + np.exp(-eta))
y_class = (np.random.uniform(0, 1, n) < p).astype(int)

# 回归目标
y_reg = 2 + 1.5 * X[:, 0] - 0.8 * X[:, 1] + 0.5 * X[:, 2] + \
        0.3 * X[:, 3] + np.random.normal(0, 0.5, n)

df['y_class'] = y_class
df['y_reg'] = y_reg

print("数据描述:")
print(df.describe())


##############################################################################
# 10.2 无监督学习：主成分分析（PCA）
##############################################################################
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

print(f"\nPCA解释方差比: {pca.explained_variance_ratio_}")
print(f"累积解释方差: {np.cumsum(pca.explained_variance_ratio_)}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(range(1, 6), pca.explained_variance_ratio_, alpha=0.7)
axes[0].set_xlabel('主成分'); axes[0].set_ylabel('解释方差比')
axes[0].set_title('PCA碎石图')
axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y_class, alpha=0.5)
axes[1].set_xlabel('PC1'); axes[1].set_ylabel('PC2')
axes[1].set_title('PCA投影')
plt.tight_layout(); plt.show()


## 因子分析
fa = FactorAnalysis(n_components=2, rotation='varimax')
X_fa = fa.fit_transform(X_scaled)
print(f"\n因子分析载荷矩阵:\n{fa.components_.T}")


##############################################################################
# 10.3 无监督学习：聚类分析
##############################################################################

## 10.3.1 K-means聚类
kmeans = KMeans(n_clusters=3, random_state=123, n_init=10)
labels_km = kmeans.fit_predict(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(X[:, 0], X[:, 1], c=labels_km, alpha=0.5)
axes[0].set_title('K-means聚类')
axes[0].set_xlabel('x1'); axes[0].set_ylabel('x2')

## 10.3.2 层次聚类
hc = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels_hc = hc.fit_predict(X_scaled)
axes[1].scatter(X[:, 0], X[:, 1], c=labels_hc, alpha=0.5)
axes[1].set_title('层次聚类')
axes[1].set_xlabel('x1'); axes[1].set_ylabel('x2')
plt.tight_layout(); plt.show()

# 肘部法则
inertias = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=123, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(1, 11), inertias, 'bo-')
ax.set_xlabel('聚类数 k'); ax.set_ylabel('惯性')
ax.set_title('肘部法则')
plt.tight_layout(); plt.show()


##############################################################################
# 10.4 监督学习：正则化回归
##############################################################################
X_train, X_test, y_train, y_test = train_test_split(
    X, y_reg, test_size=0.3, random_state=123)

## 10.4.1 岭回归
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
print(f"\n岭回归: R²={ridge.score(X_test, y_test):.4f}")
print(f"  系数: {ridge.coef_}")

## 10.4.2 LASSO回归
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
print(f"\nLASSO回归: R²={lasso.score(X_test, y_test):.4f}")
print(f"  系数: {lasso.coef_}")

## 10.4.3 Elastic Net
enet = ElasticNet(alpha=0.1, l1_ratio=0.5)
enet.fit(X_train, y_train)
print(f"\nElastic Net: R²={enet.score(X_test, y_test):.4f}")
print(f"  系数: {enet.coef_}")


##############################################################################
# 10.5 监督学习：决策树
##############################################################################
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_class, test_size=0.3, random_state=123)

## 10.5.1 分类树
dt_clf = DecisionTreeClassifier(max_depth=5, random_state=123)
dt_clf.fit(X_train_c, y_train_c)
print(f"\n分类决策树: 准确率={accuracy_score(y_test_c, dt_clf.predict(X_test_c)):.4f}")

## 10.5.2 回归树
dt_reg = DecisionTreeRegressor(max_depth=5, random_state=123)
dt_reg.fit(X_train, y_train)
print(f"回归决策树: R²={dt_reg.score(X_test, y_test):.4f}")


##############################################################################
# 10.6 监督学习：随机森林
##############################################################################
## 10.6.1 分类随机森林
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=5,
                                 random_state=123)
rf_clf.fit(X_train_c, y_train_c)
y_pred_rf = rf_clf.predict_proba(X_test_c)[:, 1]
print(f"\n随机森林（分类）: AUC={roc_auc_score(y_test_c, y_pred_rf):.4f}")

## 10.6.2 回归随机森林
rf_reg = RandomForestRegressor(n_estimators=100, max_depth=5,
                                random_state=123)
rf_reg.fit(X_train, y_train)
print(f"随机森林（回归）: R²={rf_reg.score(X_test, y_test):.4f}")

## 变量重要性
importances = rf_reg.feature_importances_
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(range(5), importances)
ax.set_xticks(range(5))
ax.set_xticklabels(['x1', 'x2', 'x3', 'x4', 'x5'])
ax.set_ylabel('重要性')
ax.set_title('随机森林变量重要性')
plt.tight_layout(); plt.show()


##############################################################################
# 10.7 监督学习：梯度提升树
##############################################################################
## 10.7.1 分类GBDT
gb_clf = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                     random_state=123)
gb_clf.fit(X_train_c, y_train_c)
y_pred_gb = gb_clf.predict_proba(X_test_c)[:, 1]
print(f"\n梯度提升树（分类）: AUC={roc_auc_score(y_test_c, y_pred_gb):.4f}")

## 10.7.2 回归GBDT
gb_reg = GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                    random_state=123)
gb_reg.fit(X_train, y_train)
print(f"梯度提升树（回归）: R²={gb_reg.score(X_test, y_test):.4f}")


##############################################################################
# 10.8 监督学习：神经网络
##############################################################################
## 10.8.1 分类神经网络
nn_clf = MLPClassifier(hidden_layer_sizes=(5, 3), max_iter=500,
                        random_state=123)
nn_clf.fit(X_train_c, y_train_c)
y_pred_nn = nn_clf.predict_proba(X_test_c)[:, 1]
print(f"\n神经网络（分类）: AUC={roc_auc_score(y_test_c, y_pred_nn):.4f}")

## 10.8.2 回归神经网络
nn_reg = MLPRegressor(hidden_layer_sizes=(5, 3), max_iter=500,
                       random_state=123)
nn_reg.fit(X_train, y_train)
print(f"神经网络（回归）: R²={nn_reg.score(X_test, y_test):.4f}")


##############################################################################
# 10.9 模型比较
##############################################################################
models = {
    'Logistic回归': LogisticRegression(random_state=123),
    '决策树': DecisionTreeClassifier(max_depth=5, random_state=123),
    '随机森林': RandomForestClassifier(n_estimators=100, random_state=123),
    '梯度提升树': GradientBoostingClassifier(n_estimators=100, random_state=123),
    '神经网络': MLPClassifier(hidden_layer_sizes=(5, 3), max_iter=500,
                               random_state=123)
}

print(f"\n分类模型比较（AUC）:")
for name, model in models.items():
    model.fit(X_train_c, y_train_c)
    y_pred = model.predict_proba(X_test_c)[:, 1]
    auc = roc_auc_score(y_test_c, y_pred)
    print(f"  {name}: AUC={auc:.4f}")


##############################################################################
# 10.10 模型解释：SHAP值
##############################################################################
try:
    import shap
    explainer = shap.TreeExplainer(rf_reg)
    shap_values = explainer.shap_values(X_test)

    fig, ax = plt.subplots(figsize=(8, 5))
    shap.summary_plot(shap_values, X_test, feature_names=['x1','x2','x3','x4','x5'],
                      show=False)
    plt.title('SHAP值变量重要性')
    plt.tight_layout(); plt.show()
except ImportError:
    print("\nSHAP库未安装，跳过SHAP分析")


##############################################################################
# 10.11 保险应用：车险定价
##############################################################################
np.random.seed(456)
n_auto = 2000
age = np.random.uniform(18, 70, n_auto)
gender = np.random.binomial(1, 0.5, n_auto)
vehicle_age = np.random.uniform(0, 15, n_auto)
region = np.random.randint(1, 5, n_auto)
no_claim_years = np.random.randint(0, 11, n_auto)

eta_freq = -1 + 0.02 * (age - 40) - 0.1 * gender + \
           0.1 * vehicle_age + 0.3 * (region == 1) - 0.1 * no_claim_years
lambda_auto = np.exp(eta_freq)
n_claims = np.random.poisson(lambda_auto)

claim_amount = np.zeros(n_auto)
for i in range(n_auto):
    if n_claims[i] > 0:
        eta_sev = 8 + 0.01 * (age[i] - 40) + 0.05 * vehicle_age[i]
        mu_sev = np.exp(eta_sev)
        claim_amount[i] = np.random.gamma(2, mu_sev/2) * n_claims[i]

df_auto = pd.DataFrame({
    'age': age, 'gender': gender, 'vehicle_age': vehicle_age,
    'region': region, 'no_claim_years': no_claim_years,
    'n_claims': n_claims, 'claim_amount': claim_amount
})

X_auto = df_auto[['age', 'gender', 'vehicle_age', 'region', 'no_claim_years']].values
y_freq = df_auto['n_claims'].values
y_sev = df_auto['claim_amount'].values

X_train_a, X_test_a, y_train_f, y_test_f = train_test_split(
    X_auto, y_freq, test_size=0.3, random_state=123)

# 随机森林预测索赔次数
rf_freq = RandomForestRegressor(n_estimators=100, random_state=123)
rf_freq.fit(X_train_a, y_train_f)
print(f"\n车险索赔次数预测: R²={rf_freq.score(X_test_a, y_test_f):.4f}")

# 梯度提升树预测索赔金额
y_sev_pos = y_sev[y_sev > 0]
X_sev_pos = X_auto[y_sev > 0]
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_sev_pos, y_sev_pos, test_size=0.3, random_state=123)
gb_sev = GradientBoostingRegressor(n_estimators=100, random_state=123)
gb_sev.fit(X_train_s, y_train_s)
print(f"车险索赔金额预测: R²={gb_sev.score(X_test_s, y_test_s):.4f}")

# 导入所需的Python模块
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

np.random.seed(123)
n = 1000
X = np.random.normal(0, 1, (n, 5))
df = pd.DataFrame(X, columns=['x1', 'x2', 'x3', 'x4', 'x5'])

# 分类目标
eta = 0.5 + 1.2 * X[:, 0] - 0.8 * X[:, 1] + 0.5 * X[:, 2]
p = 1 / (1 + np.exp(-eta))
y_class = (np.random.uniform(0, 1, n) < p).astype(int)

# 回归目标
y_reg = 2 + 1.5 * X[:, 0] - 0.8 * X[:, 1] + 0.5 * X[:, 2] + \
        0.3 * X[:, 3] + np.random.normal(0, 0.5, n)

df['y_class'] = y_class
df['y_reg'] = y_reg

print("数据描述:")
print(df.describe())


##############################################################################
# 10.2 无监督学习：主成分分析（PCA）
##############################################################################
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

print(f"\nPCA解释方差比: {pca.explained_variance_ratio_}")
print(f"累积解释方差: {np.cumsum(pca.explained_variance_ratio_)}")

##############################################################################
# 第10章 机器学习
# 对应教材：section10.tex
# 内容：无监督学习（PCA、因子分析、聚类）、监督学习
#       （正则化回归、决策树、随机森林、梯度提升树、神经网络）、SHAP值
##############################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


##############################################################################
# 10.1 数据生成
##############################################################################
np.random.seed(123)
n = 1000
X = np.random.normal(0, 1, (n, 5))
df = pd.DataFrame(X, columns=['x1', 'x2', 'x3', 'x4', 'x5'])

# 分类目标
eta = 0.5 + 1.2 * X[:, 0] - 0.8 * X[:, 1] + 0.5 * X[:, 2]
p = 1 / (1 + np.exp(-eta))
y_class = (np.random.uniform(0, 1, n) < p).astype(int)

# 回归目标
y_reg = 2 + 1.5 * X[:, 0] - 0.8 * X[:, 1] + 0.5 * X[:, 2] + \
        0.3 * X[:, 3] + np.random.normal(0, 0.5, n)

df['y_class'] = y_class
df['y_reg'] = y_reg

print("数据描述:")
print(df.describe())


##############################################################################
# 10.2 无监督学习：主成分分析（PCA）
##############################################################################
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

print(f"\nPCA解释方差比: {pca.explained_variance_ratio_}")
print(f"累积解释方差: {np.cumsum(pca.explained_variance_ratio_)}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(range(1, 6), pca.explained_variance_ratio_, alpha=0.7)
axes[0].set_xlabel('主成分'); axes[0].set_ylabel('解释方差比')
axes[0].set_title('PCA碎石图')
axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y_class, alpha=0.5)
axes[1].set_xlabel('PC1'); axes[1].set_ylabel('PC2')
axes[1].set_title('PCA投影')
plt.tight_layout(); plt.show()


## 因子分析
fa = FactorAnalysis(n_components=2, rotation='varimax')
X_fa = fa.fit_transform(X_scaled)
print(f"\n因子分析载荷矩阵:\n{fa.components_.T}")


##############################################################################
# 10.3 无监督学习：聚类分析
##############################################################################

## 10.3.1 K-means聚类
kmeans = KMeans(n_clusters=3, random_state=123, n_init=10)
labels_km = kmeans.fit_predict(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(X[:, 0], X[:, 1], c=labels_km, alpha=0.5)
axes[0].set_title('K-means聚类')
axes[0].set_xlabel('x1'); axes[0].set_ylabel('x2')

## 10.3.2 层次聚类
hc = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels_hc = hc.fit_predict(X_scaled)
axes[1].scatter(X[:, 0], X[:, 1], c=labels_hc, alpha=0.5)
axes[1].set_title('层次聚类')
axes[1].set_xlabel('x1'); axes[1].set_ylabel('x2')
plt.tight_layout(); plt.show()

# 肘部法则
inertias = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=123, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(1, 11), inertias, 'bo-')
ax.set_xlabel('聚类数 k'); ax.set_ylabel('惯性')
ax.set_title('肘部法则')
plt.tight_layout(); plt.show()


##############################################################################
# 10.4 监督学习：正则化回归
##############################################################################
X_train, X_test, y_train, y_test = train_test_split(
    X, y_reg, test_size=0.3, random_state=123)

## 10.4.1 岭回归
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
print(f"\n岭回归: R²={ridge.score(X_test, y_test):.4f}")
print(f"  系数: {ridge.coef_}")

## 10.4.2 LASSO回归
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
print(f"\nLASSO回归: R²={lasso.score(X_test, y_test):.4f}")
print(f"  系数: {lasso.coef_}")

## 10.4.3 Elastic Net
enet = ElasticNet(alpha=0.1, l1_ratio=0.5)
enet.fit(X_train, y_train)
print(f"\nElastic Net: R²={enet.score(X_test, y_test):.4f}")
print(f"  系数: {enet.coef_}")


##############################################################################
# 10.5 监督学习：决策树
##############################################################################
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_class, test_size=0.3, random_state=123)

## 10.5.1 分类树
dt_clf = DecisionTreeClassifier(max_depth=5, random_state=123)
dt_clf.fit(X_train_c, y_train_c)
print(f"\n分类决策树: 准确率={accuracy_score(y_test_c, dt_clf.predict(X_test_c)):.4f}")

## 10.5.2 回归树
dt_reg = DecisionTreeRegressor(max_depth=5, random_state=123)
dt_reg.fit(X_train, y_train)
print(f"回归决策树: R²={dt_reg.score(X_test, y_test):.4f}")


##############################################################################
# 10.6 监督学习：随机森林
##############################################################################
## 10.6.1 分类随机森林
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=5,
                                 random_state=123)
rf_clf.fit(X_train_c, y_train_c)
y_pred_rf = rf_clf.predict_proba(X_test_c)[:, 1]
print(f"\n随机森林（分类）: AUC={roc_auc_score(y_test_c, y_pred_rf):.4f}")

## 10.6.2 回归随机森林
rf_reg = RandomForestRegressor(n_estimators=100, max_depth=5,
                                random_state=123)
rf_reg.fit(X_train, y_train)
print(f"随机森林（回归）: R²={rf_reg.score(X_test, y_test):.4f}")

## 变量重要性
importances = rf_reg.feature_importances_
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(range(5), importances)
ax.set_xticks(range(5))
ax.set_xticklabels(['x1', 'x2', 'x3', 'x4', 'x5'])
ax.set_ylabel('重要性')
ax.set_title('随机森林变量重要性')
plt.tight_layout(); plt.show()


##############################################################################
# 10.7 监督学习：梯度提升树
##############################################################################
## 10.7.1 分类GBDT
gb_clf = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                     random_state=123)
gb_clf.fit(X_train_c, y_train_c)
y_pred_gb = gb_clf.predict_proba(X_test_c)[:, 1]
print(f"\n梯度提升树（分类）: AUC={roc_auc_score(y_test_c, y_pred_gb):.4f}")

## 10.7.2 回归GBDT
gb_reg = GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                    random_state=123)
gb_reg.fit(X_train, y_train)
print(f"梯度提升树（回归）: R²={gb_reg.score(X_test, y_test):.4f}")


##############################################################################
# 10.8 监督学习：神经网络
##############################################################################
## 10.8.1 分类神经网络
nn_clf = MLPClassifier(hidden_layer_sizes=(5, 3), max_iter=500,
                        random_state=123)
nn_clf.fit(X_train_c, y_train_c)
y_pred_nn = nn_clf.predict_proba(X_test_c)[:, 1]
print(f"\n神经网络（分类）: AUC={roc_auc_score(y_test_c, y_pred_nn):.4f}")

## 10.8.2 回归神经网络
nn_reg = MLPRegressor(hidden_layer_sizes=(5, 3), max_iter=500,
                       random_state=123)
nn_reg.fit(X_train, y_train)
print(f"神经网络（回归）: R²={nn_reg.score(X_test, y_test):.4f}")


##############################################################################
# 10.9 模型比较
##############################################################################
models = {
    'Logistic回归': LogisticRegression(random_state=123),
    '决策树': DecisionTreeClassifier(max_depth=5, random_state=123),
    '随机森林': RandomForestClassifier(n_estimators=100, random_state=123),
    '梯度提升树': GradientBoostingClassifier(n_estimators=100, random_state=123),
    '神经网络': MLPClassifier(hidden_layer_sizes=(5, 3), max_iter=500,
                               random_state=123)
}

print(f"\n分类模型比较（AUC）:")
for name, model in models.items():
    model.fit(X_train_c, y_train_c)
    y_pred = model.predict_proba(X_test_c)[:, 1]
    auc = roc_auc_score(y_test_c, y_pred)
    print(f"  {name}: AUC={auc:.4f}")


##############################################################################
# 10.10 模型解释：SHAP值
##############################################################################
try:
    import shap
    explainer = shap.TreeExplainer(rf_reg)
    shap_values = explainer.shap_values(X_test)

    fig, ax = plt.subplots(figsize=(8, 5))
    shap.summary_plot(shap_values, X_test, feature_names=['x1','x2','x3','x4','x5'],
                      show=False)
    plt.title('SHAP值变量重要性')
    plt.tight_layout(); plt.show()
except ImportError:
    print("\nSHAP库未安装，跳过SHAP分析")


##############################################################################
# 10.11 保险应用：车险定价
##############################################################################
np.random.seed(456)
n_auto = 2000
age = np.random.uniform(18, 70, n_auto)
gender = np.random.binomial(1, 0.5, n_auto)
vehicle_age = np.random.uniform(0, 15, n_auto)
region = np.random.randint(1, 5, n_auto)
no_claim_years = np.random.randint(0, 11, n_auto)

eta_freq = -1 + 0.02 * (age - 40) - 0.1 * gender + \
           0.1 * vehicle_age + 0.3 * (region == 1) - 0.1 * no_claim_years
lambda_auto = np.exp(eta_freq)
n_claims = np.random.poisson(lambda_auto)

claim_amount = np.zeros(n_auto)
for i in range(n_auto):
    if n_claims[i] > 0:
        eta_sev = 8 + 0.01 * (age[i] - 40) + 0.05 * vehicle_age[i]
        mu_sev = np.exp(eta_sev)
        claim_amount[i] = np.random.gamma(2, mu_sev/2) * n_claims[i]

df_auto = pd.DataFrame({
    'age': age, 'gender': gender, 'vehicle_age': vehicle_age,
    'region': region, 'no_claim_years': no_claim_years,
    'n_claims': n_claims, 'claim_amount': claim_amount
})

X_auto = df_auto[['age', 'gender', 'vehicle_age', 'region', 'no_claim_years']].values
y_freq = df_auto['n_claims'].values
y_sev = df_auto['claim_amount'].values

X_train_a, X_test_a, y_train_f, y_test_f = train_test_split(
    X_auto, y_freq, test_size=0.3, random_state=123)

# 随机森林预测索赔次数
rf_freq = RandomForestRegressor(n_estimators=100, random_state=123)
rf_freq.fit(X_train_a, y_train_f)
print(f"\n车险索赔次数预测: R²={rf_freq.score(X_test_a, y_test_f):.4f}")

# 梯度提升树预测索赔金额
y_sev_pos = y_sev[y_sev > 0]
X_sev_pos = X_auto[y_sev > 0]
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_sev_pos, y_sev_pos, test_size=0.3, random_state=123)
gb_sev = GradientBoostingRegressor(n_estimators=100, random_state=123)
gb_sev.fit(X_train_s, y_train_s)
print(f"车险索赔金额预测: R²={gb_sev.score(X_test_s, y_test_s):.4f}")

##############################################################################
# 第10章 机器学习
# 对应教材：section10.tex
# 内容：无监督学习（PCA、因子分析、聚类）、监督学习
#       （正则化回归、决策树、随机森林、梯度提升树、神经网络）、SHAP值
##############################################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


##############################################################################
# 10.1 数据生成
##############################################################################
np.random.seed(123)
n = 1000
X = np.random.normal(0, 1, (n, 5))
df = pd.DataFrame(X, columns=['x1', 'x2', 'x3', 'x4', 'x5'])

# 分类目标
eta = 0.5 + 1.2 * X[:, 0] - 0.8 * X[:, 1] + 0.5 * X[:, 2]
p = 1 / (1 + np.exp(-eta))
y_class = (np.random.uniform(0, 1, n) < p).astype(int)

# 回归目标
y_reg = 2 + 1.5 * X[:, 0] - 0.8 * X[:, 1] + 0.5 * X[:, 2] + \
        0.3 * X[:, 3] + np.random.normal(0, 0.5, n)

df['y_class'] = y_class
df['y_reg'] = y_reg

print("数据描述:")
print(df.describe())


##############################################################################
# 10.2 无监督学习：主成分分析（PCA）
##############################################################################
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

print(f"\nPCA解释方差比: {pca.explained_variance_ratio_}")
print(f"累积解释方差: {np.cumsum(pca.explained_variance_ratio_)}")

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].bar(range(1, 6), pca.explained_variance_ratio_, alpha=0.7)
axes[0].set_xlabel('主成分'); axes[0].set_ylabel('解释方差比')
axes[0].set_title('PCA碎石图')
axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y_class, alpha=0.5)
axes[1].set_xlabel('PC1'); axes[1].set_ylabel('PC2')
axes[1].set_title('PCA投影')
plt.tight_layout(); plt.show()


## 因子分析
fa = FactorAnalysis(n_components=2, rotation='varimax')
X_fa = fa.fit_transform(X_scaled)
print(f"\n因子分析载荷矩阵:\n{fa.components_.T}")


##############################################################################
# 10.3 无监督学习：聚类分析
##############################################################################

## 10.3.1 K-means聚类
kmeans = KMeans(n_clusters=3, random_state=123, n_init=10)
labels_km = kmeans.fit_predict(X_scaled)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].scatter(X[:, 0], X[:, 1], c=labels_km, alpha=0.5)
axes[0].set_title('K-means聚类')
axes[0].set_xlabel('x1'); axes[0].set_ylabel('x2')

## 10.3.2 层次聚类
hc = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels_hc = hc.fit_predict(X_scaled)
axes[1].scatter(X[:, 0], X[:, 1], c=labels_hc, alpha=0.5)
axes[1].set_title('层次聚类')
axes[1].set_xlabel('x1'); axes[1].set_ylabel('x2')
plt.tight_layout(); plt.show()

# 肘部法则
inertias = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=123, n_init=10)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(1, 11), inertias, 'bo-')
ax.set_xlabel('聚类数 k'); ax.set_ylabel('惯性')
ax.set_title('肘部法则')
plt.tight_layout(); plt.show()


##############################################################################
# 10.4 监督学习：正则化回归
##############################################################################
X_train, X_test, y_train, y_test = train_test_split(
    X, y_reg, test_size=0.3, random_state=123)

## 10.4.1 岭回归
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
print(f"\n岭回归: R²={ridge.score(X_test, y_test):.4f}")
print(f"  系数: {ridge.coef_}")

## 10.4.2 LASSO回归
lasso = Lasso(alpha=0.1)
lasso.fit(X_train, y_train)
print(f"\nLASSO回归: R²={lasso.score(X_test, y_test):.4f}")
print(f"  系数: {lasso.coef_}")

## 10.4.3 Elastic Net
enet = ElasticNet(alpha=0.1, l1_ratio=0.5)
enet.fit(X_train, y_train)
print(f"\nElastic Net: R²={enet.score(X_test, y_test):.4f}")
print(f"  系数: {enet.coef_}")


##############################################################################
# 10.5 监督学习：决策树
##############################################################################
X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
    X, y_class, test_size=0.3, random_state=123)

## 10.5.1 分类树
dt_clf = DecisionTreeClassifier(max_depth=5, random_state=123)
dt_clf.fit(X_train_c, y_train_c)
print(f"\n分类决策树: 准确率={accuracy_score(y_test_c, dt_clf.predict(X_test_c)):.4f}")

## 10.5.2 回归树
dt_reg = DecisionTreeRegressor(max_depth=5, random_state=123)
dt_reg.fit(X_train, y_train)
print(f"回归决策树: R²={dt_reg.score(X_test, y_test):.4f}")


##############################################################################
# 10.6 监督学习：随机森林
##############################################################################
## 10.6.1 分类随机森林
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=5,
                                 random_state=123)
rf_clf.fit(X_train_c, y_train_c)
y_pred_rf = rf_clf.predict_proba(X_test_c)[:, 1]
print(f"\n随机森林（分类）: AUC={roc_auc_score(y_test_c, y_pred_rf):.4f}")

## 10.6.2 回归随机森林
rf_reg = RandomForestRegressor(n_estimators=100, max_depth=5,
                                random_state=123)
rf_reg.fit(X_train, y_train)
print(f"随机森林（回归）: R²={rf_reg.score(X_test, y_test):.4f}")

## 变量重要性
importances = rf_reg.feature_importances_
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(range(5), importances)
ax.set_xticks(range(5))
ax.set_xticklabels(['x1', 'x2', 'x3', 'x4', 'x5'])
ax.set_ylabel('重要性')
ax.set_title('随机森林变量重要性')
plt.tight_layout(); plt.show()


##############################################################################
# 10.7 监督学习：梯度提升树
##############################################################################
## 10.7.1 分类GBDT
gb_clf = GradientBoostingClassifier(n_estimators=100, max_depth=3,
                                     random_state=123)
gb_clf.fit(X_train_c, y_train_c)
y_pred_gb = gb_clf.predict_proba(X_test_c)[:, 1]
print(f"\n梯度提升树（分类）: AUC={roc_auc_score(y_test_c, y_pred_gb):.4f}")

## 10.7.2 回归GBDT
gb_reg = GradientBoostingRegressor(n_estimators=100, max_depth=3,
                                    random_state=123)
gb_reg.fit(X_train, y_train)
print(f"梯度提升树（回归）: R²={gb_reg.score(X_test, y_test):.4f}")


##############################################################################
# 10.8 监督学习：神经网络
##############################################################################
## 10.8.1 分类神经网络
nn_clf = MLPClassifier(hidden_layer_sizes=(5, 3), max_iter=500,
                        random_state=123)
nn_clf.fit(X_train_c, y_train_c)
y_pred_nn = nn_clf.predict_proba(X_test_c)[:, 1]
print(f"\n神经网络（分类）: AUC={roc_auc_score(y_test_c, y_pred_nn):.4f}")

## 10.8.2 回归神经网络
nn_reg = MLPRegressor(hidden_layer_sizes=(5, 3), max_iter=500,
                       random_state=123)
nn_reg.fit(X_train, y_train)
print(f"神经网络（回归）: R²={nn_reg.score(X_test, y_test):.4f}")


##############################################################################
# 10.9 模型比较
##############################################################################
models = {
    'Logistic回归': LogisticRegression(random_state=123),
    '决策树': DecisionTreeClassifier(max_depth=5, random_state=123),
    '随机森林': RandomForestClassifier(n_estimators=100, random_state=123),
    '梯度提升树': GradientBoostingClassifier(n_estimators=100, random_state=123),
    '神经网络': MLPClassifier(hidden_layer_sizes=(5, 3), max_iter=500,
                               random_state=123)
}

print(f"\n分类模型比较（AUC）:")
for name, model in models.items():
    model.fit(X_train_c, y_train_c)
    y_pred = model.predict_proba(X_test_c)[:, 1]
    auc = roc_auc_score(y_test_c, y_pred)
    print(f"  {name}: AUC={auc:.4f}")


##############################################################################
# 10.10 模型解释：SHAP值
##############################################################################
try:
    import shap
    explainer = shap.TreeExplainer(rf_reg)
    shap_values = explainer.shap_values(X_test)

    fig, ax = plt.subplots(figsize=(8, 5))
    shap.summary_plot(shap_values, X_test, feature_names=['x1','x2','x3','x4','x5'],
                      show=False)
    plt.title('SHAP值变量重要性')
    plt.tight_layout(); plt.show()
except ImportError:
    print("\nSHAP库未安装，跳过SHAP分析")


##############################################################################
# 10.11 保险应用：车险定价
##############################################################################
np.random.seed(456)
n_auto = 2000
age = np.random.uniform(18, 70, n_auto)
gender = np.random.binomial(1, 0.5, n_auto)
vehicle_age = np.random.uniform(0, 15, n_auto)
region = np.random.randint(1, 5, n_auto)
no_claim_years = np.random.randint(0, 11, n_auto)

eta_freq = -1 + 0.02 * (age - 40) - 0.1 * gender + \
           0.1 * vehicle_age + 0.3 * (region == 1) - 0.1 * no_claim_years
lambda_auto = np.exp(eta_freq)
n_claims = np.random.poisson(lambda_auto)

claim_amount = np.zeros(n_auto)
for i in range(n_auto):
    if n_claims[i] > 0:
        eta_sev = 8 + 0.01 * (age[i] - 40) + 0.05 * vehicle_age[i]
        mu_sev = np.exp(eta_sev)
        claim_amount[i] = np.random.gamma(2, mu_sev/2) * n_claims[i]

df_auto = pd.DataFrame({
    'age': age, 'gender': gender, 'vehicle_age': vehicle_age,
    'region': region, 'no_claim_years': no_claim_years,
    'n_claims': n_claims, 'claim_amount': claim_amount
})

X_auto = df_auto[['age', 'gender', 'vehicle_age', 'region', 'no_claim_years']].values
y_freq = df_auto['n_claims'].values
y_sev = df_auto['claim_amount'].values

X_train_a, X_test_a, y_train_f, y_test_f = train_test_split(
    X_auto, y_freq, test_size=0.3, random_state=123)

# 随机森林预测索赔次数
rf_freq = RandomForestRegressor(n_estimators=100, random_state=123)
rf_freq.fit(X_train_a, y_train_f)
print(f"\n车险索赔次数预测: R²={rf_freq.score(X_test_a, y_test_f):.4f}")

# 梯度提升树预测索赔金额
y_sev_pos = y_sev[y_sev > 0]
X_sev_pos = X_auto[y_sev > 0]
X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(
    X_sev_pos, y_sev_pos, test_size=0.3, random_state=123)
gb_sev = GradientBoostingRegressor(n_estimators=100, random_state=123)
gb_sev.fit(X_train_s, y_train_s)
print(f"车险索赔金额预测: R²={gb_sev.score(X_test_s, y_test_s):.4f}")

# 导入所需的Python模块
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression, Lasso, Ridge
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

np.random.seed(123)
n = 1000
X = np.random.normal(0, 1, (n, 5))
df = pd.DataFrame(X, columns=['x1', 'x2', 'x3', 'x4', 'x5'])

# 分类目标
eta = 0.5 + 1.2 * X[:, 0] - 0.8 * X[:, 1] + 0.5 * X[:, 2]
p = 1 / (1 + np.exp(-eta))
y_class = (np.random.uniform(0, 1, n) < p).astype(int)

# 回归目标
y_reg = 2 + 1.5 * X[:, 0] - 0.8 * X[:, 1] + 0.5 * X[:, 2] + \
        0.3 * X[:, 3] + np.random.normal(0, 0.5, n)

df['y_class'] = y_class
df['y_reg'] = y_reg

print("数据描述:")
print(df.describe())


##############################################################################
# 10.2 无监督学习：主成分分析（PCA）
##############################################################################
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA()
X_pca = pca.fit_transform(X_scaled)

print(f"\nPCA解释方差比: {pca.explained_variance_ratio_}")
print(f"累积解释方差: {np.cumsum(pca.explained_variance_ratio_)}")