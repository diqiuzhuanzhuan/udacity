# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
#import visuals as vs
from IPython.display import display
from sklearn import model_selection
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA

try:
    data = pd.read_csv("customers.csv")
    data.drop(['Region', 'Channel'], axis = 1, inplace = True)
    print "Wholesale customers dataset has {} samples with {} features each.".format(*data.shape)
except:
    print "Dataset could not be loaded. Is the dataset missing?"
#
display(data.describe())
describe = data.describe()
indices = [5, 38, 100]
#
samples = pd.DataFrame(data.loc[indices], columns = data.keys()).reset_index(drop = True)
print "Chosen samples of wholesale customers dataset:"
display(samples)

temp = samples
for i in temp.columns:
    temp.loc[:][i] = temp.loc[:][i]/describe.loc['mean'][i]

display(temp)
# TODO：为DataFrame创建一个副本，用'drop'函数丢弃一些指定的特征
new_data = data.drop(['Fresh', 'Milk', 'Frozen', 'Delicatessen', 'Grocery'], axis=1)

# TODO：使用给定的特征作为目标，将数据分割成训练集和测试集
X_train, X_test, y_train, y_test = model_selection.train_test_split(new_data, data.loc[:]['Grocery'], test_size=0.25, random_state=41)

# TODO：创建一个DecisionTreeRegressor（决策树回归器）并在训练集上训练它
regressor = DecisionTreeRegressor(random_state=42)
regressor.fit(X_train, y_train)

# TODO：输出在测试集上的预测得分
score = regressor.score(X_test, y_test)
print score

# 对于数据中的每一对特征构造一个散布矩阵
#pd.scatter_matrix(data, alpha = 0.3, figsize = (14,8), diagonal = 'kde')

# TODO：使用自然对数缩放数据
log_data = np.log(data)

# TODO：使用自然对数缩放样本数据
log_samples = np.log(samples)

# 为每一对新产生的特征制作一个散射矩阵
#pd.scatter_matrix(log_data, alpha = 0.3, figsize = (14,8), diagonal = 'kde')

# 展示经过对数变换后的样本数据
display(log_samples)

# 对于每一个特征，找到值异常高或者是异常低的数据点
excep_count = {}

def calc_count(a=[], b={}):
    for i in a:
        if b.has_key(i):
            b[i] = b[i] + 1
        else:
            b[i] = 1


for feature in log_data.keys():
    # TODO：计算给定特征的Q1（数据的25th分位点）
    Q1 = np.percentile(log_data.loc[:][feature], 25)

    # TODO：计算给定特征的Q3（数据的75th分位点）
    Q3 = np.percentile(log_data.loc[:][feature], 75)

    # TODO：使用四分位范围计算异常阶（1.5倍的四分位距）
    step = Q3 - Q1

    # 显示异常点
    print "Data points considered outliers for the feature '{}':".format(feature)
    display(log_data[~((log_data[feature] >= Q1 - step) & (log_data[feature] <= Q3 + step))])
#    excep_count.append(log_data[(log_data[feature] < Q1 - step) | (log_data[feature] > Q3 + step)].index)
    calc_count(log_data[(log_data[feature] < Q1 - step) | (log_data[feature] > Q3 + step)].index, excep_count)


outliers = []
# 可选：选择你希望移除的数据点的索引
for i in excep_count.keys():
    if excep_count[i] > 1:
        print i
        outliers.append(i)

print outliers

# 如果选择了的话，移除异常点
good_data = log_data.drop(log_data.index[outliers]).reset_index(drop=True)
print good_data

# TODO：通过在good data上使用PCA，将其转换成和当前特征数一样多的维度
pca = PCA(n_components=len(good_data.columns))

# TODO：使用上面的PCA拟合将变换施加在log_samples上
pca.fit(good_data)
pca_samples = pca.transform(log_samples)
print pca_samples
# 生成PCA的结果图
#pca_results = vs.pca_results(good_data, pca)

# 展示经过PCA转换的sample log-data
#display(pd.DataFrame(np.round(pca_samples, 4), columns = pca_results.index.values))

# TODO：通过在good data上进行PCA，将其转换成两个维度
pca = PCA(n_components=2)

# TODO：使用上面训练的PCA将good data进行转换
pca.fit(good_data)
reduced_data = pca.transform(good_data)

# TODO：使用上面训练的PCA将log_samples进行转换
pca_samples = pca.transform(log_samples)

# 为降维后的数据创建一个DataFrame
reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2'])


log_centers = pca.inverse_transform(reduced_data)
np.exp(log_centers)