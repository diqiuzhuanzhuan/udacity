# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from time import time
from sklearn.metrics import f1_score

student_data = pd.read_csv("student-data.csv")
print "Student data read successfully!"

n_students = len(student_data)
n_features = len(student_data.columns)
n_passed = len(student_data.loc[student_data['passed'] == 'yes'])
n_failed = len(student_data.loc[student_data['passed'] == 'no'])
grad_rate = float(n_passed)/float(n_passed + n_failed) * 100

# 输出结果
print "Total number of students: {}".format(n_students)
print "Number of features: {}".format(n_features)
print "Number of students who passed: {}".format(n_passed)
print "Number of students who failed: {}".format(n_failed)
print "Graduation rate of the class: {:.2f}%".format(grad_rate)

# 提取特征列  :-1 第一列到倒数第一列（不包括倒数第一列）
feature_cols = list(student_data.columns[:-1])

# 提取目标列 ‘passed’  索引为-1就是最后一列的意思
target_col = student_data.columns[-1]

# 显示列的列表
print "Feature columns:\n{}".format(feature_cols)
print "\nTarget column: {}".format(target_col)

# 将数据分割成特征数据和目标数据（即X_all 和 y_all）
X_all = student_data[feature_cols]
y_all = student_data[target_col]

# 通过打印前5行显示特征信息
print "\nFeature values:"
print X_all.head()


def preprocess_features(X):
    ''' 预处理学生数据，将非数字的二元特征转化成二元值（0或1），将分类的变量转换成虚拟变量
    '''

    # 初始化一个用于输出的DataFrame
    output = pd.DataFrame(index=X.index)

    # 查看数据的每一个特征列
    for col, col_data in X.iteritems():

        # 如果数据是非数字类型，将所有的yes/no替换成1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])

        # 如果数据类型是类别的（categorical），将它转换成虚拟变量
        # 这样做的好处是多个类别的变量全部都转化成了2个类别的变量，可以都用0/1的方式来表达
        if col_data.dtype == object:
            # 例子: 'school' => 'school_GP' and 'school_MS'
            col_data = pd.get_dummies(col_data, prefix=col)

            # 收集转换后的列
        output = output.join(col_data)

    return output


X_all = preprocess_features(X_all)
print "Processed feature columns ({} total features):\n{}".format(len(X_all.columns), list(X_all.columns))

# TODO：在这里导入你可能需要使用的另外的功能
from sklearn.cross_validation import train_test_split
# TODO：设置训练集的数量
num_train = 300

# TODO：设置测试集的数量
num_test = X_all.shape[0] - num_train

# TODO：把数据集混洗和分割成上面定义的训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_all, y_all, test_size=num_test, random_state=30)

# 显示分割的结果
print "Training set has {} samples.".format(X_train.shape[0])
print "Testing set has {} samples.".format(X_test.shape[0])


def train_classifier(clf, X_train, y_train):
    ''' 用训练集训练分类器 '''

    # 开始计时，训练分类器，然后停止计时
    start = time()
    clf.fit(X_train, y_train)
    end = time()

    # Print the results
    print "Trained model in {:.4f} seconds".format(end - start)


def predict_labels(clf, features, target):
    ''' 用训练好的分类器做预测并输出F1值'''

    # 开始计时，作出预测，然后停止计时
    start = time()
    y_pred = clf.predict(features)
    end = time()

    # 输出并返回结果
    print "Made predictions in {:.4f} seconds.".format(end - start)
    return f1_score(target.values, y_pred, pos_label='yes')


def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' 用一个分类器训练和预测，并输出F1值 '''

    # 输出分类器名称和训练集大小
    print "Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train))

    # 训练一个分类器
    train_classifier(clf, X_train, y_train)

    # 输出训练和测试的预测结果
    print "F1 score for training set: {:.4f}.".format(predict_labels(clf, X_train, y_train))
    print "F1 score for test set: {:.4f}.".format(predict_labels(clf, X_test, y_test))


# TODO：从sklearn中引入三个监督学习模型
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn import neighbors
# from sklearn import model_B
# from skearln import model_C

# TODO：初始化三个模型
clf_A = GaussianNB()
clf_B = svm.SVC()
clf_C = neighbors.KNeighborsClassifier()

# TODO：设置训练集大小
X_train_100 = X_train[0:100]
y_train_100 = y_train[0:100]

train_predict(clf_A, X_train_100, y_train_100, X_test, y_test)
train_predict(clf_B, X_train_100, y_train_100, X_test, y_test)
train_predict(clf_C, X_train_100, y_train_100, X_test, y_test)

X_train_200 = X_train[0:200]
y_train_200 = y_train[0:200]
train_predict(clf_A, X_train_200, y_train_200, X_test, y_test)
train_predict(clf_B, X_train_200, y_train_200, X_test, y_test)
train_predict(clf_C, X_train_200, y_train_200, X_test, y_test)

X_train_300 = X_train[0:300]
y_train_300 = y_train[0:300]
train_predict(clf_A, X_train_300, y_train_300, X_test, y_test)
train_predict(clf_B, X_train_300, y_train_300, X_test, y_test)
train_predict(clf_C, X_train_300, y_train_300, X_test, y_test)


# TODO：对每一个分类器和每一个训练集大小运行'train_predict'
#train_predict(clf, X_train, y_train, X_test, y_test)

# TODO: 导入 'GridSearchCV' 和 'make_scorer'
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import f1_score

# TODO：创建你希望调整的参数列表
parameters =  [{'C': [1, 10, 20, 50, 100, 1000], 'kernel': ['linear']} ,{'C': [0.5, 1, 10, 20, 25, 35, 50, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.005, 0.001, 0.0001, 'auto'], 'kernel': ['rbf']}]

# TODO：初始化分类器
clf = svm.SVC()

# TODO：用'make_scorer'创建一个f1评分函数
f1_scorer = make_scorer(f1_score, pos_label='yes')

# TODO：在分类器上使用f1_scorer作为评分函数运行网格搜索
grid_obj = GridSearchCV(clf, param_grid=parameters, scoring=f1_scorer)

# TODO: Fit the grid search object to the training data and find the optimal parameters
# TODO：用训练集训练grid search object来寻找最佳参数
grid_obj = grid_obj.fit(X_train, y_train)

# Get the estimator
# 得到预测的结果
clf = grid_obj.best_estimator_
print grid_obj.best_params_

# Report the final F1 score for training and testing after parameter tuning
# 输出经过调参之后的训练集和测试集的F1值
print "Tuned model has a training F1 score of {:.4f}.".format(predict_labels(clf, X_train, y_train))
print "Tuned model has a testing F1 score of {:.4f}.".format(predict_labels(clf, X_test, y_test))
