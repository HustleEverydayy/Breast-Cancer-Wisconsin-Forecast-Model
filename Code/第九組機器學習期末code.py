from pandas import read_csv
import pandas as pd
from sklearn import datasets
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt





#資料集導入 
df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data',header=None
                              ,names = ['ID ','團塊厚度','細胞大小均勻性','細胞形狀均勻性','邊緣黏附','單上皮細胞大小'
                                        ,'裸核','平淡染色質','正常核仁','有絲分裂','分類'])
# 列印出來
print(df)

# 維度
print(df.shape)

# 資料集
df.info()
df.head(25)

# 資料集統計描述
print(df.describe())

# 資料集分佈情況
print(df.groupby('分類').size())

# 空值處理
mean_value = df[df['裸核']!="?"]['裸核'].astype(int).mean()
df = df.replace('?',mean_value)
df['裸核'] = df['裸核'].astype(np.int64)

# 顯示中文
plt.rcParams['font.sans-serif'] = ['Taipei Sans TC Beta']


# 資料可視化
# 箱線圖
df.plot(kind='box',subplots=True,layout=(3,4),sharex=False,sharey=False,figsize=(10,10))
pyplot.savefig('箱線圖')
pyplot.show()


#直方圖
df.hist(figsize=(10,10), color = 'dodgerblue')
pyplot.savefig('直方圖')
pyplot.show()

#散點矩陣圖
scatter_matrix(df,figsize=(10,10), color = 'dodgerblue')
pyplot.savefig('散點矩陣圖')
pyplot.show()

corr=df.corr()

#二維數據熱力圖
ax=sns.heatmap(corr,
            xticklabels=corr.columns,
            yticklabels=corr.columns,
            cmap="YlGnBu",
            square=True,
            linewidths=.1)
fig = ax.get_figure()
fig.savefig('熱力圖.png')



#分離資料集
array = df.values
X = array[:,1:9]
y = array[:,10]
# 十折交叉 70% 30% , validation驗證確認
validation_size = 0.3
# 亂數種子
seed = 7
# 訓練
X_train,X_validation,y_train,y_validation = train_test_split(X,y,test_size=validation_size,random_state=seed)

#評估算法,算法審查
models = {}
models['LR'] = LogisticRegression()
models['LDA'] = LinearDiscriminantAnalysis()
models['KNN'] = KNeighborsClassifier()
models['CART'] = DecisionTreeClassifier()
models['NB'] = GaussianNB()
models['SVM'] = SVC()



num_folds = 10
seed = 7
kfold = KFold(n_splits=num_folds,random_state=None)

#評估算法,評估算法 
results = []
for name in models:
    result = cross_val_score(models[name],X_train,y_train,cv=kfold,scoring='accuracy')
    results.append(result)
    msg = '%s:%.3f(%.3f)'%(name,result.mean(),result.std())
    print(msg)
#評估算法,圖標顯示 #箱線圖
fig = pyplot.figure()
fig.suptitle('算法比較')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(models.keys())
pyplot.savefig('算法比較')
pyplot.show()

#預測,使用評估數據集評估算法
svm = SVC()
svm.fit(X=X_train,y=y_train)
predictions = svm.predict(X_validation)

print('最終使用SVM算法')
# 準確度
print(accuracy_score(y_validation,predictions))
# 混淆矩陣
print(confusion_matrix(y_validation,predictions))
# 準確度、召回率(recall)、F1值(f1-score)
print(classification_report(y_validation,predictions))
