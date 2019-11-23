# Logistic Regression

## using package

~~~ python
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt
iris = load_iris()
X = iris.data[:,[2, 3]]                                     # 训练数据的X参数
y = iris.target                                             # 训练数据的分类参数y
clf = LogisticRegression()                                  # 建立逻辑回归对象
clf.fit(X, y)                                               # fit数据
x_min, x_max = X[:,0].min() - 1, X[:,0].max() + 1           # 二维化绘制分类图像
y_min, y_max = X[:,1].min() - 1, X[:,1].max() + 1           # mesh方法
xx, yy = np.meshgrid(np.arange(x_min, x_max,0.1),           # 线性插值训练集数据
                     np.arange(y_min,y_max, 0.1))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])              # 得到预测集的结果
Z = Z.reshape(xx.shape)
plt.plot()
plt.contourf(xx, yy, Z, alpha=0.4, cmap = plt.cm.RdYlBu)
plt.scatter(X[:, 0], X[:, 1], c=y,  cmap = plt.cm.brg)
plt.title("Logistic Regression")
plt.xlabel("Petal.Length")
plt.ylabel("Petal.Width")
plt.show()

~~~



