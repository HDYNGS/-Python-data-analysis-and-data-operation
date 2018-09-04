import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA

# 读取数据文件
data = np.loadtxt('data1.txt')
print(data)
x = data[:, :-1]    # 获得输入的x,所有行中第一列到倒数第二列的数据
print(x)
y = data[:, -1]     # 获得目标变量y，所有行中最后一列的数据
print(y)
print (x[0], y[0])  # 打印输出x和y的第一条记录
print()

# 使用sklearn的DecisionTreeClassifier判断变量重要性
model_tree = DecisionTreeClassifier(random_state=0)  # 建立分类决策树模型对象
model_tree.fit(x, y)  # 将数据集的维度和目标变量输入模型
feature_importance = model_tree.feature_importances_  # 获得所有变量的重要性得分
print (feature_importance)  # 打印输出
print()
'''
[ 0.03331054  0.01513967  0.02199713  0.119727    0.47930312  0.04776297
  0.17111746  0.02585441  0.02012725  0.06566044]
'''

# 使用sklearn的PCA进行维度转换
model_pca = PCA()  # 建立PCA模型对象
model_pca.fit(x)  # 将数据集输入模型
model_pca.transform(x)  # 对数据集进行转换映射
components = model_pca.components_  # 获得转换后的所有主成分
components_var = model_pca.explained_variance_  # 获得各主成分的方差
components_var_ratio = model_pca.explained_variance_ratio_  # 获得各主成分的方差占比
print (components[:2])  # 打印输出前2个主成分
'''
[[ -7.18818316e-03  -1.41619205e-02  -1.00543847e-02  -3.65097575e-01
   -6.38944537e-01   1.95750380e-02   1.73413378e-01   3.80829974e-02
    2.87413113e-03   6.52829504e-01]
 [  1.01307710e-02  -1.95270201e-04  -2.33689543e-02  -6.12915216e-01
    5.08983971e-01  -2.23429533e-02   6.02958940e-01  -1.49061329e-02
   -1.81362216e-02  -3.41623971e-03]]
'''
print (components_var[:2])  # 打印输出前2个主成分的方差
'''
[ 4.22180334  2.20928822]
'''
print (components_var_ratio)  # 打印输出所有主成分的方差占比
'''
[  3.38339364e-01   1.77054475e-01   8.92753857e-02   8.73655166e-02
   8.23542686e-02   8.03329836e-02   7.38094896e-02   7.14685179e-02
   3.80545111e-33   1.88702079e-33]
'''