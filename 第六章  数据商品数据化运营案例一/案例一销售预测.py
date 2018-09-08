'''
案例描述：通过产品的运营活动数据来预测销售量（也就是订单量）
案例过程：
        一，导入库
        二，读取数据
        三，数据审查和检验（数据探索）
            1，数据概述
            2，值域分布
            3，缺失值审查
            4，相关性分析
        四，数据预处理
            1，异常值处理
            2，分割数据集
        五，模型的最优化参数训练域和检验
            1，最优化参数训练以及检验（关键在于参数）
            2，获取最佳训练模型（关键在于模型）
        六，新数据集预测
'''
# 导入库
import numpy as np  # 导入numpy库
import pandas as pd  # 导入pandas库
from sklearn.ensemble import GradientBoostingRegressor  # 集成方法回归库
from sklearn.model_selection import GridSearchCV  # 导入交叉检验库
import matplotlib.pyplot as plt  # 导入图形展示库

# 读取数据
def data_built():
    global raw_data
    raw_data = pd.read_table('products_sales.txt', delimiter=',')  #指定分隔符为逗号
    # 数据审查和校验
    # 数据概览
    print ('{:*^60}'.format('Data overview:'))
    print (raw_data.tail(2))  # 打印原始数据后2条
    print ('{:*^60}'.format('Data dtypes:'))
    print (raw_data.dtypes)  # 打印数据类型
    print ('{:*^60}'.format('Data DESC:'))
    print (raw_data.describe().round(1).T)  # 打印原始数据基本描述性信息
    '''
    描述性信息如下：
                     count    mean     std    min     25%     50%     75%      max
    limit_infor      731.0     0.0     0.4    0.0     0.0     0.0     0.0     10.0
    campaign_type    731.0     3.0     2.0    0.0     1.0     3.0     5.0      6.0
    campaign_level   731.0     0.7     0.5    0.0     0.0     1.0     1.0      1.0
    product_level    731.0     1.4     0.5    1.0     1.0     1.0     2.0      3.0
    resource_amount  731.0     5.0     1.8    1.0     3.0     5.0     7.0      9.0
    email_rate       731.0     0.5     0.2    0.1     0.3     0.5     0.6      0.8
    price            729.0   162.8    14.3  100.0     NaN     NaN     NaN    197.0
    discount_rate    731.0     0.8     0.1    0.5     0.8     0.8     0.9      1.0
    hour_resouces    731.0   848.2   686.6    2.0   315.5   713.0  1096.0   3410.0
    campaign_fee     731.0  3696.4  1908.6   20.0  2497.0  3662.0  4795.5  33380.0
    orders           731.0  4531.1  1932.5   22.0  3199.0  4563.0  6011.5   8714.0

    '''

data_built()

# 查看值域分布
def Range_distribution():
    global col_names , unque_value
    col_names = ['limit_infor', 'campaign_type', 'campaign_level', 'product_level']  # 定义要查看的列
    for col_name in col_names:  # 循环读取每个列
        unque_value = np.sort(raw_data[col_name].unique())  # 获得列唯一值
        print ('{:*^50}'.format('{1} unique values:{0}').format(unque_value, col_name))  # 打印输出
        '''
        结果为：
        **************limit_infor unique values:[ 0  1 10]***************
        **************campaign_type unique values:[0 1 2 3 4 5 6]***************
        **************campaign_level unique values:[0 1]***************
        **************product_level unique values:[1 2 3]***************
        
        '''
Range_distribution()

# 缺失值审查
def Missing_value():
    global na_cols , na_lines
    na_cols = raw_data.isnull().any(axis=0)  # 查看每一列是否具有缺失值
    print ('{:*^60}'.format('NA Cols:'))
    print (na_cols)  # 查看具有缺失值的列
    na_lines = raw_data.isnull().any(axis=1)  # 查看每一行是否具有缺失值
    print ('Total number of NA lines is: {0}'.format(na_lines.sum()))  # 查看具有缺失值的行总记录数
    '''
    **************************NA Cols:**************************
    limit_infor        False
    campaign_type      False
    campaign_level     False
    product_level      False
    resource_amount    False
    email_rate         False
    price               True
    discount_rate      False
    hour_resouces      False
    campaign_fee       False
    orders             False
    dtype: bool
    Total number of NA lines is: 2
    '''
Missing_value()

# 相关性分析
def Correlation_analysis():
    global short_name , long_name , name_dict
    print ('{:*^60}'.format('Correlation Analyze:'))
    short_name = ['li', 'ct', 'cl', 'pl', 'ra', 'er', 'price', 'dr', 'hr', 'cf', 'orders']
    long_name = raw_data.columns
    name_dict = dict(zip(long_name, short_name))
    print (raw_data.corr().round(2).rename(index=name_dict, columns=name_dict))  # 输出所有输入特征变量以及预测变量的相关性矩阵
    print (name_dict)
    '''
        ********************Correlation Analyze:********************
              li    ct    cl    pl    ra    er  price    dr    hr    cf  orders
    li      1.00 -0.03 -0.08 -0.04  0.05  0.04  -0.02  0.00  0.01 -0.04   -0.02
    ct     -0.03  1.00  0.04  0.03  0.01 -0.01  -0.05 -0.01  0.06  0.06    0.06
    cl     -0.08  0.04  1.00  0.06  0.05  0.05   0.02  0.02 -0.52  0.26    0.05
    pl     -0.04  0.03  0.06  1.00 -0.12 -0.12   0.59 -0.04 -0.25 -0.23   -0.30
    ra      0.05  0.01  0.05 -0.12  1.00  0.98   0.13  0.15  0.54  0.46    0.62
    er      0.04 -0.01  0.05 -0.12  0.98  1.00   0.14  0.18  0.54  0.47    0.63
    price  -0.02 -0.05  0.02  0.59  0.13  0.14   1.00  0.25 -0.08 -0.11   -0.10
    dr      0.00 -0.01  0.02 -0.04  0.15  0.18   0.25  1.00  0.17  0.19    0.23
    hr      0.01  0.06 -0.52 -0.25  0.54  0.54  -0.08  0.17  1.00  0.32    0.66
    cf     -0.04  0.06  0.26 -0.23  0.46  0.47  -0.11  0.19  0.32  1.00    0.76
    orders -0.02  0.06  0.05 -0.30  0.62  0.63  -0.10  0.23  0.66  0.76    1.00
    {'discount_rate': 'dr', 'limit_infor': 'li', 'email_rate': 'er', 'resource_amount': 'ra', 'orders': 'orders', 'campaign_fee': 'cf', 'campaign_type': 'ct', 'price': 'price', 'product_level': 'pl', 'hour_resouces': 'hr', 'campaign_level': 'cl'}
   '''
Correlation_analysis()

# 数据预处理
# 异常值处理
def Outlier():
    global sales_data
    sales_data = raw_data.fillna(raw_data['price'].mean())  # 缺失值替换为均值,用filna
    # sales_data = raw_data.drop('email_rate',axis=1) # 丢弃缺失值，用drop
    sales_data = sales_data[sales_data['limit_infor'].isin((0, 1))]  # 只保留促销值为0和1的记录
    sales_data['campaign_fee'] = sales_data['campaign_fee'].replace(33380, sales_data['campaign_fee'].mean())  # 将异常极大值替换为均值
    print ('{:*^60}'.format('transformed data:'))
    print (sales_data.describe().round(2).T.rename(index=name_dict))  # 打印处理完成数据基本描述性信息
Outlier()
# 分割数据集X和y
def Split_data():
    global X , y
    X = sales_data.ix[:, :-1]  # 分割X，这就是特征量
    y = sales_data.ix[:, -1]  # 分割y
Split_data()

# 模型最优化参数训练及检验
def Parameter_training():
    global  model_gbrm,parameters,model_gs
    model_gbr = GradientBoostingRegressor()  # 建立GradientBoostingRegressor回归对象
    parameters = {'loss': ['ls', 'lad', 'huber', 'quantile'],
                  'min_samples_leaf': [1, 2, 3, 4, 5],
                  'alpha': [0.1, 0.3, 0.6, 0.9]}  # 定义要优化的参数信息
    model_gs = GridSearchCV(estimator=model_gbr, param_grid=parameters, cv=5)  # 建立交叉检验模型对象
    model_gs.fit(X, y)  # 训练交叉检验模型
    print ('Best score is:', model_gs.best_score_)  # 获得交叉检验模型得出的最优得分
    print ('Best parameter is:', model_gs.best_params_)  # 获得交叉检验模型得出的最优参数
Parameter_training()
    '''
    Best score is: 0.931455506205
    Best parameter is: {'min_samples_leaf': 3, 'alpha': 0.9, 'loss': 'huber'}
    
    '''


# 获取最佳训练模型
def Best_model():
    global model_best
    model_best = model_gs.best_estimator_  # 获得交叉检验模型得出的最优模型对象
    model_best.fit(X, y)  # 训练最优模型
    plt.style.use("ggplot")  # 应用ggplot自带样式库
    plt.figure()  # 建立画布对象
    plt.plot(np.arange(X.shape[0]), y, label='true y')  # 画出原始变量的曲线
    plt.plot(np.arange(X.shape[0]), model_best.predict(X), label='predicted y')  # 画出预测变量曲线
    plt.legend(loc=0)  # 设置图例位置
    plt.show()  # 展示图像
Best_model()


# 新数据集预测
def predition():
    New_X = np.array([[1, 1, 0, 1, 15, 0.5, 177, 0.66, 101, 798]])  # 要预测的新数据记录
    print ('{:*^60}'.format('Predicted orders:'))
    print (model_best.predict(New_X).round(0))  # 打印输出预测值
    '''
    *********************Predicted orders:**********************
    [ 779.]
    '''
predition()