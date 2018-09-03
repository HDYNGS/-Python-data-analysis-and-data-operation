import pandas as pd
import numpy as np
from sklearn.preprocessing import Imputer

#生成数据函数，并且生成缺失值函数
def data1():
    data = pd.DataFrame(np.random.randn(6,4) , columns = ['harden','james','kobe','magee'])#生成6行4列的二维数据组
    #在两个地方生成缺失值
    data.iloc[1:2,1] = np.nan
    print(data)
    '''
        harden     james      kobe     magee
    0  0.484287  0.021748 -0.419090  0.993388
    1  0.236432       NaN  1.397738 -0.160195
    2  0.008713 -0.345518 -0.031387  0.353801
    3  1.203145  1.774030  0.449158 -0.672659
    4  0.903220 -1.424713 -1.576019 -0.893289
    5 -0.109560 -0.591434 -2.284874  0.666646
    '''
    data.iloc[4,3] = np.nan
    print(data)
    '''
         harden     james      kobe     magee
    0  0.484287  0.021748 -0.419090  0.993388
    1  0.236432       NaN  1.397738 -0.160195
    2  0.008713 -0.345518 -0.031387  0.353801
    3  1.203145  1.774030  0.449158 -0.672659
    4  0.903220 -1.424713 -1.576019       NaN
    5 -0.109560 -0.591434 -2.284874  0.666646
    '''
    return data

#查看缺失值并且删除缺失值
def chakandata():
    data = data1()
    nan_all = data.isnull()
    print(nan_all)
    '''
      harden  james   kobe  magee
    0  False  False  False  False
    1  False   True  False  False
    2  False  False  False  False
    3  False  False  False  False
    4  False  False  False   True
    5  False  False  False  False
    '''
    #查看哪些列缺失
    nan_col1 = data.isnull().any()   #查看含有NA的列
    nan_col2 = data.isnull().all()   #查看全部为NA的类
    print(nan_col1)
    '''
    harden    False
    james      True
    kobe      False
    magee      True
    dtype: bool
    '''
    print(nan_col2)
    '''
    harden    False
    james     False
    kobe      False
    magee     False
    dtype: bool
    
    '''
    #丢弃缺失值
    data2 = data.dropna()  #直接丢弃含有缺失值的行记录
    print(data2)
    '''
         harden     james      kobe     magee
    0  0.005311  0.121857 -0.735128  1.834487
    2  1.053367 -0.890182  0.094548  0.042698
    3  1.353631 -0.293942  0.892748  0.021165
    5 -0.303813  0.898552  0.799187  1.219731
    '''

#数据插补
def chabu():
    data = data1()
    print(data)
    nan_model = Imputer(missing_values='NaN', strategy='mean', axis=0)  # 建立替换规则：将值为Nan的缺失值以均值做替换
    nan_result = nan_model.fit_transform(data)  # 应用模型规则
    print(nan_result)  # 打印输出
    '''
    [[ 1.6027148   0.2940447  -0.47727709  0.21849312]
     [ 0.94060422 -0.20471904 -0.36557366 -1.04459982]
     [-0.07537724  0.39486557 -1.06508238 -1.21169246]
     [-0.06039499 -0.18611966  1.1979904  -0.03219305]
     [-0.95267398 -0.65835792  1.25432293 -0.06747372]
     [-0.01629752 -0.86802788 -0.36054283  1.73262361]]
     '''
#使用pandas插补数据
def pandaschabu():
    data = data1()
    print(data)
    '''
     harden     james      kobe     magee
    0 -0.497874  1.044703  1.029007  0.251353
    1 -0.860254       NaN -0.148023 -1.543162
    2  0.313038 -0.062016 -0.510851 -0.198510
    3  1.314235 -0.264112  0.564382  1.748012
    4  0.860411  0.141933 -0.787281       NaN
    5  0.700523  0.913142 -1.553863 -1.115910
    '''
    nan_result_pd1 = data.fillna(method='backfill')  # 用后面的值替换缺失值
    nan_result_pd2 = data.fillna(method='bfill', limit=1)  # 用后面的值替代缺失值,限制每列只能替代一个缺失值
    nan_result_pd3 = data.fillna(method='pad')  # 用前面的值替换缺失值
    nan_result_pd4 = data.fillna(0)  # 用0替换缺失值
    nan_result_pd5 = data.fillna({'james': 1.1, 'magee': 1.2})  # 用不同值替换不同列的缺失值
    nan_result_pd6 = data.fillna(data.mean()['james':'magee'])  # 用平均数代替,选择各自列的均值替换缺失值
    # 打印输出
    print(nan_result_pd1)  # 打印输出
    '''
        harden     james      kobe     magee
    0 -0.497874  1.044703  1.029007  0.251353
    1 -0.860254 -0.062016 -0.148023 -1.543162
    2  0.313038 -0.062016 -0.510851 -0.198510
    3  1.314235 -0.264112  0.564382  1.748012
    4  0.860411  0.141933 -0.787281 -1.115910
    5  0.700523  0.913142 -1.553863 -1.115910
    
    '''
    print(nan_result_pd2)  # 打印输出
    '''
     harden     james      kobe     magee
    0 -0.497874  1.044703  1.029007  0.251353
    1 -0.860254 -0.062016 -0.148023 -1.543162
    2  0.313038 -0.062016 -0.510851 -0.198510
    3  1.314235 -0.264112  0.564382  1.748012
    4  0.860411  0.141933 -0.787281 -1.115910
    5  0.700523  0.913142 -1.553863 -1.115910
    '''
    print(nan_result_pd3)  # 打印输出
    '''
         harden     james      kobe     magee
    0 -0.497874  1.044703  1.029007  0.251353
    1 -0.860254  1.044703 -0.148023 -1.543162
    2  0.313038 -0.062016 -0.510851 -0.198510
    3  1.314235 -0.264112  0.564382  1.748012
    4  0.860411  0.141933 -0.787281  1.748012
    5  0.700523  0.913142 -1.553863 -1.115910
    '''
    print(nan_result_pd4)  # 打印输出
    '''
        harden     james      kobe     magee
    0 -0.497874  1.044703  1.029007  0.251353
    1 -0.860254  0.000000 -0.148023 -1.543162
    2  0.313038 -0.062016 -0.510851 -0.198510
    3  1.314235 -0.264112  0.564382  1.748012
    4  0.860411  0.141933 -0.787281  0.000000
    5  0.700523  0.913142 -1.553863 -1.115910
    '''
    print(nan_result_pd5)  # 打印输出
    '''
         harden     james      kobe     magee
    0 -0.351436  0.629751  0.205675  0.453086
    1  0.765974  1.100000 -0.168373 -1.532610
    2 -0.268235  0.492491  0.069876  1.308133
    3  0.576568 -0.523269 -0.145122  2.516401
    4  0.056818  1.314040 -0.518247  1.200000
    5  1.240566  0.424787 -0.489913  1.246508
    
    '''
    print(nan_result_pd6)  # 打印输出
    '''
         harden     james      kobe     magee
    0 -0.351436  0.629751  0.205675  0.453086
    1  0.765974  0.467560 -0.168373 -1.532610
    2 -0.268235  0.492491  0.069876  1.308133
    3  0.576568 -0.523269 -0.145122  2.516401
    4  0.056818  1.314040 -0.518247  0.798304
    5  1.240566  0.424787 -0.489913  1.246508
'''



data1()
chakandata()
chabu()
pandaschabu()
