import pandas as pd

# 生成异常数据，（数据框）
def builtdata():
    df = pd.DataFrame({'harden': [1, 120, 3, 5, 2, 12, 13],
                       'james': [12, 17, 31, 53, 22, 32, 43]})
    print(df)
    '''
       harden  james
    0       1     12
    1     120     17
    2       3     31
    3       5     53
    4       2     22
    5      12     32
    6      13     43
    '''
    return df



# 通过Z-Score方法判断异常值
def panduan():
    df = builtdata()
    df_zscore = df.copy()     # 复制一个用来存储Z-score得分的数据框
    cols = df.columns         # 获得数据框的列名
    print(cols)
    '''
    Index(['harden', 'james'], dtype='object')
    '''
    for col in cols:          # 循环读取每列
        df_col = df[col]      # 得到每列的值
        z_score = (df_col - df_col.mean()) / df_col.std()   # 计算每列的Z-score得分,零均值规范化
        print(z_score)
        '''
        0   -0.491018
        1    2.254069
        2   -0.444882
        3   -0.398746
        4   -0.467950
        5   -0.237270
        6   -0.214202
        Name: harden, dtype: float64
        0   -1.242118
        1   -0.897085
        2    0.069007
        3    1.587151
        4   -0.552052
        5    0.138013
        6    0.897085
        Name: james, dtype: float64
        '''
        df_zscore[col] = z_score.abs() > 2.2    # 判断Z-score得分是否大于2.2，(算绝对值)，如果是则是True，否则为False
    print (df_zscore)  # 打印输出

builtdata()
panduan()