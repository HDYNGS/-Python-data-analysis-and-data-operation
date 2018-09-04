import pandas as pd

# 生成重复数据
def builtdata():
    data1 = ['a', 3]
    data2 = ['b', 2]
    data3 = ['a', 3]
    data4 = ['c', 2]
    df = pd.DataFrame([data1, data2, data3, data4], columns=['col1', 'col2'])
    print (df)
    return df


def panduan():
    df = builtdata()
    # 判断重复数据
    isDuplicated = df.duplicated()  # 判断重复数据记录
    print (isDuplicated)  # 打印输出
    '''
    0    False
    1    False
    2     True
    3    False
    dtype: bool
    '''

    # 删除重复值
    new_df1 = df.drop_duplicates()  # 删除数据记录中所有列值相同的记录
    new_df2 = df.drop_duplicates(['col1'])  # 删除数据记录中col1值相同的记录
    new_df3 = df.drop_duplicates(['col2'])  # 删除数据记录中col2值相同的记录
    new_df4 = df.drop_duplicates(['col1', 'col2'])  # 删除数据记录中指定列（col1/col2）值相同的记录
    print (new_df1)  # 打印输出
    print (new_df2)  # 打印输出
    print (new_df3)  # 打印输出
    print (new_df4)  # 打印输出


builtdata()
panduan()
