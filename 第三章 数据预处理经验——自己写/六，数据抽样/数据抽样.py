import random  # 导入标准库
import numpy as np  # 导入第三方库

# 简单随机抽样
data =list(np.loadtxt('data3.txt'))  # 导入普通数据文件
data_sample = random.sample(data, 2000) # 随机抽取2000个样本
print (data_sample[:2])  # 打印输出前2条数据
print (len(data_sample))  # 打印输出抽样样本量

# 等距抽样
data = np.loadtxt('data3.txt')  # 导入普通数据文件
sample_count = 2000  # 指定抽样数量
record_count = data.shape[0]  # 获取最大样本量
width = record_count / sample_count  # 计算抽样间距
data_sample = []  # 初始化空白列表，用来存放抽样结果数据
i = 0  # 自增计数以得到对应索引值
while len(data_sample) <= sample_count and i * width <= record_count - 1:  # 当样本量小于等于指定抽样数量并且矩阵索引在有效范围内时
    data_sample.append(data[i * width])  # 新增样本
    i += 1  # 自增长
print (data_sample[:2])  # 打印输出前2条数据
print (len(data_sample))  # 打印输出样本数量

# 分层抽样
# 导入有标签的数据文件
data2 = np.loadtxt('data2.txt')  # 导入带有分层逻辑的数据
each_sample_count = 200  # 定义每个分层的抽样数量
label_data_unique = np.unique(data2[:, -1])  # 定义分层值域
sample_list = []  # 定义空列表，用于存放临时分层数据
sample_data = []  # 定义空列表，用于存放最终抽样数据
sample_dict = {}  # 定义空字典，用来显示各分层样本数量
for label_data in label_data_unique:  # 遍历每个分层标签
    for data_tmp in data2:  # 读取每条数据
        if data_tmp[-1] == label_data:  # 如果数据最后一列等于标签
            sample_list.append(data_tmp)  # 将数据加入到分层数据中
    each_sample_data = random.sample(sample_list, each_sample_count)  # 对每层数据都随机抽样
    sample_data.extend(each_sample_data)  # 将抽样数据追加到总体样本集
    sample_dict[label_data] = len(each_sample_data)  # 样本集统计结果
print (sample_dict)  # 打印输出样本集统计结果

# 整群抽样
data3 = np.loadtxt('data4.txt')  # 导入已经划分好整群的数据集
label_data_unique = list(np.unique(data3[:, -1]))  # 定义整群标签值域
print (label_data_unique)  # 打印输出所有整群标签
sample_label = random.sample(label_data_unique, 2)  # 随机抽取2个整群
sample_data = []  # 定义空列表，用来存储最终抽样数据
for each_label in sample_label:  # 遍历每个整群标签值域
    for data_tmp in data3:  # 遍历每个样本
        if data_tmp[-1] == each_label:  # 判断样本是否属于抽样整群
            sample_data.append(data_tmp)  # 样本添加到最终抽样数据集
print (sample_label)  # 打印输出样本整群标签
print (len(sample_data))  # 打印输出总抽样数据记录条数