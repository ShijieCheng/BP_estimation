import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy import signal
import torch
from sklearn.metrics import mean_squared_error,mean_absolute_error



def split_train_test(read_path, datas_name, labels_name, test_rate):
    datas = np.array(pd.read_csv(read_path+datas_name))
    labels = np.array(pd.read_csv(read_path+labels_name))

    print(datas.shape,labels.shape)

    assert(datas.shape[0] == labels.shape[0])
    total_len = datas.shape[0] #数据总长度

    test_len = int(total_len*test_rate)#分割数据长度
    test_index = np.random.choice(np.arange(total_len), test_len, replace=False)
    train_index = np.delete(np.arange(total_len),test_index)

    #保存测试数据
    test_data = []
    test_label = []
    for i in test_index:
        test_data.append(datas[i,:])
        test_label.append(labels[i,:])
    test_data = np.vstack(test_data)
    test_label = np.vstack(test_label)
    print(test_data.shape,test_label.shape)

    test_data = pd.DataFrame(test_data)
    test_label = pd.DataFrame(test_label)
    test_data.to_csv('../PPNet/val_set/test_data.csv')
    test_label.to_csv('../PPNet/val_set/test_label.csv')

    #保存训练数据
    train_data = []
    train_label = []
    for i in train_index:
        train_data.append(datas[i, :])
        train_label.append(labels[i, :])
    train_data = np.vstack(train_data)
    train_label = np.vstack(train_label)
    print(train_data.shape,train_label.shape)
    train_data = pd.DataFrame(train_data)
    train_data.to_csv('../PPNet/train_set/train_data.csv')
    train_label = pd.DataFrame(train_label)
    train_label.to_csv('../PPNet/train_set/train_label.csv')



'''
10折交叉验证的归一化
'''
def min_max_Normalize(data):
    min_features = np.min(data, axis=0)
    max_features = np.max(data, axis=0)

    nor = (data - min_features) / (max_features - min_features)

    return nor, (min_features, max_features)

def min_max_Unnormalized(nor_data, min_data, max_data):
    return nor_data * (max_data - min_data) + min_data



'''
设置了最大值和最小值的归一化
'''
def correct_Normalize(data, feature):
    dict_norm = {'PPG': [0, 4], 'SBP': [80, 180],  'DBP': [60, 130]}

    min_features = dict_norm[feature][0]
    max_features = dict_norm[feature][1]

    nor = np.round((data - min_features) / (max_features - min_features), 4)
    nor[nor < 0] = 0
    nor[nor > 1] = 1

    return nor


def correct_Unnormalized(nor_data, feature):

    dict_norm = {'PPG':[0,4], 'SBP':[80,180], 'DBP':[60,130]}
    min_features = dict_norm[feature][0]
    max_features = dict_norm[feature][1]
    return nor_data * (max_features - min_features) + min_features

#计算平均绝对误差
def cal_MAE(y,y_pred):
    return torch.mean(torch.abs(y-y_pred))

#计算均方误差
def cal_RMSE(y,y_pred):
    return torch.sqrt(torch.mean(torch.pow(y-y_pred,2)))



#计算Error的标准差STD
def cal_STD(y, y_pred):
    return torch.std(abs(y_pred-y))

#计算ME
def cal_ME(y,y_pred):
    return torch.mean(y_pred-y)


#计算相关系数Correlation



# if __name__ == '__main__':
#
#     features = pd.read_csv('D:/DeepLearning/BP-estimation/BP-python/PPNet/Datasets/train_data.csv')  # 训练数据所在位置
#     features = pd.DataFrame(features)
#     labels = pd.read_csv('D:/DeepLearning/BP-estimation/BP-python/PPNet/Datasets/labels.csv')  # 标签所在目录
#     labels = pd.DataFrame(labels)
#
#     features = np.array(features)[:, 1:]
#     labels = np.array(labels)[:, 1:]
    #
    # #训练数据归一化
    # nor1 = min_max_Normalize(features)
    # min_max_Nor = pd.DataFrame(nor1)
    # min_max_Nor.to_csv('D:/DeepLearning/BP-estimation/BP-python/PPNet/Dataset/min_max_Nor_PPG.csv')
    # nor2 = zero_score_Normalize(features)
    # nor2 = pd.DataFrame(nor2)
    # nor2.to_csv('D:/DeepLearning/BP-estimation/BP-python/PPNet/Dataset/zero_score_PPG.csv')
    #
    # #标签归一化
    # nor1_labels = min_max_Normalize(labels)
    # print(nor1_labels.shape)
    # nor1_labels = pd.DataFrame(nor1_labels)
    # nor1_labels.to_csv('D:/DeepLearning/BP-estimation/BP-python/PPNet/Dataset/labels_min_max.csv')
    # nor2_labels = zero_score_Normalize(labels)
    # print(nor2_labels.shape)
    # nor2_labels = pd.DataFrame(nor2_labels)
    # nor2_labels.to_csv('D:/DeepLearning/BP-estimation/BP-python/PPNet/Dataset/labels_zero_score.csv')

    # min_BPs = np.min(labels, axis=0)
    # max_BPs = np.max(labels, axis=0)
    # print(min_BPs,max_BPs)#[63.7708 50.    ] [199.9093 154.9724]

