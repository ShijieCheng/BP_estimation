import pandas as pd
import torch
import numpy as np
from torch.utils.data import DataLoader
from model import *
import os
import matplotlib.pyplot as plt
from ppnet_utils import *


# 定义GetLoader类，继承Dataset方法，并重写__getitem__()和__len__()方法
class GetLoader(torch.utils.data.Dataset):
	# 初始化函数，得到数据
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label
    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels
    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一脸懵逼
    def __len__(self):
        return len(self.data)

def train_model(BP_choice, batch_size,total_epoches,is_plot=True):

    # 判断是否可以使用GPU
    use_gpu = torch.cuda.is_available()
    print('Use GPU:',use_gpu)

    # 读取训练数据和测试数据
    X_train = np.array(pd.read_csv('../PPNet/train_set/train_data.csv'))[:, 2:]
    Y_train = np.array(pd.read_csv('../PPNet/train_set/train_label.csv'))[:, 2:]

    if BP_choice == 'SBP':
        Y_train = Y_train[:,0]
    elif BP_choice == 'DBP':
        Y_train = Y_train[:,1]
    else:
        print('choose right BP!')
        return
    print('shape of x,y:',X_train.shape,Y_train.shape)

    # 数据归一化
    train_data_nor = correct_Normalize(X_train, 'PPG')
    train_label_nor = correct_Normalize(Y_train,BP_choice)

    # 放入DataLoader
    train_data_nor = train_data_nor.reshape(-1, 1, 250)
    train_label_nor = train_label_nor.reshape(-1,1)
    print(train_data_nor.shape, train_label_nor.shape)

    torch_data = GetLoader(train_data_nor, train_label_nor)
    train_loader = DataLoader(torch_data, batch_size=batch_size, shuffle=True)

    """
    开始训练
    """

    # 均方误差损失函数
    model = CNN_LSTM_SDBP()
    criterion = nn.MSELoss()
    if use_gpu:
        model = model.cuda()
        criterion = criterion.cuda()

    total_epoches = total_epoches
    # Adam优化器
    optimizer = torch.optim.Adam(params=model.parameters())

    train_loss = []
    for epoch in range(total_epoches):
        for i, (ppg_data, labels) in enumerate(train_loader):

            ppg_data = ppg_data.float()
            labels = labels.type(torch.FloatTensor)

            if use_gpu:
                ppg_data = ppg_data.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()  # 清零
            outputs = model(ppg_data)
            # print(labels.size())
            # print(outputs.size())

            # 计算损失函数
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            if (i + 1) % 100 == 0:
                train_loss.append(loss.item())

        if epoch == total_epoches - 1:
            print('saving epoch%d model' % (epoch + 1))
            state = {
                'model': model.state_dict(),
                'epoch': epoch + 1
            }  # 还可以保存优化器这些
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/CNN_LSTM_%s__epoch_%d_batch_%d_droput0.1_v2.ckpt' % (BP_choice,epoch + 1, batch_size))
    if is_plot:
        plt.figure()
        plt.plot(train_loss)
        plt.savefig("CNNLSTM_loss{}__epoch{}_batch_{}.jpg".format(BP_choice, total_epoches,batch_size))
if __name__ == "__main__":

    train_model(BP_choice='SBP', batch_size=100, total_epoches=100)
    train_model(BP_choice='DBP', batch_size=100,total_epoches=100)





