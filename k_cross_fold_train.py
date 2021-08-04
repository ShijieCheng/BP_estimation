import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import pandas as pd
from torch.utils.data import DataLoader, Dataset, TensorDataset
from model import *
import time
import sys
from ppnet_utils import cal_MAE,cal_RMSE




#获取k折交叉验证某一折的训练集和验证集
def get_kfold_data(k, i, X, y):
    # 返回第 i+1 折 (i = 0 -> k-1) 交叉验证时所需要的训练和验证数据，X_train为训练集，X_valid为验证集
    fold_size = X.shape[0] // k  # 每份的个数:数据总条数/折数（组数）

    val_start = i * fold_size
    if i != k - 1:
        val_end = (i + 1) * fold_size
        X_valid, y_valid = X[val_start:val_end], y[val_start:val_end]
        X_train = torch.cat((X[0:val_start], X[val_end:]), dim=0)
        y_train = torch.cat((y[0:val_start], y[val_end:]), dim=0)
    else:  # 若是最后一折交叉验证
        X_valid, y_valid = X[val_start:], y[val_start:]  # 若不能整除，将多的case放在最后一折里
        X_train = X[0:val_start]
        y_train = y[0:val_start]

    return X_train, y_train, X_valid, y_valid



#模型训练
def traink(model, X_train, y_train, X_val, y_val, BATCH_SIZE, learning_rate, TOTAL_EPOCHS):

    #训练数据和测试数据
    train_loader = DataLoader(TensorDataset(X_train, y_train), BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, y_val), BATCH_SIZE, shuffle=True)
    #均方误差损失函数
    criterion = nn.MSELoss()
    #Adam优化器
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)

    #训练损失和测试损失
    losses = []
    val_losses = []

    for epoch in range(TOTAL_EPOCHS):
        model.train()

        for i, (ppg_data, labels) in enumerate(train_loader):
            ppg_data = ppg_data.float()
            labels = torch.squeeze(labels.type(torch.FloatTensor))
            # print(ppg_data.size(),labels.size()) #torch.Size([100, 1, 250])

            optimizer.zero_grad()  # 清零
            outputs = model(ppg_data)
            # print(outputs.size())torch.Size([100, 2])

            # 计算损失函数
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            # if (i + 1) % 1000 == 0:
                # 每10个batches打印一次loss
                # print('Epoch : %d/%d, Iter : %d/%d,  Loss: %.4f' % (epoch + 1, TOTAL_EPOCHS,
                #                                                     i + 1, len(X_train) // BATCH_SIZE,
                #                                                     loss.item()))

        #保存最后的模型
        if epoch == TOTAL_EPOCHS-1:
            print('saving epoch%d model' % (epoch + 1))
            state = {
                'model': model.state_dict(),
                'epoch': epoch + 1
            }  # 还可以保存优化器这些
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/PPNet_epoch_%d.ckpt' % (epoch + 1))


        # 每个epoch计算测试集的loss,MAE,RMSE
        model.eval()
        val_loss = 0
        mae_arr_SBP = []
        rmse_arr_SBP = []
        mae_arr_DBP = []
        rmse_arr_DBP = []
        mae_arr_BP = []
        rmse_arr_BP = []

        eval_dic = {}

        with torch.no_grad():
            for i, (ppg_data, labels) in enumerate(val_loader):
                ppg_data = ppg_data.float()
                labels = torch.squeeze(labels.type(torch.FloatTensor))
                optimizer.zero_grad()
                y_hat = model(ppg_data)

                #测试集的Loss
                loss = criterion(y_hat, labels).item()  # batch average loss
                val_loss += loss * len(labels)  # sum up batch loss
                #测试集的MAE&RMSE
                mae_arr_SBP.append(cal_MAE(y_hat[:,0],labels[:,0]))
                rmse_arr_SBP.append(cal_RMSE(y_hat[:,0],labels[:,0]))
                mae_arr_DBP.append(cal_MAE(y_hat[:, 1], labels[:, 1]))
                rmse_arr_DBP.append(cal_RMSE(y_hat[:, 1], labels[:, 1]))
                mae_arr_BP.append(cal_MAE(y_hat,labels))
                rmse_arr_BP.append(cal_RMSE(y_hat,labels))

        val_losses.append(val_loss / len(X_val))

        eval_dic['mae_SBP'] = sum(mae_arr_SBP)/len(mae_arr_SBP)
        eval_dic['rmse_SBP'] = sum(rmse_arr_SBP)/len(rmse_arr_SBP)
        eval_dic['mae_DBP'] = sum(mae_arr_DBP) / len(mae_arr_DBP)
        eval_dic['rmse_DBP'] = sum(rmse_arr_DBP) / len(rmse_arr_DBP)
        eval_dic['mae_BP'] = sum(mae_arr_BP) / len(mae_arr_BP)
        eval_dic['rmse_BP'] = sum(rmse_arr_BP) / len(rmse_arr_BP)

    # #画出训练过程中损失曲线
    # plt.figure()
    # plt.plot(losses)
    # plt.show()

    return losses, val_losses,eval_dic

def k_fold(k, X_train, y_train,model_choice, num_epochs, learning_rate, batch_size):
    train_loss_sum, valid_loss_sum,mae_sum_SBP,rmse_sum_SBP,mae_sum_DBP,rmse_sum_DBP = 0,0,0,0,0,0


    for i in range(k):
        print('*' * 25, '第', i + 1, '折', '*' * 25)
        data = get_kfold_data(k, i, X_train, y_train)  # 获取k折交叉验证的训练和验证数据

        #选择模型：CNN VS CNN_LSTM
        if model_choice == 'CNN':
            net =CNN()  # 实例化模型（某已经定义好的模型）
        elif model_choice == 'CNN_LSTM':
            net = CNN_LSTM()
        else:
            print('Choose right Model!')
            return

        # 每份数据进行训练
        train_loss, val_loss,eval_dic = traink(net, *data, batch_size, learning_rate, num_epochs)

        for i in eval_dic.items():
            print(i)

        train_loss_sum += train_loss[-1]
        valid_loss_sum += val_loss[-1]
        mae_sum_SBP += eval_dic['mae_SBP']
        rmse_sum_SBP += eval_dic['rmse_SBP']
        mae_sum_DBP += eval_dic['mae_DBP']
        rmse_sum_DBP += eval_dic['rmse_DBP']


    print('\n', '#' * 10, '最终k折交叉验证结果', '#' * 10)
    print('Average train loss for {} is {}'.format(model_choice,train_loss_sum / k))
    print('Average valid loss for {} is {}'.format(model_choice,valid_loss_sum / k))
    print('Average MAE for SBP is {}'.format(mae_sum_SBP/k))
    print('Average RMSE for SBP is{}'.format(rmse_sum_SBP/k))
    print('Average MAE for DBP is {}'.format(mae_sum_DBP / k))
    print('Average RMSE for DBP is{}'.format(rmse_sum_DBP / k))

    return

if __name__ == '__main__':
    data = pd.read_csv('../PPNet/Datasets/min_max_Nor_PPG.csv')
    data = np.array(data)[:,1:]
    X_train = torch.tensor(data).view(-1,1,250)
    print('shape of x:',X_train.size())

    labels = pd.read_csv('../PPNet/Datasets/labels_min_max.csv')
    labels = np.array(labels)[:,1:]
    y_train = torch.tensor(labels)
    print('shape of y:',y_train.size())

    # filename = 'save_result'
    # output = sys.stdout
    # outputfile = open(filename + '.txt', 'w')
    # sys.stdout = outputfile
    #
    # model=CNN_Model()
    # data = get_kfold_data(10, 6, X_train, y_train)
    # epoch_arr = [1,10,25,50]
    # for TOTAL_EPOCHS in epoch_arr:
    #     start_time = time.time()
    #     print('epoch',TOTAL_EPOCHS,file=outputfile)
    #     train_loss, val_loss,eval_dict= traink(model, *data, 100, 1e-3, TOTAL_EPOCHS)
    #     for i in eval_dict.items():
    #         print(i,file=outputfile)
    #     print('total time for CNN:{} mins'.format((time.time() - start_time) / 60),file=outputfile)
    #
    # outputfile.close()

    start_time = time.time()
    k_fold(10, X_train, y_train,model_choice='CNN',num_epochs=50, learning_rate=1e-3, batch_size=100)
    print('total time for CNN:{} mins'.format((time.time()-start_time)/60))
    start_time1 = time.time()
    k_fold(10, X_train, y_train,model_choice='CNN_LSTM',num_epochs=50, learning_rate=1e-3, batch_size=100)
    print('total time for CNN_LSTM:{} mins'.format((time.time()-start_time1)/60))
