import torch
from model import *
from k_cross_fold_train import get_kfold_data
from torch.utils.data import DataLoader, Dataset, TensorDataset
import pandas as pd
import numpy as np
from ppnet_utils import *
from train import GetLoader
from sklearn.metrics import mean_absolute_error

def test_model(BP_choice, check_point):


    # 读取测试数据
    X_val = np.array(pd.read_csv('../PPNet/val_set/test_data.csv'))[:, 2:]
    Y_val = np.array(pd.read_csv('../PPNet/val_set/test_label.csv'))[:, 2:]

    if BP_choice == 'SBP':
        Y_val = Y_val[:, 0]
    elif BP_choice == 'DBP':
        Y_val = Y_val[:, 1]
    else:
        print('choose right BP!')
        return
    print('shape of x,y:', X_val.shape, Y_val.shape)

    # 数据归一化
    test_data_nor = correct_Normalize(X_val, 'PPG')
    test_label_nor = correct_Normalize(Y_val, BP_choice)


    # 放入DataLoader
    test_data_nor = test_data_nor.reshape(-1, 1, 250)
    test_label_nor = test_label_nor.reshape(-1,1)
    print(test_label_nor.shape, test_label_nor.shape)
    torch_data = GetLoader(test_data_nor, test_label_nor)
    val_loader = DataLoader(torch_data, batch_size=100, shuffle=True)

    net = CNN_LSTM_SDBP()
    checkpoint = check_point
    net.load_state_dict(checkpoint['model'])

    nmae_arr_BP = []
    nrmse_arr_BP = []

    mae_arr_BP = []
    me_arr_BP = []
    std_arr_BP = []

    with torch.no_grad():
        for data in val_loader:
            ppg_data, labels = data
            ppg_data = ppg_data.float()
            labels = labels.type(torch.FloatTensor)

            y_pred = net(ppg_data)
            # print(y_pred.size(),labels.size())
            nmae_arr_BP.append(cal_MAE(y_pred, labels))
            nrmse_arr_BP.append(cal_RMSE(y_pred, labels))


            y_pred_unor = correct_Unnormalized(y_pred, BP_choice)
            labels_unor = correct_Unnormalized(labels, BP_choice)

            mae_arr_BP.append(cal_MAE(y_pred_unor, labels_unor))
            me_arr_BP.append(cal_ME(y_pred_unor, labels_unor))
            std_arr_BP.append(cal_STD(y_pred_unor, labels_unor))


    # print(len(me_arr_BP))
    print('NMAE', sum(nmae_arr_BP) / len(nmae_arr_BP))
    print('NRMSE', sum(nrmse_arr_BP) / len(nrmse_arr_BP))
    print('ME=', sum(me_arr_BP) / len(me_arr_BP))
    print('MAE', sum(mae_arr_BP) / len(mae_arr_BP))
    print('STD=', sum(std_arr_BP) / len(std_arr_BP))


if __name__ == "__main__":

    checkpoint = torch.load("../PPNet/checkpoint/CNN_LSTM_DBP_lr_0_epoch_100_batch_100_droput0.1_v2.ckpt")
    test_model(BP_choice='DBP',check_point=checkpoint)
