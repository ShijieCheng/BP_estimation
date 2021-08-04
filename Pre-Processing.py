import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy import signal
from ppnet_utils import split_train_test
"""
1、筛选出大于八分钟的数据
2、将PPG信号划分为6s重叠的8s窗口；按照同样方法从ABP信号获取SBP和DBP
3、对PPG进行下采样
4、分成训练集和测试集
"""


#将数据分割成8s窗口，其中6s重叠
def spilt_signal(ppg_signal, stride = 250, interval=1000):

    segmentArr = []#1000的窗宽

    start = 0
    while start+interval <= len(ppg_signal):
        segmentArr.append(ppg_signal[start:start+interval]) #截取1000个数据
        start += stride  #0,250……以此类推
    return np.array(segmentArr)


#对8s窗口数据进行下采样，变成250个
#取平均值
def down_sampling(segmentArr, scaling_factor = 4):
    downSampArr = []
    for seg in segmentArr:
        ds_seg = []
        for i in range(len(seg)):
            if i%scaling_factor == 0:
                ds_seg.append(np.mean(seg[i:i+4]))#取四个数的平均值
        downSampArr.append(ds_seg)

    return np.array(downSampArr)


#通过ABP信号获取SBP和DBP信号
def cal_BP(downSamp_ABP):

    SBP =np.max(downSamp_ABP,axis=1).reshape(-1,1)
    DBP = np.min(downSamp_ABP,axis=1).reshape(-1,1)

    return np.hstack((SBP,DBP))

if __name__ == '__main__':

    minMinutes = 8 # 8 minutes
    original_path = '../PPNet/Dataset_orig/'#原始数据路径

    for partID in range(1,5):
        cols_ppgRaw = ['partID', 'caseID', 'startT'] + ['f' + str(i) for i in range(1000)]
        df_ppg_raw = pd.DataFrame(columns=cols_ppgRaw)

        # cols_ppgDown = ['partID', 'caseID', 'startT'] + ['f' + str(i) for i in range(250)]
        # df_ppg_down = pd.DataFrame(columns=cols_ppgDown)

        cols_bpRaw = ['partID', 'caseID', 'startT', 'SBP', 'DBP']
        df_BP_raw = pd.DataFrame(columns=cols_bpRaw)

        # cols_bpDown = ['partID', 'caseID', 'startT', 'SBP', 'DBP']
        # df_BP_down = pd.DataFrame(columns=cols_bpDown)

        readPath = original_path + 'Part_' + str(partID) + '/'  # 原始数据的路径
        idNum = len(os.listdir(readPath))

        for caseID in range(1, idNum + 1):
            if caseID % 100 == 0:
                print('partID:', partID, '; caseID:', caseID)

            fileName = str(caseID) + '.csv'
            file = pd.read_csv(readPath + fileName)  # 读取文件

            # 去掉PPG和ECG少于八分钟的数据8*60*125=60000个数据
            signals = pd.DataFrame(file)
            # ECG = np.array(signals['ECG'])
            PPG = np.array(signals['PPG'])
            ABP = np.array(signals['ABP'])

            minSamples = minMinutes * 60 * 125  # 8 minutes
            if len(PPG) >= minSamples:
                # Process the ppg signals
                segment_PPG = spilt_signal(PPG, stride=250, interval=1000)
                downSamp_PPG = down_sampling(segment_PPG, scaling_factor=4)
                segment_ABP = spilt_signal(ABP, stride=250, interval=1000)
                downSamp_ABP = down_sampling(segment_ABP, scaling_factor=4)

                for i in range(segment_PPG.shape[0]):#逐列添加

                    #ppg信号
                    tmp = [partID, caseID, i * 250] + list(segment_PPG[i])
                    df_ppg_raw = df_ppg_raw.append(pd.Series(tmp, index=cols_ppgRaw), ignore_index=True)
                    # tmp = [partID, caseID, i * 250] + list(downSamp_PPG[i])
                    # df_ppg_down = df_ppg_down.append(pd.Series(tmp, index=cols_ppgDown), ignore_index=True)
                    #BP信号
                    rawSBP, rawDBP = np.round(np.max(segment_ABP[i]),4), np.round(np.min(segment_ABP[i]),4)
                    # downSBP, downDBP = np.round(np.max(downSamp_ABP[i]),4), np.round(np.min(downSamp_ABP[i]),4)

                    tmp = [partID, caseID, i * 250] + list([rawSBP, rawDBP])
                    df_BP_raw = df_BP_raw.append(pd.Series(tmp, index=cols_bpRaw), ignore_index=True)
                    # tmp = [partID, caseID, i * 250] + list([downSBP, downDBP])
                    # df_BP_down = df_BP_down.append(pd.Series(tmp, index=cols_bpDown), ignore_index=True)


        df_ppg_raw.to_csv('../PPNet/Datasets/df_PPGraw_part' + str(partID) + '.csv', index=False, float_format='%.4f')
        # df_ppg_down.to_csv('../PPNet/Datasets/df_PPGdown_part' + str(partID) + '.csv', index=False, float_format='%.4f')

        df_BP_raw.to_csv('../PPNet/Datasets/df_BPraw_part' + str(partID) + '.csv', index=False)
        # df_BP_down.to_csv('../PPNet/Datasets/df_BPdown_part' + str(partID) + '.csv', index=False)

    #
    # # 分割成训练集和测试集
    # split_train_test('../PPNet/', 'train_data.csv', 'labels.csv', 0.1)
    #
    #


















