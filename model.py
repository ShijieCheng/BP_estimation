import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import  torch.optim as optim


'''
input.permute(0,2,1) 
batch_size = first 
'''

# class CNN(nn.Module):
#     def __init__(self):
#         super(CNN, self).__init__()
#
#         #两个一维卷积层
#         self.conv1ds = nn.Sequential(
#             #Convolution1-----output:torch.Size([1, 20, 60])
#             nn.Conv1d(in_channels=1, out_channels=20, kernel_size=9),
#             # nn.BatchNorm1d(20),
#             nn.ReLU(True),
#             nn.MaxPool1d(kernel_size=4),
#             nn.Dropout(0.5),
#
#             #Convolution2-----output:torch.Size([1, 20, 13])
#             nn.Conv1d(in_channels=20, out_channels=20, kernel_size=9),
#             # nn.BatchNorm1d(20),
#             nn.ReLU(True),
#             nn.MaxPool1d(kernel_size=4),
#             nn.Dropout(0.5)
#         )
#
#         self.fc1 = nn.Linear(20*13,64)
#         self.fc2 = nn.Linear(64,128)
#         self.fc3 = nn.Linear(128,2)
#
#     def forward(self, x):
#         x = self.conv1ds(x)
#         x = x.view(x.size(0),-1)
#
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
#
# class CNN_LSTM(nn.Module):
#     def __init__(self):
#         super(CNN_LSTM, self).__init__()
#
#         # 两个一维卷积层
#         self.conv1ds = nn.Sequential(
#             # Convolution1-----output:torch.Size([1, 20, 60])
#             nn.Conv1d(in_channels=1, out_channels=20, kernel_size=9),
#
#             nn.ReLU(True),
#             nn.MaxPool1d(kernel_size=4),
#             nn.Dropout(0.5),
#
#             # Convolution2-----output:torch.Size([1, 20, 13]) 1batch,13单词，每个单词20维
#             nn.Conv1d(in_channels=20, out_channels=20, kernel_size=9),
#
#             nn.ReLU(True),
#             nn.MaxPool1d(kernel_size=4),
#             nn.Dropout(0.5)
#         )
#
#         # #两个LSTM：64cells 和128cells tanh激活函数
#         self.lstm1 = nn.LSTM(input_size=20,hidden_size=64,num_layers=1,batch_first=True,dropout=0.1)
#         self.lstm2 = nn.LSTM(input_size=64,hidden_size=128,num_layers=1,batch_first=True,dropout=0.1)
#         #
#         # #全连接层
#         self.fc = nn.Linear(128, 2)
#
#     def forward(self, x):
#         x = self.conv1ds(x)
#         x = x.view(-1,13,20)
#         out1, (h1,c1) = self.lstm1(x)
#         out2, (h2,c2) = self.lstm2(out1)
#         #
#         h2 = h2.view(h2.size(1),-1)
#         y = self.fc(h2)
#         return y


class CNN_LSTM_SDBP(nn.Module):
    def __init__(self):
        super(CNN_LSTM_SDBP, self).__init__()

        # 两个一维卷积层
        self.conv1ds = nn.Sequential(
            # Convolution1-----output:torch.Size([1, 20, 60])
            nn.Conv1d(in_channels=1, out_channels=20, kernel_size=9),

            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=4),
            nn.Dropout(0.5),

            # Convolution2-----output:torch.Size([1, 20, 13]) 1batch,13单词，每个单词20维
            nn.Conv1d(in_channels=20, out_channels=20, kernel_size=9),

            nn.ReLU(True),
            nn.MaxPool1d(kernel_size=4),
            nn.Dropout(0.5)
        )

        # #两个LSTM：64cells 和128cells tanh激活函数
        self.lstm1 = nn.LSTM(input_size=20,hidden_size=64,num_layers=1,batch_first=True,dropout=0.1)
        self.lstm2 = nn.LSTM(input_size=64,hidden_size=128,num_layers=1,batch_first=True,dropout=0.1)
        #
        # #全连接层
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x = self.conv1ds(x)
        x = x.view(-1,13,20)
        out1, (h1,c1) = self.lstm1(x)
        out2, (h2,c2) = self.lstm2(out1)
        #
        h2 = h2.view(h2.size(1),-1)
        y = self.fc(h2)
        return y



# net = CNN_LSTM_SDBP()
# x = torch.randn(100,1,250)#(100batch,250单词,一个单词的维度是1维)
# ht = net(x)
#
# print(ht.size())








