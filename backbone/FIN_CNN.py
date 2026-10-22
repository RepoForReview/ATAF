"""
Wu, Chuheng, et al. "An LSTM Feature Imitation Network for Hand Movement Recognition from sEMG Signals."
 ICASSP 2025-2025
IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2025.

"""
from typing import Tuple, Optional
import torch.nn as nn
from numpy import random
from torch.autograd.function import Function
from torch.utils.data import Dataset, DataLoader
from utils import *
import torch.nn.functional as F
import math
from sklearn.metrics import accuracy_score
from torch.autograd import Variable
import numpy as np
from loss import SupConLoss, scl
from sklearn.metrics import cohen_kappa_score
import torch
from torch import nn
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import StepLR
from scipy.stats import kurtosis
from scipy.special import softmax
from scipy.stats import entropy


def SSI(x):
    SSI = np.zeros((x.shape[0], 1))
    for k in range(x.shape[0]):
        SSI[k] = np.sum(np.square(x[k, :]))
    return SSI


def RMS(x):
    RMS = np.zeros((x.shape[0], 1))
    for k in range(x.shape[0]):
        RMS[k] = np.sqrt(np.mean(np.square(x[k, :])))
    return RMS


def ENT(x):
    ENT = np.zeros((x.shape[0], 1))
    for k in range(x.shape[0]):
        sm = softmax(x[k, :])
        ENT[k] = entropy(sm)
    return ENT


def VAR(x):
    VAR = np.zeros((x.shape[0], 1))
    for k in range(x.shape[0]):
        VAR[k] = np.var(x[k, :])
    return VAR


class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]



class fincnn_hyser(nn.Module):
    def __init__(self):
        super(fincnn_hyser, self).__init__()
        self.conv1 = nn.Conv2d(3, 256, 5)  # 16*16->12*12
        self.batchnorm1 = nn.BatchNorm2d(256)
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(256, 128, 5)  # 12*12->8*8
        self.batchnorm2 = nn.BatchNorm2d(128)
        self.prelu2 = nn.PReLU()
        self.conv3 = nn.Conv2d(128, 64, 5)  # 8*8->4*4
        self.batchnorm3 = nn.BatchNorm2d(64)
        self.prelu3 = nn.PReLU()
        self.conv4 = nn.Conv2d(64, 34, 4)  # 4->1
        self.batchnorm4 = nn.BatchNorm2d(34)
        self.prelu4 = nn.PReLU()
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = x.view(-1, 3, 16, 16)
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.prelu1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.prelu2(x)
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.prelu3(x)
        x = self.conv4(x)
        x = self.batchnorm4(x)
        out = self.prelu4(x)
        out = torch.squeeze(out)
        return out


class fincnn_ninapro(nn.Module):
    def __init__(self):
        super(fincnn_ninapro, self).__init__()
        self.conv1 = nn.Conv2d(3, 128, 5)  # 12*12->8*8
        self.batchnorm1 = nn.BatchNorm2d(128)
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(128, 64, 5)  # 8*8->4*4
        self.batchnorm2 = nn.BatchNorm2d(64)
        self.prelu2 = nn.PReLU()
        self.conv3 = nn.Conv2d(64, 21, 4)  # 4->1
        self.batchnorm3 = nn.BatchNorm2d(21)
        self.prelu3 = nn.PReLU()
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.prelu1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.prelu2(x)
        x = self.conv3(x)
        x = self.batchnorm3(x)
        out = self.prelu3(x)
        out = torch.squeeze(out)
        return out

class fincnn_senic(nn.Module):
    def __init__(self):
        super(fincnn_senic, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 5)  # 8*8->4*4
        self.batchnorm1 = nn.BatchNorm2d(64)
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(64, 8, 4)  # 4->1
        self.batchnorm2 = nn.BatchNorm2d(8)
        self.prelu2 = nn.PReLU()
        self.dropout = nn.Dropout(p=0.3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.prelu1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        out = self.prelu2(x)
        out = torch.squeeze(out)
        return out


def FIN_CNN(args, total_data):
    path = 'checkpoints/'+args.adapt_mode+'/'+args.dataset+'/'+\
           args.backbone+'/P'+str(args.source_parti)+'/best_model.pth'
    device = args.device
    data_train, data_val, data_test = total_data[0], total_data[1], total_data[2]
    emg_train = data_train['emg']
    label_train = data_train['label']
    emg_val = data_val['emg']
    label_val = data_val['label']
    emg_test = data_test['emg']
    label_test = data_test['label']

    SSI_train = SSI(np.reshape(emg_train, (-1, 200))).reshape((-1, emg_train.shape[1]))
    RMS_train = RMS(np.reshape(emg_train, (-1, 200))).reshape((-1, emg_train.shape[1]))
    # ENT_train = ENT(np.reshape(emg_train, (-1, 200))).reshape(-1, emg_train.shape[1])
    VAR_train = VAR(np.reshape(emg_train, (-1, 200))).reshape((-1, emg_train.shape[1]))

    SSI_val = SSI(np.reshape(emg_val, (-1, 200))).reshape((-1, emg_val.shape[1]))
    RMS_val = RMS(np.reshape(emg_val, (-1, 200))).reshape((-1, emg_val.shape[1]))
    # ENT_val = ENT(np.reshape(emg_val, (-1, 200))).reshape(-1, emg_val.shape[1])
    VAR_val = VAR(np.reshape(emg_val, (-1, 200))).reshape((-1, emg_val.shape[1]))

    SSI_test = SSI(np.reshape(emg_test, (-1, 200))).reshape((-1, emg_test.shape[1]))
    RMS_test = RMS(np.reshape(emg_test, (-1, 200))).reshape((-1, emg_test.shape[1]))
    # ENT_test = ENT(np.reshape(emg_test, (-1, 200))).reshape(-1, emg_test.shape[1])
    VAR_test = VAR(np.reshape(emg_test, (-1, 200))).reshape((-1, emg_test.shape[1]))

    if args.dataset == 'Hyser':
        SSI_train = np.reshape(SSI_train, (-1, 16, 16))
        RMS_train = np.reshape(RMS_train, (-1, 16, 16))
        VAR_train = np.reshape(VAR_train, (-1, 16, 16))

        SSI_val = np.reshape(SSI_val, (-1, 16, 16))
        RMS_val = np.reshape(RMS_val, (-1, 16, 16))
        VAR_val = np.reshape(VAR_val, (-1, 16, 16))

        SSI_test = np.reshape(SSI_test, (-1, 16, 16))
        RMS_test = np.reshape(RMS_test, (-1, 16, 16))
        VAR_test = np.reshape(VAR_test, (-1, 16, 16))
    else:
        SSI_train = np.tile(np.expand_dims(SSI_train, axis=-1), (1, 1, SSI_train.shape[1]))
        RMS_train = np.tile(np.expand_dims(RMS_train, axis=-1), (1, 1, RMS_train.shape[1]))
        VAR_train = np.tile(np.expand_dims(VAR_train, axis=-1), (1, 1, VAR_train.shape[1]))

        SSI_val = np.tile(np.expand_dims(SSI_val, axis=-1), (1, 1, SSI_val.shape[1]))
        RMS_val = np.tile(np.expand_dims(RMS_val, axis=-1), (1, 1, RMS_val.shape[1]))
        VAR_val = np.tile(np.expand_dims(VAR_val, axis=-1), (1, 1, VAR_val.shape[1]))

        SSI_test = np.tile(np.expand_dims(SSI_test, axis=-1), (1, 1, SSI_test.shape[1]))
        RMS_test = np.tile(np.expand_dims(RMS_test, axis=-1), (1, 1, RMS_test.shape[1]))
        VAR_test = np.tile(np.expand_dims(VAR_test, axis=-1), (1, 1, VAR_test.shape[1]))

    SSI_train = np.expand_dims(SSI_train, axis=1)
    RMS_train = np.expand_dims(RMS_train, axis=1)
    VAR_train = np.expand_dims(VAR_train, axis=1)

    SSI_val = np.expand_dims(SSI_val, axis=1)
    RMS_val = np.expand_dims(RMS_val, axis=1)
    VAR_val = np.expand_dims(VAR_val, axis=1)

    SSI_test = np.expand_dims(SSI_test, axis=1)
    RMS_test = np.expand_dims(RMS_test, axis=1)
    VAR_test = np.expand_dims(VAR_test, axis=1)

    emg_train = np.concatenate([SSI_train, RMS_train, VAR_train], axis=1)
    emg_val = np.concatenate([SSI_val, RMS_val, VAR_val], axis=1)
    emg_test = np.concatenate([SSI_test, RMS_test, VAR_test], axis=1)
    
    train_dataset = MyDataset(emg_train, label_train)
    val_dataset = MyDataset(emg_val, label_val)
    test_dataset = MyDataset(emg_test, label_test)
    if args.dataset == 'Hyser':
        batch_size = 64
        learning_rate = 0.001
    elif args.dataset == 'Ninapro':
        batch_size = 128
        learning_rate = 0.001
    elif args.dataset == 'Senic':
        batch_size = 128
        learning_rate = 0.001
    else:
        raise NotImplementedError

    total_epochs = 50
    # Define data loaders for training and testing data in this fold
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4, pin_memory=True)
    vali_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4, pin_memory=True)

    if args.dataset == 'Hyser':
        model = fincnn_hyser().float().to(device)
    elif args.dataset == 'Ninapro':
        model = fincnn_ninapro().float().to(device)
    elif args.dataset == 'Senic':
        model = fincnn_senic().float().to(device)
    else:
        raise NotImplementedError

    criteria = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.8)
    best_val_loss = float('inf')
    # Run the training loop for defined number of epochs
    for epoch in range(0, total_epochs):
        # Training
        model.train()
        # print('2')
        for batch_idx, (emg, label) in enumerate(train_loader):
            emg, label = emg.to(device, non_blocking=True), label.to(device, non_blocking=True)
            optimizer.zero_grad()
            output = model(emg)
            cross_loss = criteria(output, label)
            loss = cross_loss

            loss.backward()
            optimizer.step()
        scheduler.step()
        # print('3')
        # Validation
        model.eval()
        val_loss = []
        for i, (inputs, labels) in enumerate(vali_loader):
            emg, labels = inputs.to(device), labels.to(device)
            output = model(emg)
            cross_loss = criteria(output, labels)
            loss = cross_loss
            val_loss.append(loss.cpu().detach().numpy())
        val_loss = np.mean(val_loss)
        # print(epoch, val_loss)
        # 记录最优模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), path)
        # print(epoch, np.mean(val_loss))
    #     early_stopping(np.mean(val_loss), model)
    #     if early_stopping.early_stop:
    #         break
    # early_stopping.load_best_model(model)
    model.load_state_dict(torch.load(path))
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            output = model(inputs)
            _, preds = torch.max(output.data, 1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_targets.extend(labels.detach().cpu().numpy())
    accuracy = accuracy_score(all_targets, all_preds)
    print(f'Accuracy: {accuracy}')


if __name__ == "__main__":
    from calflops import calculate_flops
    dataset = 'Hyser'
    if dataset == 'Hyser':
        n_features = 256
        n_classes = 34
        batch_size = 256
        learning_rate = 0.001
    elif dataset == 'Ninapro':
        n_features = 12
        n_classes = 21
        batch_size = 256
        learning_rate = 0.001
    elif dataset == 'Senic':
        n_features = 8
        n_classes = 8
        batch_size = 256
        learning_rate = 0.001
    else:
        raise NotImplementedError

    model = emghandnet(input_channels=n_features, seq_length=200, num_classes=n_classes).float()
    batch_size = 1
    # input_shape = (batch_size, 14, 300) # ninapro
    input_shape = (batch_size, n_features, 200)
    flops, macs, params = calculate_flops(model=model,
                                          input_shape=input_shape,
                                          output_as_string=True,
                                          output_precision=4)
    print("FLOPs:%s   MACs:%s   Params:%s \n" % (flops, macs, params))

