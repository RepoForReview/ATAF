"""
 "EMGHandNet: A hybrid CNN and Bi-LSTM architecture for hand activity
 classification using surface EMG signals."
 Biocybernetics and Biomedical Engineering 42, no. 1 (2022): 325-340.

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

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class emghandnet(nn.Module):
    def __init__(self, input_channels=12, seq_length=200, num_classes=5):
        super(emghandnet, self).__init__()

        self.conv1 = nn.Conv1d(input_channels, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.conv4 = nn.Conv1d(256, 512, kernel_size=3, padding=1)

        self.lstm = nn.LSTM(input_size=512, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(input_size=256, hidden_size=64, num_layers=1, batch_first=True, bidirectional=True)

        self.fc1 = nn.Linear(128 * seq_length, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))

        x = x.permute(0, 2, 1)  # 调整为 (batch, seq_length, features)
        x, _ = self.lstm(x)
        x, _ = self.lstm2(x)

        x = x.reshape(x.shape[0], -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


def EMGHandNet(args, total_data):
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

    train_dataset = MyDataset(emg_train, label_train)
    val_dataset = MyDataset(emg_val, label_val)
    test_dataset = MyDataset(emg_test, label_test)

    if args.dataset == 'Hyser':
        n_features = 256
        n_classes = 34
        batch_size = 128
        learning_rate = 0.001
    elif args.dataset == 'Ninapro':
        n_features = 12
        n_classes = 21
        batch_size = 128
        learning_rate = 0.001
    elif args.dataset == 'Senic':
        n_features = 8
        n_classes = 8
        batch_size = 128
        learning_rate = 0.001
    else:
        raise NotImplementedError

    total_epochs = 50
    # Define data loaders for training and testing data in this fold
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4, pin_memory=True)
    vali_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4, pin_memory=True)

    model = emghandnet(input_channels=n_features, seq_length=200, num_classes=n_classes).float().to(device)
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

