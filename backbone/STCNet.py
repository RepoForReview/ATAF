"""
Computers in Biology and Medicine 2025
STCNet: Spatio-Temporal Cross Network with subject-aware contrastive
learning for hand gesture recognition in surface EMG✩
"""

import torch.nn as nn
import torch.nn.functional as F
import torch
import copy
import numpy as np
import math
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch
import copy
import math
from torch.optim.lr_scheduler import StepLR


class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class STCBlock(nn.Module):
    def __init__(self, in_ch):
        super(STCBlock, self).__init__()
        self.temporal_conv = TemporalConv(in_ch)
        self.spatial_conv = SpatialConv(in_ch)

    def forward(self, x):
        out1 = self.temporal_conv(x)
        out2 = self.spatial_conv(x)
        out = torch.cat((out1, out2), dim=1)
        return out


class TemporalConv(nn.Module):
    def __init__(self, in_ch):
        super(TemporalConv, self).__init__()
        self.block1 = ResidualBlock(in_ch, 16, 1)
        self.block2 = ResidualBlock(16, 32, 2)
        self.block3 = ResidualBlock(32, 64, 2)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*25, 64 * 8),
            nn.ReLU(),
            nn.Linear(64 * 8, 64)
        )

    def forward(self, x):
        x_ = self.block1(x)
        x_ = self.block2(x_)
        x_ = self.block3(x_)
        # print(x_.shape)
        out = self.fc(x_)
        return out


class SpatialConv(nn.Module):
    def __init__(self, in_ch):
        super(SpatialConv, self).__init__()
        self.block1 = ResidualBlock(100, 16, 1)
        self.block2 = ResidualBlock(16, 32, 2)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(int(in_ch/2) * 32, int(in_ch/2) * 8),
            nn.ReLU(),
            nn.Linear(int(in_ch/2) * 8, 32)
        )

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.block1(x)
        x = self.block2(x)
        out = self.fc(x)
        return out


class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first = False):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):
        # x size: batch, steps, channels, length
        batch, steps, channels, length = x.size()
        x_reshape = x.contiguous().view(batch * steps, channels, length)
        out_ = self.module(x_reshape)
        out = out_.view(batch, steps, -1)
        return out


def ConvBlock3x3(in_ch, out_ch, stride):
    return nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=3, stride=stride, padding=1,
                     padding_mode='zeros')


def ConvBlock1x1(in_ch, out_ch, stride):
    return nn.Conv1d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, stride=stride, padding=0)


class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvBlock3x3(in_ch, out_ch, stride)
        self.bat1 = nn.BatchNorm1d(out_ch)
        self.drop1 = nn.Dropout(0.3)
        self.actv1 = nn.ReLU()
        self.conv2 = ConvBlock3x3(out_ch, out_ch, 1)
        self.bat2 = nn.BatchNorm1d(out_ch)
        self.downsample = None
        if stride != 1 or in_ch != out_ch:
            self.downsample = nn.Sequential(
                ConvBlock1x1(in_ch, out_ch, stride),
                nn.BatchNorm1d(out_ch)
            )

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.bat1(x)
        x = self.actv1(x)
        x = self.drop1(x)
        x = self.conv2(x)
        x = self.bat2(x)
        if self.downsample is not None:
            identity = self.downsample(identity)
        x = x + identity
        return x


class Encoder(nn.Module):
    def __init__(self, in_ch):
        super(Encoder, self).__init__()
        self.stconv = TimeDistributed(STCBlock(in_ch), batch_first=True)
        self.longterm_encoder = nn.LSTM(96, 200, num_layers=2, batch_first=True, dropout=0.2,
                                        bidirectional=True)
        self.flat = nn.Flatten()

    def forward(self, x):
        cout = self.stconv(x)
        out, (_, _) = self.longterm_encoder(cout)
        out = self.flat(out)
        return out


class STCNetCE(nn.Module):
    def __init__(self, in_ch, cdn):
        super(STCNetCE, self).__init__()
        self.in_ch = in_ch
        self.encoder = Encoder(self.in_ch)
        self.fc = nn.Sequential(
            nn.Linear(800, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, cdn)
        )

    def forward(self, x):
        x = x.view(-1, self.in_ch, 2, 100).permute(0, 2, 1, 3)
        out = self.encoder(x)
        out = self.fc(out)
        return out

def STCNet(args, total_data):
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    vali_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # Init the neural network
    model = STCNetCE(n_features, n_classes).float().to(device)
    criteria = nn.CrossEntropyLoss()
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    # scheduler = WarmupCosineAnnealingLR(optimizer, total_epochs, warmup_steps=5)
    # early_stopping = EarlyStopping(patience=8, min_delta=0, path=path)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.8)
    best_val_loss = float('inf')

    for epoch in range(0, total_epochs):
        # Training
        model.train()
        for batch_idx, (emg, label) in enumerate(train_loader):
            emg, label = emg.to(device), label.to(device)
            optimizer.zero_grad()
            outputs = model(emg)
            loss = criteria(outputs, label)
            loss.backward()
            optimizer.step()
        scheduler.step()

        # Validation
        model.eval()
        val_loss = []
        for i, (inputs, labels) in enumerate(vali_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            loss = criteria(outputs, labels)
            val_loss.append(loss.cpu().detach().numpy())
        val_loss = np.mean(val_loss)

        # print(epoch, np.mean(val_loss))

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
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs.data, 1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_targets.extend(labels.detach().cpu().numpy())
    accuracy = accuracy_score(all_targets, all_preds)
    # precision = precision_score(all_targets, all_preds, average='macro')  # 对所有类别计算平均精确率
    # recall = recall_score(all_targets, all_preds, average='macro')  # 对所有类别计算平均召回率
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

    model = STCNetCE(n_features, n_classes).float()
    batch_size = 256
    # input_shape = (batch_size, 14, 300) # ninapro
    input_shape = (batch_size, n_features, 200)
    flops, macs, params = calculate_flops(model=model,
                                          input_shape=input_shape,
                                          output_as_string=True,
                                          output_precision=4)
    print("FLOPs:%s   MACs:%s   Params:%s \n" % (flops, macs, params))
