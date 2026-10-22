#Zhang C, Wang X, Yu Z, et al. Interpretable Dual-branch EMGNet: A transfer learning-based network for inter-subject lower limb motion intention recognition[J]. Engineering Applications of Artificial Intelligence, 2024, 130: 107761.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from utils import *
from sklearn.metrics import cohen_kappa_score
from fvcore.nn import FlopCountAnalysis
from torch.optim.lr_scheduler import StepLR

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding=kernel_size // 2)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.sigmoid(x)


class swish(nn.Module):
    def __init__(self):
        super(swish, self).__init__()
    def forward(self, x):
        return x * torch.sigmoid(x)


class Conv1dWithConstraint(nn.Conv1d):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv1dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            norm = self.weight.data.norm(2, dim=0, keepdim=True)
            desired = torch.clamp(norm, max=self.max_norm)
            self.weight.data = self.weight.data * (desired / (norm + 1e-6))
        return super(Conv1dWithConstraint, self).forward(x)


class network(nn.Module):
    def __init__(self, input_channels, n_classes=17):
        super(network, self).__init__()
        if input_channels == 256:
            self.dim_change = nn.Sequential(
                Conv1dWithConstraint(input_channels, 16, kernel_size=1, max_norm=2, doWeightNorm=True),
                nn.BatchNorm1d(16),
                swish()
            )
            input_channels = 16

        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=9, padding=4),
                nn.LeakyReLU(0.3),
                nn.MaxPool1d(4),
                nn.Dropout(0.2)
            ) for _ in range(input_channels)
        ])

        self.deep_conv = nn.Conv1d(32 * input_channels, 64, kernel_size=9, padding=4)
        self.leaky_relu = nn.LeakyReLU(0.3)
        # self.bottleneck = ImprovedBottleneck(64, 64, kernel_size=9, expand_ratio=128, stride=1, length=timesteps//4)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, n_classes)

    def forward(self, x):
        if x.shape[1] == 256:
            x = self.dim_change(x)
        out = [conv_block(x[:, i:i + 1, :]) for i, conv_block in enumerate(self.conv_blocks)]
        x = torch.cat(out, dim=1)
        x = self.leaky_relu(self.deep_conv(x))
        # x = self.bottleneck(x)
        x = self.global_avg_pool(x).squeeze(-1)
        x = F.dropout(x, p=0.2)
        x = self.fc(x)
        return x


def CNN_EMGNet(args, total_data):
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
    model = network(n_features, n_classes).float().to(device)
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

    dataset = '1'
    if dataset == 'GrabMyo':
        n_features = 16
        n_classes = 17
        timesteps = 200
        batch_size = 128
    elif dataset == 'Hyser':
        n_features = 256
        n_classes = 34
        timesteps = 200
        batch_size = 128
    else:
        n_features = 10
        n_classes = 11
        timesteps = 200
        batch_size = 128
    model = network(n_features, n_classes).float()
    batch_size = 1
    # input_shape = (batch_size, 14, 300) # ninapro
    input_shape = (batch_size, n_features, timesteps)
    flops, macs, params = calculate_flops(model=model,
                                          input_shape=input_shape,
                                          output_as_string=True,
                                          output_precision=4)
    print("FLOPs:%s   MACs:%s   Params:%s \n" % (flops, macs, params))
