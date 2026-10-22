# Y. Zou and L. Cheng, “A transfer learning model for gesture recognition based on the deep features extracted by CNN,”
# IEEE Trans. Artif. Intell. , vol. 2, no. 5, pp. 447–458, Oct. 2021.
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import accuracy_score, precision_score, recall_score
import torch
import torch.nn as nn
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from utils import *
import numpy as np
from fvcore.nn import FlopCountAnalysis
from sklearn.metrics import cohen_kappa_score
from torch.optim.lr_scheduler import StepLR

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class MKBlock(nn.Module):
    def __init__(self) -> None:
        super(MKBlock, self).__init__()
        self.streams = nn.ModuleList()
        self.streams.append(
            nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 11), stride=(1, 1),
                          padding=(1, 11 // 2)),
                nn.ELU(inplace=True),
                nn.BatchNorm2d(32),
                nn.MaxPool2d(kernel_size=(1, 16)),
                nn.Dropout(p=0.5),
                nn.Conv2d(in_channels=32, out_channels=64, \
                          kernel_size=(3, 3), stride=(1, 1),
                          padding=(1, 3 // 2)),
                nn.ELU(inplace=True),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(kernel_size=(2, 2)),
                nn.Dropout(p=0.5)
            ))
        self.streams.append(
            nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 21), stride=(1, 1),
                          padding=(1, 21 // 2)),
                nn.ELU(inplace=True),
                nn.BatchNorm2d(32),
                nn.MaxPool2d(kernel_size=(1, 16)),
                nn.Dropout(p=0.5),
                nn.Conv2d(in_channels=32, out_channels=64, \
                          kernel_size=(3, 3), stride=(1, 1),
                          padding=(1, 3 // 2)),
                nn.ELU(inplace=True),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(kernel_size=(2, 2)),
                nn.Dropout(p=0.5)
            ))
        self.streams.append(
            nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 31), stride=(1, 1),
                          padding=(1, 31 // 2)),
                nn.ELU(inplace=True),
                nn.BatchNorm2d(32),
                nn.MaxPool2d(kernel_size=(1, 16)),
                nn.Dropout(p=0.5),
                nn.Conv2d(in_channels=32, out_channels=64, \
                          kernel_size=(3, 3), stride=(1, 1),
                          padding=(1, 3 // 2)),
                nn.ELU(inplace=True),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(kernel_size=(2, 2)),
                nn.Dropout(p=0.5)
            ))
        self.streams.append(
            nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 41), stride=(1, 1),
                          padding=(1, 41 // 2)),
                nn.ELU(inplace=True),
                nn.BatchNorm2d(32),
                nn.MaxPool2d(kernel_size=(1, 16)),
                nn.Dropout(p=0.5),
                nn.Conv2d(in_channels=32, out_channels=64, \
                          kernel_size=(3, 3), stride=(1, 1),
                          padding=(1, 3 // 2)),
                nn.ELU(inplace=True),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(kernel_size=(2, 2)),
                nn.Dropout(p=0.5)
            ))
        self.streams.append(
            nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=32, kernel_size=(3, 51), stride=(1, 1),
                          padding=(1, 51 // 2)),
                nn.ELU(inplace=True),
                nn.BatchNorm2d(32),
                nn.MaxPool2d(kernel_size=(1, 16)),
                nn.Dropout(p=0.5),
                nn.Conv2d(in_channels=32, out_channels=64, \
                          kernel_size=(3, 3), stride=(1, 1),
                          padding=(1, 3 // 2)),
                nn.ELU(inplace=True),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(kernel_size=(2, 2)),
                nn.Dropout(p=0.5)
            ))

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)
        out_streams = []
        for idx in range(len(self.streams)):
            out_streams.append(self.streams[idx](x))
        out = torch.cat(out_streams, dim=1)
        return out


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


class mkcnn(nn.Module):
    def __init__(self, n_features, n_classes):
        super(mkcnn, self).__init__()
        if n_features == 256:
            self.dim_change = nn.Sequential(
                Conv1dWithConstraint(n_features, 16, kernel_size=1, max_norm=2, doWeightNorm=True),
                nn.BatchNorm1d(16),
                swish()
            )
            n_features = 16
        timesteps = 200
        self.part1 = MKBlock()
        new_timesteps_after_first_maxpool = timesteps // 16
        new_n_features_after_first_maxpool = n_features

        final_timesteps = new_timesteps_after_first_maxpool // 2
        final_n_features = new_n_features_after_first_maxpool // 2

        out_dim_block = 5 * 64
        self.part2 = nn.Sequential(
            nn.Conv2d(out_dim_block, 128, kernel_size=1, stride=1),
            nn.ELU(inplace=True)
        )
        self.part3 = nn.Sequential(
            nn.Conv2d(128, 128, groups=128, kernel_size=(3, 3), stride=1, padding=1),
            nn.ELU(inplace=True),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Dropout(p=0.5),)

        final_timesteps = final_timesteps // 2
        final_n_features = final_n_features // 2

        self.part4 = nn.Sequential(
            nn.Linear(128*final_timesteps*final_n_features, 512),
            nn.ELU(inplace=True),
            nn.Linear(512, 128),
            nn.ELU(inplace=True),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        if x.shape[1] == 256:
            x = self.dim_change(x)
        output = self.part1(x)
        output = self.part2(output)
        output = self.part3(output)
        output = output.view(output.shape[0], -1)
        output = self.part4(output)
        return output


def MKCNN(args, total_data):
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
    elif args.dataset == 'Ninapro':
        n_features = 12
        n_classes = 21
    elif args.dataset == 'Senic':
        n_features = 8
        n_classes = 8
    else:
        raise NotImplementedError

    total_epochs = 50
    learning_rate = 0.001

    # Define data loaders for training and testing data in this fold
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=False)
    vali_loader = DataLoader(val_dataset, batch_size=128, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, drop_last=False)

    # Init the neural network
    model = mkcnn(n_features, n_classes).float().to(device)
    criteria = nn.CrossEntropyLoss()
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    # scheduler = WarmupCosineAnnealingLR(optimizer, total_epochs, warmup_steps=5)
    # early_stopping = EarlyStopping(patience=8, min_delta=0, path=path)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.5)
    best_val_loss = float('inf')
    # Run the training loop for defined number of epochs
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
        all_preds = []
        all_targets = []
        for i, (inputs, labels) in enumerate(vali_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            loss = criteria(outputs, labels)
            val_loss.append(loss.cpu().detach().numpy())
            # early_stopping(np.mean(val_loss), model)
            # if early_stopping.early_stop:
            #     break
        val_loss = np.mean(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), path)
    # early_stopping.load_best_model(model)
    model.load_state_dict(torch.load(path))
    model.eval()
    # early_stopping.load_best_model(model)
    # model.eval()
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
    print(f'test Accuracy: {accuracy}')



if __name__ == "__main__":
    from calflops import calculate_flops

    dataset = '1'
    if dataset == 'GrabMyo':
        n_features = 16
        n_classes = 17
        timesteps = 200
        batch_size = 256
    elif dataset == 'Hyser':
        n_features = 256
        n_classes = 34
        timesteps = 200
        batch_size = 256
    else:
        n_features = 10
        n_classes = 11
        timesteps = 200
        batch_size = 256
    model = mkcnn(n_classes = n_classes, n_features = n_features, timesteps = timesteps)
    batch_size = 1
    # input_shape = (batch_size, 14, 300) # ninapro
    input_shape = (batch_size, n_features, timesteps)
    flops, macs, params = calculate_flops(model=model,
                                          input_shape=input_shape,
                                          output_as_string=True,
                                          output_precision=4)
    print("FLOPs:%s   MACs:%s   Params:%s \n" % (flops, macs, params))


