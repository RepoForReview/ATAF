# J. Zeng, Y. Zhou, Y. Yang, J. Wang, and H. Liu, “Feature fusion of sEMG
# and ultrasound signals in hand gesture recognition,” in Proc. IEEE Int.
# Conf. Syst., Man, Cybern. (SMC), Oct. 2020, pp. 3911–3916.


import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score
from utils import *
from fvcore.nn import FlopCountAnalysis
from sklearn.metrics import cohen_kappa_score
from torch.optim.lr_scheduler import StepLR



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


class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class eunet(nn.Module):
    def __init__(self, n_features, n_classes, timesteps):
        super().__init__()
        if n_features == 256:
            self.dim_change = nn.Sequential(
                Conv1dWithConstraint(n_features, 16, kernel_size=1, max_norm=2, doWeightNorm=True),
                nn.BatchNorm1d(16),
                swish()
            )
            n_features = 16
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3, 20), stride=(1, 1), padding=(1, 10))
        self.relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=(1, 20))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size=(1, 2))
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=0)
        self.relu3 = nn.ReLU(inplace=True)
        self.dropout1 = nn.Dropout(p=0.5)
        new_timesteps_1 = timesteps
        new_timesteps_2 = new_timesteps_1 // 20
        new_timesteps_3 = new_timesteps_2
        new_timesteps_4 = new_timesteps_3 // 2
        new_n_features_5 = n_features - 2
        new_timesteps_5 = new_timesteps_4 - 2
        flattened_size = 128 * new_n_features_5 * new_timesteps_5
        self.fc1 = nn.Linear(flattened_size, 128)
        self.relu4 = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        if x.shape[1] == 256:
            x = self.dim_change(x)
        x = torch.unsqueeze(x, dim=1)
        x = self.relu1(self.conv1(x))
        x = self.maxpool1(x)
        x = self.relu2(self.conv2(x))
        x = self.maxpool2(x)
        x = self.relu3(self.conv3(x))
        x = x.view(x.shape[0],-1)
        x = self.dropout1(x)
        x = self.relu4(self.fc1(x))
        x = self.dropout2(x)

        out = self.fc2(x)
        return out


def EUNet(args, total_data):
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
    timesteps = 200

    # Define data loaders for training and testing data in this fold
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    vali_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # Init the neural network
    model = eunet(n_features, n_classes, timesteps).float().to(device)
    criteria = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    # scheduler = WarmupCosineAnnealingLR(optimizer, total_epochs, warmup_steps=5)
    # early_stopping = EarlyStopping(patience=8, min_delta=0, path=path)
    scheduler = StepLR(optimizer, step_size=20, gamma=0.8)
    best_val_loss = float('inf')
    for epoch in range(0, total_epochs):
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
    print(f'Accuracy: {accuracy}')


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
    model = eunet(n_features, n_classes, timesteps).float()
    batch_size = 1
    # input_shape = (batch_size, 14, 300) # ninapro
    input_shape = (batch_size, n_features, timesteps)
    flops, macs, params = calculate_flops(model=model,
                                          input_shape=input_shape,
                                          output_as_string=True,
                                          output_precision=4)
    print("FLOPs:%s   MACs:%s   Params:%s \n" % (flops, macs, params))
