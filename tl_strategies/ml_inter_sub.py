
from numpy import random
import torch.nn.functional as F
import math
from sklearn.metrics import accuracy_score
from torch.autograd import Variable
import numpy as np
from loss import SupConLoss, scl
from sklearn.metrics import cohen_kappa_score
import torch
from torch import nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils import *
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA




class eca_layer(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y_1 = self.avg_pool(x)
        y_2 = self.max_pool(x)
        # Two different branches of ECA module
        y_1 = self.conv1(y_1.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y_2 = self.conv2(y_2.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y_1 = self.sigmoid(y_1)
        y_2 = self.sigmoid(y_2)
        return x * y_1.expand_as(x) * y_2.expand_as(x)+x


class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class GhostModule(nn.Module):
    def __init__(self, inp, oup, kernel_size=(1, 7), ratio=2, dw_size=(1, 7), stride=1, relu=True):
        super(GhostModule, self).__init__()
        self.oup = oup
        init_channels = math.ceil(oup / ratio)
        new_channels = init_channels * (ratio - 1)

        self.primary_conv = nn.Sequential(
            nn.Conv2d(inp, init_channels, kernel_size, stride, padding='same', bias=False),
            nn.BatchNorm2d(init_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

        self.cheap_operation = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, dw_size, 1, padding='same', groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            nn.ReLU(inplace=True) if relu else nn.Sequential(),
        )

    def forward(self, x):
        x1 = self.primary_conv(x)
        x2 = self.cheap_operation(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, :self.oup, :, :]


class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    # Conv2dWithConstraint
    def forward(self, x):
        if self.doWeightNorm and self.weight.requires_grad:
            with torch.no_grad():
                norm = self.weight.norm(2, dim=0, keepdim=True)
                desired = torch.clamp(norm, max=self.max_norm)
                self.weight.mul_(desired / (norm + 1e-6))
        return super().forward(x)


class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    # LinearWithConstraint
    def forward(self, x):
        if self.doWeightNorm and self.weight.requires_grad:
            with torch.no_grad():
                norm = self.weight.norm(2, dim=0, keepdim=True)
                desired = torch.clamp(norm, max=self.max_norm)
                self.weight.mul_(desired / (norm + 1e-6))

        return super().forward(x)

class swish(nn.Module):
    def __init__(self):
        super(swish, self).__init__()
    def forward(self, x):
        return x * torch.sigmoid(x)


class TCB(nn.ModuleDict):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(TCB, self).__init__()
        kernel_size = kernel_size if isinstance(kernel_size, list) else [kernel_size]
        num_groups = len(kernel_size)

        in_splits = [in_channels // num_groups for _ in range(num_groups)]
        in_splits[0] += in_channels - sum(in_splits)

        out_splits = [out_channels // num_groups for _ in range(num_groups)]
        out_splits[0] += out_channels - sum(out_splits)

        self.in_channels = sum(in_splits)
        self.out_channels = sum(out_splits)

        self.convs = nn.ModuleList()  # 用于存储卷积层
        self.atts = nn.ModuleList()  # 用于存储卷积层
        for idx, (k, in_ch, out_ch) in enumerate(zip(kernel_size, in_splits, out_splits)):
            self.convs.append(GhostModule(in_ch, out_ch, kernel_size=k, dw_size=k))
            self.atts.append(eca_layer())
            # self.atts.append(SCSA(out_ch))
            # SCSA
            # nn.Conv2d(in_ch, out_ch, k, padding='same', bias=False)

        self.splits = in_splits

    def forward(self, x):
        x_split = torch.split(x, self.splits, 1)
        x_out = []
        for i, (conv, att) in enumerate(zip(self.convs, self.atts)):
            x_out.append(att(conv(x_split[i])))
        x = torch.cat(x_out, 1)
        return x


class hgrnet(nn.Module):
    def __init__(self, nChan, nClass, num_feat=16, dilatability=16):
        # input_size: channel x datapoint
        super(hgrnet, self).__init__()
        if nChan == 256:
            self.dim_change = nn.Conv1d(nChan, 16, 1, bias=False)
            nChan = 16
        # self.sinc_conv = sinc_conv(N_filt=16, Filt_dim=31, fs=1000, device=self.device)
        self.first_cov = nn.Conv2d(1, num_feat, 1, bias=False)
        self.TCB = nn.Sequential(
            TCB(in_channels=num_feat, out_channels=num_feat*2, kernel_size=[(1,7), (1,14), (1,21), (1,28)]),
            nn.BatchNorm2d(num_feat*2),
            nn.GELU()
        )
        self.SCB = nn.Sequential(
            Conv2dWithConstraint(num_feat*2, num_feat*dilatability, (nChan, 1), groups=num_feat,
                                 max_norm=2, doWeightNorm=True, padding=0),
            eca_layer(),
            nn.BatchNorm2d(num_feat*dilatability),
            nn.GELU()
        )
        self.avg_pool = nn.AvgPool2d((1,50), (1,10))

        self.fc = nn.Sequential(
            LinearWithConstraint(4096, nClass, max_norm=0.5, doWeightNorm=True),
        )
        self.proj = nn.Sequential(
            LinearWithConstraint(4096, 128, max_norm=0.5, doWeightNorm=True),
        )

    def forward(self, x):
        # x = x.to(torch.float32)
        if x.shape[1] == 256:
            x = self.dim_change(x)
        x = torch.unsqueeze(x, dim=1)
        x = self.first_cov(x)
        x = self.TCB(x)
        x = self.SCB(x)
        x = self.avg_pool(x)
        f = torch.flatten(x, start_dim=1)
        c = self.fc(f)
        p = self.proj(f)
        return c, f, F.normalize(p, dim=1)


def ml_inter_sub(args, total_data):
    path = 'checkpoints/intra-subject/'+args.dataset+'/'\
           +args.backbone+'/P'+str(args.source_parti)+'/best_model.pth'
    device = args.device
    data_train, data_test = total_data[0], total_data[1]
    emg_train = data_train['emg']
    label_train = data_train['label']
    emg_test = data_test['emg']
    label_test = data_test['label']
    train_dataset = MyDataset(emg_train, label_train)
    test_dataset = MyDataset(emg_test, label_test)

    if args.dataset == 'Hyser':
        n_features = 256
        n_classes = 34
        batch_size = 128
        # learning_rate = 0.001
    elif args.dataset == 'Ninapro':
        n_features = 12
        n_classes = 21
        batch_size = 128
        # learning_rate = 0.001
    elif args.dataset == 'Senic':
        n_features = 8
        n_classes = 8
        batch_size = 128
        # learning_rate = 0.001
    else:
        raise NotImplementedError

    # Define data loaders for training and testing data in this fold
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # Init the neural network
    model = hgrnet(nChan=n_features, nClass=n_classes).float().to(device)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()

    cat_train_list = []
    adapt_label_list = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        out, f, feature = model(data)
        cat_train_list.extend(f.detach().cpu().numpy())
        adapt_label_list.extend(target.detach().cpu().numpy())
    ########################################
    cat_test_list = []
    target_label_list = []
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        out, f, feature = model(data)
        cat_test_list.extend(f.detach().cpu().numpy())
        target_label_list.extend(target.detach().cpu().numpy())

    train_y = np.array(adapt_label_list).ravel()
    test_y = np.array(target_label_list).ravel()

    cat_train_list = np.array(cat_train_list)
    cat_test_list = np.array(cat_test_list)

    # pca = PCA(n_components=5)
    # cat_train_list = pca.fit_transform(cat_train_list)
    # cat_test_list = pca.transform(cat_test_list)

    KNN = KNeighborsClassifier()
    KNN.fit(cat_train_list, train_y)
    all_preds = KNN.predict(cat_test_list)
    accuracy1 = accuracy_score(test_y, all_preds)

    SVM = SVC()
    SVM.fit(cat_train_list, train_y)
    all_preds = SVM.predict(cat_test_list)
    accuracy2 = accuracy_score(test_y, all_preds)

    RF = RandomForestClassifier()
    RF.fit(cat_train_list, train_y)
    all_preds = RF.predict(cat_test_list)
    accuracy3 = accuracy_score(test_y, all_preds)
    print(accuracy1, accuracy2, accuracy3)

