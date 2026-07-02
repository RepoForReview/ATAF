import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from loss import SupConLoss, scl
from torch import nn
from torch.utils.data import Dataset, DataLoader
from utils import *
from torch import Tensor


class eca_layer(nn.Module):
    def __init__(self, k_size=3):
        super(eca_layer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.conv1 = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.conv2 = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y_1 = self.avg_pool(x)
        y_2 = self.max_pool(x)
        y_1 = self.conv1(y_1.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y_2 = self.conv2(y_2.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
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

    def forward(self, x):
        if self.doWeightNorm:
            norm = self.weight.data.norm(2, dim=0, keepdim=True)
            desired = torch.clamp(norm, max=self.max_norm)
            self.weight.data = self.weight.data * (desired / (norm + 1e-6))
        return super(Conv2dWithConstraint, self).forward(x)


class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            norm = self.weight.data.norm(2, dim=0, keepdim=True)
            desired = torch.clamp(norm, max=self.max_norm)
            self.weight.data = self.weight.data * (desired / (norm + 1e-6))
        return super(LinearWithConstraint, self).forward(x)


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

        self.convs = nn.ModuleList()
        self.atts = nn.ModuleList()
        for idx, (k, in_ch, out_ch) in enumerate(zip(kernel_size, in_splits, out_splits)):
            self.convs.append(GhostModule(in_ch, out_ch, kernel_size=k, dw_size=k))
            self.atts.append(eca_layer())

        self.splits = in_splits

    def forward(self, x):
        x_split = torch.split(x, self.splits, 1)
        x_out = []
        for i, (conv, att) in enumerate(zip(self.convs, self.atts)):
            x_out.append(att(conv(x_split[i])))
        x = torch.cat(x_out, 1)
        return x


def channel_shuffle(x: Tensor, groups: int) -> Tensor:
    batch_size, num_channels, height, width = x.size()
    channels_per_group = num_channels // groups
    x = x.view(batch_size, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()
    x = x.view(batch_size, -1, height, width)
    return x


class GCAdapter(nn.Module):
    def __init__(self, dim=256, num_blocks=2):
        super().__init__()
        self.lam = nn.Parameter(torch.tensor(1.0), requires_grad=True)
        self.num_blocks = num_blocks

        self.gconvs = nn.ModuleList()
        self.atts = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.acts = nn.ModuleList()

        for _ in range(num_blocks):
            self.gconvs.append(nn.Conv2d(dim, dim, groups=4, kernel_size=1, stride=1, padding=0, bias=False))
            self.atts.append(eca_layer())
            self.bns.append(nn.BatchNorm2d(dim))
            self.acts.append(nn.GELU())

    def forward(self, x):
        out = x
        for i in range(self.num_blocks):
            out = self.gconvs[i](out)
            out = channel_shuffle(out, 4)
            out = self.atts[i](out)
            out = self.bns[i](out)
            out = self.acts[i](out)
        return out * self.lam + x


class hgrnet(nn.Module):
    def __init__(self, nChan, nClass, num_block, num_feat=16, dilatability=16):
        super(hgrnet, self).__init__()
        if nChan == 256:
            self.dim_change = nn.Conv1d(nChan, 16, 1, bias=False)
            nChan = 16

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

        self.adapter_a = GCAdapter(dim=256, num_blocks=num_block)

        self.fc = nn.Sequential(
            LinearWithConstraint(4096, nClass, max_norm=0.5, doWeightNorm=True),
        )
        self.proj = nn.Sequential(
            LinearWithConstraint(4096, 128, max_norm=0.5, doWeightNorm=True),
        )
        # Freeze all parameters
        for param in self.parameters():
            param.requires_grad = False
        for param in self.adapter_a.parameters():
            param.requires_grad = True


    def forward(self, x):
        # x = x.to(torch.float32)
        if x.shape[1] == 256:
            x = self.dim_change(x)
        x = torch.unsqueeze(x, dim=1)
        x = self.first_cov(x)
        x = self.TCB(x)
        x = self.SCB(x)
        x = self.avg_pool(x)
        x = self.adapter_a(x)

        f = torch.flatten(x, start_dim=1)
        c = self.fc(f)
        p = self.proj(f)
        return c, F.normalize(p, dim=1)


def gdadapter_inter_sub(args, total_data):
    path = 'checkpoints/intra-subject/'+args.dataset+args.backbone+'/P'+str(args.source_parti)+'/best_model.pth'
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
    elif args.dataset == 'Ninapro':
        n_features = 12
        n_classes = 21
        batch_size = 128
    elif args.dataset == 'Senic':
        n_features = 8
        n_classes = 8
        batch_size = 128
    else:
        raise NotImplementedError

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    total_epochs = 50
    learning_rate = 0.01

    model = hgrnet(nChan=n_features, nClass=n_classes, num_block=args.num_block).float().to(device)

    pretrained_dict = torch.load(path, map_location=device)
    model_dict = model.state_dict()
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and 'adapter' not in k}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    criteria = nn.CrossEntropyLoss()
    contrastive_loss = SupConLoss(device).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    from torch.optim.lr_scheduler import StepLR
    scheduler = StepLR(optimizer, step_size=20, gamma=0.8)

    for epoch in range(0, total_epochs):
        model.train()
        for batch_idx, (emg, label) in enumerate(train_loader):
            emg, label = emg.to(device), label.to(device)
            optimizer.zero_grad()
            output, feature = model(emg)
            cross_loss = criteria(output, label)
            p, y_scl = scl(feature, label)
            scl_loss = contrastive_loss(p, y_scl, mask=None)
            loss = cross_loss + scl_loss
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    all_preds = []
    all_targets = []
    feature_list = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            output, feature = model(inputs)
            _, preds = torch.max(output.data, 1)
            all_preds.extend(preds.detach().cpu().numpy())
            all_targets.extend(labels.detach().cpu().numpy())
            feature_list.extend(feature.cpu().numpy())
    accuracy = accuracy_score(all_targets, all_preds)
    print(f'Accuracy: {accuracy}')


if __name__ == "__main__":
    from calflops import calculate_flops
    dataset = 'Hyser'
    if dataset == 'Hyser':
        n_features = 256
        n_classes = 34
        batch_size = 128
    elif dataset == 'Ninapro':
        n_features = 12
        n_classes = 21
        batch_size = 128
    elif dataset == 'Senic':
        n_features = 8
        n_classes = 8
        batch_size = 128
    else:
        raise NotImplementedError

    model = hgrnet(nChan=n_features, nClass=n_classes, num_block=2).float()
    batch_size = 1
    input_shape = (batch_size, n_features, 200)
    flops, macs, params = calculate_flops(model=model,
                                          input_shape=input_shape,
                                          output_as_string=True,
                                          output_precision=4)
    print("FLOPs:%s   MACs:%s   Params:%s \n" % (flops, macs, params))


