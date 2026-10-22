"""
Zabihi S, Rahimian E, Asif A, et al. Trahgr: Transformer for hand gesture recognition via
electromyography[J]. IEEE Transactions on Neural Systems and Rehabilitation Engineering, 2023.
"""

# Torch
import torch
import torch.nn as nn
from einops import rearrange, repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import *
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, Subset
from fvcore.nn import FlopCountAnalysis
import numpy as np
from sklearn.metrics import cohen_kappa_score
from torch.optim.lr_scheduler import StepLR
#

class MyDataset(Dataset):
    def __init__(self, data, labels):
        self.data = torch.tensor(data, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=4, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads

        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


class ViT(nn.Module):
    # def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
    def __init__(self, num_patch, patch_size, P=8, dim=32, depth=3, heads=4, mlp_dim=128, dropout=0.2, emb_dropout=0.,
                 pool='cls', number_gesture=49, class_rest=False):
        super().__init__()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.P = P
        output_class = number_gesture + int(class_rest == True)

        self.to_patch_embedding = nn.Sequential(
            nn.Linear(int(patch_size / self.P), 4 * dim),
            nn.ReLU(),
            nn.Linear(4 * dim, dim),
        )

        # self.to_patch_embedding = nn.Linear(patch_size, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, int(num_patch * P) + 1, dim))
        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patch + 1, dim))

        self.dropout = nn.Dropout(emb_dropout)

        dim_head = int(dim / heads)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, output_class)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # shape: (B, W, C) -> shape: (B, C, W)
        batch_size, num_patch, patch_size = x.shape

        x = x.reshape(batch_size, -1, int(patch_size / self.P))
        x = self.to_patch_embedding(x)

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=batch_size)
        x = torch.cat((cls_tokens, x), dim=1)

        x += self.pos_embedding
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)
        return self.mlp_head(x)


class ViT_TNet(nn.Module):
    def __init__(self, window_size, num_channel, F=5, P=1, dim=144, depth=1, heads=8, mlp_dim=720, dropout=0.4,
                 emb_dropout=0.4, pool='cls', number_gesture=49, class_rest=False):
        super().__init__()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.conv_max_pool_factor = 4

        self.F = F  # F = output channels / input channels, in conv1d => like "F"iltering along time axis
        self.P = P  # P = num_patch / number_channel = patch_size * window_size, a scale factor to reshape the patch embedding
        self.num_channel = num_channel
        self.window_size = window_size
        self.patch_size = window_size // (P * self.conv_max_pool_factor)
        self.num_patch = num_channel * P * self.F

        output_class = number_gesture + int(class_rest == True)

        self.conv = nn.Sequential(
            nn.BatchNorm1d(num_channel),
            nn.Conv1d(num_channel, num_channel * F, kernel_size=9, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(self.conv_max_pool_factor),
        )

        self.to_patch_embedding = nn.Sequential(
            nn.Linear(self.patch_size, dim)
            # nn.Linear(self.patch_size, 4*dim),
            # nn.ReLU(),
            # nn.Linear(4*dim, dim),
        )

        # self.to_patch_embedding = nn.Linear(patch_size, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patch + 1, dim))
        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patch + 1, dim))

        self.dropout = nn.Dropout(emb_dropout)

        dim_head = int(dim / heads)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, output_class)

    def get_class_emb(self, x):
        batch_size = x.shape[0]
        x = self.conv(x)  # shape: (batch_size, F*num_channel, window_size/self.conv_max_pool_factor) = (B, F*C, W/self.conv_max_pool_factor)

        x = x.reshape(batch_size, -1, self.patch_size)
        x = self.to_patch_embedding(x)

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=batch_size)
        x = torch.cat((cls_tokens, x), dim=1)

        x += self.pos_embedding
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        return x

    def forward(self, x):
        x = self.get_class_emb(x)
        x = self.to_latent(x)

        return self.mlp_head(x)


class ViT_FNet(nn.Module):
    def __init__(self, window_size, num_channel, F=5, P=3, Q=3, dim=144, depth=1, heads=8, mlp_dim=720, dropout=0.4,
                 emb_dropout=0.4, pool='cls', number_gesture=49, class_rest=False):
        super().__init__()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.conv_max_pool_factor = 4

        self.F = F  # F = output channels / input channels, in conv1d => like "F"iltering along time axis
        self.P = P  # P = num_patch / number_channel = patch_size * window_size, a scale factor to reshape the patch embedding
        self.Q = Q
        self.num_channel = num_channel
        self.window_size = window_size
        self.patch_size = self.P * self.Q
        self.num_patch = (self.num_channel * self.F // self.P) * (
                    self.window_size // (self.conv_max_pool_factor * self.Q))

        output_class = number_gesture + int(class_rest == True)

        self.conv = nn.Sequential(
            nn.BatchNorm1d(num_channel),
            nn.Conv1d(num_channel, num_channel * F, kernel_size=9, padding='same'),
            nn.ReLU(),
            nn.MaxPool1d(self.conv_max_pool_factor),
        )

        self.to_patch_embedding = nn.Sequential(
            nn.Linear(self.patch_size, dim)
            # nn.Linear(self.patch_size, 4*dim),
            # nn.ReLU(),
            # nn.Linear(4*dim, dim),
        )

        # self.to_patch_embedding = nn.Linear(patch_size, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patch + 1, dim))
        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patch + 1, dim))

        self.dropout = nn.Dropout(emb_dropout)

        dim_head = int(dim / heads)
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, output_class)

    def get_class_emb(self, x):
        # input shape: (batch_size, window_size, num_channel) = (B, W, C)
        batch_size = x.shape[0]
        x = self.conv(x)  # shape: (batch_size, F*num_channel, window_size/self.conv_max_pool_factor) = (B, F*C, W/self.conv_max_pool_factor)

        # x = x.reshape(batch_size, -1, self.patch_size)
        x = rearrange(x, 'b (c p) (w q) -> b (c w) (p q)', p=self.P, q=self.Q)
        x = self.to_patch_embedding(x)

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=batch_size)
        x = torch.cat((cls_tokens, x), dim=1)

        x += self.pos_embedding
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        return x

    def forward(self, x):
        x = self.get_class_emb(x)
        x = self.to_latent(x)

        return self.mlp_head(x)


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


class ViT_TraHGR(nn.Module):
    def __init__(self, window_size, num_channel, F=5, Pt=1, Pf=5, Qf=1, dim=144, depth=1, heads=8, mlp_dim=720,
                 dropout=0.2, emb_dropout=0.2, pool='cls', number_gesture=49, class_rest=False):
        super().__init__()
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'
        if num_channel == 256:
            self.dim_change = nn.Conv1d(num_channel, 16, 1, bias=False)
            num_channel = 16

        output_class = number_gesture + int(class_rest == True)

        self.ViT_TNet = ViT_TNet(window_size, num_channel, F=F, P=Pt, dim=dim, depth=depth, heads=heads,
                                 mlp_dim=mlp_dim, dropout=dropout, emb_dropout=emb_dropout, pool=pool,
                                 number_gesture=number_gesture, class_rest=class_rest)
        self.ViT_FNet = ViT_FNet(window_size, num_channel, F=F, P=Pf, Q=Qf, dim=dim, depth=depth, heads=heads,
                                 mlp_dim=mlp_dim, dropout=dropout, emb_dropout=emb_dropout, pool=pool,
                                 number_gesture=number_gesture, class_rest=class_rest)

        self.mlp_head_TNet = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, output_class)
        )

        self.mlp_head_FNet = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, output_class)
        )

        self.mlp_head_TraHGR = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, output_class)
        )

        # self.input_conv = torch.nn.Conv1d(256, 16, kernel_size=1)

    def forward(self, x):
        if x.shape[1] == 256:
            x = self.dim_change(x)
        z_TNet = self.ViT_TNet.get_class_emb(x)  # (num_batch, dim, 1)
        z_FNet = self.ViT_FNet.get_class_emb(x)  # (num_batch, dim, 1)
        return self.mlp_head_TNet(z_TNet), self.mlp_head_FNet(z_FNet), self.mlp_head_TraHGR(z_TNet + z_FNet)


def TraHGR(args, total_data):
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
        batch_size = 64
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
    timesteps = 200
    total_epochs = 50

    # Define data loaders for training and testing data in this fold
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    vali_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    # Init the neural network
    model = ViT_TraHGR(timesteps, n_features, number_gesture=n_classes).to(device)
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
            gesture_pred_TNet, gesture_pred_FNet, outputs = model(emg)
            loss_TNet   = criteria(gesture_pred_TNet,label)
            loss_FNet   = criteria(gesture_pred_FNet,label)
            loss_TraHGR = criteria(outputs,label)
            loss = loss_TNet + loss_FNet + loss_TraHGR
            loss.backward()
            optimizer.step()

        scheduler.step()
        # Validation
        model.eval()
        val_loss = []
        for i, (inputs, labels) in enumerate(vali_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            gesture_pred_TNet, gesture_pred_FNet, outputs = model(inputs)
            loss_TNet   = criteria(gesture_pred_TNet,labels)
            loss_FNet   = criteria(gesture_pred_FNet,labels)
            loss_TraHGR = criteria(outputs, labels)
            loss = loss_TNet + loss_FNet + loss_TraHGR
            val_loss.append(loss.cpu().detach().numpy())

        val_loss = np.mean(val_loss)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), path)
    # early_stopping.load_best_model(model)
    model.load_state_dict(torch.load(path))
    model.eval()
    from sklearn.metrics import accuracy_score, precision_score, recall_score
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            _, _, outputs = model(inputs)
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
    model = ViT_TraHGR(200, n_features, number_gesture=n_classes)
    batch_size = 1
    # input_shape = (batch_size, 14, 300) # ninapro
    input_shape = (batch_size, n_features, timesteps)
    flops, macs, params = calculate_flops(model=model,
                                          input_shape=input_shape,
                                          output_as_string=True,
                                          output_precision=4)
    print("FLOPs:%s   MACs:%s   Params:%s \n" % (flops, macs, params))













