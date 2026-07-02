import torch
import torch.nn as nn
from numpy import random
import numpy as np
from torch.autograd.function import Function


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(self, device, temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.device = device

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf
        Args:
        features: hidden vector of shape [bsz, n_views, ...].
        labels: ground truth of shape [bsz].
        mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
        has the same class as sample i. Can be asymmetric.
        Returns:
        A loss scalar.
        """
        device = self.device

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...], at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)

        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')

            mask = torch.eq(labels, labels.T).float().to(device)

        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]  # number of positives per sample

        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        anchor_feature = contrast_feature
        anchor_count = contrast_count

        # compute logits - calculates the dot product of every two vectors divided by temperature
        anchor_dot_contrast = torch.div(torch.matmul(anchor_feature, contrast_feature.T), self.temperature)

        # for numerical stability  (some kind of normalization!)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask as much as number of positives per sample
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(torch.ones_like(mask), 1,
                                    torch.arange(batch_size * anchor_count).view(-1, 1).to(device), 0)

        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        eps = 1e-30
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + eps)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1) + eps)
        # loss
        loss = -  mean_log_prob_pos

        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, size_average=True):
        super(CenterLoss, self).__init__()
        centers = random.randn(num_classes, feat_dim).astype('float32')
        self.centers = nn.Parameter(torch.from_numpy(centers))
        self.centerlossfunc = CenterlossFunc.apply
        self.feat_dim = feat_dim
        self.size_average = size_average

    def forward(self, label, feat):
        batch_size = feat.size(0)
        feat = feat.view(batch_size, -1)
        # To check the dim of centers and features
        if feat.size(1) != self.feat_dim:
            raise ValueError("Center's dim: {0} should be equal to input feature's \
                            dim: {1}".format(self.feat_dim,feat.size(1)))
        batch_size_tensor = feat.new_empty(1).fill_(batch_size if self.size_average else 1)
        loss = self.centerlossfunc(feat, label, self.centers, batch_size_tensor)
        return loss


class CenterlossFunc(Function):
    @staticmethod
    def forward(ctx, feature, label, centers, batch_size):
        ctx.save_for_backward(feature, label, centers, batch_size)
        centers_batch = centers.index_select(0, label.long())
        return (feature - centers_batch).pow(2).sum() / 2.0 / batch_size

    @staticmethod
    def backward(ctx, grad_output):
        feature, label, centers, batch_size = ctx.saved_tensors
        centers_batch = centers.index_select(0, label.long())
        diff = centers_batch - feature
        # init every iteration
        counts = centers.new_ones(centers.size(0))
        ones = centers.new_ones(label.size(0))
        grad_centers = centers.new_zeros(centers.size())

        counts = counts.scatter_add_(0, label.long(), ones)
        grad_centers.scatter_add_(0, label.unsqueeze(1).expand(feature.size()).long(), diff)
        grad_centers = grad_centers/counts.view(-1, 1)
        return - grad_output * diff / batch_size, None, grad_centers / batch_size, None


def scl(feature, label):
    lam = np.random.uniform(0.9, 1.0)
    sorted_y, indices = torch.sort(label)
    sorted_proj = torch.zeros_like(feature)
    for idx, order in enumerate(indices):
        sorted_proj[idx] = feature[order]
    intervals = []
    ex = 0
    for idx, val in enumerate(sorted_y):
        if ex == val:
            continue
        intervals.append(idx)
        ex = val
    batch_size = label.size()[0]
    intervals.append(batch_size)

    proj = sorted_proj
    y_scl = sorted_y

    # shuffle
    mix1 = torch.zeros_like(proj)
    mix2 = torch.zeros_like(proj)
    ex = 0
    for end in intervals:
        shuffle_indices = torch.randperm(end - ex) + ex
        shuffle_indices2 = torch.randperm(end - ex) + ex
        for idx in range(end - ex):
            mix1[idx + ex] = proj[shuffle_indices[idx]]
            mix2[idx + ex] = proj[shuffle_indices2[idx]]
        ex = end
    p1 = lam * proj + (1 - lam) * mix1
    p2 = lam * proj + (1 - lam) * mix2
    p = torch.cat([p1.unsqueeze(1), p2.unsqueeze(1)], dim=1)
    return p, y_scl