# 把深度回归问题 → 深度分类问题
# 将深度划分为 K 个 bin
# MVS 和 DA3 各自对 bin 有 soft prior
# 融合网络输出一个 新的 bin 分布
# 用 one-hot / soft-label loss
# 👉 这是 CVPR / ICCV 深度估计的主流方向
# 因为DA3的conf不是真实的cof，DA3 / MVS 的 conf 只影响 logits

import torch
import torch.nn as nn
import torch.nn.functional as F

class DepthBinFusionNet(nn.Module):
    """
    Depth bin fusion with confidence-aware conditioning
    """
    def __init__(self, num_bins=64, hidden=64):
        super().__init__()
        self.num_bins = num_bins

        self.encoder = nn.Sequential(
            nn.Conv2d(4, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.bin_logits = nn.Conv2d(hidden, num_bins, 1)

    def forward(self, D_mvs, C_mvs, D_da, C_da):
        """
        conf is treated as raw feature
        """
        x = torch.cat([D_mvs, D_da, C_mvs, C_da], dim=1)
        feat = self.encoder(x)

        logits = self.bin_logits(feat)
        prob = F.softmax(logits, dim=1)

        # depth confidence from entropy
        conf = -torch.sum(prob * torch.log(prob + 1e-8), dim=1, keepdim=True)

        return prob, conf


def depth_bin_loss(prob, gt_depth, depth_bins):
    """
    prob: (B,K,H,W)
    """
    B, K, H, W = prob.shape

    # hard assignment
    gt_bin = torch.argmin(
        torch.abs(gt_depth - depth_bins.view(1, K, 1, 1)),
        dim=1
    )

    loss_ce = F.cross_entropy(prob, gt_bin)

    # EMD regularization
    cdf_pred = torch.cumsum(prob, dim=1)
    gt_onehot = F.one_hot(gt_bin, K).permute(0, 3, 1, 2).float()
    cdf_gt = torch.cumsum(gt_onehot, dim=1)

    loss_emd = torch.mean(torch.abs(cdf_pred - cdf_gt))

    return loss_ce + 0.1 * loss_emd
