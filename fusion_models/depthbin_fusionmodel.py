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
    prob: (B, K, H, W)
    gt_depth: (B, H, W)
    depth_bins: (B, K) or (1, K)
    """
    B, K, H, W = prob.shape
    # gt_depth: (B, H, W) -> (B, 1, H, W)
    # depth_bins: (B, K) -> (B, K, 1, 1)
    # 这样广播后得到: (B, K, H, W)
    gt_depth_expanded = gt_depth.unsqueeze(1)  # (B, 1, H, W)
    depth_bins_expanded = depth_bins.view(B, K, 1, 1)  # (B, K, 1, 1)
    
    # hard assignment
    # 计算每个像素点与每个深度区间的真实距离
    dist = torch.abs(gt_depth_expanded - depth_bins_expanded)  # (B, K, H, W)
    # 距离到某个bin距离最小说明这个bin是真实bin
    gt_bin = torch.argmin(dist, dim=1)  # (B, H, W)
    # 那么这里可以计算一个prob的猜测对不对了，可以把gtbing看作标签，计算出prob的取bin的交叉熵，就知道有没有计算对位置
    loss_ce = F.cross_entropy(prob, gt_bin)

    # EMD regularization
    # 在 bin 维度（dim=1）做累积求和
    cdf_pred = torch.cumsum(prob, dim=1)
    # 把(2,296,400)的 bin 编号转成独热编码，维度变成(2,296,400,64)（比如 bin=49 的位置是 1，其余是 0）
    gt_onehot = F.one_hot(gt_bin, K).permute(0, 3, 1, 2).float()
    # 对真实独热分布做累积求和，比如真实 bin 是 49，则cdf_gt中前 49 个 bin 的累积和是 0，第 49 个及之后是 1
    cdf_gt = torch.cumsum(gt_onehot, dim=1)
    # 计算预测累积分布和真实累积分布的绝对差，再求所有元素的平均值；这相当于计算两个分布之间的地球移动距离（EMD）
    loss_emd = torch.mean(torch.abs(cdf_pred - cdf_gt))

    return loss_ce + 0.1 * loss_emd


def depth_bin_loss(prob, gt_depth, depth_bins, gt_mask):
    """
    修正版：加入gt_mask过滤无效像素，仅计算有效区域的损失
    Args:
        prob: (B, K, H, W) - 模型预测的bin概率
        gt_depth: (B, H, W) - 真实深度图（含无效值，如0/NaN）
        depth_bins: (B, K) or (1, K) - 深度bin的数值
        gt_mask: (B, H, W) - 有效像素掩码，1=有效，0=无效（float/bool类型均可）
    Returns:
        total_loss: 仅有效像素的损失值
    """
    B, K, H, W = prob.shape
    
    # 扩展维度用于广播计算
    gt_depth_expanded = gt_depth.unsqueeze(1)  # (B, 1, H, W)
    depth_bins_expanded = depth_bins.view(B, K, 1, 1)  # (B, K, 1, 1)
    
    # -------------------------- 2. 计算真实bin（和原逻辑一致） --------------------------
    dist = torch.abs(gt_depth_expanded - depth_bins_expanded)  # (B, K, H, W)
    gt_bin = torch.argmin(dist, dim=1)  # (B, H, W)
    
    # -------------------------- 3. 核心修改：应用mask过滤无效像素 --------------------------
    # 统一mask类型：转为float，确保维度是(B, H, W)，1=有效，0=无效
    if gt_mask.dtype == torch.bool:
        gt_mask = gt_mask.float()
    # 扩展mask维度，匹配prob的维度（B, 1, H, W），方便后续广播
    gt_mask_expanded = gt_mask.unsqueeze(1)  # (B, 1, H, W)
    
    # -------------------------- 4. 计算带mask的交叉熵损失 --------------------------
    # 方法：先计算所有像素的交叉熵，再用mask加权平均（仅有效像素参与）
    # 步骤1：计算每个像素的交叉熵（不做mean）
    # F.cross_entropy的reduction='none'表示返回每个像素的损失值，维度(B, H, W)
    loss_ce_per_pixel = F.cross_entropy(prob, gt_bin, reduction='none')  # (B, H, W)
    # 步骤2：用mask过滤无效像素（无效像素损失置0）
    loss_ce_masked = loss_ce_per_pixel * gt_mask  # (B, H, W)
    # 步骤3：计算有效像素的平均损失（避免除以0）
    valid_pixel_num = torch.clamp(gt_mask.sum(), min=1)  # 有效像素数，最小为1防止除以0
    loss_ce = loss_ce_masked.sum() / valid_pixel_num  # 标量
    
    # -------------------------- 5. 计算带mask的EMD损失 --------------------------
    # 步骤1：计算累积分布（和原逻辑一致）
    cdf_pred = torch.cumsum(prob, dim=1)  # (B, K, H, W)
    gt_onehot = F.one_hot(gt_bin, K).permute(0, 3, 1, 2).float()  # (B, K, H, W)
    cdf_gt = torch.cumsum(gt_onehot, dim=1)  # (B, K, H, W)
    
    # 步骤2：计算每个bin、每个像素的EMD绝对差
    emd_per_pixel_per_bin = torch.abs(cdf_pred - cdf_gt)  # (B, K, H, W)
    # 步骤3：用mask过滤无效像素（扩展后的mask广播到K个bin维度）
    emd_masked = emd_per_pixel_per_bin * gt_mask_expanded  # (B, K, H, W)
    # 步骤4：计算有效像素的平均EMD损失
    loss_emd = emd_masked.sum() / (valid_pixel_num * K)  # 标量（除以有效像素数×bin数）
    
    # -------------------------- 6. 总损失 --------------------------
    total_loss = loss_ce + 0.1 * loss_emd
    
    return total_loss
