# 以 MVS 深度为 anchor

# 网络只预测一个 残差 ΔD

# 同时预测一个 新的置信度图 Ĉ

# 利用 DA3 & MVS 的置信度作为 不确定性先验
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualFusionNet(nn.Module):
    """
    Residual-based fusion with learned uncertainty handling
    """
    def __init__(self, hidden=48):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(4, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.ReLU(inplace=True),

            nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden),
            nn.ReLU(inplace=True),
        )

        self.depth_head = nn.Conv2d(hidden, 1, 3, padding=1)
        self.conf_head  = nn.Conv2d(hidden, 1, 3, padding=1)

    def forward(self, D_mvs, C_mvs, D_da, C_da):
        """
        所有 conf 都作为 raw feature 输入
        """
        x = torch.cat([D_mvs, D_da, C_mvs, C_da], dim=1)
        feat = self.encoder(x)

        delta_depth = self.depth_head(feat)
        fused_depth = D_mvs + delta_depth

        # 新预测的置信度（正值，不做任何先验假设）
        fused_conf = F.softplus(self.conf_head(feat))

        return fused_depth, fused_conf



def fusion_loss_residual(
    D_fused,
    C_fused,
    D_gt=None,
    D_mvs=None,
    C_mvs=None,  # 新增：MVS光度一致性置信度
    gt_mask=None,
    lambda_cons_base=0.1,  # MVS一致性基础权重
    lambda_uncert=1.0,     # 不确定性正则项权重
    cons_weight_type='exp' # MVS置信度加权方式：'exp'/'linear'/'sigmoid'
):
    """
    最终优化版：
    1. 加入gt_mask过滤无效像素
    2. 基于MVS置信度（C_mvs）动态加权一致性损失：高C_mvs区域更贴近MVS，低C_mvs区域允许修正
    3. 保留异方差回归的不确定性学习
    
    Args:
        D_fused: (B,1,H,W) or (B,H,W) - 融合深度
        C_fused: (B,1,H,W) or (B,H,W) - 融合置信度（learned uncertainty）
        D_gt: (B,1,H,W) or (B,H,W) - 真实深度（可选）
        D_mvs: (B,1,H,W) or (B,H,W) - MVS深度（可选）
        C_mvs: (B,1,H,W) or (B,H,W) - MVS光度一致性置信度（必须和D_mvs同时传入）
        gt_mask: (B,1,H,W) or (B,H,W) - 有效像素掩码，1=有效，0=无效（bool/float）
        lambda_cons_base: float - 一致性损失基础权重（最终权重=base×动态权重）
        lambda_uncert: float - 不确定性正则项权重
        cons_weight_type: str - 动态加权方式：
            'exp'：指数加权（高C_mvs权重陡增，更激进）
            'linear'：线性加权（权重随C_mvs线性增长，更温和）
            'sigmoid'：S型加权（权重饱和，避免极端值）
    
    Returns:
        loss: 标量，仅有效像素的总损失
    """
    # -------------------------- 0. 统一张量维度（确保都是(B,1,H,W)） --------------------------
    def ensure_4d(tensor):
        if tensor is None:
            return None
        if tensor.dim() == 3:  # (B,H,W) → (B,1,H,W)
            return tensor.unsqueeze(1)
        return tensor
    
    D_fused = ensure_4d(D_fused)
    C_fused = ensure_4d(C_fused)
    D_gt = ensure_4d(D_gt)
    D_mvs = ensure_4d(D_mvs)
    C_mvs = ensure_4d(C_mvs)

    loss = 0.0
    B, _, H, W = D_fused.shape

    # -------------------------- 1. 预处理gt_mask（过滤无效像素） --------------------------
    if gt_mask is not None:
        gt_mask = ensure_4d(gt_mask)
        if gt_mask.dtype == torch.bool:
            gt_mask = gt_mask.float()
        gt_mask = gt_mask.to(D_fused.device)
        valid_pixel_num = torch.clamp(gt_mask.sum(), min=1)  # 防止除以0
    else:
        valid_pixel_num = B * H * W  # 所有像素有效

    # -------------------------- 2. 异方差回归损失（仅有效像素） --------------------------
    if D_gt is not None:
        # 异方差回归核心：对确定的像素（C_fused小）严格，不确定的像素（C_fused大）宽松
        abs_error = torch.abs(D_fused - D_gt)  # (B,1,H,W)
        heteroscedastic_term = torch.exp(-C_fused) * abs_error  # 加权绝对误差
        uncert_term = lambda_uncert * C_fused  # 防止C_fused无限大
        
        # 合并损失并过滤无效像素
        loss_depth_per_pixel = heteroscedastic_term + uncert_term
        if gt_mask is not None:
            loss_depth_per_pixel = loss_depth_per_pixel * gt_mask
        
        # 有效像素平均损失
        loss_depth = loss_depth_per_pixel.sum() / valid_pixel_num
        loss += loss_depth

    # -------------------------- 3. 动态加权的MVS一致性损失（核心优化） --------------------------
    if D_mvs is not None and C_mvs is not None:
        # 步骤1：归一化C_mvs到[0,1]（消除置信度数值范围差异）
        C_mvs_min = C_mvs.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]  # (B,1,1,1)
        C_mvs_max = C_mvs.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]  # (B,1,1,1)
        C_mvs_norm = (C_mvs - C_mvs_min) / (C_mvs_max - C_mvs_min + 1e-8)  # (B,1,H,W)，归一化到[0,1]
        
        # 步骤2：根据C_mvs_norm计算动态权重（高C_mvs区域权重高）
        if cons_weight_type == 'exp':
            # 指数加权：权重 = exp(α×C_mvs_norm) - 1，α=3（可调整），权重范围[0, exp(3)-1≈20]
            dynamic_weight = torch.exp(3 * C_mvs_norm) - 1
        elif cons_weight_type == 'linear':
            # 线性加权：权重 = C_mvs_norm，范围[0,1]
            dynamic_weight = C_mvs_norm
        elif cons_weight_type == 'sigmoid':
            # S型加权：权重 = sigmoid(10×(C_mvs_norm-0.5))，范围[0.006, 0.994]，中间值0.5对应权重0.5
            dynamic_weight = torch.sigmoid(10 * (C_mvs_norm - 0.5))
        else:
            raise ValueError(f"不支持的加权方式：{cons_weight_type}")
        
        # 步骤3：计算带动态权重的一致性误差（高C_mvs区域误差惩罚更重）
        cons_error = torch.abs(D_fused - D_mvs)  # (B,1,H,W)
        weighted_cons_error = cons_error * dynamic_weight  # 动态加权
        
        # 步骤4：过滤无效像素
        if gt_mask is not None:
            weighted_cons_error = weighted_cons_error * gt_mask
        
        # 步骤5：计算最终一致性损失（基础权重×加权误差的平均）
        loss_cons = (weighted_cons_error.sum() / valid_pixel_num) * lambda_cons_base
        loss += loss_cons

    return loss