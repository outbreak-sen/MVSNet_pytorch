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
    lambda_cons=0.1
):
    """
    C_fused 被解释为 learned uncertainty (log variance)
    """
    loss = 0.0

    if D_gt is not None:
        # heteroscedastic regression
        loss_depth = torch.mean(
            torch.exp(-C_fused) * torch.abs(D_fused - D_gt) + C_fused
        )
        loss += loss_depth

    if D_mvs is not None:
        # consistency regularization
        loss_cons = torch.mean(torch.abs(D_fused - D_mvs))
        loss += lambda_cons * loss_cons

    return loss
