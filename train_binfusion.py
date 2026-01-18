"""
Training script for DepthBinFusionNet
Fuses MVSNet depth predictions with DA3 depth predictions
"""

import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import numpy as np
import time
from tensorboardX import SummaryWriter
import sys
import datetime

from datasets import find_dataset_def
from models import MVSNet
from fusion_models.depthbin_fusionmodel import DepthBinFusionNet, depth_bin_loss
from utils import *

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Depth Bin Fusion Network Training')
parser.add_argument('--model', default='mvsnet', help='base model')
parser.add_argument('--dataset', default='dtu_raw_forSparseMVS', help='select dataset')

parser.add_argument('--trainpath', required=True, help='train datapath')
parser.add_argument('--testpath', help='test datapath')
parser.add_argument('--trainlist', required=True, help='train list')
parser.add_argument('--testlist', help='test list')

parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--lrepochs', type=str, default="10,15,18:2", help='epoch ids to downscale lr and the downscale rate')
parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')

parser.add_argument('--batch_size', type=int, default=4, help='train batch size')
parser.add_argument('--numdepth', type=int, default=192, help='number of depth values')
parser.add_argument('--interval_scale', type=float, default=1.06, help='depth interval scale')

parser.add_argument('--mvsnet_ckpt', required=True, help='pretrained MVSNet checkpoint path')
parser.add_argument('--fusion_ckpt', default=None, help='load a pretrained fusion model checkpoint')
parser.add_argument('--logdir', default='./checkpoints/fusion', help='directory to save checkpoints/logs')
parser.add_argument('--resume', action='store_true', help='continue to train the model')

parser.add_argument('--summary_freq', type=int, default=20, help='print and summary frequency')
parser.add_argument('--save_freq', type=int, default=1, help='save checkpoint frequency')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')

parser.add_argument('--num_bins', type=int, default=64, help='number of depth bins')
parser.add_argument('--hidden_dim', type=int, default=64, help='hidden dimension of fusion network')

# parse arguments
args = parser.parse_args()
if args.testpath is None:
    args.testpath = args.trainpath
if args.testlist is None:
    args.testlist = args.trainlist

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

# create directory
if not os.path.isdir(args.logdir):
    os.makedirs(args.logdir)

current_time_str = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
print("Current time:", current_time_str)
print("Creating summary file...")
logger = SummaryWriter(args.logdir)

print("argv:", sys.argv[1:])
print_args(args)

# ==================== Dataset and DataLoader ====================
print("\n" + "="*80)
print("Loading dataset...")
print("="*80)

MVSDataset = find_dataset_def(args.dataset)
train_dataset = MVSDataset(
    args.trainpath, args.trainlist, "train", 
    nviews=3, ndepths=args.numdepth, interval_scale=args.interval_scale
)
test_dataset = MVSDataset(
    args.testpath, args.testlist, "val", 
    nviews=3, ndepths=args.numdepth, interval_scale=args.interval_scale
)

TrainImgLoader = DataLoader(
    train_dataset, args.batch_size, shuffle=True, 
    num_workers=4, drop_last=True
)
TestImgLoader = DataLoader(
    test_dataset, args.batch_size, shuffle=False, 
    num_workers=2, drop_last=False
)

print(f"✓ Train dataset: {len(train_dataset)} samples")
print(f"✓ Test dataset: {len(test_dataset)} samples")

# ==================== Models ====================
print("\n" + "="*80)
print("Loading models...")
print("="*80)

# Load pretrained MVSNet (frozen)
print("Loading pretrained MVSNet from:", args.mvsnet_ckpt)
mvsnet = MVSNet(refine=False)
mvsnet = nn.DataParallel(mvsnet)
mvsnet.cuda()

state_dict = torch.load(args.mvsnet_ckpt)
if 'model' in state_dict:
    mvsnet.load_state_dict(state_dict['model'])
else:
    mvsnet.load_state_dict(state_dict)
    
mvsnet.eval()
# Freeze MVSNet
for param in mvsnet.parameters():
    param.requires_grad = False
print("✓ MVSNet loaded and frozen")

# Create fusion network
print(f"Creating DepthBinFusionNet (num_bins={args.num_bins}, hidden={args.hidden_dim})")
fusion_model = DepthBinFusionNet(num_bins=args.num_bins, hidden=args.hidden_dim)
fusion_model = nn.DataParallel(fusion_model)
fusion_model.cuda()
print("✓ Fusion model created")

# ==================== Optimizer ====================
optimizer = optim.Adam(
    fusion_model.parameters(), 
    lr=args.lr, 
    betas=(0.9, 0.999), 
    weight_decay=args.wd
)

# ==================== Load checkpoint ====================
start_epoch = 0
if args.resume or (args.fusion_ckpt and os.path.exists(args.fusion_ckpt)):
    if args.resume:
        # Find latest checkpoint
        saved_models = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
        if saved_models:
            saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
            loadckpt = os.path.join(args.logdir, saved_models[-1])
            print(f"Resuming from: {loadckpt}")
        else:
            loadckpt = None
    else:
        loadckpt = args.fusion_ckpt
        print(f"Loading fusion model from: {loadckpt}")
    
    if loadckpt and os.path.exists(loadckpt):
        state_dict = torch.load(loadckpt)
        if 'model' in state_dict:
            fusion_model.load_state_dict(state_dict['model'])
            if 'optimizer' in state_dict:
                optimizer.load_state_dict(state_dict['optimizer'])
            if 'epoch' in state_dict:
                start_epoch = state_dict['epoch'] + 1
        else:
            fusion_model.load_state_dict(state_dict)
        print(f"✓ Model loaded, starting from epoch {start_epoch}")

print(f"✓ Total fusion model parameters: {sum([p.data.nelement() for p in fusion_model.parameters()])}")

# ==================== Helper Functions ====================
def depth_to_bins(depth_gt, depth_values, num_bins):
    """Convert depth ground truth to bin indices"""
    B, H, W = depth_gt.shape
    num_depths = depth_values.shape[1]
    
    # Create bin edges
    depth_min = depth_values[:, 0:1].view(B, 1, 1)  # (B, 1, 1)
    depth_max = depth_values[:, -1:].view(B, 1, 1)  # (B, 1, 1)
    
    # Normalize depth to [0, 1]
    depth_normalized = (depth_gt - depth_min) / (depth_max - depth_min + 1e-8)
    depth_normalized = torch.clamp(depth_normalized, 0, 1)
    
    # Convert to bin indices
    bin_indices = (depth_normalized * (num_bins - 1)).long()
    bin_indices = torch.clamp(bin_indices, 0, num_bins - 1)
    
    return bin_indices

def depth_from_prob(prob, depth_bins):
    """Estimate depth from probability distribution"""
    # prob: (B, num_bins, H, W)
    # depth_bins: (B, num_bins)
    B, num_bins, H, W = prob.shape
    
    # 确保 depth_bins 有正确的形状用于广播
    if depth_bins.dim() == 2:
        # depth_bins: (B, num_bins) -> (B, num_bins, 1, 1)
        depth_bins_expanded = depth_bins.view(B, num_bins, 1, 1)
    else:
        # 如果已经是 (B, num_bins, 1, 1) 或类似形状
        depth_bins_expanded = depth_bins
    
    # 计算期望深度（期望值）
    prob_soft = F.softmax(prob, dim=1)  # 归一化概率分布
    depth_est = (prob_soft * depth_bins_expanded).sum(dim=1)
    return depth_est

@torch.no_grad()
def mvsnet_inference(sample):
    """Run MVSNet inference to get depth and confidence"""
    sample_cuda = tocuda(sample)
    imgs = sample_cuda["imgs"]
    proj_matrices = sample_cuda["proj_matrices"]
    depth_values = sample_cuda["depth_values"]
    print("in mvsnet_inference:", imgs.shape, proj_matrices.shape, depth_values.shape)
    outputs = mvsnet(imgs, proj_matrices, depth_values)
    depth_mvs = outputs["depth"]
    conf_mvs = outputs["photometric_confidence"]
    print("in mvsnet_inference:", conf_mvs.shape, depth_mvs.shape)
    
    return depth_mvs, conf_mvs


def resize_da3_to_mvs(da3_depth, da3_conf, target_hw):
    """
    da3_depth: [B, H, W]
    da3_conf : [B, H, W]
    target_hw: (H_t, W_t)
    return:
        da3_depth_rs: [B, H_t, W_t]
        da3_conf_rs : [B, H_t, W_t]
    """
    H_t, W_t = target_hw

    # [B, H, W] -> [B, 1, H, W]
    da3_depth = da3_depth.unsqueeze(1)
    da3_conf  = da3_conf.unsqueeze(1)

    da3_depth_rs = F.interpolate(
        da3_depth,
        size=(H_t, W_t),
        mode="bilinear",
        align_corners=False
    )

    da3_conf_rs = F.interpolate(
        da3_conf,
        size=(H_t, W_t),
        mode="bilinear",
        align_corners=False
    )

    # back to [B, H, W]
    da3_depth_rs = da3_depth_rs.squeeze(1)
    da3_conf_rs  = da3_conf_rs.squeeze(1)

    return da3_depth_rs, da3_conf_rs
def ensure_4d(x):
    """
    Ensure tensor shape is (B, 1, H, W)
    """
    if x.dim() == 2:
        # (H, W)
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.dim() == 3:
        # (B, H, W)
        x = x.unsqueeze(1)
    elif x.dim() == 4:
        # already OK
        pass
    else:
        raise ValueError(f"Invalid tensor shape: {x.shape}")
    return x

def train_sample(sample, detailed_summary=False):
    """Train one sample"""
    fusion_model.train()
    optimizer.zero_grad()
    
    sample_cuda = tocuda(sample)
    
    # MVSNet inference (frozen)
    with torch.no_grad():
        depth_mvs, conf_mvs = mvsnet_inference(sample)
    
    # Get DA3 depth and confidence
    da3_depth = sample_cuda["da3_depth"]
    da3_conf = sample_cuda["da3_conf"]
    B, Hm, Wm = depth_mvs.shape
    da3_depth, da3_conf = resize_da3_to_mvs(
        da3_depth,
        da3_conf,
        target_hw=(Hm, Wm)
    )
    depth_gt = sample_cuda["depth"]
    depth_gt, mask = resize_da3_to_mvs(
        depth_gt,
        sample_cuda["mask"],
        target_hw=(Hm, Wm)
    )
    mask = mask > 0.5
    # Handle missing DA3 data
    if da3_depth is None:
        da3_depth = depth_mvs.clone().detach()
    if da3_conf is None:
        da3_conf = conf_mvs.clone().detach()
    depth_values = sample_cuda["depth_values"]
    print("shapes:", depth_mvs.shape, conf_mvs.shape, da3_depth.shape, da3_conf.shape, depth_gt.shape,depth_values.shape)
    
    # Ensure all inputs have batch dimension
    
    # 修复：为每个batch样本计算各自的深度范围
    depth_mvs = ensure_4d(depth_mvs)
    conf_mvs  = ensure_4d(conf_mvs)
    da3_depth = ensure_4d(da3_depth)
    da3_conf  = ensure_4d(da3_conf)
    depth_values = depth_values.unsqueeze(1)
    print("shapes:", depth_mvs.shape, conf_mvs.shape, da3_depth.shape, da3_conf.shape, depth_gt.shape,depth_values.shape)
    
    # Fusion model forward
    prob, conf_fused = fusion_model(depth_mvs, conf_mvs, da3_depth, da3_conf)
    print("fusionshapes:", prob.shape, conf_fused.shape)
    K = prob.shape[1]  # 64
    
    # 修复：为每个batch样本计算各自的深度区间
    depth_bins = []
    for b in range(B):
        mask_b = mask[b] if mask.dim() > 1 else mask
        depth_min = depth_gt[b][mask_b].min()
        depth_max = depth_gt[b][mask_b].max()
        bins = torch.linspace(
            depth_min,
            depth_max,
            K,
            device=prob.device
        )
        depth_bins.append(bins)
    depth_bins = torch.stack(depth_bins, dim=0)  # (B, K)
    
    # Compute loss
    print("before cal loss, shape:", prob.shape, depth_gt.shape, depth_bins.shape,mask.shape)
    loss = depth_bin_loss(prob, depth_gt, depth_bins, mask)
    
    loss.backward()
    optimizer.step()
    
    # Get fused depth for visualization
    with torch.no_grad():
        print("depth_bins:", depth_bins.shape)
        depth_fused = depth_from_prob(prob, depth_bins)
    
    scalar_outputs = {"loss": loss}
    image_outputs = {
        "depth_mvs": depth_mvs * mask.unsqueeze(1),
        "depth_fused": depth_fused * mask,
        "depth_gt": depth_gt,
        "depth_da3": da3_depth * mask.unsqueeze(1),
        "conf_da3": da3_conf ,
        "conf_mvs": conf_mvs,
        "conf_fused": conf_fused,
    }
    
    if detailed_summary:
        with torch.no_grad():
            depth_error = torch.abs(depth_fused - depth_gt) * mask
            scalar_outputs["abs_depth_error"] = depth_error[mask].mean()
            scalar_outputs["rmse"] = torch.sqrt((depth_error ** 2).mean())
    
    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs

@torch.no_grad()
def test_sample(sample, detailed_summary=True):
    """Test one sample"""
    fusion_model.eval()
    
    sample_cuda = tocuda(sample)
    depth_gt = sample_cuda["depth"]
    mask = sample_cuda["mask"] > 0.5
    
    # MVSNet inference
    depth_mvs, conf_mvs = mvsnet_inference(sample)
    
    # Get DA3 data
    da3_depth = sample_cuda["da3_depth"]
    da3_conf = sample_cuda["da3_conf"]
    
    if da3_depth is None:
        da3_depth = depth_mvs.clone()
    if da3_conf is None:
        da3_conf = conf_mvs.clone()
    
    # Ensure batch dimension
    if len(depth_mvs.shape) == 3:
        depth_mvs = depth_mvs.unsqueeze(0)
    if len(conf_mvs.shape) == 3:
        conf_mvs = conf_mvs.unsqueeze(0)
    if len(da3_depth.shape) == 3:
        da3_depth = da3_depth.unsqueeze(0)
    if len(da3_conf.shape) == 3:
        da3_conf = da3_conf.unsqueeze(0)
    
    # Fusion forward
    prob, conf_fused = fusion_model(depth_mvs, conf_mvs, da3_depth, da3_conf)
    
    # Loss
    depth_values = sample_cuda["depth_values"]
    loss = depth_bin_loss(prob, depth_gt, depth_values)
    
    # Get fused depth
    depth_fused = depth_from_prob(prob, depth_values)
    
    scalar_outputs = {"loss": loss}
    image_outputs = {
        "depth_mvs": depth_mvs * mask.unsqueeze(1),
        "depth_fused": depth_fused * mask,
        "depth_gt": depth_gt,
        "conf_mvs": conf_mvs,
        "conf_fused": conf_fused,
    }
    
    if detailed_summary:
        depth_error = torch.abs(depth_fused - depth_gt) * mask
        scalar_outputs["abs_depth_error"] = depth_error[mask].mean()
        scalar_outputs["rmse"] = torch.sqrt((depth_error ** 2).mean())
        
        # Threshold metrics
        for threshold in [2, 4, 8]:
            error_mask = depth_error < threshold
            acc = error_mask[mask].float().mean()
            scalar_outputs[f"acc_{threshold}mm"] = acc
    
    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs


def train():
    """Main training loop"""
    milestones = [int(epoch_idx) for epoch_idx in args.lrepochs.split(':')[0].split(',')]
    lr_gamma = 1 / float(args.lrepochs.split(':')[1])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones, gamma=lr_gamma, last_epoch=start_epoch - 1
    )
    start_reverse_epoch = int(args.epochs / 2)
    for epoch_idx in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch_idx}:")
        print("="*80)
        lr_scheduler.step()
        global_step = len(TrainImgLoader) * epoch_idx
        
        if epoch_idx == start_reverse_epoch:
            print("=> Start using reverse view selection for training")
            train_dataset.set_pair_reverse_model(True, N=3)
        # Training
        for batch_idx, sample in enumerate(TrainImgLoader):
            start_time = time.time()
            global_step = len(TrainImgLoader) * epoch_idx + batch_idx
            do_summary = global_step % args.summary_freq == 0
            
            loss, scalar_outputs, image_outputs = train_sample(sample, detailed_summary=do_summary) # scalar是标量的意思
            
            if do_summary:
                save_scalars(logger, 'train', scalar_outputs, global_step)
                save_images(logger, 'train', image_outputs, global_step)
            
            del scalar_outputs, image_outputs
            
            print(
                f'Epoch {epoch_idx}/{args.epochs}, Iter {batch_idx}/{len(TrainImgLoader)}, '
                f'train loss = {loss:.6f}, time = {time.time() - start_time:.3f}s'
            )
        
        # Save checkpoint
        if (epoch_idx + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(args.logdir, f'model_{epoch_idx:06d}.ckpt')
            torch.save({
                'epoch': epoch_idx,
                'model': fusion_model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, checkpoint_path)
            print(f"✓ Checkpoint saved: {checkpoint_path}")
        
        # Testing
        print("\nTesting...")
        # avg_test_scalars = DictAverageMeter()
        # for batch_idx, sample in enumerate(TestImgLoader):
        #     start_time = time.time()
        #     do_summary = (batch_idx % args.summary_freq == 0)
            
        #     loss, scalar_outputs, image_outputs = test_sample(sample, detailed_summary=do_summary)
            
        #     if do_summary:
        #         save_scalars(logger, 'test', scalar_outputs, global_step)
        #         save_images(logger, 'test', image_outputs, global_step)
            
        #     avg_test_scalars.update(scalar_outputs)
        #     del scalar_outputs, image_outputs
            
        #     print(
        #         f'Epoch {epoch_idx}/{args.epochs}, Iter {batch_idx}/{len(TestImgLoader)}, '
        #         f'test loss = {loss:.6f}, time = {time.time() - start_time:.3f}s'
        #     )
        
        # # Log average test metrics
        # save_scalars(logger, 'test_avg', avg_test_scalars.mean(), epoch_idx)
        # print(f"\nAverage test metrics: {avg_test_scalars.mean()}")


def test():
    """Test mode"""
    avg_test_scalars = DictAverageMeter()
    fusion_model.eval()
    
    for batch_idx, sample in enumerate(TestImgLoader):
        start_time = time.time()
        loss, scalar_outputs, image_outputs = test_sample(sample, detailed_summary=True)
        avg_test_scalars.update(scalar_outputs)
        del scalar_outputs, image_outputs
        
        print(
            f'Iter {batch_idx}/{len(TestImgLoader)}, '
            f'test loss = {loss:.6f}, time = {time.time() - start_time:.3f}s'
        )
        
        if batch_idx % 50 == 0:
            print(f"Iter {batch_idx}/{len(TestImgLoader)}, test results: {avg_test_scalars.mean()}")
    
    print(f"\nFinal test results: {avg_test_scalars.mean()}")


if __name__ == '__main__':
    if not args.mvsnet_ckpt:
        print("Error: --mvsnet_ckpt is required!")
        sys.exit(1)
    
    train()
