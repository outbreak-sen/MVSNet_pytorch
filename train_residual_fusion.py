"""
Training script for ResidualFusionNet
Fuses MVSNet depth predictions with DA3 depth predictions using residual learning
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
from fusion_models.residual_fusionmodel import ResidualFusionNet, fusion_loss_residual
from utils import *

cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Residual Fusion Network Training')
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
parser.add_argument('--logdir', default='./checkpoints/residual_fusion', help='directory to save checkpoints/logs')
parser.add_argument('--resume', action='store_true', help='continue to train the model')

parser.add_argument('--summary_freq', type=int, default=20, help='print and summary frequency')
parser.add_argument('--save_freq', type=int, default=1, help='save checkpoint frequency')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed')

parser.add_argument('--hidden_dim', type=int, default=48, help='hidden dimension of fusion network')
parser.add_argument('--lambda_cons_base', type=float, default=0.1, help='MVS consistency loss weight')
parser.add_argument('--lambda_uncert', type=float, default=1.0, help='uncertainty regularization weight')
parser.add_argument('--cons_weight_type', type=str, default='exp', help='consistency weight type: exp/linear/sigmoid')

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
print(f"Creating ResidualFusionNet (hidden={args.hidden_dim})")
fusion_model = ResidualFusionNet(hidden=args.hidden_dim)
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
def ensure_4d(x):
    """Ensure tensor shape is (B, 1, H, W)"""
    if x is None:
        return None
    if x.dim() == 2:
        x = x.unsqueeze(0).unsqueeze(0)
    elif x.dim() == 3:
        x = x.unsqueeze(1)
    elif x.dim() == 4:
        pass
    else:
        raise ValueError(f"Invalid tensor shape: {x.shape}")
    return x

def resize_da3_to_mvs(da3_depth, da3_conf, target_hw):
    """
    Resize DA3 depth/conf to MVSNet output resolution
    da3_depth: [B, H, W]
    da3_conf : [B, H, W]
    target_hw: (H_t, W_t)
    """
    import torch.nn.functional as F
    H_t, W_t = target_hw
    
    if da3_depth is None:
        return None, None
    
    da3_depth = da3_depth.unsqueeze(1)
    da3_conf  = da3_conf.unsqueeze(1) if da3_conf is not None else None
    
    da3_depth_rs = F.interpolate(
        da3_depth,
        size=(H_t, W_t),
        mode="bilinear",
        align_corners=False
    )
    
    if da3_conf is not None:
        da3_conf_rs = F.interpolate(
            da3_conf,
            size=(H_t, W_t),
            mode="bilinear",
            align_corners=False
        )
    else:
        da3_conf_rs = None
    
    da3_depth_rs = da3_depth_rs.squeeze(1)
    da3_conf_rs = da3_conf_rs.squeeze(1) if da3_conf_rs is not None else None
    
    return da3_depth_rs, da3_conf_rs

@torch.no_grad()
def mvsnet_inference(sample):
    """Run MVSNet inference to get depth and confidence"""
    sample_cuda = tocuda(sample)
    imgs = sample_cuda["imgs"]
    proj_matrices = sample_cuda["proj_matrices"]
    depth_values = sample_cuda["depth_values"]
    
    outputs = mvsnet(imgs, proj_matrices, depth_values)
    depth_mvs = outputs["depth"]
    conf_mvs = outputs["photometric_confidence"]
    
    return depth_mvs, conf_mvs


def train_sample(sample, detailed_summary=False):
    """Train one sample"""
    fusion_model.train()
    optimizer.zero_grad()
    
    sample_cuda = tocuda(sample)
    
    # MVSNet inference (frozen)
    with torch.no_grad():
        depth_mvs, conf_mvs = mvsnet_inference(sample)
    
    # Get GT and mask
    depth_gt = sample_cuda["depth"]
    mask = sample_cuda["mask"] > 0.5
    
    # Get DA3 depth and confidence
    da3_depth = sample_cuda["da3_depth"]
    da3_conf = sample_cuda["da3_conf"]
    
    # Resize DA3 to MVSNet resolution
    B, Hm, Wm = depth_mvs.shape
    da3_depth, da3_conf = resize_da3_to_mvs(
        da3_depth,
        da3_conf,
        target_hw=(Hm, Wm)
    )
    
    # Resize GT and mask to MVSNet resolution
    import torch.nn.functional as F
    depth_gt_rs = F.interpolate(
        depth_gt.unsqueeze(1),
        size=(Hm, Wm),
        mode="bilinear",
        align_corners=False
    ).squeeze(1)
    mask_rs = F.interpolate(
        mask.float().unsqueeze(1),
        size=(Hm, Wm),
        mode="nearest"
    ).squeeze(1) > 0.5
    
    # Handle missing DA3 data
    if da3_depth is None:
        da3_depth = depth_mvs.clone().detach()
    if da3_conf is None:
        da3_conf = conf_mvs.clone().detach()
    
    # Ensure all inputs have batch dimension
    depth_mvs = ensure_4d(depth_mvs)
    conf_mvs  = ensure_4d(conf_mvs)
    da3_depth = ensure_4d(da3_depth)
    da3_conf  = ensure_4d(da3_conf)
    
    # Fusion model forward
    depth_fused, conf_fused = fusion_model(depth_mvs, conf_mvs, da3_depth, da3_conf)
    
    # Loss
    loss = fusion_loss_residual(
        depth_fused, conf_fused,
        D_gt=depth_gt_rs,
        D_mvs=depth_mvs.squeeze(1),
        C_mvs=conf_mvs.squeeze(1),
        gt_mask=mask_rs,
        lambda_cons_base=args.lambda_cons_base,
        lambda_uncert=args.lambda_uncert,
        cons_weight_type=args.cons_weight_type
    )
    
    loss.backward()
    optimizer.step()
    
    scalar_outputs = {"loss": loss}
    image_outputs = {
        "depth_mvs": depth_mvs * mask_rs.unsqueeze(1),
        "depth_fused": depth_fused * mask_rs.unsqueeze(1),
        "depth_gt": depth_gt_rs,
        "depth_da3": da3_depth * mask_rs.unsqueeze(1),
        "conf_da3": da3_conf ,
        "conf_mvs": conf_mvs,
        "conf_fused": conf_fused,
    }
    
    if detailed_summary:
        with torch.no_grad():
            depth_error = torch.abs(depth_fused.squeeze(1) - depth_gt_rs) * mask_rs
            scalar_outputs["abs_depth_error"] = depth_error[mask_rs].mean()
            scalar_outputs["rmse"] = torch.sqrt((depth_error ** 2)[mask_rs].mean())
    
    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs


@torch.no_grad()
def test_sample(sample, detailed_summary=True):
    """Test one sample"""
    fusion_model.eval()
    
    sample_cuda = tocuda(sample)
    
    # MVSNet inference
    depth_mvs, conf_mvs = mvsnet_inference(sample)
    
    # Get GT and mask
    depth_gt = sample_cuda["depth"]
    mask = sample_cuda["mask"] > 0.5
    
    # Get DA3 data
    da3_depth = sample_cuda["da3_depth"]
    da3_conf = sample_cuda["da3_conf"]
    
    # Resize DA3 to MVSNet resolution
    B, Hm, Wm = depth_mvs.shape
    da3_depth, da3_conf = resize_da3_to_mvs(
        da3_depth,
        da3_conf,
        target_hw=(Hm, Wm)
    )
    
    # Resize GT and mask to MVSNet resolution
    import torch.nn.functional as F
    depth_gt_rs = F.interpolate(
        depth_gt.unsqueeze(1),
        size=(Hm, Wm),
        mode="bilinear",
        align_corners=False
    ).squeeze(1)
    mask_rs = F.interpolate(
        mask.float().unsqueeze(1),
        size=(Hm, Wm),
        mode="nearest"
    ).squeeze(1) > 0.5
    
    # Handle missing DA3 data
    if da3_depth is None:
        da3_depth = depth_mvs.clone()
    if da3_conf is None:
        da3_conf = conf_mvs.clone()
    
    # Ensure batch dimension
    depth_mvs = ensure_4d(depth_mvs)
    conf_mvs  = ensure_4d(conf_mvs)
    da3_depth = ensure_4d(da3_depth)
    da3_conf  = ensure_4d(da3_conf)
    
    # Fusion forward
    depth_fused, conf_fused = fusion_model(depth_mvs, conf_mvs, da3_depth, da3_conf)
    
    # Loss
    loss = fusion_loss_residual(
        depth_fused, conf_fused,
        D_gt=depth_gt_rs,
        D_mvs=depth_mvs.squeeze(1),
        C_mvs=conf_mvs.squeeze(1),
        gt_mask=mask_rs,
        lambda_cons_base=args.lambda_cons_base,
        lambda_uncert=args.lambda_uncert,
        cons_weight_type=args.cons_weight_type
    )
    
    scalar_outputs = {"loss": loss}
    image_outputs = {
        "depth_mvs": depth_mvs * mask_rs.unsqueeze(1),
        "depth_fused": depth_fused * mask_rs.unsqueeze(1),
        "depth_gt": depth_gt_rs,
        "conf_mvs": conf_mvs,
        "conf_fused": conf_fused,
    }
    
    if detailed_summary:
        depth_error = torch.abs(depth_fused.squeeze(1) - depth_gt_rs) * mask_rs
        scalar_outputs["abs_depth_error"] = depth_error[mask_rs].mean()
        scalar_outputs["rmse"] = torch.sqrt((depth_error ** 2)[mask_rs].mean())
    
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
            
            loss, scalar_outputs, image_outputs = train_sample(sample, detailed_summary=do_summary)
            
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
            torch.save({
                'epoch': epoch_idx,
                'model': fusion_model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, os.path.join(args.logdir, f'model_{epoch_idx:06d}.ckpt'))
            print(f"✓ Checkpoint saved")
        
        # Testing
        print("\nTesting...")
        avg_test_scalars = DictAverageMeter()
        for batch_idx, sample in enumerate(TestImgLoader):
            start_time = time.time()
            do_summary = (batch_idx % args.summary_freq == 0)
            
            loss, scalar_outputs, image_outputs = test_sample(sample, detailed_summary=do_summary)
            
            if do_summary:
                save_scalars(logger, 'test', scalar_outputs, global_step)
                save_images(logger, 'test', image_outputs, global_step)
            
            avg_test_scalars.update(scalar_outputs)
            del scalar_outputs, image_outputs
            
            print(
                f'Epoch {epoch_idx}/{args.epochs}, Iter {batch_idx}/{len(TestImgLoader)}, '
                f'test loss = {loss:.6f}, time = {time.time() - start_time:.3f}s'
            )
        
        # Log average test metrics
        save_scalars(logger, 'test_avg', avg_test_scalars.mean(), epoch_idx)
        print(f"\nAverage test metrics: {avg_test_scalars.mean()}")


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
            print(f"Current avg metrics: {avg_test_scalars.mean()}")
    
    print(f"\nFinal test results: {avg_test_scalars.mean()}")


if __name__ == '__main__':
    if not args.mvsnet_ckpt:
        print("Error: --mvsnet_ckpt is required!")
        sys.exit(1)
    
    train()
