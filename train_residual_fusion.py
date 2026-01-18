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
parser.add_argument('--dataset', default='dtu_raw_forSparseMVS', help='select dataset')

parser.add_argument('--trainpath', required=True, help='train datapath')
parser.add_argument('--testpath', help='test datapath')
parser.add_argument('--trainlist', required=True, help='train list')
parser.add_argument('--testlist', help='test list')

parser.add_argument('--epochs', type=int, default=20, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
parser.add_argument('--lrepochs', type=str, default="10,15,18:2", help='epoch ids to downscale lr')
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
parser.add_argument('--seed', type=int, default=1, help='random seed')
parser.add_argument('--hidden_dim', type=int, default=48, help='hidden dimension of fusion network')

args = parser.parse_args()
if args.testpath is None:
    args.testpath = args.trainpath
if args.testlist is None:
    args.testlist = args.trainlist

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

if not os.path.isdir(args.logdir):
    os.makedirs(args.logdir)

current_time_str = str(datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
print("Current time:", current_time_str)
logger = SummaryWriter(args.logdir)

print("argv:", sys.argv[1:])
print_args(args)

# ==================== Dataset ====================
print("\nLoading dataset...")
MVSDataset = find_dataset_def(args.dataset)
train_dataset = MVSDataset(args.trainpath, args.trainlist, "train", nviews=3, ndepths=args.numdepth, interval_scale=args.interval_scale)
test_dataset = MVSDataset(args.testpath, args.testlist, "val", nviews=3, ndepths=args.numdepth, interval_scale=args.interval_scale)

TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=4, drop_last=True)
TestImgLoader = DataLoader(test_dataset, args.batch_size, shuffle=False, num_workers=2, drop_last=False)

print(f"✓ Train: {len(train_dataset)}, Test: {len(test_dataset)}")

# ==================== Models ====================
print("\nLoading models...")

# MVSNet (frozen)
print(f"Loading MVSNet from: {args.mvsnet_ckpt}")
mvsnet = MVSNet(refine=False)
mvsnet = nn.DataParallel(mvsnet)
mvsnet.cuda()
state_dict = torch.load(args.mvsnet_ckpt)
if 'model' in state_dict:
    mvsnet.load_state_dict(state_dict['model'])
else:
    mvsnet.load_state_dict(state_dict)
mvsnet.eval()
for param in mvsnet.parameters():
    param.requires_grad = False
print("✓ MVSNet loaded (frozen)")

# Fusion model
print(f"Creating ResidualFusionNet (hidden={args.hidden_dim})")
fusion_model = ResidualFusionNet(hidden=args.hidden_dim)
fusion_model = nn.DataParallel(fusion_model)
fusion_model.cuda()
print("✓ Fusion model created")

# ==================== Optimizer ====================
optimizer = optim.Adam(fusion_model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.wd)

# ==================== Load checkpoint ====================
start_epoch = 0
if args.resume or (args.fusion_ckpt and os.path.exists(args.fusion_ckpt)):
    if args.resume:
        saved_models = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
        if saved_models:
            saved_models = sorted(saved_models, key=lambda x: int(x.split('_')[-1].split('.')[0]))
            loadckpt = os.path.join(args.logdir, saved_models[-1])
        else:
            loadckpt = None
    else:
        loadckpt = args.fusion_ckpt
    
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

print(f"✓ Total params: {sum([p.data.nelement() for p in fusion_model.parameters()])}")

# ==================== Functions ====================
@torch.no_grad()
def mvsnet_inference(sample):
    """Run MVSNet inference"""
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
    depth_gt = sample_cuda["depth"]
    mask = sample_cuda["mask"] > 0.5
    
    # MVSNet inference
    with torch.no_grad():
        depth_mvs, conf_mvs = mvsnet_inference(sample)
    
    # Get DA3
    da3_depth = sample_cuda["da3_depth"]
    da3_conf = sample_cuda["da3_conf"]
    
    if da3_depth is None:
        da3_depth = depth_mvs.clone().detach()
    if da3_conf is None:
        da3_conf = conf_mvs.clone().detach()
    
    # Ensure batch dimension
    if len(depth_mvs.shape) == 3:
        depth_mvs = depth_mvs.unsqueeze(1)
    if len(conf_mvs.shape) == 3:
        conf_mvs = conf_mvs.unsqueeze(1)
    if len(da3_depth.shape) == 3:
        da3_depth = da3_depth.unsqueeze(1)
    if len(da3_conf.shape) == 3:
        da3_conf = da3_conf.unsqueeze(1)
    
    # Fusion forward
    depth_fused, conf_fused = fusion_model(depth_mvs, conf_mvs, da3_depth, da3_conf)
    
    # Loss
    loss = fusion_loss_residual(depth_fused, conf_fused, D_gt=depth_gt, D_mvs=depth_mvs, lambda_cons=0.1)
    
    loss.backward()
    optimizer.step()
    
    scalar_outputs = {"loss": loss}
    image_outputs = {
        "depth_mvs": depth_mvs * mask.unsqueeze(1),
        "depth_fused": depth_fused * mask.unsqueeze(1),
        "depth_gt": depth_gt,
        "conf_mvs": conf_mvs,
        "conf_fused": conf_fused,
    }
    
    if detailed_summary:
        with torch.no_grad():
            depth_error = torch.abs(depth_fused - depth_gt.unsqueeze(1)) * mask.unsqueeze(1)
            scalar_outputs["abs_depth_error"] = depth_error[mask].mean()
            scalar_outputs["rmse"] = torch.sqrt((depth_error ** 2)[mask].mean())
    
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
    
    # Get DA3
    da3_depth = sample_cuda["da3_depth"]
    da3_conf = sample_cuda["da3_conf"]
    
    if da3_depth is None:
        da3_depth = depth_mvs.clone()
    if da3_conf is None:
        da3_conf = conf_mvs.clone()
    
    # Ensure batch dimension
    if len(depth_mvs.shape) == 3:
        depth_mvs = depth_mvs.unsqueeze(1)
    if len(conf_mvs.shape) == 3:
        conf_mvs = conf_mvs.unsqueeze(1)
    if len(da3_depth.shape) == 3:
        da3_depth = da3_depth.unsqueeze(1)
    if len(da3_conf.shape) == 3:
        da3_conf = da3_conf.unsqueeze(1)
    
    # Fusion forward
    depth_fused, conf_fused = fusion_model(depth_mvs, conf_mvs, da3_depth, da3_conf)
    
    # Loss
    loss = fusion_loss_residual(depth_fused, conf_fused, D_gt=depth_gt, D_mvs=depth_mvs, lambda_cons=0.1)
    
    scalar_outputs = {"loss": loss}
    image_outputs = {
        "depth_mvs": depth_mvs * mask.unsqueeze(1),
        "depth_fused": depth_fused * mask.unsqueeze(1),
        "depth_gt": depth_gt,
        "conf_mvs": conf_mvs,
        "conf_fused": conf_fused,
    }
    
    if detailed_summary:
        depth_error = torch.abs(depth_fused - depth_gt.unsqueeze(1)) * mask.unsqueeze(1)
        scalar_outputs["abs_depth_error"] = depth_error[mask].mean()
        scalar_outputs["rmse"] = torch.sqrt((depth_error ** 2)[mask].mean())
    
    return tensor2float(loss), tensor2float(scalar_outputs), image_outputs


def train():
    """Main training loop"""
    milestones = [int(epoch_idx) for epoch_idx in args.lrepochs.split(':')[0].split(',')]
    lr_gamma = 1 / float(args.lrepochs.split(':')[1])
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=lr_gamma, last_epoch=start_epoch - 1)
    start_reverse_epoch = int(args.epochs / 2)
    for epoch_idx in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch_idx}/{args.epochs}")
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
            print(f'Train {batch_idx}/{len(TrainImgLoader)}, loss={loss:.6f}, time={time.time()-start_time:.3f}s')
        
        # Save checkpoint
        if (epoch_idx + 1) % args.save_freq == 0:
            torch.save({
                'epoch': epoch_idx,
                'model': fusion_model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, os.path.join(args.logdir, f'model_{epoch_idx:06d}.ckpt'))
            print(f"✓ Checkpoint saved")
        
        # Testing
        print("Testing...")
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
            print(f'Test {batch_idx}/{len(TestImgLoader)}, loss={loss:.6f}')
        
        save_scalars(logger, 'test_avg', avg_test_scalars.mean(), epoch_idx)
        print(f"Test avg: {avg_test_scalars.mean()}")


if __name__ == '__main__':
    if not args.mvsnet_ckpt:
        print("Error: --mvsnet_ckpt is required!")
        sys.exit(1)
    
    train()
