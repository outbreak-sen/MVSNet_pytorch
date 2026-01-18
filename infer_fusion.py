"""
Inference script for DepthBinFusionNet
Fuses MVSNet and DA3 predictions on test data
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import time

from datasets import find_dataset_def
from models import MVSNet
from fusion_models.depthbin_fusionmodel import DepthBinFusionNet
from utils import tocuda

parser = argparse.ArgumentParser(description='Depth Bin Fusion Inference')
parser.add_argument('--dataset', default='dtu_raw_forSparseMVS', help='dataset name')
parser.add_argument('--testpath', required=True, help='testing data path')
parser.add_argument('--testlist', required=True, help='testing scan list')

parser.add_argument('--batch_size', type=int, default=1, help='testing batch size')
parser.add_argument('--numdepth', type=int, default=192, help='number of depth values')
parser.add_argument('--interval_scale', type=float, default=1.06, help='depth interval scale')

parser.add_argument('--mvsnet_ckpt', required=True, help='pretrained MVSNet checkpoint')
parser.add_argument('--fusion_ckpt', required=True, help='trained fusion model checkpoint')
parser.add_argument('--outdir', default='./outputs_fusion', help='output directory')

parser.add_argument('--num_bins', type=int, default=64, help='number of depth bins')
parser.add_argument('--hidden_dim', type=int, default=64, help='hidden dimension')
parser.add_argument('--save_depth', action='store_true', help='save depth maps as pfm')
parser.add_argument('--save_conf', action='store_true', help='save confidence maps')
parser.add_argument('--display', action='store_true', help='display depth visualizations')

args = parser.parse_args()

# Create output directory
os.makedirs(args.outdir, exist_ok=True)

print("="*80)
print("DepthBinFusionNet Inference")
print("="*80)
print(f"Test path: {args.testpath}")
print(f"Test list: {args.testlist}")
print(f"Output directory: {args.outdir}")

# ==================== Load Dataset ====================
print("\nLoading dataset...")
MVSDataset = find_dataset_def(args.dataset)
test_dataset = MVSDataset(
    args.testpath, args.testlist, "test",
    nviews=3, ndepths=args.numdepth, interval_scale=args.interval_scale
)
TestImgLoader = DataLoader(
    test_dataset, args.batch_size, shuffle=False, num_workers=2, drop_last=False
)
print(f"✓ Loaded {len(test_dataset)} test samples")

# ==================== Load Models ====================
print("\nLoading models...")

# MVSNet
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
print("✓ MVSNet loaded")

# Fusion Model
print(f"Loading Fusion Model from: {args.fusion_ckpt}")
fusion_model = DepthBinFusionNet(num_bins=args.num_bins, hidden=args.hidden_dim)
fusion_model = nn.DataParallel(fusion_model)
fusion_model.cuda()
state_dict = torch.load(args.fusion_ckpt)
if 'model' in state_dict:
    fusion_model.load_state_dict(state_dict['model'])
else:
    fusion_model.load_state_dict(state_dict)
fusion_model.eval()
print("✓ Fusion Model loaded")

# ==================== Helper Functions ====================
@torch.no_grad()
def depth_from_prob(prob, depth_values):
    """Estimate depth from probability distribution"""
    B, num_bins, H, W = prob.shape
    
    # Interpolate depth values to bins
    depth_min = depth_values[:, 0].view(B, 1, 1)
    depth_max = depth_values[:, -1].view(B, 1, 1)
    bin_depths = torch.linspace(0, 1, num_bins, device=prob.device).view(1, num_bins, 1, 1)
    bin_depths = depth_min + (depth_max - depth_min) * bin_depths
    
    # Compute expected depth
    prob_soft = prob.softmax(dim=1)
    depth_est = (prob_soft * bin_depths).sum(dim=1)
    
    return depth_est


def save_pfm(filename, data):
    """Save depth map as PFM file"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    data = np.asarray(data, dtype=np.float32)
    
    with open(filename, 'wb') as f:
        if data.ndim == 3:
            data = data[:, :, 0]
        h, w = data.shape
        f.write(b'Pf\n')
        f.write(f'{w} {h}\n'.encode('ascii'))
        f.write(b'-1\n')
        data.astype(np.float32).tofile(f)


def save_confidence(filename, conf):
    """Save confidence map as NPY file"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    conf = np.asarray(conf, dtype=np.float32)
    if len(conf.shape) == 3:
        conf = conf[0]
    np.save(filename, conf)


def visualize_depth(depth, filename, colormap='viridis'):
    """Save depth visualization"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    depth = np.asarray(depth, dtype=np.float32)
    
    if len(depth.shape) == 3:
        depth = depth[0]
    
    # Normalize to [0, 1]
    valid_mask = depth > 0
    if valid_mask.sum() > 0:
        vmin = depth[valid_mask].min()
        vmax = depth[valid_mask].max()
        depth_norm = (depth - vmin) / (vmax - vmin + 1e-8)
    else:
        depth_norm = depth
    
    # Apply colormap
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    
    cmap = cm.get_cmap(colormap)
    depth_vis = cmap(np.clip(depth_norm, 0, 1))
    depth_vis = (depth_vis[:, :, :3] * 255).astype(np.uint8)
    
    Image.fromarray(depth_vis).save(filename)


# ==================== Inference ====================
print("\n" + "="*80)
print("Running inference...")
print("="*80)

total_time = 0
num_samples = 0

with torch.no_grad():
    for batch_idx, sample in enumerate(TestImgLoader):
        start_time = time.time()
        
        sample_cuda = tocuda(sample)
        
        # MVSNet inference
        imgs = sample_cuda["imgs"]
        proj_matrices = sample_cuda["proj_matrices"]
        depth_values = sample_cuda["depth_values"]
        
        outputs = mvsnet(imgs, proj_matrices, depth_values)
        depth_mvs = outputs["depth"]
        conf_mvs = outputs["photometric_confidence"]
        
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
        
        # Fusion inference
        prob, conf_fused = fusion_model(depth_mvs, conf_mvs, da3_depth, da3_conf)
        
        # Get fused depth
        depth_fused = depth_from_prob(prob, depth_values)
        
        # Convert to numpy
        depth_fused_np = depth_fused.cpu().numpy()
        conf_fused_np = conf_fused.cpu().numpy()
        depth_mvs_np = depth_mvs.cpu().numpy()
        conf_mvs_np = conf_mvs.cpu().numpy()
        
        # Get filename
        filename = sample['filename']
        
        # Save results
        if args.save_depth:
            out_depth_file = os.path.join(args.outdir, f'depth_fused_{batch_idx:06d}.pfm')
            save_pfm(out_depth_file, depth_fused_np)
            
            out_mvs_file = os.path.join(args.outdir, f'depth_mvs_{batch_idx:06d}.pfm')
            save_pfm(out_mvs_file, depth_mvs_np)
        
        if args.save_conf:
            out_conf_file = os.path.join(args.outdir, f'conf_fused_{batch_idx:06d}.npy')
            save_confidence(out_conf_file, conf_fused_np)
            
            out_conf_mvs_file = os.path.join(args.outdir, f'conf_mvs_{batch_idx:06d}.npy')
            save_confidence(out_conf_mvs_file, conf_mvs_np)
        
        if args.display:
            out_vis_file = os.path.join(args.outdir, f'depth_fused_{batch_idx:06d}.png')
            visualize_depth(depth_fused_np, out_vis_file)
        
        elapsed_time = time.time() - start_time
        total_time += elapsed_time
        num_samples += 1
        
        print(
            f'Iter {batch_idx}/{len(TestImgLoader)}, '
            f'depth_fused: {depth_fused_np.shape}, '
            f'conf_fused: {conf_fused_np.shape}, '
            f'time: {elapsed_time:.3f}s'
        )

# ==================== Summary ====================
print("\n" + "="*80)
print("Inference completed!")
print("="*80)
print(f"Total samples: {num_samples}")
print(f"Total time: {total_time:.2f}s")
print(f"Average time per sample: {total_time / max(num_samples, 1):.3f}s")
print(f"Output directory: {args.outdir}")
print("\nOutput files:")
if args.save_depth:
    print("  - depth_fused_*.pfm (fused depth maps)")
    print("  - depth_mvs_*.pfm (MVSNet depth maps)")
if args.save_conf:
    print("  - conf_fused_*.npy (fused confidence maps)")
    print("  - conf_mvs_*.npy (MVSNet confidence maps)")
if args.display:
    print("  - depth_fused_*.png (depth visualizations)")
