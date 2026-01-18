from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
# from datasets.data_io import *
try:
    # 当直接运行 python datasets/dataloader.py 时
    from data_io import *
except ImportError:
    # 当从外部导入时
    from .data_io import *
# 这个文件是针对想要训练一个对于SparseMVSNet的的深度融合模块，需要输出raw原始尺寸的DTU图像和DA3的深度图和置信度图
# 需要注意的是这里的DTU数据集原始为1600*1200的，然后需要剪切[:16]然后因为推理获得下采样四倍的深度图所以内参除以4
# 然后注意DA3和DA置信度图需要改成一样的尺度
class MVSDataset(Dataset):
    def __init__(self, datapath, listfile, mode, nviews, ndepths=192, interval_scale=1.06, **kwargs):
        super(MVSDataset, self).__init__()
        self.datapath = datapath
        self.listfile = listfile
        self.mode = mode
        self.nviews = nviews
        self.ndepths = ndepths
        self.interval_scale = interval_scale
        self.reversemodel = False

        assert self.mode in ["train", "val", "test"]
        self.metas = self.build_list()

    def build_list(self):
        metas = []
        with open(self.listfile) as f:
            scans = f.readlines()
            scans = [line.rstrip() for line in scans]

        # scans
        for scan in scans:
            pair_file = "Cameras/pair.txt"
            # read the pair file
            with open(os.path.join(self.datapath, pair_file)) as f:
                num_viewpoint = int(f.readline())
                # viewpoints (49)
                for view_idx in range(num_viewpoint):
                    ref_view = int(f.readline().rstrip())
                    src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                    # light conditions 0-6
                    for light_idx in range(1):
                        metas.append((scan, light_idx, ref_view, src_views))
        print("dataset", self.mode, "metas:", len(metas))
        return metas

    def __len__(self):
        return len(self.metas)

    def read_cam_file(self, filename):
        with open(filename) as f:
            lines = f.readlines()
            lines = [line.rstrip() for line in lines]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ').reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ').reshape((3, 3))
        intrinsics[:2, :] /= 4
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        depth_interval = float(lines[11].split()[1]) * self.interval_scale
        return intrinsics, extrinsics, depth_min, depth_interval

    def read_img(self, filename):
        img = Image.open(filename)
        # scale 0~255 to 0~1
        np_img = np.array(img, dtype=np.float32) / 255.
        assert np_img.shape[:2] == (1200, 1600)
        # crop to (1184, 1600)
        np_img = np_img[:-16, :] 
        return np_img

    def read_depth(self, filename):
        # read pfm depth file
        np_depth = np.array(read_pfm(filename)[0], dtype=np.float32)
        np_depth = np_depth[:-16, :] 
        return np_depth
    
    def read_da3_depth(self, filename):
        # read npy depth file
        if os.path.exists(filename):
            depth = np.load(filename).astype(np.float32)
            depth = depth[:-16, :] 
            return depth
        else:
            return None

    def read_da3_conf(self, filename):
        # read npy confidence file
        if os.path.exists(filename):
            conf = np.load(filename).astype(np.float32)
            conf = conf[:-16, :] 
            return conf
        else:
            return None
    def set_pair_reverse_model(self,reverse_model,N = None):
        if N is not None:
            self.nviews = N
        self.reversemodel = reverse_model
        
        
    def __getitem__(self, idx):
        meta = self.metas[idx]
        scan, light_idx, ref_view, src_views = meta
        # use only the reference view and first nviews-1 source views
        if not self.reversemodel:
            view_ids = [ref_view] + src_views[:self.nviews - 1]
        else:
            view_ids = [ref_view] + src_views[len(src_views)-self.nviews +1:-1]

        imgs = []
        mask = None
        depth = None
        depth_values = None
        proj_matrices = []
        da3_depth = None
        da3_conf = None

        for i, vid in enumerate(view_ids):
            # NOTE that the id in image file names is from 1 to 49 (not 0~48)
            img_filename = os.path.join(self.datapath,'Rectified/{}/rect_{:0>3}_{}_r5000.png'.format(scan, vid + 1, light_idx))
            mask_filename = os.path.join(self.datapath, 'Depths_raw/{}/depth_visual_{:0>4}.png'.format(scan, vid))
            depth_filename = os.path.join(self.datapath, 'Depths_raw/{}/depth_map_{:0>4}.pfm'.format(scan, vid))
            proj_mat_filename = os.path.join(self.datapath, 'Cameras/train/{:0>8}_cam.txt').format(vid)
            da3_depth_path = os.path.join(self.datapath, 'DA3Depth', scan, f'{vid:08d}.npy')
            da3_conf_path = os.path.join(self.datapath, 'DA3Conf', scan, f'{vid:08d}.npy')
            
            imgs.append(self.read_img(img_filename))
            intrinsics, extrinsics, depth_min, depth_interval = self.read_cam_file(proj_mat_filename)

            # multiply intrinsics and extrinsics to get projection matrix
            proj_mat = extrinsics.copy()
            proj_mat[:3, :4] = np.matmul(intrinsics, proj_mat[:3, :4])
            proj_matrices.append(proj_mat)

            if i == 0:  # reference view
                depth_values = np.arange(depth_min, depth_interval * self.ndepths + depth_min, depth_interval,
                                         dtype=np.float32)
                mask = self.read_img(mask_filename)
                depth = self.read_depth(depth_filename)
                da3_depth = self.read_da3_depth(da3_depth_path)
                da3_conf = self.read_da3_conf(da3_conf_path)

        imgs = np.stack(imgs).transpose([0, 3, 1, 2])
        proj_matrices = np.stack(proj_matrices)

        return {"imgs": imgs,
                "proj_matrices": proj_matrices,
                "depth_values": depth_values,
                "depth": depth,
                "mask": mask,
                "filename": scan + '/{}/' + '{:0>8}'.format(view_ids[0]) + "{}",
                "da3_depth":da3_depth,
                "da3_conf" :da3_conf,}


# ==================== Test Main ====================
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import sys
    
    # Configuration
    m_show = False  # Set to True to visualize images
    
    # Dataset configuration   
    # datapath = "/media/outbreak/68E1-B517/Dataset/processed/dtu_training_raw"  # Replace with actual path
    datapath = "/data1/local_userdata/houbosen/dtu_training_raw"
    listfile = "lists/dtu/train.txt"  # Replace with actual path
    mode = "val"
    nviews = 3
    ndepths = 192
    
    print("=" * 80)
    print("MVSDataset Test Program")
    print("=" * 80)
    print(f"m_show (Visualization): {m_show}")
    print(f"Dataset path: {datapath}")
    print(f"List file: {listfile}")
    print(f"Mode: {mode}")
    print(f"Number of views: {nviews}")
    print(f"Number of depths: {ndepths}")
    print("-" * 80)
    
    try:
        # Initialize dataset
        dataset = MVSDataset(
            datapath=datapath,
            listfile=listfile,
            mode=mode,
            nviews=nviews,
            ndepths=ndepths
        )
        
        print(f"\n✓ Dataset initialized successfully!")
        print(f"✓ Total samples in dataset: {len(dataset)}")
        
        # Test a few samples
        num_test_samples = min(3, len(dataset))
        
        for sample_idx in range(num_test_samples):
            print("\n" + "=" * 80)
            print(f"Sample {sample_idx + 1}/{num_test_samples}")
            print("-" * 80)
            
            try:
                sample = dataset[sample_idx]
                
                # Print shape information for all variables
                print("\n【Data Shapes】")
                print(f"  imgs:              {sample['imgs'].shape} (dtype: {sample['imgs'].dtype})")
                print(f"  proj_matrices:     {sample['proj_matrices'].shape} (dtype: {sample['proj_matrices'].dtype})")
                print(f"  depth_values:      {sample['depth_values'].shape} (dtype: {sample['depth_values'].dtype})")
                print(f"  depth:             {sample['depth'].shape if sample['depth'] is not None else 'None'} (dtype: {sample['depth'].dtype if sample['depth'] is not None else 'N/A'})")
                print(f"  mask:              {sample['mask'].shape if sample['mask'] is not None else 'None'} (dtype: {sample['mask'].dtype if sample['mask'] is not None else 'N/A'})")
                if sample['da3_depth'] is not None:
                    print(f"  da3_depth:         {sample['da3_depth'].shape} (dtype: {sample['da3_depth'].dtype})")
                else:
                    print(f"  da3_depth:         None")
                if sample['da3_conf'] is not None:
                    print(f"  da3_conf:          {sample['da3_conf'].shape} (dtype: {sample['da3_conf'].dtype})")
                else:
                    print(f"  da3_conf:          None")
                print(f"  filename:          {sample['filename']}")
                
                # Print memory size information
                print("\n【Memory Size】")
                imgs_size = sample['imgs'].nbytes / (1024 ** 2)
                print(f"  imgs:              {imgs_size:.2f} MB")
                print(f"  proj_matrices:     {sample['proj_matrices'].nbytes / 1024:.2f} KB")
                print(f"  depth_values:      {sample['depth_values'].nbytes / 1024:.2f} KB")
                if sample['depth'] is not None:
                    print(f"  depth:             {sample['depth'].nbytes / (1024 ** 2):.2f} MB")
                if sample['mask'] is not None:
                    print(f"  mask:              {sample['mask'].nbytes / (1024 ** 2):.2f} MB")
                if sample['da3_depth'] is not None:
                    print(f"  da3_depth:         {sample['da3_depth'].nbytes / (1024 ** 2):.2f} MB")
                if sample['da3_conf'] is not None:
                    print(f"  da3_conf:          {sample['da3_conf'].nbytes / (1024 ** 2):.2f} MB")
                
                # Print value statistics
                print("\n【Value Statistics】")
                print(f"  imgs:              min={sample['imgs'].min():.4f}, max={sample['imgs'].max():.4f}, mean={sample['imgs'].mean():.4f}")
                print(f"  depth_values:      min={sample['depth_values'].min():.4f}, max={sample['depth_values'].max():.4f}")
                if sample['depth'] is not None:
                    valid_depth = sample['depth'][sample['depth'] > 0]
                    if len(valid_depth) > 0:
                        print(f"  depth:             min={valid_depth.min():.4f}, max={valid_depth.max():.4f}, mean={valid_depth.mean():.4f}")
                    else:
                        print(f"  depth:             (all zeros or invalid)")
                if sample['mask'] is not None:
                    print(f"  mask:              min={sample['mask'].min():.4f}, max={sample['mask'].max():.4f}, mean={sample['mask'].mean():.4f}")
                if sample['da3_depth'] is not None:
                    valid_da3 = sample['da3_depth'][sample['da3_depth'] > 0]
                    if len(valid_da3) > 0:
                        print(f"  da3_depth:         min={valid_da3.min():.4f}, max={valid_da3.max():.4f}, mean={valid_da3.mean():.4f}")
                if sample['da3_conf'] is not None:
                    print(f"  da3_conf:          min={sample['da3_conf'].min():.4f}, max={sample['da3_conf'].max():.4f}, mean={sample['da3_conf'].mean():.4f}")
                
                # Visualization
                if m_show:
                    print("\n【Visualization】")
                    num_views = sample['imgs'].shape[0]
                    num_cols = min(num_views, 3)
                    num_rows = 2 + (1 if sample['depth'] is not None else 0) + (1 if sample['da3_depth'] is not None else 0)
                    
                    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 4 * num_rows))
                    if num_rows == 1:
                        axes = axes.reshape(1, -1)
                    
                    # Display images
                    for view_idx in range(min(num_views, num_cols)):
                        img = sample['imgs'][view_idx].transpose(1, 2, 0)
                        axes[0, view_idx].imshow(np.clip(img, 0, 1))
                        axes[0, view_idx].set_title(f'Image View {view_idx}')
                        axes[0, view_idx].axis('off')
                    
                    # Display mask
                    for view_idx in range(num_cols):
                        if view_idx == 0 and sample['mask'] is not None:
                            axes[1, view_idx].imshow(sample['mask'], cmap='gray')
                            axes[1, view_idx].set_title('Mask')
                        else:
                            axes[1, view_idx].axis('off')
                    
                    # Display depth
                    row_idx = 2
                    if sample['depth'] is not None:
                        im = axes[row_idx, 0].imshow(sample['depth'], cmap='viridis')
                        axes[row_idx, 0].set_title('Depth Map')
                        plt.colorbar(im, ax=axes[row_idx, 0])
                        for view_idx in range(1, num_cols):
                            axes[row_idx, view_idx].axis('off')
                        row_idx += 1
                    
                    # Display DA3 depth
                    if sample['da3_depth'] is not None:
                        im = axes[row_idx, 0].imshow(sample['da3_depth'], cmap='viridis')
                        axes[row_idx, 0].set_title('DA3 Depth')
                        plt.colorbar(im, ax=axes[row_idx, 0])
                        for view_idx in range(1, num_cols):
                            axes[row_idx, view_idx].axis('off')
                    
                    plt.tight_layout()
                    plt.savefig(f'test_sample_{sample_idx}.png', dpi=100, bbox_inches='tight')
                    print(f"  ✓ Visualization saved to: test_sample_{sample_idx}.png")
                    plt.show()
                    
            except Exception as e:
                print(f"✗ Error loading sample {sample_idx}: {str(e)}")
                import traceback
                traceback.print_exc()
        
        print("\n" + "=" * 80)
        print("✓ Test completed successfully!")
        print("=" * 80)
        
    except Exception as e:
        print(f"✗ Error initializing dataset: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
