import os
import json
import numpy as np
from lidm.dataset.base import DatasetBase
from ..utils.lidar_utils import pcd2range, pcd2coord2d, range2pcd

class nuScenesBase(DatasetBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.dataset_name = 'nuScenes'
        self.num_sem_cats = kwargs['dataset_config'].num_sem_cats + 1 # 17
        self.return_remission = (kwargs['dataset_config'].num_channels == 2)

    @staticmethod
    def load_lidar_sweep(path):
        scan = np.fromfile(path, dtype=np.float32)
        scan = scan.reshape((-1, 5))
        points = scan[:, 0:4]  # get xyz & intensity
        points[:, 3] = points[:, 3] / 255.0
        return points
    
    def load_semantic_map(self, path, pcd):
        raise NotImplementedError

    def load_camera(self, path):
        raise NotImplementedError

    def get_pth_path(self, pts_path):
        return pts_path.replace('sweeps', 'sweeps_range').replace('.bin', '.pth')

    def process_remission(selef, range_feature):
        range_feature = np.clip(range_feature, 0, 1.0)
        range_feature = np.expand_dims(range_feature, axis = 0)
        return range_feature

    def __getitem__(self, idx):
        example = dict()
        data_path = self.data[idx]
        # lidar point cloud
        sweep = self.load_lidar_sweep(data_path)

        if self.lidar_transform:
            sweep, _ = self.lidar_transform(sweep, None)

        if self.condition_key == 'segmentation':
            # semantic maps
            proj_range, sem_map = self.load_semantic_map(data_path, sweep)
            example[self.condition_key] = sem_map
        else:
            proj_range, proj_feature = pcd2range(sweep[:,:3], self.img_size, self.fov, self.depth_range,remission=sweep[:,-1])
        proj_range, proj_mask = self.process_scan(proj_range)

        if self.return_remission:
            proj_feature = self.process_remission(proj_feature)
            proj_range = np.concatenate((proj_range, proj_feature), axis = 0)

        example['image'], example['mask'] = proj_range, proj_mask
        if self.return_pcd:
            reproj_sweep, _, _ = range2pcd(proj_range[0] * .5 + .5, self.fov, self.depth_range, self.depth_scale, self.log_scale)
            example['raw'] = sweep
            example['reproj'] = reproj_sweep.astype(np.float32)
        # image degradation
        if self.degradation_transform:
            degraded_proj_range = self.degradation_transform(proj_range)
            example['degraded_image'] = degraded_proj_range

        # cameras
        if self.condition_key == 'camera':
            cameras = self.load_camera(data_path)
            example[self.condition_key] = cameras
        return example

class nuScenesImageTrain(nuScenesBase):
    def __init__(self, **kwargs):
        super().__init__(split='train', **kwargs)

    def prepare_data(self):
        with open(os.path.join(self.data_root, 'v1.0-trainval/v1.0-trainval/sample_data.json')) as f:
            sample_data = json.load(f)

        custom_path = 'v1.0-trainval'
        file_paths = [os.path.join(self.data_root, custom_path, x['filename']) 
                           for x in sample_data 
                           if 'sweeps/LIDAR_TOP' in x['filename']]
        self.data = sorted(file_paths)

class nuScenesImageValidation(nuScenesBase):
    def __init__(self, **kwargs):
        super().__init__(split='val', **kwargs)

    def prepare_data(self):
        with open(os.path.join(self.data_root, 'v1.0-trainval/v1.0-mini/sample_data.json')) as f:
            sample_data = json.load(f)

        custom_path = 'v1.0-trainval'
        file_paths = [os.path.join(self.data_root, custom_path, x['filename']) 
                           for x in sample_data 
                           if 'sweeps/LIDAR_TOP' in x['filename']]
        self.data = sorted(file_paths)