import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import DeformConv2d 

from lidm.utils.lidar_utils import pcd2range_gpu

class Box2Condition(nn.Module):
    def __init__(self, config, kernel_size=3):
        super(Box2Condition, self).__init__()
        self.config = config
        self.box_range = torch.Tensor(config.box_range).cuda()
        self.fov = torch.Tensor(config.fov).cuda()
        size = config.size
        in_channels = config.in_channels + 7
        out_channels = config.out_channels

        self.size = torch.Tensor([size_factor / config.project_ds_scale for size_factor in size]).cuda()
        self.up_factor = config.up_factor
        self.channel_reduce = nn.Conv2d(in_channels, 128, kernel_size=1)
        self.offset_conv = nn.Conv2d(128, 2 * kernel_size * kernel_size, 
                                     kernel_size=3, padding=1)
        self.deform_conv = DeformConv2d(128, out_channels, kernel_size=3, padding=1)
        self.upsample = nn.ConvTranspose2d(
            out_channels,
            out_channels,
            kernel_size=2 * self.up_factor,
            stride=self.up_factor,
            padding=self.up_factor // 2
        )
    
    def rescale_box(self, boxes, features):
        condition_feature = []
        idx_row = torch.where(boxes[:,0] == -1)[0]
        x_min,y_min,z_min,x_max,y_max,z_max = self.box_range
        boxes[1:,0] = boxes[1:,0] * (x_max - x_min) + x_min
        boxes[1:,1] = boxes[1:,1] * (y_max - y_min) + y_min
        boxes[1:,2] = boxes[1:,2] * (z_max - z_min) + z_min
        points = boxes[:,:3]
        # points --> range
        for i in range(idx_row.shape[0]):
            if i == idx_row.shape[0] - 1:
                batch_points = points[(idx_row[i]+1):, :]
                batch_features = features[(idx_row[i]+1):, :]
            else:
                batch_points = points[(idx_row[i]+1):(idx_row[i+1]), :]
                batch_features = features[(idx_row[i]+1):(idx_row[i+1]), :]
            batch_features  = pcd2range_gpu(
                pcd = batch_points,
                size= self.size,
                fov = self.fov,
                remission=batch_features,
            )
            condition_feature.append(batch_features.permute(2,0,1).unsqueeze(dim=0))

        return torch.cat(condition_feature, dim=0)

    def project_box2range(self, dec_boxes, dec_angles, features):
        features = torch.cat([features, dec_boxes, dec_angles.unsqueeze(-1)],dim=-1)
        condition_feature_map = self.rescale_box(dec_boxes, features)
        return condition_feature_map

    def forward(self, dec_boxes, dec_angles, features):
        condition_feature_map = self.project_box2range(dec_boxes, dec_angles, features)
        condition_feature_map = self.channel_reduce(condition_feature_map) 
        offsets = self.offset_conv(condition_feature_map)               # shape: [N, 2*K*K, H, W]
        x_prop = self.deform_conv(condition_feature_map, offsets)       # shape: [N, out_channels, H, W]
        out = self.upsample(x_prop)                 # shape: [N, out_channels, 2H, 2W]
        return out
