from roi_align.roi_align import RoIAlign
from roi_align.roi_align import CropAndResize
import torch

def run_roi_align(frame_feature_map, bbox_in, crop_size=(7, 7)):
    roi_align = RoIAlign(crop_size[0], crop_size[1])

    B, C, H, W = frame_feature_map.size()
    input_size = bbox_in.size()
    boxes_in_flat = torch.reshape(bbox_in, (-1, 4))  #B*T*N, 4

    boxes_idx = [i * torch.ones(input_size[2], dtype=torch.int)   for i in range(input_size[0]*input_size[1]) ]
    boxes_idx = torch.stack(boxes_idx).to(device=bbox_in.device)  # B*T, N
    boxes_idx_flat = torch.reshape(boxes_idx, (input_size[0]*input_size[1]*input_size[2],))  #B*T*N,
    del boxes_idx
    boxes_in_flat.requires_grad = False
    boxes_idx_flat.requires_grad = False

    # RoI Align
    boxes_features = roi_align(frame_feature_map,
                                        boxes_in_flat,
                                        boxes_idx_flat)
    del boxes_in_flat
    boxes_features = torch.reshape(boxes_features, (input_size[0], C, -1, crop_size[0], crop_size[1]))
    return boxes_features
