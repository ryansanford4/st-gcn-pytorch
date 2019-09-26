import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

from st_gcn import Model
from gwnet import gwnet
from P3D import P3D199, get_optim_policies

class GGCN(nn.Module):
	def __init__(self, cfg):
		super(GGCN, self).__init__()

		self.cfg = cfg

		self.p3d = P3D199(pretrained=True, num_classes=cfg.num_activities, dropout=cfg.train_dropout_prob, config=cfg)
		self.p3d.apply(get_optim_policies)
		self.maxpool2d = nn.AdaptiveMaxPool2d((None, 1))
		self.avgpool = nn.AdaptiveMaxPool1d(1)
		self.fc_drop = nn.Sequential(nn.Linear(2048*self.cfg.crop_size[0]*self.cfg.crop_size[1], 1024), nn.ReLU(), nn.Dropout(cfg.train_dropout_prob))
		self.fc_cls = nn.Linear(1024, cfg.num_activities)
		# self.gcn = Model(in_channels=2048, num_class=cfg.num_activities, edge_importance_weighting=False)
		self.gcn = gwnet(device=torch.device('cuda'), num_nodes=cfg.num_boxes, in_dim=1024, out_dim=cfg.num_activities, blocks=2)
	
	def forward(self, x, bbox_in):
		roi_align = self.p3d(x, bbox_in)
		if self.cfg.stgcn:
			roi_align = torch.reshape(roi_align, (-1, 2048*self.cfg.crop_size[0]*self.cfg.crop_size[1]))
			roi_align = self.fc_drop(roi_align)
			roi_align = torch.reshape(roi_align, (-1, self.cfg.num_frames, self.cfg.num_boxes, 1024))
			roi_align = roi_align.permute(0, 3, 2, 1)
			output = self.gcn(roi_align)
			# output = self.gcn(roi_align, {"bbox_in": bbox_in, "strategy": 'uniform'})
		elif self.cfg.roi_align:
			# output from p3d is [B, C, N*T, H, W]
			roi_align = torch.reshape(roi_align, (-1, 2048*self.cfg.crop_size[0]*self.cfg.crop_size[1]))
			roi_align = self.fc_drop(roi_align)
			roi_align = torch.reshape(roi_align, (-1, self.cfg.num_frames, self.cfg.num_boxes, 1024))
			roi_align, _ = torch.max(roi_align, dim=2)
			roi_align = self.fc_cls(roi_align)
			roi_align = roi_align.permute(0, 2, 1)
			output = torch.mean(roi_align, dim=-1)
		else:
			output = roi_align
		return output