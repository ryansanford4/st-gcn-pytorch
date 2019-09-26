import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

from st_gcn import Model
from gwnet import gwnet
from resnet import resnet50

class GGCN(nn.Module):
	def __init__(self, cfg):
		super(GGCN, self).__init__()

		self.cfg = cfg

		self.resnet_model = resnet50(pretrained=False, num_classes=cfg.num_activities)
		self.stgcn = Model(in_channels=1024, num_class=cfg.num_activities, edge_importance_weighting=False)
		self.gwnet = gwnet(device=torch.device('cuda'), num_nodes=cfg.num_boxes, in_dim=1024, out_dim=cfg.num_activities, blocks=2)
		self.fc_emb = nn.Sequential(nn.Linear(2048, 1024), nn.ReLU(), nn.Dropout(cfg.train_dropout_prob))
		self.fc_cls = nn.Linear(1024, cfg.num_activities)
		self.avgpool = nn.AdaptiveAvgPool1d(1)
	
	def forward(self, x):
		if len(x) == 1:
			x, bbox_coords = x[0], x[1]
		x = self.resnet_model(x)
		x = self.fc_emb(x)
		if self.cfg.stgcn:
			x = torch.reshape(x, (-1, 1024, self.cfg.num_frames, self.cfg.num_boxes))
			output = self.stgcn(x, {"bbox_in": bbox_coords, "strategy": 'uniform'})
		elif self.cfg.gwnet:
			x = torch.reshape(x, (-1, 1024, self.cfg.num_boxes, self.cfg.num_frames))
			output = self.gwnet(x)
		else:
			x = torch.reshape(x, (-1, self.cfg.num_frames, self.cfg.num_boxes, 1024))
			# Find max activation in each bbox
			x, _ = torch.max(x, dim=2)
			x = self.fc_cls(x)
			x = x.permute(0, 2, 1)
			output = torch.squeeze(self.avgpool(x), dim=-1)
		return output