from functools import reduce
from operator import add

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet
from torchvision.models import vgg

from .base.feature import extract_feat_vgg, extract_feat_res
from .base.correlation import Correlation
from .leaner import HPNLearner
import numpy as np
import cv2
class ASNet(nn.Module):
    def __init__(self, backbone, use_original_imgsize = False):
        super(ASNet, self).__init__()

        # 1. Backbone network initialization
        self.backbone_type = backbone
        self.use_original_imgsize = use_original_imgsize
        if backbone == 'vgg16':
            self.backbone = vgg.vgg16(pretrained=True)
            self.feat_ids = [17, 19, 21, 24, 26, 28, 30]
            self.extract_feats = extract_feat_vgg
            nbottlenecks = [2, 2, 3, 3, 3, 1]
        elif backbone == 'resnet50':
            self.backbone = resnet.resnet50(pretrained=True)
            self.feat_ids = list(range(4, 17))
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 6, 3]
        elif backbone == 'resnet101':
            self.backbone = resnet.resnet101(pretrained=True)
            self.feat_ids = list(range(4, 34))
            self.extract_feats = extract_feat_res
            nbottlenecks = [3, 4, 23, 3]
        else:
            raise Exception('Unavailable backbone: %s' % backbone)

        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), nbottlenecks)))
        self.lids = reduce(add, [[i + 1] * x for i, x in enumerate(nbottlenecks)])
        self.stack_ids = torch.tensor(self.lids).bincount().__reversed__().cumsum(dim=0)[:3]
        self.backbone.eval()
        self.hpn_learner = HPNLearner(list(reversed(nbottlenecks[-3:])))
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, query_img, support_img, support_mask,query_mask,training,epoch):
        with torch.no_grad():
