
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
class PCAS(nn.Module):
    def __init__(self, backbone, use_original_imgsize = False):
        super(PCAS, self).__init__()
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
            overlap_mask = support_mask + query_mask
            overlap_mask[overlap_mask!=0] = 1
            support_target = support_mask.unsqueeze(1) * support_img
            overlap_img = query_img.clone()
            overlap_img[support_target!=0] = 0
            overlap_img += support_target
            query_feats = self.extract_feats(query_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
            query_feats_mask_ = self.mask_feature(query_feats, query_mask.clone())
            support_feats1 = self.extract_feats(support_img, self.backbone, self.feat_ids, self.bottleneck_ids, self.lids)
            support_feats1_mask_ = self.mask_feature(support_feats1, support_mask.clone())
            overlap_img_feature_aux = support_feats2
            corr1 = Correlation.multilayer_correlation(query_feats, support_feats1_mask_, self.stack_ids)
        logit_mask1 = self.hpn_learner(corr1)
        pesudo_mask1 = F.interpolate(logit_mask1, support_img.size()[2:], mode='bilinear', align_corners=True)
        pesudo_mask1 = pesudo_mask1.argmax(dim=1)
        if epoch >=1 or training is True:
            pesudo_mask1_idx = pesudo_mask1.clone()
            support_mask_idx = support_mask.clone()
            b, h, w = pesudo_mask1.shape 
            median_p = []
            median_s = []
            b_pidxs, h_pidxs, w_pidxs = torch.where(pesudo_mask1_idx.cpu()==1)
            b_sidxs, h_sidxs, w_sidxs = torch.where(support_mask_idx.cpu() == 1)
            for i in range(b):
                p_mh = np.median(h_pidxs[b_pidxs==i].numpy())
                p_mw = np.median(w_pidxs[b_pidxs == i].numpy())
                median_p.append([p_mh,p_mw])
                s_mh = np.median(h_sidxs[b_sidxs == i].numpy())
                s_mw = np.median(w_sidxs[b_sidxs == i].numpy())
                median_s.append([s_mh, s_mw])
                h_distance = np.abs(p_mh - s_mh)
                w_distance = np.abs(p_mw - s_mw)
                shift_size_h = 0
                shift_size_w = 0
                if h_distance < h/3.0:
                    shift_size_h = int(h_distance + h/3.0)
                    if s_mh > h/2.0 and p_mh>h/2.0:
                        shift_size_h = -np.min(h_sidxs[b_sidxs == i].numpy())
                    else:
                        shift_size_h = h - np.max(h_sidxs[b_sidxs == i].numpy())
                if w_distance < w/3.0:
                    shift_size_w = int(w_distance + w/3.0)
                    if s_mw >w/2.0 and p_mw > w/2.0:
                        shift_size_w = -np.min(w_sidxs[b_sidxs == i].numpy())
                    else:
                        shift_size_w = w - np.max(w_sidxs[b_sidxs == i].numpy())
                shift_size_h = int(shift_size_h)
                shift_size_w = int(shift_size_w)
                next_s_img = torch.roll(support_img[i,...], shifts=(shift_size_h, shift_size_w), dims=(1, 2))
                next_s_mask = torch.roll(support_mask[i,...], shifts=(shift_size_h, shift_size_w), dims=(0,1))
                support_img[i,...]  = next_s_img
                support_mask[i,...] = next_s_mask
        mask_multi = support_mask * pesudo_mask1
        query_multi_target =mask_multi.unsqueeze(1) * query_img
        pesudo_mask1 = support_mask + pesudo_mask1
        pesudo_mask1[pesudo_mask1!=0] = 1
        with torch.no_grad():
            support_target = support_mask.unsqueeze(1) * support_img
            overlap_img = query_img.clone()
            overlap_img[support_target!=0] = 0
            overlap_img += support_target
            overlap_img[query_multi_target!=0]=0
            overlap_img += query_multi_target
            support_feats2 = self.extract_feats(overlap_img, self.backbone, self.feat_ids, self.bottleneck_ids,
                                               self.lids)
            overlap_img_feature = support_feats2
            support_feats3_mask_ = self.mask_feature(overlap_img_feature, pesudo_mask1.clone())
            corr3 = Correlation.multilayer_correlation(query_feats, support_feats3_mask_, self.stack_ids)
        logit_mask2 = self.hpn_learner(corr3)
        if not self.use_original_imgsize:
            logit_mask1 = F.interpolate(logit_mask1, support_img.size()[2:], mode='bilinear', align_corners=True)
            logit_mask2 = F.interpolate(logit_mask2, support_img.size()[2:], mode='bilinear', align_corners=True)
        if training: 
            with torch.no_grad():
                corr4 = Correlation.multilayer_correlation(overlap_img_feature_aux, query_feats_mask_, self.stack_ids)
                corr5 = Correlation.multilayer_correlation(overlap_img_feature_aux, support_feats1_mask_, self.stack_ids)
            logit_mask3 = self.hpn_learner(corr4, corr5)
            if not self.use_original_imgsize:
                logit_mask3 = F.interpolate(logit_mask3, support_img.size()[2:], mode='bilinear', align_corners=True)
        if training:
            loss = self.compute_objective(logit_mask1,query_mask) + self.compute_objective(logit_mask2,query_mask) + self.compute_objective(logit_mask3,overlap_mask)
        else:
            loss = self.compute_objective(logit_mask1, query_mask) + self.compute_objective(logit_mask2, query_mask)
        return loss,logit_mask2, logit_mask1
    def mask_feature(self, features, support_mask):
        res_features = []
        for idx, feature in enumerate(features):
            mask = F.interpolate(support_mask.unsqueeze(1).float(), feature.size()[2:], mode='bilinear', align_corners=True)
            res_features.append(features[idx] * mask)
        return res_features
    def predict_mask_nshot(self, batch, nshot):
        logit_mask_agg = 0
        pesudo_mask_agg = 0
        for s_idx in range(nshot):
            current_s_img, current_s_mask = batch['support_imgs'][:, s_idx], batch['support_masks'][:, s_idx]
            h,w = current_s_img.shape[2:]
            loss, logit_mask, pesudo_mask = self(batch['query_img'], current_s_img,
                                                 current_s_mask, batch['query_mask'], False,1)
            if self.use_original_imgsize:
                org_qry_imsize = tuple([batch['org_query_imsize'][1].item(), batch['org_query_imsize'][0].item()])
                logit_mask = F.interpolate(logit_mask, org_qry_imsize, mode='bilinear', align_corners=True)
                pesudo_mask = F.interpolate(pesudo_mask, org_qry_imsize, mode='bilinear', align_corners=True)
            pesudo_mask_agg += pesudo_mask.argmax(dim=1).clone()
            logit_mask_agg += logit_mask.argmax(dim=1).clone()
            if nshot == 1: return logit_mask_agg,pesudo_mask_agg
        bsz = logit_mask_agg.size(0)
        max_vote = logit_mask_agg.view(bsz, -1).max(dim=1)[0]
        max_vote = torch.stack([max_vote, torch.ones_like(max_vote).long()])
        max_vote = max_vote.max(dim=0)[0].view(bsz, 1, 1)
        pred_mask = logit_mask_agg.float() / max_vote
        pred_mask[pred_mask < 0.5] = 0
        pred_mask[pred_mask >= 0.5] = 1
        bsz = pesudo_mask_agg.size(0)
        max_vote = pesudo_mask_agg.view(bsz, -1).max(dim=1)[0]
        max_vote = torch.stack([max_vote, torch.ones_like(max_vote).long()])
        max_vote = max_vote.max(dim=0)[0].view(bsz, 1, 1)
        pesudo_mask = pesudo_mask_agg.float() / max_vote
        pesudo_mask[pesudo_mask < 0.5] = 0
        pesudo_mask[pesudo_mask >= 0.5] = 1
        return pred_mask,pesudo_mask
    def compute_objective(self, logit_mask, gt_mask):
        bsz = logit_mask.size(0)
        logit_mask = logit_mask.view(bsz, 2, -1)
        gt_mask = gt_mask.view(bsz, -1).long()
        return self.cross_entropy_loss(logit_mask, gt_mask)
    def train_mode(self):
        self.train()
        self.backbone.eval()
