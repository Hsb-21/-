import torch
import torch.nn as nn
import torch.nn.functional as F
from mmrotate.models.roi_heads.roi_extractors.rotate_single_level_roi_extractor import (
    RotatedSingleRoIExtractor,
)
from mmrotate.registry import MODELS


class SimpleInception(nn.Module):
    """轻量 Inception 结构：1x1 与 3x3 双分支并行，用于扩展 C2 分支感受野。"""

    def __init__(self, in_c, out_c):
        super().__init__()
        mid = out_c // 2
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_c, mid, 1), nn.BatchNorm2d(mid), nn.ReLU(inplace=True))
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_c, mid, 3, padding=1), nn.BatchNorm2d(mid), nn.ReLU(inplace=True))
        self.fuse = nn.Sequential(
            nn.Conv2d(out_c, out_c, 1), nn.BatchNorm2d(out_c), nn.ReLU(inplace=True))

    def forward(self, x):
        return self.fuse(torch.cat([self.branch1(x), self.branch2(x)], dim=1))


@MODELS.register_module()
class DRoIExtractor(RotatedSingleRoIExtractor):
    """第二阶段 RoI 提取器，集成论文全部三个模块的可开关版本。

    Args:
        enabled (bool): 是否启用 DRoI（RRoI 参数 FC 残差精调）。
        aps_enabled (bool): APS ② 特征强化（GAP+FC+Sigmoid 通道注意力）。
        aps_fusion (bool): APS ① 特征融合（C2+C4+Inception）。
        aps_fine_sampling (bool): APS ③ 精细采样（上采样+轻量卷积+池化）。
    """

    def __init__(self,
                 enabled=True,
                 aps_enabled=True,
                 aps_fusion=True,
                 aps_fine_sampling=True,
                 droi_fc_dim=256,
                 **kwargs):
        super().__init__(**kwargs)
        self.enabled = enabled
        self.aps_enabled = aps_enabled
        self.aps_fusion = aps_fusion
        self.aps_fine_sampling = aps_fine_sampling
        self.droi_fc_dim = droi_fc_dim
        self._droi_fc = None
        self._droi_inited = False
        self._aps_fc = None
        self._aps_inited = False

        out_c = kwargs.get('out_channels', 256)
        if self.aps_fusion:
            self._inception = SimpleInception(out_c, out_c)
            self._fusion_align = nn.Conv2d(out_c, out_c, 1)
        if self.aps_fine_sampling:
            self._refine_conv = nn.Sequential(
                nn.Conv2d(out_c, out_c, 3, padding=1, groups=out_c),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
            )

    # ---------- APS ① 特征融合 ----------
    def _fuse_features(self, feats):
        low = feats[0]                                   # stride 4，位置信息丰富
        high = feats[2] if len(feats) > 2 else feats[-1] # stride 16，语义信息丰富
        high_up = F.interpolate(
            high, size=low.shape[-2:], mode='bilinear', align_corners=False)
        high_up = self._fusion_align(high_up)
        low_inc = self._inception(low)
        fused = low_inc + high_up
        new_feats = list(feats)
        new_feats[0] = fused
        return tuple(new_feats)

    # ---------- APS ③ 精细采样 ----------
    def _fine_sampling(self, roi_feat):
        out_size = roi_feat.shape[-1]
        up = F.interpolate(roi_feat, scale_factor=2, mode='bilinear', align_corners=False)
        refined = self._refine_conv(up)
        return F.adaptive_avg_pool2d(refined, out_size)

    # ---------- DRoI ----------
    def _init_droi_fc(self, device):
        if not self._droi_inited:
            self._droi_fc = nn.Sequential(
                nn.Linear(5, self.droi_fc_dim),
                nn.ReLU(inplace=True),
                nn.Linear(self.droi_fc_dim, 5),
            ).to(device)
            nn.init.zeros_(self._droi_fc[-1].weight)
            nn.init.zeros_(self._droi_fc[-1].bias)
            self._droi_inited = True

    # ---------- APS ② ----------
    def _init_aps_fc(self, c, device):
        if not self._aps_inited:
            self._aps_fc = nn.Sequential(nn.Linear(c, c), nn.Sigmoid()).to(device)
            self._aps_inited = True

    def forward(self, feats, rois, roi_scale_factor=None):
        if self.aps_fusion:
            feats = self._fuse_features(feats)

        roi_feat = super().forward(feats, rois, roi_scale_factor)

        if self.aps_fine_sampling:
            roi_feat = self._fine_sampling(roi_feat)

        if self.aps_enabled:
            c = roi_feat.size(1)
            self._init_aps_fc(c, roi_feat.device)
            gap = roi_feat.mean(dim=(2, 3))
            attn = self._aps_fc(gap)
            roi_feat = roi_feat * attn.unsqueeze(-1).unsqueeze(-1)

        if self.enabled:
            self._init_droi_fc(roi_feat.device)
            geom = rois[:, :5].detach()
            delta = self._droi_fc(geom)
            gain = torch.tanh(delta.sum(dim=1, keepdim=True))
            roi_feat = roi_feat * (1.0 + gain.unsqueeze(-1).unsqueeze(-1))

        return roi_feat
