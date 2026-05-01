# Copyright (c) OpenRobotLab. All rights reserved.

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import build_conv_layer
from mmengine.model import BaseModule
from torch import Tensor

from embodiedscan.models.losses.occ_loss import (geo_scal_loss,
                                                 occ_multiscale_supervision,
                                                 sem_scal_loss)
from embodiedscan.registry import MODELS
from embodiedscan.utils.typing_config import SampleList

from .cross_att import CrossSwin3DBlock


@MODELS.register_module()
class ImVoxelOccHead_drocc_imgq(BaseModule):
    """Occupancy prediction head compatible with ImVoxelNeck outputs.

    Args:
        num_classes (int): Number of categories. Defaults to 21.
        volume_h (int): Size along h of the 3D volume. Defaults to 40.
        volume_w (int): Size along w of the 3D volume. Defaults to 40.
        volume_z (int): Size along z of the 3D volume. Defaults to 16.
        in_channels (int): Input channels. Defaults to 128.
        use_semantic (bool): Whether to use semantic predictions.
            Defaults to True.
    """

    def __init__(self,
                 *args,
                 num_classes=21,
                 volume_h=40,
                 volume_w=40,
                 volume_z=16,
                 in_channels=[256,256,256],
                 use_semantic=True,
                 swin_num_heads=[8, 8, 8],
                 window_size=[[4, 4, 4], [4, 4, 4], [2, 2, 2]],
                 shift=[(0, 0, 0), (0, 0, 0), (0, 0, 0)],
                 **kwargs):
        super(ImVoxelOccHead_drocc_imgq, self).__init__()
        self.num_classes = num_classes
        self.volume_h = volume_h
        self.volume_w = volume_w
        self.volume_z = volume_z

        self.swin_num_heads = swin_num_heads
        self.window_size = window_size
        self.shift = shift

        self.in_channels = in_channels
        self.use_semantic = use_semantic

        self._init_layers()

    def _init_layers(self):
        conv_cfg = dict(type='Conv3d', bias=False)
        self.occ = nn.ModuleList()

        self.imgcroatt = nn.ModuleList()


        self.upconvs = nn.ModuleList()

        for i in range(len(self.in_channels)):
            imgcroatt = CrossSwin3DBlock(dim=self.in_channels[i], num_heads=self.swin_num_heads[i],
                                         window_size=self.window_size[i],
                                         shift=self.shift[i])
            self.imgcroatt.append(imgcroatt)


        for i in range(len(self.in_channels) - 1):
            upconv = nn.ConvTranspose3d(in_channels=self.in_channels[0], out_channels=self.in_channels[0], kernel_size=4, stride=2, padding=1)
            self.upconvs.append(upconv)

        for i in range(len(self.in_channels)):
            if self.use_semantic:
                occ = build_conv_layer(conv_cfg,
                                       in_channels=self.in_channels[i],
                                       out_channels=self.num_classes,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0)
                self.occ.append(occ)
            else:
                occ = build_conv_layer(conv_cfg,
                                       in_channels=self.in_channels[i],
                                       out_channels=1,
                                       kernel_size=1,
                                       stride=1,
                                       padding=0)
                self.occ.append(occ)

    def forward(self, img_volumes, point_volumes, input_metas):
        """Forward function.

        Args:
            mlvl_feats (list[Tensor]): Multi-level features.
            input_metas (list[dict]): Input meta infos.

        Returns:
            list[Tensor]: Occupancy predicted maps.
        """
        occ_preds = []

        imgouts = []

        scales = len(img_volumes)

        for i in range(scales):
            imgout = self.imgcroatt[i](img_volumes[i], point_volumes[i])

            imgouts.append(imgout)

        for i in range(scales):
            occ_preds.append(self.occ[i](imgouts[i]))

        return occ_preds

    def predict(self, img_volumes, point_volumes, batch_data_samples: SampleList):
        """Predict/Inference function.

        Args:
            x (Tuple[Tensor]): Multi-level features.
            batch_data_samples (`SampleList`): Batch of data samples.

        Returns:
            Tensor: Occupancy predictions.
        """
        batch_input_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        pred = self.forward(img_volumes, point_volumes, batch_input_metas)[0]
        if self.use_semantic:
            _, pred_occ = torch.max(torch.softmax(pred, dim=1), dim=1)
        else:
            pred_occ = torch.sigmoid(pred[:, 0])
        return pred_occ

    def loss(self, img_volumes, point_volumes, batch_data_samples: SampleList):
        """Loss function..

        Args:
            x (Tuple[Tensor]): Multi-level features.
            batch_data_samples (`SampleList`): Batch of data samples.

        Returns:
            Dict: Multi-scale occupancy prediction losses.
        """
        batch_input_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]
        occ_preds = self.forward(img_volumes, point_volumes, batch_input_metas)

        """
        (Pdb) p occ_preds[0].shape
        torch.Size([1, 81, 40, 40, 16])
        (Pdb) p occ_preds[1].shape
        torch.Size([1, 81, 20, 20, 8])
        (Pdb) p occ_preds[2].shape
        torch.Size([1, 81, 10, 10, 4])
        """
        gt_occupancy = [
            data_samples.gt_occupancy for data_samples in batch_data_samples
        ]  # B * N * 4, xyz+label, TODO: put data in load process
        gt_occupancy_masks = None
        if 'gt_occupancy_masks' in batch_data_samples[0]:
            gt_occupancy_masks = [
                data_samples.gt_occupancy_masks
                for data_samples in batch_data_samples
            ]

        if not self.use_semantic:
            loss_dict = {}
            for i in range(len(occ_preds)):
                pred = occ_preds[i][:, 0]
                ratio = 2 ** i
                # downsample occ_masks accordingly
                if gt_occupancy_masks is not None:
                    pooling = nn.MaxPool3d(ratio, stride=ratio)
                    pooled_masks = []
                    for mask in gt_occupancy_masks:
                        float_mask = mask.float()[None]
                        pooled_masks.append(pooling(float_mask)[0].bool())
                else:
                    pooled_masks = gt_occupancy_masks
                gt = occ_multiscale_supervision(gt_occupancy, ratio,
                                                occ_preds[i].shape,
                                                pooled_masks)
                loss_occ_i = (F.binary_cross_entropy_with_logits(pred, gt) +
                              geo_scal_loss(pred, gt.long(), semantic=False))
                loss_occ_i = loss_occ_i * ((0.5) ** i)
                loss_dict['loss_occ_{}'.format(i)] = loss_occ_i

        else:
            pred = occ_preds
            criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
            loss_dict = {}

            for i in range(len(occ_preds)):
                pred = occ_preds[i]
                ratio = 2 ** i
                # downsample occ_masks accordingly
                if gt_occupancy_masks is not None:
                    # a little hack: use maxpool to achieve downsample
                    pooling = nn.MaxPool3d(ratio, stride=ratio)
                    pooled_masks = []
                    for mask in gt_occupancy_masks:
                        float_mask = mask.float()[None]
                        pooled_masks.append(pooling(float_mask)[0].bool())
                else:
                    pooled_masks = gt_occupancy_masks
                gt = occ_multiscale_supervision(gt_occupancy, ratio,
                                                occ_preds[i].shape,
                                                pooled_masks)
                loss_occ_i = (criterion(pred, gt.long()) +
                              sem_scal_loss(pred, gt.long()) +
                              geo_scal_loss(pred, gt.long()))
                loss_occ_i = loss_occ_i * ((0.5) ** i)
                loss_dict['loss_occ_{}'.format(i)] = loss_occ_i

        return loss_dict
