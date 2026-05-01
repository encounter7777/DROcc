# Copyright (c) OpenRobotLab. All rights reserved.
from typing import List, Optional, Tuple, Union
import ipdb
import torch
from mmengine.structures import InstanceData

import torch.nn as nn

try:
    import MinkowskiEngine as ME
except ImportError:
    # Please follow getting_started.md to install MinkowskiEngine.
    ME = None
    pass

from mmengine.model import BaseModel

from embodiedscan.registry import MODELS, TASK_UTILS
from embodiedscan.structures.bbox_3d import get_proj_mat_by_coord_type
from embodiedscan.utils import ConfigType, OptConfigType
from embodiedscan.utils.typing_config import (ForwardResults, InstanceList,
                                              SampleList)

from ..layers.fusion_layers.point_fusion import (batch_point_sample,
                                                 point_sample)



@MODELS.register_module()
class DenseFusionOccPredictor_drocc_depth_only(BaseModel):
    """Dense Fusion framework for occupancy prediction.

    Args:
        backbone (:obj:`ConfigDict` or dict): The image backbone config.
        backbone_3d (:obj:`ConfigDict` or dict): The 3D backbone config.
        neck (:obj:`ConfigDict` or dict): The image neck config.
        neck_3d (:obj:`ConfigDict` or dict): The 3D neck config.
        bbox_head (:obj:`ConfigDict` or dict): The bbox head config.
        prior_generator (:obj:`ConfigDict` or dict): The prior grid generator
            config.
        n_voxels (list): Number of voxels along x, y, z axis.
        coord_type (str): The type of coordinates of points cloud:
            'DEPTH', 'LIDAR', or 'CAMERA'.
        use_valid_mask (bool): Whether to use valid masks to handle
            visible voxels. Defaults to False.
        use_xyz_feat (bool): Whether to use xyz features.
            Defaults to False.
        point_cloud_range (list]): Point cloud range, [x_min, y_min, z_min,
            x_max, y_max, z_max], e.g., [-3.2, -3.2, -0.78, 3.2, 3.2, 1.78].
        train_cfg (:obj:`ConfigDict` or dict, optional): Config dict of
            training hyper-parameters. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): Config dict of test
            hyper-parameters. Defaults to None.
        data_preprocessor (dict or ConfigDict, optional): The pre-process
            config of :class:`BaseDataPreprocessor`.  it usually includes,
                ``pad_size_divisor``, ``pad_value``, ``mean`` and ``std``.
        init_cfg (:obj:`ConfigDict` or dict, optional): The initialization
            config. Defaults to None.
    """

    def __init__(self,
                 backbone: ConfigType,
                 backbone_3d: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 prior_generator: ConfigType,
                 n_voxels: List,
                 coord_type: str,
                 use_valid_mask=True,
                 use_xyz_feat: bool = False,
                 point_cloud_range=None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptConfigType = None):
        super().__init__(data_preprocessor=data_preprocessor,
                         init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        self.backbone_3d = MODELS.build(backbone_3d)
        if neck is not None:
            self.neck = MODELS.build(neck)

        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = MODELS.build(bbox_head)
        self.n_voxels = n_voxels
        self.point_cloud_range = point_cloud_range
        prior_range = prior_generator['ranges'][0]
        if backbone_3d['type'] == 'MinkResNet':
            self.voxel_stride = 2 ** 6
        else:
            self.voxel_stride = 1
        self.voxel_size = [(prior_range[3] - prior_range[0]) /
                           self.n_voxels[0] / self.voxel_stride,
                           (prior_range[4] - prior_range[1]) /
                           self.n_voxels[1] / self.voxel_stride,
                           (prior_range[5] - prior_range[2]) /
                           self.n_voxels[2] / self.voxel_stride]
        self.prior_generator = TASK_UTILS.build(prior_generator)
        self.coord_type = coord_type
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.use_valid_mask = use_valid_mask
        self.use_xyz_feat = use_xyz_feat

        if ME is None:
            raise ImportError(
                'Please follow `getting_started.md` to install MinkowskiEngine.`'  # noqa: E501
            )

        self.imgdown1 = nn.Conv3d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.imgdown2 = nn.Conv3d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1)

        self.pointch = nn.Conv3d(in_channels=512, out_channels=256, kernel_size=1, stride=1, padding=0)
        self.pointdown1 = nn.Conv3d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.pointdown2 = nn.Conv3d(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1)



    @property
    def with_neck(self):
        """Whether the detector has a 2D backbone."""
        return hasattr(self, 'neck') and self.neck is not None

    @property
    def with_neck_3d(self):
        """Whether the detector has a 3D neck."""
        return hasattr(self, 'neck_3d') and self.neck_3d is not None

    def extract_feat(self, batch_inputs_dict: dict,
                     batch_data_samples: SampleList):
        """Extract 3d features from the backbone -> fpn -> 3d projection.

        -> 3d neck -> bbox_head.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                the 'imgs' key.

                    - imgs (torch.Tensor, optional): Image of each sample.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            Tuple:
             - torch.Tensor: Features of shape (N, C_out, N_x, N_y, N_z).
             - torch.Tensor: Valid mask of shape (N, 1, N_x, N_y, N_z).
        """

        # 1. Extract the feature volume from images
        img = batch_inputs_dict['imgs']  # img.shape : torch.size([1,10,3,480,480])
        batch_img_metas = [
            data_samples.metainfo for data_samples in batch_data_samples
        ]

        """
        将batch_data_samples转化为dict
        ipdb> p len(batch_img_metas)
        1

        ipdb> p len(batch_img_metas[0]['depth2img']['extrinsic'])
        10
        ipdb> p batch_img_metas[0]['depth2img']['extrinsic'][0].shape
        (4, 4)
        ipdb> p batch_img_metas[0]['depth2img']['intrinsic'][0].shape
        (4, 4)
        """
        batch_size = img.shape[0]

        if len(img.shape) > 4:  # (B, n_views, C, H, W)
            img = img.reshape([-1] + list(img.shape)[2:])  # img.shape : torch.size([10,3,480,480])
            # 这行代码的作用是将图像数据的前两个维度（通常是批量大小和通道数）合并为一个维度

            x = self.backbone(img)
            """
            x: tuple  , len(x) : 4
            ipdb> p x[0].shape
            torch.Size([10, 256, 120, 120])
            ipdb> p x[1].shape
            torch.Size([10, 512, 60, 60])
            ipdb> p x[2].shape
            torch.Size([10, 1024, 30, 30])
            ipdb> p x[3].shape
            torch.Size([10, 2048, 15, 15])
            """

            x = self.neck(x)[0]
            """
            neck : mmdet.FPN
            x : torch.Size([10, 256, 120, 120])
            """

            x = x.reshape([batch_size, -1] + list(x.shape)[1:])  # x : torch.Size([1,10, 256, 120, 120])

        else:
            x = self.backbone(img)
            x = self.neck(x)[0]

        prior_points = self.prior_generator.grid_anchors(
            [self.n_voxels[::-1]], device=img.device)[0][:, :3]
        """
        ipdb> p prior_points.shape
        torch.Size([25600, 3])
        """

        if 'origin' in batch_img_metas[0]['depth2img'].keys():
            assert len(batch_img_metas) == 1, 'only support batch_size=1 here'
            prior_points += prior_points.new_tensor(
                batch_img_metas[0]['depth2img']['origin'])
            # For calibration with original ImVoxelNet implementation
            # prior_points += prior_points.new_tensor([-0.08, -0.08, -0.08])

        volumes, valid_preds = [], []
        for feature, img_meta in zip(x, batch_img_metas):
            """
            ipdb> p x.shape
            torch.Size([1, 10, 256, 120, 120])
            ipdb> p feature.shape
            torch.Size([10, 256, 120, 120])
            ipdb> p len(batch_img_metas)
            1
            """
            img_scale_factor = (prior_points.new_tensor(
                img_meta['scale_factor'][:2])
                                if 'scale_factor' in img_meta.keys() else 1)
            img_flip = img_meta['flip'] if 'flip' in img_meta.keys() else False
            img_crop_offset = (prior_points.new_tensor(
                img_meta['img_crop_offset'])
                               if 'img_crop_offset' in img_meta.keys() else 0)
            proj_mat = get_proj_mat_by_coord_type(img_meta, self.coord_type)
            # Multi-View ImVoxelNet
            """
            ipdb> p type(proj_mat)
            <class 'dict'>
            ipdb> p proj_mat
            {'extrinsic': [array([[ 0.97753465, -0.20978756, -0.02041809, -0.58578783],

                   ..........

                   [ 0.        ,  0.        ,  0.        ,  1.        ]],
                  dtype=float32)], 
                  'intrinsic': [array([[1.07717e+03, 0.00000e+00, 6.34483e+02, 0.00000e+00],

                   .........

                   [0.00000e+00, 0.00000e+00, 0.00000e+00, 1.00000e+00]],
                  dtype=float32)], 
                  'origin': array([0. , 0. , 0.5], dtype=float32)}

            """
            if isinstance(proj_mat, dict):
                assert 'extrinsic' in proj_mat.keys()
                assert 'intrinsic' in proj_mat.keys()
                projection = []
                # Support different intrinsic matrices for different images
                # if the original intrinsic is only a matrix
                # we will simply copy it to construct the intrinsic matrix list
                # in MultiViewPipeline
                assert isinstance(proj_mat['intrinsic'], list)
                for proj_idx in range(len(proj_mat['extrinsic'])):
                    """
                    ipdb> p len(proj_mat['extrinsic'])
                    10
                    """
                    intrinsic = img.new_tensor(proj_mat['intrinsic'][proj_idx])
                    extrinsic = img.new_tensor(proj_mat['extrinsic'][proj_idx])
                    projection.append(intrinsic @ extrinsic)
                    """
                    first iter
                    ipdb> p intrinsic @ extrinsic
                    tensor([[ 1.1856e+03,  3.9425e+02, -3.7373e+01, -1.6906e+03],
                            [ 7.7569e+01,  4.6759e+02, -1.0896e+03,  6.4665e+02],
                            [ 2.0923e-01,  9.7754e-01, -2.4246e-02, -1.6699e+00],
                            [ 0.0000e+00,  0.0000e+00,  0.0000e+00,  1.0000e+00]], device='cuda:0')

                    注：在 PyTorch 中，@ 符号用于矩阵乘法（matrix multiplication）或矩阵与向量的乘法（matrix-vector multiplication）。它等价于使用 torch.matmul() 函数。
                    """
                proj_mat = torch.stack(projection)

                """
                ipdb> p proj_mat.shape
                torch.Size([10, 4, 4])
                """
                # 将 3D 点投影到单视图图像上
                volume = batch_point_sample(
                    img_meta,
                    img_features=feature,
                    points=prior_points,
                    proj_mat=proj_mat,
                    coord_type=self.coord_type,
                    img_scale_factor=img_scale_factor,
                    img_crop_offset=img_crop_offset,
                    img_flip=img_flip,
                    img_pad_shape=img.shape[-2:],
                    img_shape=img_meta['img_shape'][:2],
                    aligned=False)
                """
                ipdb> p volume.shape
                torch.Size([25600, 256])
                """
            else:
                proj_mat = prior_points.new_tensor(proj_mat)
                volume = point_sample(img_meta,
                                      img_features=feature[None, ...],
                                      points=prior_points,
                                      proj_mat=proj_mat,
                                      coord_type=self.coord_type,
                                      img_scale_factor=img_scale_factor,
                                      img_crop_offset=img_crop_offset,
                                      img_flip=img_flip,
                                      img_pad_shape=img.shape[-2:],
                                      img_shape=img_meta['img_shape'][:2],
                                      aligned=False)
            volumes.append(
                volume.reshape(self.n_voxels[::-1] + [-1]).permute(3, 2, 1, 0))
            """
            ipdb> p self.n_voxels
            [40, 40, 16]
            ipdb> p len(volumes)
            1
            ipdb> p volumes[0].shape
            torch.Size([256, 40, 40, 16])
            ipdb>
            """
            valid_preds.append(
                ~torch.all(volumes[-1] == 0, dim=0, keepdim=True))
            """
            ipdb> p len(valid_preds)
            1
            ipdb> p valid_preds[0].shape
            torch.Size([1, 40, 40, 16])

            注：torch.all(..., dim=0, keepdim=True)：
            dim=0：在第 0 维（索引为 0）上检查所有元素是否均为 True（即在该维度上是否所有元素都为 0）。
            keepdim=True：保持输出张量的维度与输入张量的维度一致，除了在 dim=0 上会被压缩为 1。
            """
        img_volume = torch.stack(volumes)
        """
        ipdb> p img_volume.shape
        torch.Size([1, 256, 40, 40, 16])
        """

        # 2. Extract sparse point feats and scatter to the feat volume
        points = batch_inputs_dict['points']  # batch_inputs_dict['points'][0].shape : torch.size([100000,3])
        voxel_size = prior_points.new_tensor(self.voxel_size)
        point_cloud_range = prior_points.new_tensor(self.point_cloud_range)
        # construct sparse tensor and features
        if self.use_xyz_feat:
            coordinates, features = ME.utils.batch_sparse_collate(
                [((p[:, :3] - point_cloud_range[:3]) / voxel_size, p)
                 for p in points],
                device=points[0].device)
        else:
            coordinates, features = ME.utils.batch_sparse_collate(
                [((p[:, :3] - point_cloud_range[:3] / voxel_size), p[:, 3:])
                 for p in points],
                device=points[0].device)

        coordinates[:, 1:] = coordinates[:, 1:].clamp(
            min=torch.tensor([0, 0, 0],
                             dtype=coordinates.dtype,
                             device=coordinates.device),
            max=torch.tensor(
                [n * self.voxel_stride - 1 for n in self.n_voxels],
                dtype=coordinates.dtype,
                device=coordinates.device))

        sparse_point_feat = ME.SparseTensor(coordinates=coordinates,
                                            features=features)

        """
        ipdb> p sparse_point_feat
        SparseTensor(
          coordinates=tensor([[   0, 1064,  897,  293],
                [   0,  996,  921,  292],
                [   0,  999,  573,  629],
                ...,
                [   0, 1014, 1206,  530],
                [   0,  818, 1143,  295],
                [   0,  934, 1163,  314]], device='cuda:0', dtype=torch.int32)
          features=tensor([[-0.5393, -0.9568, -0.0461],
                [-0.7084, -0.8962, -0.0498],
                [-0.7011, -1.7652,  0.7945],
                ...,
                [-0.6644, -0.1839,  0.5472],
                [-1.1533, -0.3402, -0.0409],
                [-0.8630, -0.2922,  0.0056]], device='cuda:0')
          coordinate_map_key=coordinate map key:[1, 1, 1]
          coordinate_manager=CoordinateMapManagerGPU_c10(
                [1, 1, 1]:      CoordinateMapGPU:62282x4
                algorithm=MinkowskiAlgorithm.DEFAULT
          )
          spatial dimension=3)
        ipdb>

        ipdb> p sparse_point_feat.shape
        torch.Size([62282, 3])

        ipdb> p sparse_point_feat.features.shape
        torch.Size([62282, 3])
        ipdb> p sparse_point_feat.coordinates.shape
        torch.Size([62282, 4])
        """

        sparse_point_feat = self.backbone_3d(sparse_point_feat)

        """
        ipdb> p len(sparse_point_feat)
        4
        ipdb> p sparse_point_feat[0].shape
        torch.Size([12473, 64])
        ipdb> p sparse_point_feat[1].shape
        torch.Size([4885, 128])
        ipdb> p sparse_point_feat[2].shape
        torch.Size([1661, 256])
        ipdb> p sparse_point_feat[3].shape
        torch.Size([521, 512])
        ipdb> p sparse_point_feat[-1].shape
        torch.Size([521, 512])

        ipdb> p sparse_point_feat[-1].features.shape
        torch.Size([521, 512])

        """

        assert len(points) == 1, 'Only support batch size 1 for now!!'
        point_volume = sparse_point_feat[-1].dense(
            shape=torch.Size(
                [1, sparse_point_feat[-1].features.shape[-1], *self.n_voxels]),
            min_coordinate=torch.IntTensor([0, 0, 0]))[0]
        """
        ipdb> p point_volume.shape
        torch.Size([1, 512, 40, 40, 16])
        """

        """
        ipdb> p img_volume.shape
        torch.Size([1, 256, 40, 40, 16])
        ipdb> p point_volume.shape
        torch.Size([1, 512, 40, 40, 16])
        """

        """
        ipdb> p len(x)
        3
        ipdb> p x[0].shape
        torch.Size([1, 128, 40, 40, 16])
        ipdb> p x[1].shape
        torch.Size([1, 128, 20, 20, 8])
        ipdb> p x[2].shape
        torch.Size([1, 128, 10, 10, 4])
        """
        """
        ipdb> p valid_preds[0].shape
        torch.Size([1, 40, 40, 16])
        bool型
        """
        #

        img_volumes, point_volumes = [], []
        point_volume = self.pointch(point_volume)

        imgdown1 = self.imgdown1(img_volume)
        imgdown2 = self.imgdown2(imgdown1)

        pointdown1 = self.pointdown1(point_volume)
        pointdown2 = self.pointdown2(pointdown1)

        img_volumes.append(img_volume)
        img_volumes.append(imgdown1)
        img_volumes.append(imgdown2)
        point_volumes.append(point_volume)
        point_volumes.append(pointdown1)
        point_volumes.append(pointdown2)

        return point_volumes, point_volumes, torch.stack(valid_preds).float()

    def loss(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
             **kwargs) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                the 'imgs' key.

                    - imgs (torch.Tensor, optional): Image of each sample.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """

        """
        batch_inputs_dict:dict
        batch_inputs_dict['points'][0].shape : torch.size([100000,3])
        batch_inputs_dict['imgs'].shape : torch.size([1,10,3,480,480])
        """

        img_volumes, point_volumes, valid_preds = self.extract_feat(batch_inputs_dict,
                                                                    batch_data_samples)

        """
        x:list  len(x) = 3
        x[0].shape : torch.size([1,120,40,40,16])
        x[1].shape : torch.size([1,120,20,20,8])
        x[2].shape : torch.size([1,120,10,10,4])

        valid_preds.shape : torch.size([1,1,40,40,16])

        """

        # For indoor datasets ImVoxelNet uses ImVoxelHead that handles
        # mask of visible voxels.
        if self.coord_type in ('DEPTH', 'CAMERA') and self.use_valid_mask:
            x += (valid_preds, )

        losses = self.bbox_head.loss(img_volumes, point_volumes, batch_data_samples, **kwargs)
        """
        p losses
        {'loss_occ_0': tensor(18.8484, device='cuda:0', grad_fn=<MulBackward0>), 
        'loss_occ_1': tensor(9.2756, device='cuda:0', grad_fn=<MulBackward0>), 
        'loss_occ_2': tensor(4.3035, device='cuda:0', grad_fn=<MulBackward0>)}
        """
        return losses

    def predict(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
                **kwargs) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                the 'imgs' key.

                    - imgs (torch.Tensor, optional): Image of each sample.

            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, `gt_panoptic_seg_3d` and `gt_sem_seg_3d`.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input images. Each Det3DDataSample usually contain
            'pred_instances_3d'. And the ``pred_instances_3d`` usually
            contains following keys.

                - scores_3d (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels_3d (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes_3d (Tensor): Contains a tensor with shape
                    (num_instances, C) where C >=7.
        """
        img_volumes, point_volumes, valid_preds = self.extract_feat(batch_inputs_dict,
                                                                    batch_data_samples)
        # For indoor datasets ImVoxelNet uses ImVoxelHead that handles
        # mask of visible voxels.
        if self.coord_type in ('DEPTH', 'CAMERA') and self.use_valid_mask:
            x += (valid_preds, )

        results_list = self.bbox_head.predict(img_volumes, point_volumes, batch_data_samples, **kwargs)
        predictions = self.add_occupancy_to_data_sample(
            batch_data_samples, results_list)

        return predictions

    def add_occupancy_to_data_sample(self, data_samples: SampleList, pred):
        for i, data_sample in enumerate(data_samples):
            data_sample.pred_occupancy = pred[i]
        return data_samples

    def _forward(self, batch_inputs_dict: dict, batch_data_samples: SampleList,
                 *args, **kwargs) -> Tuple[List[torch.Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            batch_inputs_dict (dict): The model input dict which include
                the 'imgs' key.

                    - imgs (torch.Tensor, optional): Image of each sample.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`, `gt_panoptic_seg_3d` and `gt_sem_seg_3d`.

        Returns:
            tuple[list]: A tuple of features from ``bbox_head`` forward.
        """
        x, valid_preds = self.extract_feat(batch_inputs_dict,
                                           batch_data_samples)
        # For indoor datasets ImVoxelNet uses ImVoxelHead that handles
        # mask of visible voxels.
        if self.coord_type in ('DEPTH', 'CAMERA') and self.use_valid_mask:
            x += (valid_preds,)
        results = self.bbox_head.forward(x)

        return results

    def forward(self,
                inputs: Union[dict, List[dict]],
                data_samples: Optional[List] = None,
                mode: str = 'tensor',
                **kwargs) -> ForwardResults:
        """The unified entry for a forward process in both training and test.

        The method should accept three modes: "tensor", "predict" and "loss":

        - "tensor": Forward the whole network and return tensor or tuple of
        tensor without any post-processing, same as a common nn.Module.
        - "predict": Forward and return the predictions, which are fully
        processed to a list of :obj:`Det3DDataSample`.
        - "loss": Forward and return a dict of losses according to the given
        inputs and data samples.

        Note that this method doesn't handle neither back propagation nor
        optimizer updating, which are done in the :meth:`train_step`.

        Args:
            inputs  (dict | list[dict]): When it is a list[dict], the
                outer list indicate the test time augmentation. Each
                dict contains batch inputs
                which include 'points' and 'imgs' keys.

                - points (list[torch.Tensor]): Point cloud of each sample.
                - imgs (torch.Tensor): Image tensor has shape (B, C, H, W).
            data_samples (list[:obj:`Det3DDataSample`],
                list[list[:obj:`Det3DDataSample`]], optional): The
                annotation data of every samples. When it is a list[list], the
                outer list indicate the test time augmentation, and the
                inter list indicate the batch. Otherwise, the list simply
                indicate the batch. Defaults to None.
            mode (str): Return what kind of value. Defaults to 'tensor'.

        Returns:
            The return type depends on ``mode``.

            - If ``mode="tensor"``, return a tensor or a tuple of tensor.
            - If ``mode="predict"``, return a list of :obj:`Det3DDataSample`.
            - If ``mode="loss"``, return a dict of tensor.
        """
        if mode == 'loss':
            return self.loss(inputs, data_samples, **kwargs)
        elif mode == 'predict':
            return self.predict(inputs, data_samples, **kwargs)
        else:
            raise RuntimeError(f'Invalid mode "{mode}". '
                               'Only supports loss, predict and tensor mode')

    def add_pred_to_datasample(
            self,
            data_samples: SampleList,
            data_instances_3d: Optional[InstanceList] = None,
            data_instances_2d: Optional[InstanceList] = None,
    ) -> SampleList:
        """Convert results list to `Det3DDataSample`.

        Subclasses could override it to be compatible for some multi-modality
        3D detectors.

        Args:
            data_samples (list[:obj:`Det3DDataSample`]): The input data.
            data_instances_3d (list[:obj:`InstanceData`], optional): 3D
                Detection results of each sample.
            data_instances_2d (list[:obj:`InstanceData`], optional): 2D
                Detection results of each sample.

        Returns:
            list[:obj:`Det3DDataSample`]: Detection results of the
            input. Each Det3DDataSample usually contains
            'pred_instances_3d'. And the ``pred_instances_3d`` normally
            contains following keys.

            - scores_3d (Tensor): Classification scores, has a shape
              (num_instance, )
            - labels_3d (Tensor): Labels of 3D bboxes, has a shape
              (num_instances, ).
            - bboxes_3d (Tensor): Contains a tensor with shape
              (num_instances, C) where C >=7.

            When there are image prediction in some models, it should
            contains  `pred_instances`, And the ``pred_instances`` normally
            contains following keys.

            - scores (Tensor): Classification scores of image, has a shape
              (num_instance, )
            - labels (Tensor): Predict Labels of 2D bboxes, has a shape
              (num_instances, ).
            - bboxes (Tensor): Contains a tensor with shape
              (num_instances, 4).
        """

        assert (data_instances_2d is not None) or \
               (data_instances_3d is not None), \
            'please pass at least one type of data_samples'

        if data_instances_2d is None:
            data_instances_2d = [
                InstanceData() for _ in range(len(data_instances_3d))
            ]
        if data_instances_3d is None:
            data_instances_3d = [
                InstanceData() for _ in range(len(data_instances_2d))
            ]

        for i, data_sample in enumerate(data_samples):
            data_sample.pred_instances_3d = data_instances_3d[i]
            data_sample.pred_instances = data_instances_2d[i]
        return data_samples


