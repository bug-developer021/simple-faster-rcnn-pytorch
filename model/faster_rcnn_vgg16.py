from __future__ import  absolute_import
import torch as t
from torch import nn
from torchvision.models import vgg16
from torchvision.ops import RoIPool

from model.region_proposal_network import RegionProposalNetwork
from model.faster_rcnn import FasterRCNN
from utils import array_tool as at
from utils.config import opt

# 将vgg16拆分为特征提取网络和分类网络
def decom_vgg16():
    # the 30th layer of features is relu of conv5_3
    if opt.caffe_pretrain:
        model = vgg16(pretrained=False)
        if not opt.load_path:
            model.load_state_dict(t.load(opt.caffe_pretrain_path))
    else:
        model = vgg16(not opt.load_path)
    
    # 加载预训练模型vgg16的conv5_3之前的部分
    features = list(model.features)[:30]
    classifier = model.classifier
    
    # classifier层结构如下
    #     Sequential(
    #   (0): Linear(in_features=25088, out_features=4096, bias=True)
    #   (1): ReLU(inplace=True)
    #   (2): Dropout(p=0.5, inplace=False)
    #   (3): Linear(in_features=4096, out_features=4096, bias=True)
    #   (4): ReLU(inplace=True)
    #   (5): Dropout(p=0.5, inplace=False)
    #   (6): Linear(in_features=4096, out_features=1000, bias=True)
    # )
    classifier = list(classifier)
    # 删除输出分类结果层，后面改写输出维度为n_classes
    del classifier[6]
    #  删除两个dropout
    if not opt.use_drop:
        del classifier[5]
        del classifier[2]
    classifier = nn.Sequential(*classifier)
    
    # 冻结vgg16前2个stage,不进行反向传播
    # freeze top4 conv
    for layer in features[:10]:
        for p in layer.parameters():
            p.requires_grad = False
    # 拆分为特征提取网络和分类网络
    return nn.Sequential(*features), classifier


# 分别对 VGG16 的特征提取部分、分类部分、RPN 网络、VGG16RoIHead 网络进行了实例化。
class FasterRCNNVGG16(FasterRCNN):
    """Faster R-CNN based on VGG-16.
    For descriptions on the interface of this model, please refer to
    :class:`model.faster_rcnn.FasterRCNN`.

    Args:
        n_fg_class (int): The number of classes excluding the background.
        ratios (list of floats): This is ratios of width to height of
            the anchors.
        anchor_scales (list of numbers): This is areas of anchors.
            Those areas will be the product of the square of an element in
            :obj:`anchor_scales` and the original area of the reference
            window.

    """
    # vgg16通过5个stage下采样16倍
    feat_stride = 16  # downsample 16x for output of conv5 in vgg16
    
    # 总类别数为20类，三种尺度三种比例的anchor
    def __init__(self,
                 n_fg_class=20,
                 ratios=[0.5, 1, 2],
                 anchor_scales=[8, 16, 32]
                 ):
                 
        extractor, classifier = decom_vgg16()
        
        #  返回rpn_locs, rpn_scores, rois, roi_indices, anchor
        rpn = RegionProposalNetwork(
            512, 512,
            ratios=ratios,
            anchor_scales=anchor_scales,
            feat_stride=self.feat_stride,
        )

        head = VGG16RoIHead(
            n_class=n_fg_class + 1,
            roi_size=7,
            spatial_scale=(1. / self.feat_stride),
            classifier=classifier
        )

        super(FasterRCNNVGG16, self).__init__(
            extractor,
            rpn,
            head,
        )


# 输入特征图、rois、roi_indices
# 输出roi_cls_locs和roi_scores
class VGG16RoIHead(nn.Module):
    """Faster R-CNN Head for VGG-16 based implementation.
    This class is used as a head for Faster R-CNN.
    This outputs class-wise localizations and classification based on feature
    maps in the given RoIs.
    
    Args:
        n_class (int): The number of classes possibly including the background.
        roi_size (int): Height and width of the feature maps after RoI-pooling.
        spatial_scale (float): Scale of the roi is resized.
        classifier (nn.Module): Two layer Linear ported from vgg16

    """

    def __init__(self, n_class, roi_size, spatial_scale,
                 classifier):
        # n_class includes the background
        super(VGG16RoIHead, self).__init__()
        
        # vgg16中的最后两个全连接层
        self.classifier = classifier
        self.cls_loc = nn.Linear(4096, n_class * 4)
        self.score = nn.Linear(4096, n_class)

        # 全连接层权重初始化
        normal_init(self.cls_loc, 0, 0.001)
        normal_init(self.score, 0, 0.01)

        # 加上背景21类
        self.n_class = n_class
        # roi-pooling后变为7x7大小
        self.roi_size = roi_size

        # roi对应的是缩放后图片的尺度M*N，将其映射回M/16 * N/16大小的featuremap尺度
        self.spatial_scale = spatial_scale
        # 将大小不同的roi变成大小一致，得到pooling后的特征，
        self.roi = RoIPool( (self.roi_size, self.roi_size),self.spatial_scale)

    def forward(self, x, rois, roi_indices):
        """Forward the chain.

        We assume that there are :math:`N` batches.

        Args:
            x (Variable): 4D image variable.
            rois (Tensor): A bounding box array containing coordinates of
                proposal boxes.  This is a concatenation of bounding box
                arrays from multiple images in the batch.
                Its shape is :math:`(R', 4)`. Given :math:`R_i` proposed
                RoIs from the :math:`i` th image,
                :math:`R' = \\sum _{i=1} ^ N R_i`.
            roi_indices (Tensor): An array containing indices of images to
                which bounding boxes correspond to. Its shape is :math:`(R',)`.

        """
        # in case roi_indices is  ndarray
        roi_indices = at.totensor(roi_indices).float()
        rois = at.totensor(rois).float()
        indices_and_rois = t.cat([roi_indices[:, None], rois], dim=1)
        # NOTE: important: yx->xy
        xy_indices_and_rois = indices_and_rois[:, [0, 2, 1, 4, 3]]
        indices_and_rois =  xy_indices_and_rois.contiguous()

        pool = self.roi(x, indices_and_rois)
        # flat操作
        pool = pool.view(pool.size(0), -1)
        # decom_vgg16（）得到的calssifier,得到4096
        fc7 = self.classifier(pool)
        # （4096->84）
        roi_cls_locs = self.cls_loc(fc7)
        # （4096->21）
        roi_scores = self.score(fc7)
        # roi回归输出的是128*84，即对每个roi都输出了对应每种类别的回归参数，21*4共84种
        # 然而真实位置参数是128*4，后面通过_suppress转换为真正的bbox, label, score
        return roi_cls_locs, roi_scores


def normal_init(m, mean, stddev, truncated=False):
    """
    weight initalizer: truncated normal and random normal.
    """
    # x is a parameter
    if truncated:
        m.weight.data.normal_().fmod_(2).mul_(stddev).add_(mean)  # not a perfect approximation
    else:
        m.weight.data.normal_(mean, stddev)
        m.bias.data.zero_()
