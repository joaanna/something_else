import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
import numpy as np

from functools import partial

__all__ = [
    'ResNet', 'resnet10', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
    'resnet152', 'resnet200',
]


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    conv_op = None
    offset_groups = 1

    def __init__(self, dim_in, dim_out, stride, dim_inner, group=1, use_temp_conv=1, temp_stride=1, dcn=False,
                 shortcut_type='B'):
        super(Bottleneck, self).__init__()
        # 1 x 1 layer
        self.with_dcn = dcn
        self.conv1 = self.Conv3dBN(dim_in, dim_inner, (1 + use_temp_conv * 2, 1, 1), (temp_stride, 1, 1),
                                   (use_temp_conv, 0, 0))
        self.relu = nn.ReLU(inplace=True)
        # 3 x 3 layer
        self.conv2 = self.Conv3dBN(dim_inner, dim_inner, (1, 3, 3), (1, stride, stride), (0, 1, 1))
        # 1 x 1 layer
        self.conv3 = self.Conv3dBN(dim_inner, dim_out, (1, 1, 1), (1, 1, 1), (0, 0, 0))

        self.shortcut_type = shortcut_type
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.temp_stride = temp_stride
        self.stride = stride
        # nn.Conv3d(dim_in, dim_out, (1,1,1),(temp_stride,stride,stride),(0,0,0))
        if self.shortcut_type == 'B':
            if self.dim_in == self.dim_out and self.temp_stride == 1 and self.stride == 1:  # or (self.dim_in == self.dim_out and self.dim_in == 64 and self.stride ==1):

                pass
            else:
                # pass
                self.shortcut = self.Conv3dBN(dim_in, dim_out, (1, 1, 1), (temp_stride, stride, stride), (0, 0, 0))

        # nn.Conv3d(dim_in,dim_inner,kernel_size=(1+use_temp_conv*2,1,1),stride = (temp_stride,1,1),padding = )

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.relu(out)
        out = self.conv3(out)
        if self.dim_in == self.dim_out and self.temp_stride == 1 and self.stride == 1:
            pass
        else:
            residual = self.shortcut(residual)
        out += residual
        out = self.relu(out)
        return out

    def Conv3dBN(self, dim_in, dim_out, kernels, strides, pads, group=1):
        if self.with_dcn and kernels[0] > 1:
            # use deformable conv
            return nn.Sequential(
                self.conv_op(dim_in, dim_out, kernel_size=kernels, stride=strides, padding=pads, bias=False,
                             offset_groups=self.offset_groups),
                nn.BatchNorm3d(dim_out)
            )
        else:
            return nn.Sequential(
                nn.Conv3d(dim_in, dim_out, kernel_size=kernels, stride=strides, padding=pads, bias=False),
                nn.BatchNorm3d(dim_out)
            )


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 use_temp_convs_set,
                 temp_strides_set,
                 sample_size,
                 sample_duration,
                 shortcut_type='B',
                 num_classes=400,
                 stage_with_dcn=(False, False, False, False),
                 extract_features=False,
                 loss_type='softmax'):
        super(ResNet, self).__init__()
        self.extract_features = extract_features
        self.stage_with_dcn = stage_with_dcn
        self.group = 1
        self.width_per_group = 64
        self.dim_inner = self.group * self.width_per_group
        # self.shortcut_type = shortcut_type
        self.conv1 = nn.Conv3d(
            3,
            64,
            kernel_size=(1 + use_temp_convs_set[0][0] * 2, 7, 7),
            stride=(temp_strides_set[0][0], 2, 2),
            padding=(use_temp_convs_set[0][0], 3, 3),
            bias=False)
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 0, 0))
        with_dcn = True if self.stage_with_dcn[0] else False
        self.layer1 = self._make_layer(block, 64, 256, shortcut_type, stride=1, num_blocks=layers[0],
                                       dim_inner=self.dim_inner, group=self.group, use_temp_convs=use_temp_convs_set[1],
                                       temp_strides=temp_strides_set[1], dcn=with_dcn)
        self.maxpool2 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0))
        with_dcn = True if self.stage_with_dcn[1] else False
        self.layer2 = self._make_layer(block, 256, 512, shortcut_type, stride=2, num_blocks=layers[1],
                                       dim_inner=self.dim_inner * 2, group=self.group,
                                       use_temp_convs=use_temp_convs_set[2], temp_strides=temp_strides_set[2],
                                       dcn=with_dcn)
        with_dcn = True if self.stage_with_dcn[2] else False
        self.layer3 = self._make_layer(block, 512, 1024, shortcut_type, stride=2, num_blocks=layers[2],
                                       dim_inner=self.dim_inner * 4, group=self.group,
                                       use_temp_convs=use_temp_convs_set[3], temp_strides=temp_strides_set[3],
                                       dcn=with_dcn)
        with_dcn = True if self.stage_with_dcn[3] else False
        self.layer4 = self._make_layer(block, 1024, 2048, shortcut_type, stride=1, num_blocks=layers[3],
                                       dim_inner=self.dim_inner * 8, group=self.group,
                                       use_temp_convs=use_temp_convs_set[4], temp_strides=temp_strides_set[4],
                                       dcn=with_dcn)
        last_duration = int(math.ceil(sample_duration / 2))  # int(math.ceil(sample_duration / 8))
        last_size = int(math.ceil(sample_size / 16))
        # self.avgpool = nn.AvgPool3d((last_duration, last_size, last_size), stride=1) #nn.AdaptiveAvgPool3d((1, 1, 1)) #
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = torch.nn.Dropout(p=0.5)
        self.classifier = nn.Linear(2048, num_classes)

        for m in self.modules():
            # if isinstance(m, nn.Conv3d):
            #     m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            # elif isinstance(m,nn.Linear):
            #    m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            # elif 
            if isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, dim_in, dim_out, shortcut_type, stride, num_blocks, dim_inner=None, group=None,
                    use_temp_convs=None, temp_strides=None, dcn=False):
        if use_temp_convs is None:
            use_temp_convs = np.zeros(num_blocks).astype(int)
        if temp_strides is None:
            temp_strides = np.ones(num_blocks).astype(int)
        if len(use_temp_convs) < num_blocks:
            for _ in range(num_blocks - len(use_temp_convs)):
                use_temp_convs.append(0)
                temp_strides.append(1)
        layers = []
        for idx in range(num_blocks):
            block_stride = 2 if (idx == 0 and stride == 2) else 1

            layers.append(
                block(dim_in, dim_out, block_stride, dim_inner, group, use_temp_convs[idx], temp_strides[idx], dcn))
            dim_in = dim_out
        return nn.Sequential(*layers)

    def forward_single(self, x):
        x = self.conv1(x)

        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool1(x)

        x = self.layer1(x)
        x = self.maxpool2(x)
        x = self.layer2(x)

        x = self.layer3(x)
        features = self.layer4(x)

        x = self.avgpool(features)

        x = x.view(x.size(0), -1)
        x = self.dropout(x)

        y = self.classifier(x)
        if self.extract_features:
            return y, features
        else:
            return y

    def forward_multi(self, x):
        clip_preds = []
        # import ipdb;ipdb.set_trace()
        for clip_idx in range(x.shape[1]):  # B, 10, 3, 3, 32, 224, 224
            spatial_crops = []
            for crop_idx in range(x.shape[2]):
                clip = x[:, clip_idx, crop_idx]
                clip = self.forward_single(clip)
                spatial_crops.append(clip)
            spatial_crops = torch.stack(spatial_crops, 1).mean(1)  # (B, 400)
            clip_preds.append(spatial_crops)
        clip_preds = torch.stack(clip_preds, 1).mean(1)  # (B, 400)
        return clip_preds

    def forward(self, x):

        # 5D tensor == single clip
        if x.dim() == 5:
            pred = self.forward_single(x)

        # 7D tensor == 3 crops/10 clips
        elif x.dim() == 7:
            pred = self.forward_multi(x)

        # loss_dict = {}
        # if 'label' in batch:
        #     loss = F.cross_entropy(pred, batch['label'], reduction='none')
        #     loss_dict = {'clf': loss}

        return pred


def get_fine_tuning_parameters(model, ft_begin_index):
    if ft_begin_index == 0:
        return model.parameters()

    ft_module_names = []
    for i in range(ft_begin_index, 5):
        ft_module_names.append('layer{}'.format(i))
    ft_module_names.append('fc')
    # import ipdb;ipdb.set_trace()
    parameters = []
    for k, v in model.named_parameters():
        for ft_module in ft_module_names:
            if ft_module in k:
                parameters.append({'params': v})
                break
        else:
            parameters.append({'params': v, 'lr': 0.0})

    return parameters


def obtain_arc(arc_type):
    # c2d, ResNet50
    if arc_type == 1:
        use_temp_convs_1 = [0]
        temp_strides_1 = [2]
        use_temp_convs_2 = [0, 0, 0]
        temp_strides_2 = [1, 1, 1]
        use_temp_convs_3 = [0, 0, 0, 0]
        temp_strides_3 = [1, 1, 1, 1]
        use_temp_convs_4 = [0, ] * 6
        temp_strides_4 = [1, ] * 6
        use_temp_convs_5 = [0, 0, 0]
        temp_strides_5 = [1, 1, 1]

    # i3d, ResNet50
    if arc_type == 2:
        use_temp_convs_1 = [2]
        temp_strides_1 = [1]
        use_temp_convs_2 = [1, 1, 1]
        temp_strides_2 = [1, 1, 1]
        use_temp_convs_3 = [1, 0, 1, 0]
        temp_strides_3 = [1, 1, 1, 1]
        use_temp_convs_4 = [1, 0, 1, 0, 1, 0]
        temp_strides_4 = [1, 1, 1, 1, 1, 1]
        use_temp_convs_5 = [0, 1, 0]
        temp_strides_5 = [1, 1, 1]

    # c2d, ResNet101
    if arc_type == 3:
        use_temp_convs_1 = [0]
        temp_strides_1 = [2]
        use_temp_convs_2 = [0, 0, 0]
        temp_strides_2 = [1, 1, 1]
        use_temp_convs_3 = [0, 0, 0, 0]
        temp_strides_3 = [1, 1, 1, 1]
        use_temp_convs_4 = [0, ] * 23
        temp_strides_4 = [1, ] * 23
        use_temp_convs_5 = [0, 0, 0]
        temp_strides_5 = [1, 1, 1]

    # i3d, ResNet101
    if arc_type == 4:
        use_temp_convs_1 = [2]
        temp_strides_1 = [2]
        use_temp_convs_2 = [1, 1, 1]
        temp_strides_2 = [1, 1, 1]
        use_temp_convs_3 = [1, 0, 1, 0]
        temp_strides_3 = [1, 1, 1, 1]
        use_temp_convs_4 = []
        for i in range(23):
            if i % 2 == 0:
                use_temp_convs_4.append(1)
            else:
                use_temp_convs_4.append(0)

        temp_strides_4 = [1, ] * 23
        use_temp_convs_5 = [0, 1, 0]
        temp_strides_5 = [1, 1, 1]

    use_temp_convs_set = [use_temp_convs_1, use_temp_convs_2, use_temp_convs_3, use_temp_convs_4, use_temp_convs_5]
    temp_strides_set = [temp_strides_1, temp_strides_2, temp_strides_3, temp_strides_4, temp_strides_5]

    return use_temp_convs_set, temp_strides_set


def resnet10(**kwargs):
    """Constructs a ResNet-18 model.
    """
    use_temp_convs_set = []
    temp_strides_set = []
    model = ResNet(BasicBlock, [1, 1, 1, 1], use_temp_convs_set, temp_strides_set, **kwargs)
    return model


def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    use_temp_convs_set = []
    temp_strides_set = []
    model = ResNet(BasicBlock, [2, 2, 2, 2], use_temp_convs_set, temp_strides_set, **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    use_temp_convs_set = []
    temp_strides_set = []
    model = ResNet(BasicBlock, [3, 4, 6, 3], use_temp_convs_set, temp_strides_set, **kwargs)
    return model


def resnet50(extract_features, **kwargs):
    """Constructs a ResNet-50 model.
    """
    use_temp_convs_set, temp_strides_set = obtain_arc(2)
    model = ResNet(Bottleneck, [3, 4, 6, 3], use_temp_convs_set, temp_strides_set,
                   extract_features=extract_features, **kwargs)
    return model


def resnet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    use_temp_convs_set, temp_strides_set = obtain_arc(4)
    model = ResNet(Bottleneck, [3, 4, 23, 3], use_temp_convs_set, temp_strides_set, **kwargs)
    return model


def resnet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    use_temp_convs_set = []
    temp_strides_set = []
    model = ResNet(Bottleneck, [3, 8, 36, 3], use_temp_convs_set, temp_strides_set, **kwargs)
    return model


def resnet200(**kwargs):
    """Constructs a ResNet-101 model.
    """
    use_temp_convs_set = []
    temp_strides_set = []
    model = ResNet(Bottleneck, [3, 24, 36, 3], use_temp_convs_set, temp_strides_set, **kwargs)
    return model


def Net(num_classes, extract_features=False, loss_type='softmax',
        weights=None, freeze_all_but_cls=False):
    net = globals()['resnet' + str(50)](
        num_classes=num_classes,
        sample_size=50,
        sample_duration=32,
        extract_features=extract_features,
        loss_type=loss_type,
    )

    if weights is not None:
        kinetics_weights = torch.load(weights)['state_dict']
        print("Found weights in {}.".format(weights))
        cls_name = 'fc'
    else:
        kinetics_weights = torch.load('model/pretrained_weights/kinetics-res50.pth')
        cls_name = 'fc'
        print('\n Restoring Kintetics \n')

    new_weights = {}
    for k, v in kinetics_weights.items():
        if not k.startswith('module.' + cls_name):
            new_weights[k.replace('module.', '')] = v
    net.load_state_dict(new_weights, strict=False)

    if freeze_all_but_cls:
        for name, par in net.named_parameters():
            if not name.startswith('classifier'):
                par.requires_grad = False
    return net
