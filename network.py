import math
import numpy as np
from os.path import join

import torch.nn as nn
import torch
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torchvision import models

BatchNorm = nn.BatchNorm2d


def get_model_url(data='imagenet', name='dla34', hash='ba72cf86'):
    return join('http://dl.yf.io/dla/models', data, '{}-{}.pth'.format(name, hash))


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=3,
                               stride=stride, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn1 = BatchNorm(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=dilation,
                               bias=False, dilation=dilation)
        self.bn2 = BatchNorm(planes)
        self.stride = stride

    def forward(self, x, residual=None):
        if residual is None:
            residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += residual
        out = self.relu(out)

        return out


class Root(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, residual):
        super(Root, self).__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels, 1,
            stride=1, bias=False, padding=(kernel_size - 1) // 2)
        self.bn = BatchNorm(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual = residual

    def forward(self, *x):
        children = x
        x = self.conv(torch.cat(x, 1))
        x = self.bn(x)
        if self.residual:
            x += children[0]
        x = self.relu(x)

        return x


class Tree(nn.Module):
    def __init__(self, levels, block, in_channels, out_channels, stride=1,
                 level_root=False, root_dim=0, root_kernel_size=1,
                 dilation=1, root_residual=False):
        super(Tree, self).__init__()
        if root_dim == 0:
            root_dim = 2 * out_channels
        if level_root:
            root_dim += in_channels
        if levels == 1:
            self.tree1 = block(in_channels, out_channels, stride,
                               dilation=dilation)
            self.tree2 = block(out_channels, out_channels, 1,
                               dilation=dilation)
        else:
            self.tree1 = Tree(levels - 1, block, in_channels, out_channels,
                              stride, root_dim=0,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
            self.tree2 = Tree(levels - 1, block, out_channels, out_channels,
                              root_dim=root_dim + out_channels,
                              root_kernel_size=root_kernel_size,
                              dilation=dilation, root_residual=root_residual)
        if levels == 1:
            self.root = Root(root_dim, out_channels, root_kernel_size,
                             root_residual)
        self.level_root = level_root
        self.root_dim = root_dim
        self.downsample = None
        self.project = None
        self.levels = levels
        if stride > 1:
            self.downsample = nn.MaxPool2d(stride, stride=stride)
        if in_channels != out_channels:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=1, bias=False),
                BatchNorm(out_channels)
            )

    def forward(self, x, residual=None, children=None):
        children = [] if children is None else children

        bottom = self.downsample(x) if self.downsample else x
        residual = self.project(bottom) if self.project else bottom

        if self.level_root:
            children.append(bottom)

        x1 = self.tree1(x, residual)
        if self.levels == 1:
            x2 = self.tree2(x1)
            x = self.root(x2, x1, *children)
        else:
            children.append(x1)
            x = self.tree2(x1, children=children)
        return x


class DLA(nn.Module):
    def __init__(self, levels, channels, num_classes=1000, block=BasicBlock, residual_root=False, return_levels=False, pool_size=7):
        super(DLA, self).__init__()
        self.channels = channels
        self.return_levels = return_levels
        self.num_classes = num_classes
        self.base_layer = nn.Sequential(
            nn.Conv2d(3, channels[0], kernel_size=7, stride=1,
                      padding=3, bias=False),
            BatchNorm(channels[0]),
            nn.ReLU(inplace=True))

        self.level0 = self._make_conv_level(
            channels[0], channels[0], levels[0])
        self.level1 = self._make_conv_level(
            channels[0], channels[1], levels[1], stride=2)

        self.level2 = Tree(levels[2], block, channels[1], channels[2], 2,
                           level_root=False, root_residual=residual_root)
        self.level3 = Tree(levels[3], block, channels[2], channels[3], 2,
                           level_root=True, root_residual=residual_root)
        self.level4 = Tree(levels[4], block, channels[3], channels[4], 2,
                           level_root=True, root_residual=residual_root)
        self.level5 = Tree(levels[5], block, channels[4], channels[5], 2,
                           level_root=True, root_residual=residual_root)

        self.avgpool = nn.AvgPool2d(pool_size)
        self.fc = nn.Conv2d(channels[-1], num_classes, kernel_size=1,
                            stride=1, padding=0, bias=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    @staticmethod
    def _make_level(block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                nn.MaxPool2d(stride, stride=stride),
                nn.Conv2d(inplanes, planes,
                          kernel_size=1, stride=1, bias=False),
                BatchNorm(planes),
            )

        layers = [block(inplanes, planes, stride, downsample=downsample)]
        for i in range(1, blocks):
            layers.append(block(inplanes, planes))

        return nn.Sequential(*layers)

    @staticmethod
    def _make_conv_level(inplanes, planes, convs, stride=1, dilation=1):
        modules = []
        for i in range(convs):
            modules.extend([
                nn.Conv2d(inplanes, planes, kernel_size=3,
                          stride=stride if i == 0 else 1,
                          padding=dilation, bias=False, dilation=dilation),
                BatchNorm(planes),
                nn.ReLU(inplace=True)])
            inplanes = planes
        return nn.Sequential(*modules)

    def forward(self, x):
        y = []
        x = self.base_layer(x)
        for i in range(6):
            x = getattr(self, 'level{}'.format(i))(x)
            y.append(x)
        if self.return_levels:
            return y
        else:
            x = self.avgpool(x)
            x = self.fc(x)
            x = x.view(x.size(0), -1)
            return x

    def load_pretrained_model(self,  data='imagenet', name='dla34', hash='ba72cf86'):
        fc = self.fc
        if name.endswith('.pth'):
            model_weights = torch.load(data + name)
        else:
            model_url = get_model_url(data, name, hash)
            model_weights = model_zoo.load_url(model_url)
        num_classes = len(model_weights[list(model_weights.keys())[-1]])
        self.fc = nn.Conv2d(
            self.channels[-1], num_classes,
            kernel_size=1, stride=1, padding=0, bias=True)
        self.load_state_dict(model_weights)
        self.fc = fc


def dla34(pretrained, **kwargs):
    model = DLA([1, 1, 1, 2, 2, 1], [16, 32, 64, 128, 256, 512], block=BasicBlock, **kwargs)
    if pretrained:
        model.load_pretrained_model(data='imagenet', name='dla34', hash='ba72cf86')
    return model


class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    @staticmethod
    def forward(x):
        return x


def fill_up_weights(up):
    w = up.weight.data
    f = math.ceil(w.size(2) / 2)
    c = (2 * f - 1 - f % 2) / (2. * f)
    for i in range(w.size(2)):
        for j in range(w.size(3)):
            w[0, 0, i, j] = \
                (1 - math.fabs(i / f - c)) * (1 - math.fabs(j / f - c))
    for c in range(1, w.size(0)):
        w[c, 0, :, :] = w[0, 0, :, :]


class IDAUp(nn.Module):
    def __init__(self, node_kernel, out_dim, channels, up_factors):
        super(IDAUp, self).__init__()
        self.channels = channels
        self.out_dim = out_dim
        for i, c in enumerate(channels):
            if c == out_dim:
                proj = Identity()
            else:
                proj = nn.Sequential(
                    nn.Conv2d(c, out_dim,
                              kernel_size=1, stride=1, bias=False),
                    BatchNorm(out_dim),
                    nn.ReLU(inplace=True))
            f = int(up_factors[i])
            if f == 1:
                up = Identity()
            else:
                up = nn.ConvTranspose2d(
                    out_dim, out_dim, f * 2, stride=f, padding=f // 2,
                    output_padding=0, groups=out_dim, bias=False)
                fill_up_weights(up)
            setattr(self, 'proj_' + str(i), proj)
            setattr(self, 'up_' + str(i), up)

        for i in range(1, len(channels)):
            node = nn.Sequential(
                nn.Conv2d(out_dim * 2, out_dim,
                          kernel_size=node_kernel, stride=1,
                          padding=node_kernel // 2, bias=False),
                BatchNorm(out_dim),
                nn.ReLU(inplace=True))
            setattr(self, 'node_' + str(i), node)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, BatchNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, layers):
        assert len(self.channels) == len(layers), \
            '{} vs {} layers'.format(len(self.channels), len(layers))
        layers = list(layers)
        for i, l in enumerate(layers):
            upsample = getattr(self, 'up_' + str(i))
            project = getattr(self, 'proj_' + str(i))
            layers[i] = upsample(project(l))
        x = layers[0]
        y = []
        for i in range(1, len(layers)):
            node = getattr(self, 'node_' + str(i))
            x = node(torch.cat([x, layers[i]], 1))
            y.append(x)
        return x, y


class DLAUp(nn.Module):
    def __init__(self, channels, scales=(1, 2, 4, 8, 16), in_channels=None):
        super(DLAUp, self).__init__()
        if in_channels is None:
            in_channels = channels
        self.channels = channels
        channels = list(channels)
        scales = np.array(scales, dtype=int)
        for i in range(len(channels) - 1):
            j = -i - 2
            setattr(self, 'ida_{}'.format(i),
                    IDAUp(3, channels[j], in_channels[j:],
                          scales[j:] // scales[j]))
            scales[j + 1:] = scales[j]
            in_channels[j + 1:] = [channels[j] for _ in channels[j + 1:]]

    def forward(self, layers):
        x = None
        layers = list(layers)
        assert len(layers) > 1
        for i in range(len(layers) - 1):
            ida = getattr(self, 'ida_{}'.format(i))
            x, y = ida(layers[-i - 2:])
            layers[-i - 1:] = y
        return x


class SegVexPlusOriDLA(nn.Module):
    def __init__(self, base_name='dla34', pretrained=True, down_ratio=2, eval_mode=False):
        super(SegVexPlusOriDLA, self).__init__()
        assert down_ratio in [2, 4, 8, 16]
        self.eval_mode = eval_mode
        self.first_level = int(np.log2(down_ratio))
        self.base = globals()[base_name](pretrained=pretrained, return_levels=True)
        channels = self.base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(channels[self.first_level:], scales=scales)

        self.upsample = nn.ConvTranspose2d(channels[self.first_level], 64, (4, 4), (2, 2), (1, 1))

        self.seg_branch = nn.Sequential(nn.Conv2d(64, 32, (3, 3), padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 32, (3, 3), padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 1, (3, 3), padding=1), nn.Sigmoid())

        self.vex_branch = nn.Sequential(nn.Conv2d(64, 32, (3, 3), padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 32, (3, 3), padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 2, (3, 3), padding=1), nn.Sigmoid())

        self.ori_branch = nn.Sequential(nn.Conv2d(66, 32, (3, 3), padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 32, (3, 3), padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 2, (3, 3), padding=1), nn.Tanh())

    def forward(self, x):
        x = self.base(x)
        x = self.dla_up(x[self.first_level:])
        x = self.upsample(x)
        seg = self.seg_branch(x)
        vex = self.vex_branch(x)
        ori = self.ori_branch(torch.cat([x, vex], 1))
        return {'seg': seg, 'vex': vex, 'ori': ori}


class SegVexPlusDLA(nn.Module):
    def __init__(self, base_name='dla34', pretrained=True, down_ratio=2, eval_mode=False):
        super(SegVexPlusDLA, self).__init__()
        assert down_ratio in [2, 4, 8, 16]
        self.eval_mode = eval_mode
        self.first_level = int(np.log2(down_ratio))
        self.base = globals()[base_name](pretrained=pretrained, return_levels=True)
        channels = self.base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(channels[self.first_level:], scales=scales)

        self.upsample = nn.ConvTranspose2d(channels[self.first_level], 64, (4, 4), (2, 2), (1, 1))

        self.seg_branch = nn.Sequential(nn.Conv2d(64, 32, (3, 3), padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 32, (3, 3), padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 1, (3, 3), padding=1), nn.Sigmoid())

        self.vex_branch = nn.Sequential(nn.Conv2d(64, 32, (3, 3), padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 32, (3, 3), padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 2, (3, 3), padding=1), nn.Sigmoid())

    def forward(self, x):
        x = self.base(x)
        x = self.dla_up(x[self.first_level:])
        x = self.upsample(x)
        seg = self.seg_branch(x)
        vex = self.vex_branch(x)
        return {'seg': seg, 'vex': vex}


class SegVexDLA(nn.Module):
    def __init__(self, base_name='dla34', pretrained=True, down_ratio=2, eval_mode=False):
        super(SegVexDLA, self).__init__()
        assert down_ratio in [2, 4, 8, 16]
        self.eval_mode = eval_mode
        self.first_level = int(np.log2(down_ratio))
        self.base = globals()[base_name](pretrained=pretrained, return_levels=True)
        channels = self.base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(channels[self.first_level:], scales=scales)

        self.upsample = nn.ConvTranspose2d(channels[self.first_level], 64, (4, 4), (2, 2), (1, 1))

        self.seg_branch = nn.Sequential(nn.Conv2d(64, 32, (3, 3), padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 32, (3, 3), padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 1, (3, 3), padding=1), nn.Sigmoid())

        self.vex_branch = nn.Sequential(nn.Conv2d(64, 32, (3, 3), padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 32, (3, 3), padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 1, (3, 3), padding=1), nn.Sigmoid())

    def forward(self, x):
        x = self.base(x)
        x = self.dla_up(x[self.first_level:])
        x = self.upsample(x)
        seg = self.seg_branch(x)
        vex = self.vex_branch(x)
        return {'seg': seg, 'vex': vex}


class SegDLA(nn.Module):
    def __init__(self, base_name='dla34', pretrained=True, down_ratio=2, eval_mode=False):
        super(SegDLA, self).__init__()
        assert down_ratio in [2, 4, 8, 16]
        self.eval_mode = eval_mode
        self.first_level = int(np.log2(down_ratio))
        self.base = globals()[base_name](pretrained=pretrained, return_levels=True)
        channels = self.base.channels
        scales = [2 ** i for i in range(len(channels[self.first_level:]))]
        self.dla_up = DLAUp(channels[self.first_level:], scales=scales)

        self.upsample = nn.ConvTranspose2d(channels[self.first_level], 64, (4, 4), (2, 2), (1, 1))

        self.seg_branch = nn.Sequential(nn.Conv2d(64, 32, (3, 3), padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 32, (3, 3), padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 1, (3, 3), padding=1), nn.Sigmoid())

    def forward(self, x):
        x = self.base(x)
        x = self.dla_up(x[self.first_level:])
        x = self.upsample(x)
        seg = self.seg_branch(x)
        return {'seg': seg}


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=(3, 3), padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=(2, 2), stride=(2, 2))
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(1, 1))

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=True, eval_mode=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return {'seg': torch.sigmoid(logits)}


class DilateBlock(nn.Module):
    def __init__(self, channel):
        super(DilateBlock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=(3, 3), dilation=(1, 1), padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=(3, 3), dilation=(2, 2), padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=(3, 3), dilation=(4, 4), padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=(3, 3), dilation=(8, 8), padding=8)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = F.relu(self.dilate1(x), inplace=True)
        dilate2_out = F.relu(self.dilate2(dilate1_out), inplace=True)
        dilate3_out = F.relu(self.dilate3(dilate2_out), inplace=True)
        dilate4_out = F.relu(self.dilate4(dilate3_out), inplace=True)
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out
        return out


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, (1, 1))
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nn.ReLU(inplace=True)

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, (3, 3), stride=(2, 2),
                                          padding=(1, 1), output_padding=(1, 1))
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, (1, 1))
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)
        return x


class DLinkNet34(nn.Module):
    def __init__(self, num_classes=1, eval_mode=False):
        super(DLinkNet34, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = DilateBlock(512)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 32, (4, 4), (2, 2), (1, 1))
        self.finalrelu1 = nn.ReLU(inplace=True)
        self.finalconv2 = nn.Conv2d(32, 32, (3, 3), padding=1)
        self.finalrelu2 = nn.ReLU(inplace=True)
        self.finalconv3 = nn.Conv2d(32, num_classes, (3, 3), padding=1)

    def forward(self, x):
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        e4 = self.dblock(e4)

        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        out = self.finaldeconv1(d1)
        out = self.finalrelu1(out)
        out = self.finalconv2(out)
        out = self.finalrelu2(out)
        out = self.finalconv3(out)

        return {'seg': torch.sigmoid(out)}


class ResUNet(nn.Module):
    def __init__(self, eval_mode=False):
        super(ResUNet, self).__init__()
        self.res_path1 = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, (3, 3), padding=1)
        )
        self.shortcut1 = nn.Sequential(
            nn.Conv2d(3, 64, (1, 1)),
            nn.BatchNorm2d(64)
        )
        self.res_block2 = ResBlock([64, 128], [2, 1])
        self.res_block3 = ResBlock([128, 256], [2, 1])
        self.res_block4 = ResBlock([256, 512], [2, 1])
        self.res_block5 = ResBlock([768, 256], [1, 1])
        self.res_block6 = ResBlock([384, 128], [1, 1])
        self.res_block7 = ResBlock([192, 64], [1, 1])
        self.output = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=(1, 1)),
            nn.Sigmoid()
        )

    def forward(self, x):
        block1 = self.res_path1(x) + self.shortcut1(x)
        block2 = self.res_block2(block1)
        block3 = self.res_block3(block2)
        block4 = self.res_block4(block3)
        block5 = self.res_block5(torch.cat([F.interpolate(block4, scale_factor=2, mode='bilinear'), block3], 1))
        block6 = self.res_block6(torch.cat([F.interpolate(block5, scale_factor=2, mode='bilinear'), block2], 1))
        block7 = self.res_block7(torch.cat([F.interpolate(block6, scale_factor=2, mode='bilinear'), block1], 1))

        return {'seg': self.output(block7)}


class ResBlock(nn.Module):
    def __init__(self, channels, strides):
        super(ResBlock, self).__init__()
        self.res_path = nn.Sequential(
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(),
            nn.Conv2d(channels[0], channels[1], kernel_size=(3, 3), stride=strides[0], padding=1),
            nn.BatchNorm2d(channels[1]),
            nn.ReLU(),
            nn.Conv2d(channels[1], channels[1], kernel_size=(3, 3), stride=strides[1], padding=1)
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(channels[0], channels[1], kernel_size=(1, 1), stride=strides[0]),
            nn.BatchNorm2d(channels[1])
        )

    def forward(self, x):
        residual = self.res_path(x)
        x = self.shortcut(x)

        return x + residual


class SegVexPlusOriUNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, bilinear=True, eval_mode=False):
        super(SegVexPlusOriUNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        # self.outc = OutConv(64, n_classes)

        # self.upsample = nn.ConvTranspose2d(channels[self.first_level], 64, (4, 4), (2, 2), (1, 1))

        self.seg_branch = nn.Sequential(nn.Conv2d(64, 32, (3, 3), padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 32, (3, 3), padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 1, (3, 3), padding=1), nn.Sigmoid())

        self.vex_branch = nn.Sequential(nn.Conv2d(64, 32, (3, 3), padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 32, (3, 3), padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 2, (3, 3), padding=1), nn.Sigmoid())

        self.ori_branch = nn.Sequential(nn.Conv2d(66, 32, (3, 3), padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 32, (3, 3), padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 2, (3, 3), padding=1), nn.Tanh())

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        seg = self.seg_branch(x)
        vex = self.vex_branch(x)
        ori = self.ori_branch(torch.cat([x, vex], 1))
        return {'seg': seg, 'vex': vex, 'ori': ori}


class SegVexPlusOriDLinkNet34(nn.Module):
    def __init__(self, num_classes=1, eval_mode=False):
        super(SegVexPlusOriDLinkNet34, self).__init__()

        filters = [64, 128, 256, 512]
        resnet = models.resnet34(pretrained=True)
        self.firstconv = resnet.conv1
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

        self.dblock = DilateBlock(512)

        self.decoder4 = DecoderBlock(filters[3], filters[2])
        self.decoder3 = DecoderBlock(filters[2], filters[1])
        self.decoder2 = DecoderBlock(filters[1], filters[0])
        self.decoder1 = DecoderBlock(filters[0], filters[0])

        self.finaldeconv1 = nn.ConvTranspose2d(filters[0], 64, (4, 4), (2, 2), (1, 1))

        self.seg_branch = nn.Sequential(nn.Conv2d(64, 32, (3, 3), padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 32, (3, 3), padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 1, (3, 3), padding=1), nn.Sigmoid())

        self.vex_branch = nn.Sequential(nn.Conv2d(64, 32, (3, 3), padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 32, (3, 3), padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 2, (3, 3), padding=1), nn.Sigmoid())

        self.ori_branch = nn.Sequential(nn.Conv2d(66, 32, (3, 3), padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 32, (3, 3), padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 2, (3, 3), padding=1), nn.Tanh())

    def forward(self, x):
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        e1 = self.encoder1(x)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        e4 = self.dblock(e4)

        d4 = self.decoder4(e4) + e3
        d3 = self.decoder3(d4) + e2
        d2 = self.decoder2(d3) + e1
        d1 = self.decoder1(d2)

        x = self.finaldeconv1(d1)
        seg = self.seg_branch(x)
        vex = self.vex_branch(x)
        ori = self.ori_branch(torch.cat([x, vex], 1))
        return {'seg': seg, 'vex': vex, 'ori': ori}


class SegVexPlusOriResUNet(nn.Module):
    def __init__(self, eval_mode=False):
        super(SegVexPlusOriResUNet, self).__init__()
        self.res_path1 = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, (3, 3), padding=1)
        )
        self.shortcut1 = nn.Sequential(
            nn.Conv2d(3, 64, (1, 1)),
            nn.BatchNorm2d(64)
        )
        self.res_block2 = ResBlock([64, 128], [2, 1])
        self.res_block3 = ResBlock([128, 256], [2, 1])
        self.res_block4 = ResBlock([256, 512], [2, 1])
        self.res_block5 = ResBlock([768, 256], [1, 1])
        self.res_block6 = ResBlock([384, 128], [1, 1])
        self.res_block7 = ResBlock([192, 64], [1, 1])

        self.seg_branch = nn.Sequential(nn.Conv2d(64, 32, (3, 3), padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 32, (3, 3), padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 1, (3, 3), padding=1), nn.Sigmoid())

        self.vex_branch = nn.Sequential(nn.Conv2d(64, 32, (3, 3), padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 32, (3, 3), padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 2, (3, 3), padding=1), nn.Sigmoid())

        self.ori_branch = nn.Sequential(nn.Conv2d(66, 32, (3, 3), padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 32, (3, 3), padding=1), nn.ReLU(inplace=True), nn.Conv2d(32, 2, (3, 3), padding=1), nn.Tanh())

    def forward(self, x):
        block1 = self.res_path1(x) + self.shortcut1(x)
        block2 = self.res_block2(block1)
        block3 = self.res_block3(block2)
        block4 = self.res_block4(block3)
        block5 = self.res_block5(torch.cat([F.interpolate(block4, scale_factor=2, mode='bilinear', align_corners=True), block3], 1))
        block6 = self.res_block6(torch.cat([F.interpolate(block5, scale_factor=2, mode='bilinear', align_corners=True), block2], 1))
        x = self.res_block7(torch.cat([F.interpolate(block6, scale_factor=2, mode='bilinear', align_corners=True), block1], 1))

        seg = self.seg_branch(x)
        vex = self.vex_branch(x)
        ori = self.ori_branch(torch.cat([x, vex], 1))
        return {'seg': seg, 'vex': vex, 'ori': ori}
