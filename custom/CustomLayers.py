import torch
from torch import nn
from torch.nn import functional as F


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class SeparableConv2D(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: tuple = (3, 3),
                 stride: tuple = (1, 1),
                 padding: tuple = (1, 1),
                 normalize: bool = True,
                 activation: bool = False):
        super(SeparableConv2D, self).__init__()
        self.normalize = normalize
        self.activation = activation

        self.depthwise_conv = nn.Conv2d(in_channels=in_channels, out_channels=in_channels,
                                        kernel_size=kernel_size, stride=stride, padding=padding,
                                        groups=in_channels, bias=False)
        self.pointwise_conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=(1, 1))

        if normalize:
            self.batch_norm = nn.BatchNorm2d(num_features=out_channels,
                                             momentum=1e-2, eps=1e-3)

        if activation:
            self.swish = Swish()

    def forward(self, x):

        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)

        if self.normalize:
            x = self.batch_norm(x)

        if self.activation:
            x = self.swish(x)

        return x


class ConvBlock(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: tuple = (3, 3),
                 stride: tuple = (1, 1),
                 padding: tuple = (1, 1),
                 upsample: bool = False):
        super(ConvBlock, self).__init__()

        self.upsample = upsample

        self.sequential = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                  kernel_size=kernel_size, stride=stride,
                                                  padding=padding, bias=False),
                                        nn.BatchNorm2d(num_features=out_channels, momentum=1e-2, eps=1e-3),
                                        Swish())

        self.sep_conv = SeparableConv2D(in_channels=out_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=stride,
                                        padding=padding)

    def forward(self, x):
        x = self.sequential(x)

        x = self.sep_conv(x)

        if self.upsample:
            x = F.interpolate(x, scale_factor=(2, 2), mode='bilinear', align_corners=True)

        return x


class SegmentationBlock(nn.Sequential):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 upsample_num: int = 0):
        super(SegmentationBlock, self).__init__()

        modules = nn.ModuleList()
        modules.add_module(name='0', module=ConvBlock(in_channels=in_channels,
                                                      out_channels=out_channels))

        for i in range(upsample_num):
            modules.add_module(name="%d" % (i + 1), module=ConvBlock(in_channels=out_channels,
                                                                     out_channels=out_channels,
                                                                     upsample=True))
        super(SegmentationBlock, self).__init__(*modules)


class BiFPN(nn.Module):
    def __init__(self,
                 num_features: int,
                 conv_channels: list = None,
                 attention: bool = True,
                 first_time: bool = True,
                 epsilon: float = 1e-4):
        """
        Similar to You only look once
        In general, backbone only outputs the features under three receptive fields
        Since HybridNets uses five receptive fields, it is necessary to down sample
        to obtain the 4th and 5th receptive field features when calling this module for the first time
        """
        super(BiFPN, self).__init__()

        self.epsilon = epsilon
        self.first_time = first_time

        self.conv6_up = SeparableConv2D(in_channels=num_features, out_channels=num_features)
        self.conv5_up = SeparableConv2D(in_channels=num_features, out_channels=num_features)
        self.conv4_up = SeparableConv2D(in_channels=num_features, out_channels=num_features)
        self.conv3_up = SeparableConv2D(in_channels=num_features, out_channels=num_features)

        self.conv4_down = SeparableConv2D(in_channels=num_features, out_channels=num_features)
        self.conv5_down = SeparableConv2D(in_channels=num_features, out_channels=num_features)
        self.conv6_down = SeparableConv2D(in_channels=num_features, out_channels=num_features)
        self.conv7_down = SeparableConv2D(in_channels=num_features, out_channels=num_features)

        self.p6_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p5_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p4_upsample = nn.Upsample(scale_factor=2, mode='nearest')
        self.p3_upsample = nn.Upsample(scale_factor=2, mode='nearest')

        self.p4_downsample = nn.MaxPool2d(kernel_size=(3, 3),
                                          stride=(2, 2),
                                          padding=(1, 1))
        self.p5_downsample = nn.MaxPool2d(kernel_size=(3, 3),
                                          stride=(2, 2),
                                          padding=(1, 1))
        self.p6_downsample = nn.MaxPool2d(kernel_size=(3, 3),
                                          stride=(2, 2),
                                          padding=(1, 1))
        self.p7_downsample = nn.MaxPool2d(kernel_size=(3, 3),
                                          stride=(2, 2),
                                          padding=(1, 1))

        self.swish = Swish()

        if attention:
            self.p6_w1 = nn.Parameter(data=torch.ones(size=(2,)).float(), requires_grad=True)
            self.p5_w1 = nn.Parameter(data=torch.ones(size=(2,)).float(), requires_grad=True)
            self.p4_w1 = nn.Parameter(data=torch.ones(size=(2,)).float(), requires_grad=True)
            self.p3_w1 = nn.Parameter(data=torch.ones(size=(2,)).float(), requires_grad=True)

            self.p4_w2 = nn.Parameter(data=torch.ones(size=(3,)).float(), requires_grad=True)
            self.p5_w2 = nn.Parameter(data=torch.ones(size=(3,)).float(), requires_grad=True)
            self.p6_w2 = nn.Parameter(data=torch.ones(size=(3,)).float(), requires_grad=True)
            self.p7_w2 = nn.Parameter(data=torch.ones(size=(2,)).float(), requires_grad=True)

        if self.first_time:
            self.p5_to_p6 = nn.Sequential(nn.Conv2d(in_channels=conv_channels[2],
                                                    out_channels=num_features,
                                                    kernel_size=(1, 1)),
                                          nn.BatchNorm2d(num_features=num_features,
                                                         momentum=1e-2, eps=1e-3),
                                          nn.MaxPool2d(kernel_size=(3, 3),
                                                       stride=(2, 2),
                                                       padding=(1, 1)))
            self.p6_to_p7 = nn.Sequential(nn.MaxPool2d(kernel_size=(3, 3),
                                                       stride=(2, 2),
                                                       padding=(1, 1)))

            self.p3_down_channel = nn.Sequential(nn.Conv2d(in_channels=conv_channels[0],
                                                           out_channels=num_features,
                                                           kernel_size=(1, 1)),
                                                 nn.BatchNorm2d(num_features=num_features,
                                                                momentum=1e-2, eps=1e-3))

            self.p4_down_channel = nn.Sequential(nn.Conv2d(in_channels=conv_channels[1],
                                                           out_channels=num_features,
                                                           kernel_size=(1, 1)),
                                                 nn.BatchNorm2d(num_features=num_features,
                                                                momentum=1e-2, eps=1e-3))

            self.p5_down_channel = nn.Sequential(nn.Conv2d(in_channels=conv_channels[2],
                                                           out_channels=num_features,
                                                           kernel_size=(1, 1)),
                                                 nn.BatchNorm2d(num_features=num_features,
                                                                momentum=1e-2, eps=1e-3))

            self.p4_down_channel_2 = nn.Sequential(nn.Conv2d(in_channels=conv_channels[1],
                                                             out_channels=num_features,
                                                             kernel_size=(1, 1)),
                                                   nn.BatchNorm2d(num_features=num_features,
                                                                  momentum=1e-2, eps=1e-3))

            self.p5_down_channel_2 = nn.Sequential(nn.Conv2d(in_channels=conv_channels[2],
                                                             out_channels=num_features,
                                                             kernel_size=(1, 1)),
                                                   nn.BatchNorm2d(num_features=num_features,
                                                                  momentum=1e-2, eps=1e-3))

    def forward(self, inputs):
        """
        :param inputs: Extracted by backbone, usually 3 elements
        :return:
        """
        if self.first_time:
            p3, p4, p5 = inputs

            p3_in = self.p3_down_channel(p3)
            p4_in = self.p4_down_channel(p4)
            p5_in = self.p5_down_channel(p5)

            p6_in = self.p5_to_p6(p5)
            p7_in = self.p6_to_p7(p6_in)
        else:
            p3_in, p4_in, p5_in, p6_in, p7_in = inputs

        # upsample blocks
        # p6_w1 Normalization, avoid parameters<=0
        p6_w1 = torch.maximum(self.p6_w1, torch.zeros(size=()))
        p6_w1 = p6_w1 / (torch.sum(p6_w1) + self.epsilon)

        p6_up = self.conv6_up(self.swish(p6_w1[0] * p6_in + p6_w1[1] * self.p6_upsample(p7_in)))

        # p5_w1 Normalization, avoid parameters<=0
        p5_w1 = torch.maximum(self.p5_w1, torch.zeros(size=()))
        p5_w1 = p5_w1 / (torch.sum(p5_w1) + self.epsilon)

        p5_up = self.conv5_up(self.swish(p5_w1[0] * p5_in + p5_w1[1] * self.p5_upsample(p6_in)))

        # p4_w1 Normalization, avoid parameters<=0
        p4_w1 = torch.maximum(self.p4_w1, torch.zeros(size=()))
        p4_w1 = p4_w1 / (torch.sum(p4_w1) + self.epsilon)

        p4_up = self.conv4_up(self.swish(p4_w1[0] * p4_in + p4_w1[1] * self.p4_upsample(p5_in)))

        # p3_w1 Normalization, avoid parameters<=0
        p3_w1 = torch.maximum(self.p3_w1, torch.zeros(size=()))
        p3_w1 = p3_w1 / (torch.sum(p3_w1) + self.epsilon)

        p3_out = self.conv3_up(self.swish(p3_w1[0] * p3_in + p3_w1[1] * self.p3_upsample(p4_in)))

        if self.first_time:
            p4_in = self.p4_down_channel_2(p4)
            p5_in = self.p5_down_channel_2(p5)

        # downsample blocks
        # p4_w2 Normalization, avoid parameters<=0
        p4_w2 = torch.maximum(self.p4_w2, torch.zeros(size=()))
        p4_w2 = p4_w2 / (torch.sum(p4_w2) + self.epsilon)
        p4_out = self.conv4_down(
            self.swish(p4_w2[0] * p4_in + p4_w2[1] * p4_up + p4_w2[2] * self.p4_downsample(p3_out)))

        # p5_w2 Normalization, avoid parameters<=0
        p5_w2 = torch.maximum(self.p5_w2, torch.zeros(size=()))
        p5_w2 = p5_w2 / (torch.sum(p5_w2) + self.epsilon)
        p5_out = self.conv5_down(
            self.swish(p5_w2[0] * p5_in + p5_w2[1] * p5_up + p5_w2[2] * self.p5_downsample(p4_out)))

        # p6_w2 Normalization, avoid parameters<=0
        p6_w2 = torch.maximum(self.p6_w2, torch.zeros(size=()))
        p6_w2 = p6_w2 / (torch.sum(p6_w2) + self.epsilon)
        p6_out = self.conv6_down(
            self.swish(p6_w2[0] * p6_in + p6_w2[1] * p6_up + p6_w2[2] * self.p6_downsample(p5_out)))

        # p7_w2 Normalization, avoid parameters<=0
        p7_w2 = torch.maximum(self.p7_w2, torch.zeros(size=()))
        p7_w2 = p7_w2 / (torch.sum(p7_w2) + self.epsilon)
        p7_out = self.conv7_down(
            self.swish(p7_w2[0] * p7_in + p7_w2[1] * self.p7_downsample(p6_out)))

        return p3_out, p4_out, p5_out, p6_out, p7_out


class Classifier(nn.Module):
    def __init__(self,
                 num_features: int,
                 num_anchors: int,
                 num_classes: int,
                 num_layers: int,
                 pyramid_level: int = 5):
        super(Classifier, self).__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        self.conv_blocks = nn.ModuleList([SeparableConv2D(in_channels=num_features,
                                                          out_channels=num_features,
                                                          normalize=False)
                                          for i in range(num_layers)])
        self.bn_list = nn.ModuleList([nn.ModuleList([nn.BatchNorm2d(num_features=num_features)
                                                     for j in range(num_layers)])
                                      for i in range(pyramid_level)])
        self.header = SeparableConv2D(in_channels=num_features,
                                      out_channels=num_classes * num_anchors,
                                      normalize=False)
        self.swish = Swish()

    def forward(self, inputs):

        features = []
        for batch_norms, feat in zip(self.bn_list, inputs):
            for batch_norm, conv in zip(batch_norms, self.conv_blocks):
                feat = conv(feat)
                feat = batch_norm(feat)
                feat = self.swish(feat)
            feat = self.header(feat)

            feat = feat.permute(0, 2, 3, 1)
            feat = feat.reshape(*feat.size()[:-1], self.num_anchors, self.num_classes)
            feat = feat.reshape(feat.size(0), -1, self.num_anchors, self.num_classes)

            features.append(feat)

        features = torch.cat(features, dim=1)
        # features = torch.sigmoid(features)
        features = torch.softmax(features, dim=-1)

        return features


class Regressor(nn.Module):
    def __init__(self,
                 num_features: int,
                 num_anchors: int,
                 num_layers: int,
                 pyramid_level: int = 5):
        super(Regressor, self).__init__()
        self.num_anchors = num_anchors
        self.conv_blocks = nn.ModuleList([SeparableConv2D(in_channels=num_features,
                                                          out_channels=num_features,
                                                          normalize=False)
                                          for i in range(num_layers)])
        self.bn_list = nn.ModuleList([nn.ModuleList([nn.BatchNorm2d(num_features=num_features)
                                                     for j in range(num_layers)])
                                      for i in range(pyramid_level)])
        self.header = SeparableConv2D(in_channels=num_features,
                                      out_channels=4 * num_anchors,
                                      normalize=False)
        self.swish = Swish()

    def forward(self, inputs):

        features = []
        for batch_norms, feat in zip(self.bn_list, inputs):
            for batch_norm, conv in zip(batch_norms, self.conv_blocks):
                feat = conv(feat)
                feat = batch_norm(feat)
                feat = self.swish(feat)
            feat = self.header(feat)

            feat = feat.permute(0, 2, 3, 1)
            feat = feat.reshape(*feat.size()[:-1], self.num_anchors, 4)
            feat = feat.reshape(feat.size(0), -1, self.num_anchors, 4)

            features.append(feat)

        features = torch.cat(features, dim=1)

        return features


class Activation(nn.Module):
    def __init__(self,
                 name):
        super(Activation, self).__init__()
        if name is None:
            self.activation = nn.Identity()
        elif name == "sigmoid":
            self.activation = nn.Sigmoid()
        elif name == "softmax":
            self.activation = nn.Softmax(dim=1)
        elif name == "log_softmax":
            self.activation = nn.LogSoftmax()
        else:
            raise ValueError('Activation should be sigmoid/softmax/logsoftmax/None; got {}'.format(name))

    def forward(self, x):

        return self.activation(x)


class MergeBlock(nn.Module):
    def __init__(self,
                 mode: str):
        super(MergeBlock, self).__init__()
        assert mode in ['add', 'cat']
        self.mode = mode

    def forward(self, x):

        if self.mode == 'add':
            x = sum(x)
        else:
            x = torch.cat(x, dim=1)

        return x


class BiFPNDecoder(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 embed_channels: int,
                 upsample_nums: list = [5, 4, 3, 2, 1],
                 merge_mode: str = 'add',
                 drop_rate: float = .2):
        """
        According to the size of inputs,
        the features under the five receptive fields are upsampled for different times and superposed
        """
        super(BiFPNDecoder, self).__init__()

        # self.drop_rate = drop_rate

        self.seg_blocks = nn.ModuleList([SegmentationBlock(in_channels=in_channels,
                                                           out_channels=out_channels,
                                                           upsample_num=upsample_num)
                                         for upsample_num in upsample_nums])

        self.seg_p2 = SegmentationBlock(in_channels=embed_channels,
                                        out_channels=out_channels)

        self.drop_out = nn.Dropout2d(p=drop_rate, inplace=True)

        self.merge = MergeBlock(mode=merge_mode)

    def forward(self, inputs: list):
        p2 = inputs[0]

        features = [seg_block(input) for seg_block, input in zip(self.seg_blocks,
                                                                 list(reversed(inputs))[:-1])]

        p2 = self.seg_p2(p2)

        x = self.merge([p2, *features])

        # x = F.dropout2d(x, p=self.drop_rate, inplace=True)
        x = self.drop_out(x)

        return x

class SegmentationHead(nn.Sequential):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: tuple = (1, 1),
                 scale_factor: tuple = (2, 2),
                 activation: str = 'sigmoid'):
        super(SegmentationHead, self).__init__()

        conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                           kernel_size=kernel_size, padding=(kernel_size[0]//2, kernel_size[1]//2))
        upsample = nn.UpsamplingBilinear2d(scale_factor=scale_factor)
        activation = Activation(name=activation)

        super(SegmentationHead, self).__init__(conv2d, upsample, activation)
