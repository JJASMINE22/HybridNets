import timm
import torch
from torch import nn
from custom.CustomLayers import BiFPN, BiFPNDecoder, SegmentationHead, Classifier, Regressor

class CreateModel(nn.Module):
    def __init__(self,
                 fpn_cells: int,
                 num_layers: int,
                 num_anchors: int,
                 num_classes: int,
                 seg_classes: int,
                 num_features: int,
                 conv_channels: list,
                 out_indices: tuple,
                 up_scale: tuple,
                 backbone: str):
        super(CreateModel, self).__init__()
        assert out_indices.__len__() == conv_channels.__len__()

        self.encoder = timm.create_model(backbone, pretrained=True, features_only=True,
                                         out_indices=out_indices)  # P3,P4,P5

        self.bifpn = nn.Sequential(*[BiFPN(num_features=num_features,
                                           conv_channels=conv_channels[1:] if not i else None,
                                           attention=True,
                                           first_time=True if not i else False)
                                     for i in range(fpn_cells)])

        self.bifpn_decoder = BiFPNDecoder(in_channels=num_features,
                                          out_channels=num_features,
                                          embed_channels=conv_channels[0])

        self.classifer = Classifier(num_features=num_features,
                                    num_anchors=num_anchors,
                                    num_classes=num_classes,
                                    num_layers=num_layers)

        self.regressor = Regressor(num_features=num_features,
                                   num_anchors=num_anchors,
                                   num_layers=num_layers)

        self.segmentation_head = SegmentationHead(in_channels=num_features,
                                                  out_channels=seg_classes,
                                                  scale_factor=up_scale)

        self.initialize_decoder(self.bifpn)
        self.initialize_decoder(self.bifpn_decoder)
        self.initialize_header(self.segmentation_head)

    def forward(self, input):

        p2, p3, p4, p5 = self.encoder(input)

        p3, p4, p5, p6, p7 = self.bifpn([p3, p4, p5])

        feats = self.bifpn_decoder([p2, p3, p4, p5, p6, p7])

        segmentations = self.segmentation_head(feats)

        classifications = self.classifer([p3, p4, p5, p6, p7])

        regressions = self.regressor([p3, p4, p5, p6, p7])

        return regressions, classifications, segmentations

    def initialize_decoder(self, model):

        for module in model.modules():

            if isinstance(module, nn.Conv2d):

                nn.init.kaiming_uniform_(module.weight, mode="fan_in", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

            elif isinstance(module, nn.BatchNorm2d):

                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

            elif isinstance(module, nn.Linear):

                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def initialize_header(self, model):

        for module in model.modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def split_weights(self):

        weights, bias = [], []
        for named_param in self.named_parameters():
            name, param = named_param
            if param.requires_grad:
                if name.split('.')[-1] == 'weight':
                    weights.append(param)
                elif name.split('.')[-1] == 'bias':
                    bias.append(param)
                else:
                    weights.append(param)

        return weights, bias
