import os
import onnx
import torch
import _utils
import numpy as np
import onnxruntime as ort
from net.networks import CreateModel
from configure import config as cfg

class ONNX:
    def __init__(self,
                 fpn_cells,
                 num_layers,
                 num_anchors,
                 num_classes,
                 seg_classes,
                 num_features,
                 conv_channels,
                 out_indices,
                 up_scale,
                 backbone,
                 batch_size,
                 onnx_path,
                 opset_version,
                 providers):
        """
        Generation and use of onnx model
        """

        self.batch_size = batch_size
        self.onnx_path = onnx_path
        self.opset_version = opset_version
        self.providers = providers

        self.model = CreateModel(fpn_cells=fpn_cells,
                                 num_layers=num_layers,
                                 num_anchors=num_anchors,
                                 num_classes=num_classes,
                                 seg_classes=seg_classes,
                                 num_features=num_features,
                                 conv_channels=conv_channels,
                                 out_indices=out_indices,
                                 up_scale=up_scale,
                                 backbone=backbone)

        try:
            ckpt = torch.load(os.path.join(cfg.ckpt_path, "模型文件"))
            self.model.load_state_dict(ckpt['state_dict'])
            self.model.eval()
            print("model successfully loaded, loss is {:.3f}".format(ckpt['loss']))
        except FileNotFoundError:
            raise ("please enter the right params path")

        self.load_onnx()

    def generate_onnx(self):

        input_names = ["input"]
        output_names = ["regressions", "classifications", "segmentations"]

        sources = torch.randn(self.batch_size, 3, *cfg.input_size)

        torch.onnx.export(model=self.model, args=sources, f=self.onnx_path,
                          operator_export_type=torch.onnx.OperatorExportTypes.ONNX_ATEN,
                          input_names=input_names, output_names=output_names,
                          opset_version=self.opset_version)

    def load_onnx(self):

        self.generate_onnx()

        model = onnx.load(f=self.onnx_path)
        onnx.checker.check_model(model)

        self.session = ort.InferenceSession(path_or_bytes=self.onnx_path,
                                            providers=self.providers)

    def __call__(self, x, *args, **kwargs):

        output_names = [output.name for output in self.session.get_outputs()]

        outputs = self.session.run(output_names, input_feed={'input': x})

        return outputs

if __name__ == '__main__':

    onnx_model = ONNX(fpn_cells=cfg.fpn_cells,
                      num_layers=cfg.num_layers,
                      num_anchors=cfg.num_anchors,
                      num_classes=cfg.class_names.__len__() + 1,
                      seg_classes=cfg.segmentation_class_names.__len__(),
                      num_features=cfg.num_features,
                      conv_channels=cfg.conv_channels,
                      out_indices=cfg.out_indices,
                      up_scale=cfg.up_scale,
                      backbone=cfg.backbone,
                      batch_size=1,
                      onnx_path='.\\onnx_model\\hybridnet.onnx',
                      opset_version=11,
                      providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
