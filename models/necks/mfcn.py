import torch
import torch.nn as nn

# MFCN: multi-scale feature concat network
__all__ = ["MFCN"]


class MFCN(nn.Module):
    def __init__(self, inplanes, outplanes, instrides, outstrides):
        super(MFCN, self).__init__()

        assert isinstance(inplanes, list)
        assert isinstance(outplanes, list) and len(outplanes) == 1
        assert isinstance(outstrides, list) and len(outstrides) == 1
        assert outplanes[0] == sum(inplanes)  # concat
        self.inplanes = inplanes
        self.outplanes = outplanes
        self.instrides = instrides
        self.outstrides = outstrides

        # Debug: print initialization info
        print(f"MFCN init: inplanes={inplanes}, instrides={instrides}, outstrides={outstrides}")
        self.scale_factors = [
            in_stride / outstrides[0] for in_stride in instrides
        ]  # for resize
        self.upsample_list = [
            nn.UpsamplingBilinear2d(scale_factor=scale_factor)
            for scale_factor in self.scale_factors
        ]

    def forward(self, input):
        features = input["features"]
        assert len(self.inplanes) == len(features)

        # Get target size from outstrides
        # Assume input image size is 224x224 (can be inferred from first feature)
        target_size = 224 // self.outstrides[0]  # e.g., 224 // 16 = 14

        feature_list = []
        # resize & concatenate
        for i in range(len(features)):
            feature = features[i]
            h, w = feature.shape[2], feature.shape[3]

            # Calculate scale factor based on actual feature size
            scale_h = target_size / h
            scale_w = target_size / w

            # Use adaptive resize instead of fixed scale_factor
            if scale_h != 1.0 or scale_w != 1.0:
                import torch.nn.functional as F
                feature_resize = F.interpolate(
                    feature,
                    size=(target_size, target_size),
                    mode='bilinear',
                    align_corners=False
                )
            else:
                feature_resize = feature

            feature_list.append(feature_resize)

        feature_align = torch.cat(feature_list, dim=1)

        return {"feature_align": feature_align, "outplane": self.get_outplanes()}

    def get_outplanes(self):
        return self.outplanes

    def get_outstrides(self):
        return self.outstrides
