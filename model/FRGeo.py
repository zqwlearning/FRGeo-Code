import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random
from .convnext import convnext_tiny

class FRGeo(nn.Module):
    """
    Simple Siamese baseline with avgpool
    """
    def __init__(self,  args, base_encoder=None):
        """
        dim: feature dimension (default: 512)
        """
        super(FRGeo, self).__init__()
        self.dim = args.dim

        # create the encoders
        # num_classes is the output fc dimension

        if args.dataset == 'vigor':
            self.size_sat = [320, 320]
            self.size_sat_default = [320, 320]
            self.size_grd = [320, 640]
        elif args.dataset == 'cvusa':
            self.size_sat = [256, 256]
            self.size_sat_default = [256, 256]
            self.size_grd = [128, 512]
        elif args.dataset == 'cvact':
            self.size_sat = [256, 256]
            self.size_sat_default = [256, 256]
            self.size_grd = [128, 512]

        if args.fov != 0:
            self.size_grd[1] = int(args.fov / 360. * self.size_grd[1])

        self.ratio = self.size_sat[0]/self.size_sat_default[0]
        base_model = convnext_tiny

        self.query_net = base_model(pretrained=True, in_22k=False)
        self.reference_net = base_model(pretrained=True, in_22k=False)
        self.polar = None

    def forward(self, im_q, im_k):
        return self.query_net(im_q), self.reference_net(x=im_k)
