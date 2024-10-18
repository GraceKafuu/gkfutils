# -*- coding: utf-8 -*-

# @Time     : 2023/8/10 15:18
# @Author   : GraceKafuu
# @File     : cls_seg_net_v2.py
# @IDE      : PyCharm

"""
Description:
1.
2.
3.

Implementation Steps:
1.
2.
3.

"""

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F


class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', inplace=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
                              groups=groups, bias=bias, padding_mode=padding_mode)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class CLS_SEG(nn.Module):
    def __init__(self, cls_n, seg_n, in_channels, inner_channels=256, **kwargs):
        """
        :param in_channels: 基础网络输出的维度
        :param kwargs:
        """
        super().__init__()
        self.cls_n = cls_n
        self.seg_n = seg_n
        inplace = True
        self.conv_out = inner_channels
        inner_channels = inner_channels // 4
        # reduce layers
        self.reduce_conv_c2 = ConvBnRelu(in_channels[0], inner_channels, kernel_size=1, inplace=inplace)
        self.reduce_conv_c3 = ConvBnRelu(in_channels[1], inner_channels, kernel_size=1, inplace=inplace)
        self.reduce_conv_c4 = ConvBnRelu(in_channels[2], inner_channels, kernel_size=1, inplace=inplace)
        self.reduce_conv_c5 = ConvBnRelu(in_channels[3], inner_channels, kernel_size=1, inplace=inplace)
        # Smooth layers
        self.smooth_p4 = ConvBnRelu(inner_channels, inner_channels, kernel_size=3, padding=1, inplace=inplace)
        self.smooth_p3 = ConvBnRelu(inner_channels, inner_channels, kernel_size=3, padding=1, inplace=inplace)
        self.smooth_p2 = ConvBnRelu(inner_channels, inner_channels, kernel_size=3, padding=1, inplace=inplace)

        up_in = 256
        self.conv = nn.Sequential(
            nn.Conv2d(self.conv_out, self.conv_out, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(self.conv_out),
            nn.ReLU(inplace=inplace)
        )

        self.up = nn.Sequential(
            nn.Conv2d(self.conv_out, self.conv_out // 4, 3, padding=1),  # [1, 64, 160, 160]
            nn.BatchNorm2d(self.conv_out // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.conv_out // 4, self.conv_out // 4, 2, 2),  # [1, 64, 320, 320]
            nn.BatchNorm2d(self.conv_out // 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.conv_out // 4, self.seg_n, 2, 2),
            nn.Sigmoid())

        self.cls_conv = nn.Sequential(
            nn.Conv2d(self.conv_out, self.conv_out // 2, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(self.conv_out // 2),
            nn.ReLU(inplace=True)
        )
        self.fc = nn.Linear(self.conv_out // 2 * 160 * 160, self.cls_n)

    def forward(self, x):
        c2, c3, c4, c5 = x
        # Top-down
        p5 = self.reduce_conv_c5(c5)
        p4 = self._upsample_add(p5, self.reduce_conv_c4(c4))
        p4 = self.smooth_p4(p4)
        p3 = self._upsample_add(p4, self.reduce_conv_c3(c3))
        p3 = self.smooth_p3(p3)
        p2 = self._upsample_add(p3, self.reduce_conv_c2(c2))
        p2 = self.smooth_p2(p2)

        x = self._upsample_cat(p2, p3, p4, p5)
        x = self.conv(x)

        seg_out = self.up(x)
        cls_conv_out = self.cls_conv(x)
        cls_conv_out = cls_conv_out.view(-1, 128 * 160 * 160)
        cls_out = self.fc(cls_conv_out)

        return seg_out, cls_out
    
    def _upsample_add(self, x, y):
        return F.interpolate(x, size=y.size()[2:]) + y

    def _upsample_cat(self, p2, p3, p4, p5):
        h, w = p2.size()[2:]
        p3 = F.interpolate(p3, size=(h, w))
        p4 = F.interpolate(p4, size=(h, w))
        p5 = F.interpolate(p5, size=(h, w))
        return torch.cat([p2, p3, p4, p5], dim=1)


if __name__ == '__main__':
    cls_seg = CLS_SEG(cls_n=10, seg_n=5, in_channels=[64, 128, 256, 512])
    xn = [torch.rand(1, 64, 160, 160), torch.rand(1, 128, 80, 80), torch.rand(1, 256, 40, 40), torch.rand(1, 512, 20, 20)]
    out = cls_seg(xn)
    print(out.shape)