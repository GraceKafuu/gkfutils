import torch
import torch.nn as nn
import warnings
import math


# ==============================================================================================================
# ==============================================================================================================
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class UpSample(nn.Module):

    def __init__(self):
        super(UpSample, self).__init__()
        self.up_sample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        return self.up_sample(x)


class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class DWConv(Conv):
    # Depth-wise convolution
    def __init__(self, c1, c2, k=1, s=1, d=1, act=True):  # ch_in, ch_out, kernel, stride, dilation, activation
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), d=d, act=act)


class DWConvTranspose2d(nn.ConvTranspose2d):
    # Depth-wise transpose convolution
    def __init__(self, c1, c2, k=1, s=1, p1=0, p2=0):  # ch_in, ch_out, kernel, stride, padding, padding_out
        super().__init__(c1, c2, k, s, p1, p2, groups=math.gcd(c1, c2))


class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), 1))))


class CrossConv(nn.Module):
    # Cross Convolution Downsample
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        # ch_in, ch_out, kernel, stride, groups, expansion, shortcut
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, (1, k), (1, s))
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # optional act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), 1))


class C3x(C3):
    # C3 module with cross-convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)))


class C3TR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3SPP(C3):
    # C3 module with SPP()
    def __init__(self, c1, c2, k=(5, 9, 13), n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = SPP(c_, c_, k)


class C3Ghost(C3):
    # C3 module with GhostBottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class SPP(nn.Module):
    # Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act=act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat((x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]), 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act=act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act=act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat((y, self.cv2(y)), 1)


class GhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1,
                                                                            act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert (h / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(b, c, h // s, s, w // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(b, c * s * s, h // s, w // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(b, s, s, c // s ** 2, h, w)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(b, c // s ** 2, h * s, w * s)  # x(1,16,160,160)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)
    

class CSPDarkNet(nn.Module):
    def __init__(self, gd=0.33, gw=0.5):
        super(CSPDarkNet, self).__init__()
        self.small = nn.Sequential(
            Conv(3, round(64 * gw), k=6, s=2, p=2),
            Conv(round(64 * gw), round(128 * gw), k=3, s=2),
            C3(round(128 * gw), round(128 * gw), n=1, shortcut=True, e=gw),
            Conv(round(128 * gw), round(256 * gw), k=3, s=2),
            C3(round(256 * gw), round(256 * gw), n=1, shortcut=True, e=gw)
        )
        self.medium = nn.Sequential(
            Conv(round(256 * gw), round(512 * gw), k=3, s=2),
            C3(round(512 * gw), round(512 * gw), n=1, shortcut=True, e=gw)
        )
        self.large = nn.Sequential(
            Conv(round(512 * gw), round(1024 * gw), k=3, s=2),
            C3(round(1024 * gw), round(1024 * gw), n=1, shortcut=True, e=gw),
            SPPF(round(1024 * gw), round(1024 * gw), k=5)
        )

    def forward(self, x):
        small = self.small(x)
        medium = self.medium(small)
        large = self.large(medium)
        return small, medium, large


# ==============================================================================================================
# ==============================================================================================================

def darknet53(gd, gw, pretrained, **kwargs):
    model = CSPDarkNet(gd, gw)
    if pretrained:
        if isinstance(pretrained, str):
            model.load_state_dict(torch.load(pretrained))
        else:
            raise Exception(f"darknet request a pretrained path. got[{pretrained}]")
    return model


class yolov5(nn.Module):
    """添加一个分类分支"""
    def __init__(self, nc=80, gd=0.33, gw=0.5, clsFlag=True, num_class=3):
        super(yolov5, self).__init__()
        # ------------------------------Backbone--------------------------------
        self.backbone = darknet53(gd, gw, None)
        self.clsFlag = clsFlag
        self.num_class = num_class
        self.nc = nc
        self.gd = gd
        self.gw = gw

        # ------------------------------Neck------------------------------------
        self.up_sample = UpSample()
        self.large_1 = nn.Sequential(
            Conv(round(1024 * gw), round(512 * gw), k=1, s=1)
        )
        self.medium_1 = nn.Sequential(
            Conv(round(1024 * gw), round(512 * gw), k=1, s=1),
            UpSample()
        )
        self.small_1 = nn.Sequential(
            C3(round(1024 * gw), round(512 * gw), n=1, shortcut=False),
            Conv(round(512 * gw), round(256 * gw), k=1, s=1),
            UpSample()
        )
        self.small_2 = nn.Sequential(
            C3(round(512 * gw), round(256 * gw), n=1, shortcut=False)
        )
        self.medium_21 = nn.Sequential(
            C3(round(1024 * gw), round(512 * gw), n=1, shortcut=False),
            Conv(round(512 * gw), round(256 * gw), k=1, s=1)
        )
        self.medium_22 = nn.Sequential(
            Conv(round(256 * gw), round(256 * gw), k=3, s=2)
        )
        self.medium_23 = nn.Sequential(
            C3(round(512 * gw), round(512 * gw), n=1, shortcut=False)
        )
        self.large_21 = nn.Sequential(
            Conv(round(512 * gw), round(512 * gw), k=3, s=2)
        )
        self.large_22 = nn.Sequential(
            C3(round(1024 * gw), round(1024 * gw), n=1, shortcut=False)
        )

        # ------------------------------Prediction--------------------------------
        self.small_out = nn.Sequential(
            Conv(round(256 * gw), 3 * (5 + nc), 1, 1, 0)
        )
        self.medium_out = nn.Sequential(
            Conv(round(512 * gw), 3 * (5 + nc), 1, 1, 0)
        )
        self.large_out = nn.Sequential(
            Conv(round(1024 * gw), 3 * (5 + nc), 1, 1, 0)
        )

        self.conv_befor_cls = nn.Sequential(
            # Conv(512, 128, k=3, s=2),
            # Conv(128, 32, k=3, s=2)
            Conv(round(1024 * gw), round(256 * gw), k=3, s=2),
            Conv(round(256 * gw), round(128 * gw), k=3, s=2)
        )
        self.classifier = nn.Sequential(
            nn.Linear(round(128 * gw) * 3 * 3, round(256 * gw)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(round(256 * gw), round(128 * gw)),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(round(128 * gw), self.num_class)
        )

    def forward(self, x):
        small, medium, large = self.backbone(x)

        # -----------------------------------------------------------
        conv_10 = self.large_1(large)
        up1 = self.up_sample(conv_10)
        cat1 = torch.cat([medium, up1], dim=1)
        medium_in1 = self.medium_21(cat1)
        up2 = self.up_sample(medium_in1)
        cat2 = torch.cat([small, up2], dim=1)

        # -----------------------------------------------------------
        c3_small = self.small_2(cat2)
        medium_in2 = self.medium_22(c3_small)
        cat3 = torch.cat([medium_in1, medium_in2], dim=1)
        c3_medium = self.medium_23(cat3)

        large_in1 = self.large_21(c3_medium)
        cat4 = torch.cat([conv_10, large_in1], dim=1)
        c3_large = self.large_22(cat4)

        # ---------------------prediction----------------------------
        small_out = self.small_out(c3_small)
        medium_out = self.medium_out(c3_medium)
        large_out = self.large_out(c3_large)

        if self.clsFlag:
            conv_befor_cls = self.conv_befor_cls(cat4)
            conv_befor_cls = conv_befor_cls.view(-1, round(128 * self.gw) * 3 * 3)
            cls_out = self.classifier(conv_befor_cls)
            return small_out, medium_out, large_out, cls_out
        else:
            return small_out, medium_out, large_out


    

if __name__ == '__main__':
    # 配置文件的写法
    config = {
        #            gd    gw
        'yolov5s': [0.33, 0.50],
        'yolov5m': [0.67, 0.75],
        'yolov5l': [1.00, 1.00],
        'yolov5x': [1.33, 1.25]
    }
    # 修改一次文件名字

    # 单模态多任务

    net_size = config['yolov5s']
    clsFlag = True
    num_class = 3
    net = yolov5(nc=80, gd=net_size[0], gw=net_size[1], clsFlag=clsFlag, num_class=num_class)
    print(net)
    a = torch.randn(1, 3, 384, 384)
    # a = torch.randn(1, 3, 640, 640)
    y = net(a)

    if clsFlag:
        print(y[0].shape, y[1].shape, y[2].shape, y[3].shape)

        y0 = y[0].view(1, -1, 85)
        y1 = y[1].view(1, -1, 85)
        y2 = y[2].view(1, -1, 85)
        print(y0.shape, y1.shape, y2.shape)

        out = torch.cat([y0, y1, y2], dim=1)
        print(out.shape)
    else:
        print(y[0].shape, y[1].shape, y[2].shape)

        y0 = y[0].view(1, -1, 85)
        y1 = y[1].view(1, -1, 85)
        y2 = y[2].view(1, -1, 85)
        print(y0.shape, y1.shape, y2.shape)

        out = torch.cat([y0, y1, y2], dim=1)
        print(out.shape)



