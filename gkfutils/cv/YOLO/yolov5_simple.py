import torch
import torch.nn as nn


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class UpSample(nn.Module):

    def __init__(self):
        super(UpSample, self).__init__()
        self.up_sample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        return self.up_sample(x)
    

class CBL(nn.Module):

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, e=1.0):
        super(CBL, self).__init__()
        c1 = round(c1 * e)
        c2 = round(c2 * e)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class Focus(nn.Module):

    def __init__(self, c1, c2, k=3, s=1, p=1, g=1, act=True, e=1.0):
        super(Focus, self).__init__()
        c2 = round(c2 * e)
        self.conv = CBL(c1 * 4, c2, k, s, p, g, act)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        flatten_channel = torch.cat([x[..., 0::2, 0::2],
                                     x[..., 1::2, 0::2],
                                     x[..., 0::2, 1::2],
                                     x[..., 1::2, 1::2]], dim=1)
        print(self.conv(flatten_channel).shape)

        return self.conv(flatten_channel)


class SPP(nn.Module):

    def __init__(self, c1, c2, k=(5, 9, 13), e=1.0):
        super(SPP, self).__init__()
        c1 = round(c1 * e)
        c2 = round(c2 * e)
        c_ = c1 // 2
        self.cbl_before = CBL(c1, c_, 1, 1)
        self.max_pool = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cbl_after = CBL(c_ * 4, c2, 1, 1)

    def forward(self, x):  
        x = self.cbl_before(x)
        x_cat = torch.cat([x] + [m(x) for m in self.max_pool], 1)
        return self.cbl_after(x_cat)


class ResUnit_n(nn.Module):

    def __init__(self, c1, c2, n):
        super(ResUnit_n, self).__init__()
        self.shortcut = c1 == c2
        res_unit = nn.Sequential(
            CBL(c1, c1, k=1, s=1, p=0),
            CBL(c1, c2, k=3, s=1, p=1)
        )
        self.res_unit_n = nn.Sequential(*[res_unit for _ in range(n)])

    def forward(self, x):
        return x + self.res_unit_n(x) if self.shortcut else self.res_unit_n(x)


class CSP1_n(nn.Module):

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True, n=1, e=None):
        super(CSP1_n, self).__init__()

        c1 = round(c1 * e[1])
        c2 = round(c2 * e[1])
        n = round(n * e[0])
        c_ = c2 // 2
        self.up = nn.Sequential(
            CBL(c1, c_, k, s, autopad(k, p), g, act),
            ResUnit_n(c_, c_, n),
            # nn.Conv2d(c_, c_, 1, 1, 0, bias=False) 这里最新yolov5结构中去掉了，与网上的结构图稍微有些区别
        )
        self.bottom = nn.Conv2d(c1, c_, 1, 1, 0)
        self.tie = nn.Sequential(
            nn.BatchNorm2d(c_ * 2),
            nn.LeakyReLU(),
            nn.Conv2d(c_ * 2, c2, 1, 1, 0, bias=False)
        )
    def forward(self, x):
        total = torch.cat([self.up(x), self.bottom(x)], dim=1)
        out = self.tie(total)
        return out


class CSP2_n(nn.Module):

    def __init__(self, c1, c2, e=0.5, n=1):
        super(CSP2_n, self).__init__()
        c_ = int(c1 * e)
        cbl_2 = nn.Sequential(
            CBL(c1, c_, 1, 1, 0),
            CBL(c_, c_, 1, 1, 0),
        )
        self.cbl_2n = nn.Sequential(*[cbl_2 for _ in range(n)])
        self.conv_up = nn.Conv2d(c_, c_, 1, 1, 0)
        self.conv_bottom = nn.Conv2d(c1, c_, 1, 1, 0)
        self.tie = nn.Sequential(
            nn.BatchNorm2d(c_ * 2),
            nn.LeakyReLU(),
            nn.Conv2d(c_ * 2, c2, 1, 1, 0)
        )

    def forward(self, x):
        up = self.conv_up(self.cbl_2n(x))
        total = torch.cat([up, self.conv_bottom(x)], dim=1)
        out = self.tie(total)
        return out


class CSPDarkNet(nn.Module):

    def __init__(self, gd=0.33, gw=0.5):
        super(CSPDarkNet, self).__init__()
        self.truck_big = nn.Sequential(
            Focus(3, 64, e=gw),
            CBL(64, 128, k=3, s=2, p=1, e=gw),
            CSP1_n(128, 128, n=3, e=[gd, gw]),
            CBL(128, 256, k=3, s=2, p=1, e=gw),
            CSP1_n(256, 256, n=9, e=[gd, gw]),

        )
        self.truck_middle = nn.Sequential(
            CBL(256, 512, k=3, s=2, p=1, e=gw),
            CSP1_n(512, 512, n=9, e=[gd, gw]),
        )
        self.truck_small = nn.Sequential(
            CBL(512, 1024, k=3, s=2, p=1, e=gw),
            SPP(1024, 1024, e=gw)
        )

    def forward(self, x):
        h_big = self.truck_big(x)  # torch.Size([2, 128, 76, 76])
        h_middle = self.truck_middle(h_big)
        h_small = self.truck_small(h_middle)
        return h_big, h_middle, h_small


def darknet53(gd, gw, pretrained, **kwargs):
    model = CSPDarkNet(gd, gw)
    if pretrained:
        if isinstance(pretrained, str):
            model.load_state_dict(torch.load(pretrained))
        else:
            raise Exception(f"darknet request a pretrained path. got[{pretrained}]")
    return model


class yolov5(nn.Module):

    def __init__(self, nc=80, gd=0.33, gw=0.5):
        super(yolov5, self).__init__()
        # ------------------------------Backbone--------------------------------
        self.backbone = darknet53(gd, gw, None)

        # ------------------------------Neck------------------------------------
        self.neck_small = nn.Sequential(
            CSP1_n(1024, 1024, n=3, e=[gd, gw]),
            CBL(1024, 512, 1, 1, 0, e=gw)
        )
        self.up_middle = nn.Sequential(
            UpSample()
        )
        self.out_set_middle = nn.Sequential(
            CSP1_n(1024, 512, n=3, e=[gd, gw]),
            CBL(512, 256, 1, 1, 0, e=gw),
        )
        self.up_big = nn.Sequential(
            UpSample()
        )
        self.out_set_tie_big = nn.Sequential(
            CSP1_n(512, 256, n=3, e=[gd, gw])
        )

        self.pan_middle = nn.Sequential(
            CBL(256, 256, 3, 2, 1, e=gw)
        )
        self.out_set_tie_middle = nn.Sequential(
            CSP1_n(512, 512, n=3, e=[gd, gw])
        )
        self.pan_small = nn.Sequential(
            CBL(512, 512, 3, 2, 1, e=gw)
        )
        self.out_set_tie_small = nn.Sequential(
            CSP1_n(1024, 1024, n=3, e=[gd, gw])
        )
        # ------------------------------Prediction--------------------------------
        # prediction
        big_ = round(256 * gw)
        middle = round(512 * gw)
        small_ = round(1024 * gw)
        self.out_big = nn.Sequential(
            nn.Conv2d(big_, 3 * (5 + nc), 1, 1, 0)
        )
        self.out_middle = nn.Sequential(
            nn.Conv2d(middle, 3 * (5 + nc), 1, 1, 0)
        )
        self.out_small = nn.Sequential(
            nn.Conv2d(small_, 3 * (5 + nc), 1, 1, 0)
        )

    def forward(self, x):
        h_big, h_middle, h_small = self.backbone(x)
        neck_small = self.neck_small(h_small)  
        # ----------------------------up sample 38*38-------------------------------
        up_middle = self.up_middle(neck_small)
        middle_cat = torch.cat([up_middle, h_middle], dim=1)
        out_set_middle = self.out_set_middle(middle_cat)

        # ----------------------------up sample 76*76-------------------------------
        up_big = self.up_big(out_set_middle)  # torch.Size([2, 128, 76, 76])
        big_cat = torch.cat([up_big, h_big], dim=1)
        out_set_tie_big = self.out_set_tie_big(big_cat)

        # ----------------------------PAN 36*36-------------------------------------
        neck_tie_middle = torch.cat([self.pan_middle(out_set_tie_big), out_set_middle], dim=1)
        up_middle = self.out_set_tie_middle(neck_tie_middle)

        # ----------------------------PAN 18*18-------------------------------------
        neck_tie_small = torch.cat([self.pan_small(up_middle), neck_small], dim=1)
        out_set_small = self.out_set_tie_small(neck_tie_small)

        # ----------------------------prediction-------------------------------------
        out_small = self.out_small(out_set_small)
        out_middle = self.out_middle(up_middle)
        out_big = self.out_big(out_set_tie_big)

        return out_small, out_middle, out_big
    

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
    net_size = config['yolov5s']
    net = yolov5(nc=80, gd=net_size[0], gw=net_size[1])
    print(net)
    a = torch.randn(1, 3, 640, 640)
    y = net(a)
    print(y[0].shape, y[1].shape, y[2].shape)

    y0 = y[0].view(1, -1, 85)
    y1 = y[1].view(1, -1, 85)
    y2 = y[2].view(1, -1, 85)
    print(y0.shape, y1.shape, y2.shape)

    out = torch.cat([y0, y1, y2], dim=1)
    print(out.shape)





