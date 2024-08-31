# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""
YOLO-specific modules.

Usage:
    $ python models/yolo.py --cfg yolov5s.yaml
"""

import argparse
import contextlib
import math
import os
import platform
import sys
from copy import deepcopy
from pathlib import Path

import torch
import torch.nn as nn

FILE = Path(__file__).resolve()
ROOT = FILE.parents[1]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if platform.system() != "Windows":
    ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import (
    SE_Block,
    C3,
    C3SPP,
    C3TR,
    SPP,
    SPPF,
    Bottleneck,
    BottleneckCSP,
    C3Ghost,
    C3x,
    Classify,
    Concat,
    Contract,
    Conv,
    CrossConv,
    DetectMultiBackend,
    DWConv,
    DWConvTranspose2d,
    Expand,
    Focus,
    GhostBottleneck,
    GhostConv,
    Proto,
    FEM,
    PSA,
    SPPFCSPC,
    C2,
    C2f,
    FEMS,
    FFM_Concat2,
    FFM_Concat3,
    C2fCIBAttention,
    SE,
    G_bneck,
    C3_DCN,
    LSKblock,
    CoordAtt,
    C3_CA,
    DSConv,
    DSConv_C3,
    SCAM,
    SCAMS,
    SCAMSE,
)

from models.CSPPC import CSPPC
from models.mobile import MobileViTv2_Block

from models.C2FGhost import C2fGhost

from models.MLLA import MLLAttention
from models.experimental import MixConv2d
from models.SEAttention import SEAttention
from models.ema import EMA
from models.selfattention import SelfAttention
from models.mul import MultiSpectralAttentionLayer
from models.SPPELAN import SPPELAN
from utils.autoanchor import check_anchor_order
from utils.general import LOGGER, check_version, check_yaml, colorstr, make_divisible, print_args
from utils.plots import feature_visualization
from utils.torch_utils import (
    fuse_conv_and_bn,#合并卷积层和批量归一化层。
    initialize_weights,#初始化模型的权重。
    model_info,#获取模型的信息，比如层次结构、参数数量等
    profile,#对模型进行性能分析，通常是为了找出性能瓶颈
    scale_img,#调整图像大小。
    select_device,#选择运行模型的设备，比如 CPU 或 GPU。
    time_sync,#时间同步，可能是用于在分布式系统中保持时间同步。
)

try:
    import thop  # for FLOPs computation
except ImportError:  #处理导入失败的情况
    thop = None


class Detect(nn.Module):
    # YOLOv5 Detect head for detection models
    stride = None  # strides computed during build   (类属性，用于指示特征图的步幅。)
    dynamic = False  # force grid reconstruction   (类属性，它用于控制是否强制使用动态网格重建。)
    export = False  # export mode  （export 是一个类属性，初始化为 False。它用于指示模型是否处于导出模式。）

    def __init__(self, nc=80, anchors=(), ch=(), inplace=True):
        """Initializes YOLOv5 detection layer with specified classes, anchors, channels, and inplace operations."""
        super().__init__()
        self.nc = nc  # number of classes
        self.no = nc + 5  # number of outputs per anchor
        self.nl = len(anchors)  # number of detection layers
        self.na = len(anchors[0]) // 2  # number of anchors
        self.grid = [torch.empty(0) for _ in range(self.nl)]  # init grid
        self.anchor_grid = [torch.empty(0) for _ in range(self.nl)]  # init anchor grid
        self.register_buffer("anchors", torch.tensor(anchors).float().view(self.nl, -1, 2))  # shape(nl,na,2)
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        self.inplace = inplace  # use inplace ops (e.g. slice assignment)

    def forward(self, x):     #这段代码完成了 Detect 类的前向传播方法的定义，用于对输入数据进行处理，并生成检测结果。
        #这是 Detect 类的前向传播方法 forward() 的定义，它接受一个参数 x，代表输入数据
        """Processes input through YOLOv5 layers, altering shape for detection: `x(bs, 3, ny, nx, 85)`."""
        #这是对 forward() 方法的文档字符串，说明该方法的作用和输入数据的格式
        z = []  # inference output
        #创建了一个空列表 z，用于存储推理输出。
        for i in range(self.nl):     #对每个检测层进行迭代处理。
            x[i] = self.m[i](x[i])  # conv     (对输入数据 x 经过第 i 个卷积层进行卷积操作，更新 x[i])
            bs, _, ny, nx = x[i].shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
            #获取 x[i] 的形状信息，其中 bs 表示批量大小，ny 和 nx 分别表示特征图的高度和宽度
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
                #调整 x[i] 的形状，以便后续处理。将其视为形状为 (bs, na, no, ny, nx) 的张量，并重新排列维度顺序
            if not self.training:  # inference  如果模型处于推理模式下
                if self.dynamic or self.grid[i].shape[2:4] != x[i].shape[2:4]:
                    self.grid[i], self.anchor_grid[i] = self._make_grid(nx, ny, i)
                        #如果需要动态网格重建或者网格形状与输入张量不匹配，则重新创建网格。

                if isinstance(self, Segment):  # (boxes + masks)  如果 self 类型为 Segment，即含有分割功能
                    xy, wh, conf, mask = x[i].split((2, 2, self.nc + 1, self.no - self.nc - 5), 4)
                    #将输入张量 x[i] 拆分为坐标 xy、宽高 wh、置信度 conf 和掩码 mask
                    xy = (xy.sigmoid() * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh.sigmoid() * 2) ** 2 * self.anchor_grid[i]  # wh
                    #对坐标和宽高进行解码，得到真实的边界框坐标和尺寸
                    y = torch.cat((xy, wh, conf.sigmoid(), mask), 4)
                    #将解码后的结果拼接为输出张量 y。
                else:  # Detect (boxes only)    否则，即执行目标检测任务
                    xy, wh, conf = x[i].sigmoid().split((2, 2, self.nc + 1), 4)
                    #将输入张量 x[i] 的预测结果拆分为坐标 xy、宽高 wh 和置信度 conf。
                    xy = (xy * 2 + self.grid[i]) * self.stride[i]  # xy
                    wh = (wh * 2) ** 2 * self.anchor_grid[i]  # wh
                    #对坐标和宽高进行解码，得到真实的边界框坐标和尺寸
                    y = torch.cat((xy, wh, conf), 4)
                    #将解码后的结果拼接为输出张量 y
                z.append(y.view(bs, self.na * nx * ny, self.no))
                #将解码后的结果拼接为输出张量 y

        return x if self.training else (torch.cat(z, 1),) if self.export else (torch.cat(z, 1), x)
                    #如果模型处于训练模式下，返回处理后的输入张量 x；如果处于导出模式下，返回推理输出列表 z；否则返回推理输出列表 z 和处理后的输入张量 x
    def _make_grid(self, nx=20, ny=20, i=0, torch_1_10=check_version(torch.__version__, "1.10.0")):
        """Generates a mesh grid for anchor boxes with optional compatibility for torch versions < 1.10."""
        d = self.anchors[i].device
        t = self.anchors[i].dtype
        shape = 1, self.na, ny, nx, 2  # grid shape
        y, x = torch.arange(ny, device=d, dtype=t), torch.arange(nx, device=d, dtype=t)
        #使用 torch.arange 创建表示沿 y 轴和 x 轴的索引的张量 y 和 x
        yv, xv = torch.meshgrid(y, x, indexing="ij") if torch_1_10 else torch.meshgrid(y, x)  # torch>=0.7 compatibility
        #使用 torch.meshgrid 构造网格 yv 和 xv，如果 Torch 版本为 1.10 或更高，则使用 'ij' 索引，否则使用默认行为
        grid = torch.stack((xv, yv), 2).expand(shape) - 0.5  # add grid offset, i.e. y = 2.0 * x - 0.5
        #通过堆叠 xv 和 yv 并添加网格偏移量 -0.5 来构造网格张量。
        anchor_grid = (self.anchors[i] * self.stride[i]).view((1, self.na, 1, 1, 2)).expand(shape)
        return grid, anchor_grid
        #将锚框张量乘以其相应的步长，并重塑以匹配网格张量的形状，构造锚框网格张量。
        #返回网格张量和锚框网格张量作为一个元组

        #该方法实质上生成了一组网格点和相应的锚框，这在目标检测算法（如 YOLO 或 SSD）中使用。网格由 nx * ny 个点组成，对于每个点，都定义了 self.na 个锚框


class Segment(Detect):
    # YOLOv5 Segment head for segmentation models
    def __init__(self, nc=80, anchors=(), nm=32, npr=256, ch=(), inplace=True):
        #nc：类别数量（默认值为 80）。anchors：锚框。nm：掩膜数量（默认值为 32）。npr：原型数量（默认值为 256）。ch：通道数。inplace：一个布尔值，表示是否进行原地操作（默认值为 True）
        """Initializes YOLOv5 Segment head with options for mask count, protos, and channel adjustments."""
        super().__init__(nc, anchors, ch, inplace)
        self.nm = nm  # number of masks
        self.npr = npr  # number of protos
        self.no = 5 + nc + self.nm  # number of outputs per anchor
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)  # output conv
        #m：是一个模块列表，其中的每个元素都是一个 nn.Conv2d 对象，用于输出预测结果。这些卷积层的输入通道数分别来自参数 ch 中的每个值，输出通道数为 self.no * self.na
        self.proto = Proto(ch[0], self.npr, self.nm)  # protos proto：是一个 Proto 对象，它的初始化函数接受 ch[0]（ch 中的第一个值）、self.npr 和 self.nm 作为参数。
        self.detect = Detect.forward

        #这个类似乎是用于实现 YOLOv5 的分割头部，它包括了用于分割模型的掩膜数量、原型数量和通道调整等选项

    def forward(self, x):
        """Processes input through the network, returning detections and prototypes; adjusts output based on
        training/export mode.
        """
        p = self.proto(x[0])
        x = self.detect(self, x)
        return (x, p) if self.training else (x[0], p) if self.export else (x[0], p, x[1])


class BaseModel(nn.Module):   #它的 forward 方法用于执行单尺度的推理或训练过程，同时支持性能分析和可视化。
    """YOLOv5 base model."""

    def forward(self, x, profile=False, visualize=False):
        #profile：一个布尔值，表示是否进行性能分析（默认为 False）。visualize：一个布尔值，表示是否进行可视化（默认为 False）。
        """Executes a single-scale inference or training pass on the YOLOv5 base model, with options for profiling and
        visualization.
        """
        return self._forward_once(x, profile, visualize)  # single-scale inference, train
        #方法内部调用了 _forward_once 方法，传递了参数 x、profile 和 visualize，并返回其结果。这表明 _forward_once 方法可能是类中的另一个方法，负责执行单次前向传播的具体逻辑。
    def _forward_once(self, x, profile=False, visualize=False):
        """Performs a forward pass on the YOLOv5 model, enabling profiling and feature visualization options."""
        y, dt = [], []  # outputs
        for m in self.model:
            if m.f != -1:  # if not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]  # from earlier layers
            #如果 m.f != -1，即不是从前一层获取输入，则根据 m.f 来确定输入 x。如果 m.f 是整数，则表示从先前的层获取输入；如果 m.f 是列表，则根据列表中的索引获取输入。
            if profile:   #如果开启了性能分析 (profile)，则调用 _profile_one_layer 方法来对当前层进行性能分析。
                self._profile_one_layer(m, x, dt)
            x = m(x)  # run  执行当前层的前向传播操作，将结果保存在 x 中。
            y.append(x if m.i in self.save else None)  # save output
            #如果当前层的索引 m.i 在 self.save 中，则将当前层的输出保存到列表 y 中。
            if visualize:   #如果开启了特征可视化 (visualize)，则调用 feature_visualization 方法对当前层的输出进行可视化。
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    def _profile_one_layer(self, m, x, dt):   #这个 _profile_one_layer 方法用于对单个层的性能进行分析，包括计算 GFLOPs（十亿次浮点运算）、执行时间和参数数量
        """Profiles a single layer's performance by computing GFLOPs, execution time, and parameters."""
        c = m == self.model[-1]  # is final layer, copy input as inplace fix
        #首先，方法检查当前层是否是模型的最后一层（通过判断 m == self.model[-1]）。如果是最后一层，则将输入数据进行复制，以修复就地操作的问题（inplace fix）
        o = thop.profile(m, inputs=(x.copy() if c else x,), verbose=False)[0] / 1e9 * 2 if thop else 0  # FLOPs
        #方法使用 thop 库来计算当前层的 GFLOPs。如果 thop 不可用，则将 GFLOPs 设为 0。
        t = time_sync()
        #方法通过调用 time_sync() 函数计算当前层的执行时间。在这里，使用了一个循环来多次运行当前层，以获得更准确的执行时间。计算得到的执行时间单位为毫秒。
        for _ in range(10):
            m(x.copy() if c else x)
        dt.append((time_sync() - t) * 100)
        if m == self.model[0]:
            LOGGER.info(f"{'time (ms)':>10s} {'GFLOPs':>10s} {'params':>10s}  module")
        LOGGER.info(f"{dt[-1]:10.2f} {o:10.2f} {m.np:10.0f}  {m.type}")
        #方法输出当前层的执行时间、GFLOPs 和参数数量。如果当前层是模型的第一层，则输出一个表头。
        if c:
            LOGGER.info(f"{sum(dt):10.2f} {'-':>10s} {'-':>10s}  Total")
            #如果当前层是模型的最后一层，方法还会输出总的执行时间，但不会输出 GFLOPs 和参数数量。


            #这个方法用于对单个层的性能进行分析，并输出执行时间、GFLOPs 和参数数量。

    def fuse(self):  #这段代码定义了一个名为 fuse 的方法，用于在模型中融合 Conv2d() 和 BatchNorm2d() 层，以提高推理速度。
        """Fuses Conv2d() and BatchNorm2d() layers in the model to improve inference speed."""
        LOGGER.info("Fusing layers... ")
        for m in self.model.modules():  #方法内部遍历模型中的每个模块，通过 self.model.modules() 获取模型的所有模块。
            if isinstance(m, (Conv, DWConv)) and hasattr(m, "bn"):
                #对于每个模块 m，检查它是否是 Conv 或 DWConv 类型，并且具有属性 bn。Conv 和 DWConv 应该是某些特定类型的卷积层
                m.conv = fuse_conv_and_bn(m.conv, m.bn)  # update conv
                #调用 fuse_conv_and_bn 函数来融合卷积层和批标准化层，更新卷积层 m.conv
                delattr(m, "bn")  # remove batchnorm   删除模块 m 的 bn 属性，以移除批标准化层
                m.forward = m.forward_fuse  # update forward
                #更新模块 m 的 forward 方法为 m.forward_fuse，这可能是一个替代的前向传播方法，用于在融合后的模块中使用。
        self.info()  #在遍历完成后，输出信息以显示融合后的模型的详细信息，并返回 self，以便可以连续调用其他方法。
        return self  #这个方法用于在模型中融合卷积层和批标准化层，以提高推理速度。

    def info(self, verbose=False, img_size=640):  #这段代码定义了一个名为 info 的方法，用于打印模型的信息，根据 verbosity（详细程度）和图像大小进行调整
        #verbose：一个布尔值，表示是否打印详细信息（默认为 False）img_size：一个整数，表示图像的大小（默认为 640）。
        """Prints model information given verbosity and image size, e.g., `info(verbose=True, img_size=640)`."""
        model_info(self, verbose, img_size)
        #方法调用了一个名为 model_info 的函数，传递了模型自身 self、verbose 和 img_size 作为参数。这个函数可能是一个外部定义的函数，用于打印模型的信息。
        #在 model_info 函数内部，根据传入的参数，打印了模型的信息。可能会包括模型的层次结构、参数数量、推理速度等。这个方法用于打印模型的信息，可以根据需要设置详细程度和图像大小

    def _apply(self, fn):
        #这段代码定义了一个名为 _apply 的方法，用于对模型张量应用转换，例如 to()、cpu()、cuda()、half()，但不包括参数或已注册的缓冲区
        """Applies transformations like to(), cpu(), cuda(), half() to model tensors excluding parameters or registered
        buffers.
        """
        self = super()._apply(fn) #方法接受一个参数 fn，代表了应用的转换函数。
        #方法调用了父类的 _apply 方法，并将返回结果赋值给 self，以确保模型的张量都应用了相同的转换。
        m = self.model[-1]  # Detect()
        #方法获取模型的最后一个模块 m，通常是 Detect() 或 Segment() 类型。
        if isinstance(m, (Detect, Segment)):#如果 m 是 Detect() 或 Segment() 类型的实例，即为检测或分割模型：
            m.stride = fn(m.stride) #将 m 的步长（stride）应用转换函数 fn
            m.grid = list(map(fn, m.grid)) #对 m 的网格（grid）中的每个张量都应用转换函数 fn
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))  #如果 m 的锚框网格（anchor_grid）是一个列表，也对其中的每个张量应用转换函数 fn。
        return self   #这个方法用于对模型的张量应用转换，但不包括模型的参数或已注册的缓冲区。


class DetectionModel(BaseModel):
    # YOLOv5 detection model
    def __init__(self, cfg="yolov5s.yaml", ch=3, nc=None, anchors=None):
        """Initializes YOLOv5 model with configuration file, input channels, number of classes, and custom anchors."""
        super().__init__()
        if isinstance(cfg, dict):   #如果 cfg 是字典类型，则直接将其作为模型的配置信息
            self.yaml = cfg  # model dict   如果 cfg 是 .yaml 文件路径，则使用 PyYAML 库加载该文件并将其作为模型的配置信息。
        else:  # is *.yaml
            import yaml  # for torch hub

            self.yaml_file = Path(cfg).name
            with open(cfg, encoding="ascii", errors="ignore") as f:
                self.yaml = yaml.safe_load(f)  # model dict

        # Define model
        ch = self.yaml["ch"] = self.yaml.get("ch", ch)  # input channels 获取配置文件中的输入通道数 ch，并覆盖默认值。
        if nc and nc != self.yaml["nc"]: #如果传入了类别数 nc，则将配置文件中的类别数覆盖为传入值。
            LOGGER.info(f"Overriding model.yaml nc={self.yaml['nc']} with nc={nc}")
            self.yaml["nc"] = nc  # override yaml value
        if anchors:
            LOGGER.info(f"Overriding model.yaml anchors with anchors={anchors}")
            self.yaml["anchors"] = round(anchors)  # override yaml value 如果传入了自定义锚框 anchors，则使用传入值覆盖配置文件中的锚框信息。
        self.model, self.save = parse_model(deepcopy(self.yaml), ch=[ch])  # model, savelist
        #解析模型结构，并根据输入通道数构建模型。
        self.names = [str(i) for i in range(self.yaml["nc"])]  # default names
        self.inplace = self.yaml.get("inplace", True)

        # Build strides, anchors
        m = self.model[-1]  # Detect()   根据模型的最后一层类型（Detect 或 Segment），设置模型的步长（stride）和锚框（anchors）。
        if isinstance(m, (Detect, Segment)):
            s = 256  # 2x min stride
            m.inplace = self.inplace
            forward = lambda x: self.forward(x)[0] if isinstance(m, Segment) else self.forward(x)
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])  # forward
            check_anchor_order(m)
            m.anchors /= m.stride.view(-1, 1, 1)
            self.stride = m.stride
            self._initialize_biases()  # only run once

        # Init weights, biases   初始化权重和偏置
        initialize_weights(self)
        self.info()
        LOGGER.info("")            #DetectionModel 类用于初始化和构建 YOLOv5 检测模型，并对其进行配置和初始化。

    def forward(self, x, augment=False, profile=False, visualize=False):
        #这个 forward 方法用于执行单尺度或增强推理，并可以包括性能分析或可视化
        #augment：一个布尔值，表示是否进行增强推理（默认为 False）。
        """Performs single-scale or augmented inference and may include profiling or visualization."""
        if augment:   #如果 augment 为真，则调用 _forward_augment 方法执行增强推理，并返回结果。这个方法可能是类中的另一个方法，用于执行增强推理。
            return self._forward_augment(x)  # augmented inference, None
        return self._forward_once(x, profile, visualize)  # single-scale inference, train
        #如果 augment 为假，则调用 _forward_once 方法执行单尺度推理，同时传递 profile 和 visualize 参数，并返回结果。这个方法可能是类中的另一个方法，用于执行单尺度推理和训练。
    def _forward_augment(self, x):
        """Performs augmented inference across different scales and flips, returning combined detections."""
        img_size = x.shape[-2:]  # height, width
        s = [1, 0.83, 0.67]  # scales   s 包含了三个尺度系数：1、0.83 和 0.67。
        f = [None, 3, None]  # flips (2-ud, 3-lr)  f 包含了对应的翻转方式：None（无翻转）、3（左右翻转）和 None（无翻转）。
        y = []  # outputs   空列表用于输出
        for si, fi in zip(s, f):    #接下来，对于每个尺度系数 si 和对应的翻转方式 fi
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            #如果有翻转，就对输入数据进行翻转操作，使用 flip 方法，将左右翻转的情况用翻转参数 fi 表示
            #对输入数据进行尺度变换，使用 scale_img 方法将图像缩放到目标尺度，并根据模型的最大步长进行网格抽样，其中 gs=int(self.stride.max()) 表示使用模型中最大步长来设置抽样间隔。
            yi = self._forward_once(xi)[0]  # forward
            #对缩放后的输入数据进行单尺度推理，调用 _forward_once 方法，并取其返回的结果的第一个元素（通常是预测结果）
            # cv2.imwrite(f'img_{si}.jpg', 255 * xi[0].cpu().numpy().transpose((1, 2, 0))[:, :, ::-1])  # save
            yi = self._descale_pred(yi, fi, si, img_size) #对预测结果进行反缩放操作，使用 _descale_pred 方法，将预测结果映射回原始图像尺寸。
            y.append(yi)
        y = self._clip_augmented(y)  # clip augmented tails
        #最后，对增强后的预测结果列表进行处理，使用 _clip_augmented 方法剪裁多余的部分，保留有效的检测结果。
        return torch.cat(y, 1), None  # augmented inference, train
        #这个方法用于执行增强推理，包括在不同尺度和翻转情况下对输入图像进行处理，并将多个尺度的预测结果组合在一起。

    def _descale_pred(self, p, flips, scale, img_size):
        #这个 _descale_pred 方法用于将增强推理后的预测结果进行反缩放，根据翻转和图像尺寸进行调整。
        #p：预测结果。lips：翻转方式。scale：尺度系数。img_size：图像尺寸，格式为 (height, width)。
        """De-scales predictions from augmented inference, adjusting for flips and image size."""
        if self.inplace:  #如果模型设置为原地操作（self.inplace 为真），则直接对预测结果的坐标部分进行反缩放：
            p[..., :4] /= scale  # de-scale 将预测结果的前四个元素（坐标信息）除以尺度系数 scale，以进行反缩放。
            if flips == 2:
                p[..., 1] = img_size[0] - p[..., 1]  # de-flip ud
                #如果翻转方式为上下翻转（flips == 2），则将预测结果中的 y 坐标取反，即 img_size[0] - p[..., 1]。
            elif flips == 3:
                p[..., 0] = img_size[1] - p[..., 0]  # de-flip lr
                #如果翻转方式为左右翻转（flips == 3），则将预测结果中的 x 坐标取反，即 img_size[1] - p[..., 0]。
        else:   #如果模型不是原地操作，则创建新的张量来存储反缩放后的结果：
            x, y, wh = p[..., 0:1] / scale, p[..., 1:2] / scale, p[..., 2:4] / scale  # de-scale
            #将预测结果的 x、y 坐标和宽高信息除以尺度系数 scale，以进行反缩放。
            if flips == 2:
                y = img_size[0] - y  # de-flip ud
            elif flips == 3:
                x = img_size[1] - x  # de-flip lr
            p = torch.cat((x, y, wh, p[..., 4:]), -1)#将处理后的坐标信息和原始的类别置信度信息拼接起来。
        return p
            #这个方法用于根据输入的翻转方式和图像尺寸对预测结果进行反缩放，将其映射回原始图像尺寸。

    def _clip_augmented(self, y): #这个方法用于修剪增强推理过程中产生的多余部分，确保输出结果的正确性和一致性。
        #这个 _clip_augmented 方法用于修剪增强推理过程中产生的多余部分，特别是对于 YOLOv5 模型，影响第一个和最后一个张量，基于网格点和层次计数。
        #方法接受一个参数 y，表示增强推理的输出结果
        """Clips augmented inference tails for YOLOv5 models, affecting first and last tensors based on grid points and
        layer counts.
        """
        nl = self.model[-1].nl  # number of detection layers (P3-P5)
        #首先，获取模型中最后一个检测层的数量 nl，即 P3-P5 层的数量。
        g = sum(4**x for x in range(nl))  # grid points
        #接下来，计算网格点数 g，通过对 4 的幂级数求和，范围从 0 到 nl - 1。
        e = 1  # exclude layer count   设置一个变量 e，用于指定要排除的层次计数
        i = (y[0].shape[1] // g) * sum(4**x for x in range(e))  # indices
        #计算要保留的索引 i，通过将张量的列数除以网格点数 g，再乘以 4 的幂级数求和，范围从 0 到 e - 1
        y[0] = y[0][:, :-i]  # large   使用计算出的索引，修剪张量的列，即保留前 :-i 列。
        i = (y[-1].shape[1] // g) * sum(4 ** (nl - 1 - x) for x in range(e))  # indices
        #计算要保留的索引 i，通过将张量的列数除以网格点数 g，再乘以从 nl - 1 - e 到 nl - 1 的 4 的幂级数求和。
        y[-1] = y[-1][:, i:]  # small    使用计算出的索引，修剪张量的列，即保留从第 i 列开始到末尾的所有列
        return y

    def _initialize_biases(self, cf=None):
        #这个 _initialize_biases 方法用于为 YOLOv5 的 Detect() 模块初始化偏置项，并可选择使用类别频率（cf），接受一个参数 cf，表示类别频率（可选）
        """
        Initializes biases for YOLOv5's Detect() module, optionally using class frequencies (cf).

        For details see https://arxiv.org/abs/1708.02002 section 3.3.
        """
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.model[-1]  # Detect() module  首先，获取模型中最后一个模块 m，即 Detect() 模块。
        #对于每个模块 mi 和对应的步长 s，进行以下操作
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            #将当前模块的偏置项 mi.bias 重新形状为 (na, -1)，其中 na 是锚框的数量
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            #将偏置项的第 4 列（索引从 0 开始）加上一个偏置值，该偏置值是根据对象数量和图像尺寸计算得出的。
            b.data[:, 5 : 5 + m.nc] += (
                math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())
            )  # cls   将偏置项的第 5 到第 5 + m.nc 列（索引从 0 开始）加上另一个偏置值，该偏置值是根据类别数量和类别频率（如果提供了）计算得出的
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            #最后将经过调整的偏置项重新赋值给模块的偏置参数，并将其设置为可训练状态

Model = DetectionModel  # retain YOLOv5 'Model' class for backwards compatibility
#这个方法用于根据对象数量、类别数量、图像尺寸和类别频率（可选）来初始化 Detect() 模块的偏置项，以帮助模型更好地适应训练数据和任务要求。


class SegmentationModel(DetectionModel): #用于初始化 YOLOv5 分割模型
    # YOLOv5 segmentation model    调用了父类 DetectionModel 的构造函数,将参数传递给父类来初始化分割模型。
    def __init__(self, cfg="yolov5s-seg.yaml", ch=3, nc=None, anchors=None):
        #cfg：配置文件路径，默认为 "yolov5s-seg.yaml"。nc：类别数，默认为 None。
        """Initializes a YOLOv5 segmentation model with configurable params: cfg (str) for configuration, ch (int) for channels, nc (int) for num classes, anchors (list)."""
        super().__init__(cfg, ch, nc, anchors)


class ClassificationModel(BaseModel): #用于初始化 YOLOv5 分类模型
    # YOLOv5 classification model    ClassificationModel 类用于根据给定的检测模型或配置文件初始化 YOLOv5 分类模型，并提供了灵活的参数配置选项。
    def __init__(self, cfg=None, model=None, nc=1000, cutoff=10):#cutoff：截断索引，默认为 10。
        """Initializes YOLOv5 model with config file `cfg`, input channels `ch`, number of classes `nc`, and `cuttoff`
        index.
        """
        super().__init__()
        self._from_detection_model(model, nc, cutoff) if model is not None else self._from_yaml(cfg)
        #如果提供了已有的检测模型 model，则调用 _from_detection_model 方法来初始化分类模型，并传递给定的类别数和截断索引
        #如果提供了配置文件路径 cfg，则调用 _from_yaml 方法来从配置文件中加载模型的配置信息，并初始化分类模型。
    def _from_detection_model(self, model, nc=1000, cutoff=10):
        #这个 _from_detection_model 方法用于从一个 YOLOv5 检测模型创建一个分类模型，通过指定的截断索引 cutoff 切片模型，并添加一个分类层。
        """Creates a classification model from a YOLOv5 detection model, slicing at `cutoff` and adding a classification
        layer.
        """
        if isinstance(model, DetectMultiBackend): #如果 model 是 DetectMultiBackend 类的实例，则将其解包以获取内部的检测模型
            model = model.model  # unwrap DetectMultiBackend
        model.model = model.model[:cutoff]  # backbone
        #从 model 中提取出要保留的部分，即 model.model[:cutoff]，即截取到指定的截断索引处，作为分类模型的主干骨架。
        m = model.model[-1]  # last layer   获取主干骨架的最后一层，并提取其输入通道数 ch。
        ch = m.conv.in_channels if hasattr(m, "conv") else m.cv1.conv.in_channels  # ch into module
        c = Classify(ch, nc)  # Classify() 创建一个 Classify 实例 c，用于分类，其中传入参数是输入通道数 ch 和类别数 nc
        c.i, c.f, c.type = m.i, m.f, "models.common.Classify"  # index, from, type设置分类层的索引、来源和类型。
        model.model[-1] = c  # replace 将分类层替换为原模型的最后一层，并更新分类模型的属性（如主干骨架、步长、保存列表和类别数）
        self.model = model.model
        self.stride = model.stride
        self.save = []
        self.nc = nc
        #这个方法用于从一个 YOLOv5 检测模型创建一个分类模型，通过切片截断模型并添加一个分类层来完成转换。

    def _from_yaml(self, cfg):#这个 _from_yaml 方法用于从指定的 YAML 配置文件创建一个 YOLOv5 分类模型
        """Creates a YOLOv5 classification model from a specified *.yaml configuration file."""
        self.model = None   #设置了 self.model 为 None，表示暂时不从配置文件中创建模型。
        #这个方法的实现可能是由于在 YOLOv5 分类模型中，通常会在构造函数中使用配置文件来初始化模型，而不是在这个方法中。

def parse_model(d, ch):
    #这个 parse_model 函数用于解析一个 YOLOv5 模型的配置字典 d，根据输入通道 ch 和模型架构配置模型的层次结构
    """Parses a YOLOv5 model from a dict `d`, configuring layers based on input channels `ch` and model architecture."""
    LOGGER.info(f"\n{'':>3}{'from':>18}{'n':>3}{'params':>10}  {'module':<40}{'arguments':<30}")
    #这段代码的作用是打印一个包含模型详细信息的表头，用于在日志中展示每个模块的相关信息
    anchors, nc, gd, gw, act, ch_mul = (
        d["anchors"],
        d["nc"],
        d["depth_multiple"],
        d["width_multiple"],
        d.get("activation"),
        d.get("channel_multiple"),
    )
    if act:
        Conv.default_act = eval(act)  # redefine default activation, i.e. Conv.default_act = nn.SiLU()
        LOGGER.info(f"{colorstr('activation:')} {act}")  # print
        #如果激活函数 act 存在（即不为 None），则重新定义了默认的卷积层激活函数 Conv.default_act 为给定的激活函数。使用 eval(act) 来将字符串形式的激活函数名转换为实际的函数对象。然后通过 LOGGER.info() 打印了激活函数的信息。这一步是为了根据配置文件中提供的激活函数类型重新定义默认激活函数。
    if not ch_mul:
        ch_mul = 8
        #如果通道倍数 ch_mul 不存在或为 0，则将其设为默认值 8。通道倍数用于调整模型的通道数量，以增加或减少模型的复杂度
    na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors  # number of anchors
    #计算了锚框的数量 na。如果 anchors 是一个列表，则表示具有多个尺度的锚框，这里取第一个尺度的锚框数量。如果 anchors 是一个数值，则表示锚框的数量。锚框用于目标检测模型中的预测。
    no = na * (nc + 5)  # number of outputs = anchors * (classes + 5)
    #计算了模型输出的通道数 no。模型的输出通道数等于锚框数量乘以（类别数加上 5），其中 5 表示目标的坐标信息（中心点坐标、宽度、高度）以及置信度分数

    layers, save, c2 = [], [], ch[-1]  # layers, savelist, ch out
    #layers 用于存储模型的每一层，save 用于存储需要保存输出的层次索引，c2 则初始化为输入通道列表 ch 的最后一个值，即模型的输出通道数
    for i, (f, n, m, args) in enumerate(d["backbone"] + d["head"]):  # from, number, module, args
        #使用 enumerate() 遍历了配置字典中的 "backbone" 和 "head" 部分，其中 "backbone" 表示模型的主干网络部分，"head" 表示模型的头部网络部分。对于每个部分，提取了来源 f、数量 n、模块类型 m 和参数 args。
        m = eval(m) if isinstance(m, str) else m  # eval strings
        #对于模块类型 m，如果是字符串类型，则使用 eval() 函数将其转换为实际的类对象；然后遍历参数列表 args，如果参数是字符串类型，则同样使用 eval() 函数将其转换为实际的值。这一步是为了确保模块类型和参数的正确性，并将字符串形式的参数转换为实际的对象。
        for j, a in enumerate(args):
            with contextlib.suppress(NameError):
                args[j] = eval(a) if isinstance(a, str) else a  # eval strings

        n = n_ = max(round(n * gd), 1) if n > 1 else n  # depth gain
        #计算深度增益 n_，其中 gd 是深度倍数，表示了模型的深度相对于原始设计的增益。如果输入的 n 大于 1（即有多个重复模块），则将其乘以深度倍数 gd 并向上取整，否则深度增益为 1。这一步是为了根据配置文件中提供的深度倍数调整模型的深度。
        if m in {
            Conv,
            GhostConv,
            Bottleneck,
            GhostBottleneck,
            SPP,
            SPPF,
            DWConv,
            MixConv2d,
            Focus,
            CrossConv,
            BottleneckCSP,
            C3,
            C3TR,
            C3SPP,
            C3Ghost,
            nn.ConvTranspose2d,
            DWConvTranspose2d,
            C3x,
            SE_Block,
            SEAttention,
            FEM,
            PSA,
            SPPELAN,

            SPPFCSPC,
            MultiSpectralAttentionLayer,
            FEMS,
            EMA,

            FFM_Concat3,
            C2fCIBAttention,
            SelfAttention,
            SE,
            G_bneck,
            C3_DCN,
            CoordAtt,
            C3_CA,
            CSPPC,
            C2fGhost
        }:
            c1, c2 = ch[f], args[0]
            if c2 != no:  # if not output
                c2 = make_divisible(c2 * gw, ch_mul)
                #提取输入通道数 c1 和输出通道数 c2。输入通道数 c1 取自输入通道列表 ch 的索引 f，输出通道数 c2 取自参数列表 args 的第一个参数
                #如果输出通道数 c2 不等于模型的总输出通道数 no，则将其乘以宽度倍数 gw 并调用 make_divisible 函数，确保输出通道数是可以被通道倍数 ch_mul 整除的。这一步是为了根据配置文件中提供的宽度倍数调整模型的宽度，并确保输出通道数是合理的。
            args = [c1, c2, *args[1:]]
            if m in {BottleneckCSP, C3, C3TR, C3Ghost, C3x}:
                args.insert(2, n)  # number of repeats
                n = 1
    #如果模块类型是 BottleneckCSP、C3、C3TR、C3Ghost 或 C3x 中的一种，则在参数列表 args 的第三个位置（索引为 2）插入重复次数 n，然后将 n 设为 1。这是因为这些模块是需要重复多次的模块，而重复次数已经在之前的操作中进行了调整，因此需要在参数列表中插入这个值。
        elif m is nn.BatchNorm2d:
            args = [ch[f]]
        elif m in {MLLAttention}:
            c2=ch[f]
            args=[c2,*args]
        elif m in (DSConv,DSConv_C3):
            c1,c2=ch[f],args[0]
            if c2 !=nc:
                c2=make_divisible(c2* gw,8)
            args=[c1,c2, *args[1:]]
            if m is DSConv_C3:
                args.insert(2,n)
                n=1
    #如果模块类型是 nn.BatchNorm2d，则将参数列表 args 设置为包含输入通道数 ch[f] 的列表。这是因为批量归一化层的参数只与输入通道数相关。
        elif m is Concat:
            c2 = sum(ch[x] for x in f)
    #如果模块类型是 Concat，则将输出通道数 c2 设置为输入通道列表 ch 中索引为 f 的所有通道数的总和。这是因为拼接操作会将所有输入通道连接起来。
        elif m in [MobileViTv2_Block]:
            c1,c2=ch[f],args[0]
            if c2!=no:
                c2=make_divisible(c2*gw,8)
                args=[c1,c2]
            if m in [MobileViTv2_Block]:
                args.insert(2,n)
                n=1
        elif m in [C2f,C2]:
            c1,c2=ch[f],args[0]
            if c2!=no:
                c2=make_divisible(c2*gw,8)
                args=[c1,c2]
            if m in [C2f,C2]:
                args.insert(2,n)
                n=1
        elif m in {SCAM}:
            c2 = ch[f]
            args = [c2]
        elif m in {SCAMS}:
            c2 = ch[f]
            args = [c2]
        elif m in {SCAMSE}:
            c2 = ch[f]
            args = [c2]
        elif m is LSKblock:
            c1=ch[f]
            args=[c1,*args[0:]]
        elif m is FFM_Concat2:
            c2 = sum(ch[x] for x in f)
            args = [args[0], c2 // 2, c2 // 2]
        elif m is FFM_Concat3:
            c2 = sum(ch[x] for x in f)
            args = [args[0], c2 // 4, c2 // 2, c2 // 4]
        # TODO: channel, gw, gd
        elif m in {Detect, Segment}:
            args.append([ch[x] for x in f])
            if isinstance(args[1], int):  # number of anchors
                args[1] = [list(range(args[1] * 2))] * len(f)
            if m is Segment:
                args[3] = make_divisible(args[3] * gw, ch_mul)
    #如果模块类型是 Detect 或 Segment，则将输入通道列表 ch 中索引为 f 的所有通道数添加到参数列表 args 中。如果参数列表中的第二个参数是整数，则将其转换为锚框列表。如果模块类型是 Segment，还会根据宽度倍数 gw 和通道倍数 ch_mul 调整参数列表中的值。
        elif m is Contract:
            c2 = ch[f] * args[0] ** 2
    #如果模块类型是 Contract，则将输出通道数 c2 设置为输入通道数 ch[f] 乘以第一个参数的平方。
        elif m is Expand:
            c2 = ch[f] // args[0] ** 2
    #如果模块类型是 Expand，则将输出通道数 c2 设置为输入通道数 ch[f] 除以第一个参数的平方。
        else:
            c2 = ch[f]  #对于其他模块类型，直接将输出通道数 c2 设置为输入通道列表 ch 中索引为 f 的通道数。

        m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)  # module
        #根据重复次数 n，使用列表推导式构建了一个包含多个模块 m(*args) 的 nn.Sequential 容器 m_。如果重复次数 n 大于 1，则使用 nn.Sequential 将多个相同的模块堆叠起来；否则直接使用单个模块。这一步是为了根据配置文件中指定的重复次数构建模型的深度
        t = str(m)[8:-2].replace("__main__.", "")  # module type
        np = sum(x.numel() for x in m_.parameters())  # number params
        m_.i, m_.f, m_.type, m_.np = i, f, t, np  # attach index, 'from' index, type, number params
        LOGGER.info(f"{i:>3}{str(f):>18}{n_:>3}{np:10.0f}  {t:<40}{str(args):<30}")  # print
        #将需要保存的层的索引添加到保存列表 save 中。如果 f 是整数，则直接将其添加到保存列表中；如果 f 是列表，则遍历其中的每个元素，如果不是 -1，则将其添加到保存列表中。这一步是为了后续对模型进行保存操作做准备。
        save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)  # append to savelist
        layers.append(m_)
        if i == 0:
            ch = []
        ch.append(c2) #最后，将构建好的模型的层列表 layers 转换为 nn.Sequential 容器，并返回给调用者。同时，保存列表 save 也按照索引排序，以确保保存操作的正确性。
    return nn.Sequential(*layers), sorted(save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, default="yolov5s.yaml", help="model.yaml")
    parser.add_argument("--batch-size", type=int, default=1, help="total batch size for all GPUs")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--profile", action="store_true", help="profile model speed")
    parser.add_argument("--line-profile", action="store_true", help="profile model speed layer by layer")
    parser.add_argument("--test", action="store_true", help="test all yolo*.yaml")
    opt = parser.parse_args()
    opt.cfg = check_yaml(opt.cfg)  # check YAML
    print_args(vars(opt))
    device = select_device(opt.device)

    # Create model
    im = torch.rand(opt.batch_size, 3, 640, 640).to(device)
    model = Model(opt.cfg).to(device)

    # Options
    if opt.line_profile:  # profile layer by layer
        model(im, profile=True)

    elif opt.profile:  # profile forward-backward
        results = profile(input=im, ops=[model], n=3)

    elif opt.test:  # test all models
        for cfg in Path(ROOT / "models").rglob("yolo*.yaml"):
            try:
                _ = Model(cfg)
            except Exception as e:
                print(f"Error in {cfg}: {e}")

    else:  # report fused model summary
        model.fuse()
