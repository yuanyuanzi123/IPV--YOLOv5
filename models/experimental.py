# YOLOv5 🚀 by Ultralytics, AGPL-3.0 license
"""Experimental modules."""

import math

import numpy as np
import torch
import torch.nn as nn

from utils.downloads import attempt_download

#Sum类实现了多个层输出的加权和操作，支持两个或两个以上的输入。在初始化过程中，可以选择是否启用权重。如果启用了权重，则会学习到一组权重参数，用于对各个输入进行加权求和。
# 在前向传播过程中，如果启用了权重，则会对每个输入乘以相应的权重，然后进行求和；否则，直接对所有输入进行求和。
class Sum(nn.Module):
    """Weighted sum of 2 or more layers https://arxiv.org/abs/1911.09070."""

    def __init__(self, n, weight=False):
        """Initializes a module to sum outputs of layers with number of inputs `n` and optional weighting, supporting 2+
        inputs.
        """
        super().__init__()
        self.weight = weight  # apply weights boolean
        self.iter = range(n - 1)  # iter object
        if weight:
            self.w = nn.Parameter(-torch.arange(1.0, n) / 2, requires_grad=True)  # layer weights

    def forward(self, x):
        """Processes input through a customizable weighted sum of `n` inputs, optionally applying learned weights."""
        y = x[0]  # no weight
        if self.weight:
            w = torch.sigmoid(self.w) * 2
            for i in self.iter:
                y = y + x[i + 1] * w[i]
        else:
            for i in self.iter:
                y = y + x[i + 1]
        return y

#MixConv2d类实现了混合深度卷积（Mixed Depth-wise Conv）操作，通过结合多个不同大小的卷积核，从而增加了网络的多样性和表征能力。在初始化过程中，可以指定输入通道数 c1、
# 输出通道数 c2、卷积核大小 k、步长 s，以及是否采用相等通道分配策略 equal_ch。在前向传播过程中，将输入通过多个混合深度卷积层，并在通道维度上进行拼接，最后经过批归一化和激活函数输出。
class MixConv2d(nn.Module):
    """Mixed Depth-wise Conv https://arxiv.org/abs/1907.09595."""

    def __init__(self, c1, c2, k=(1, 3), s=1, equal_ch=True):
        """Initializes MixConv2d with mixed depth-wise convolutional layers, taking input and output channels (c1, c2),
        kernel sizes (k), stride (s), and channel distribution strategy (equal_ch).
        """
        super().__init__()
        n = len(k)  # number of convolutions
        if equal_ch:  # equal c_ per group
            i = torch.linspace(0, n - 1e-6, c2).floor()  # c2 indices
            c_ = [(i == g).sum() for g in range(n)]  # intermediate channels
        else:  # equal weight.numel() per group
            b = [c2] + [0] * n
            a = np.eye(n + 1, n, k=-1)
            a -= np.roll(a, 1, axis=1)
            a *= np.array(k) ** 2
            a[0] = 1
            c_ = np.linalg.lstsq(a, b, rcond=None)[0].round()  # solve for equal weight indices, ax = b

        self.m = nn.ModuleList(
            [nn.Conv2d(c1, int(c_), k, s, k // 2, groups=math.gcd(c1, int(c_)), bias=False) for k, c_ in zip(k, c_)]
        )
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()

    def forward(self, x):
        """Performs forward pass by applying SiLU activation on batch-normalized concatenated convolutional layer
        outputs.
        """
        return self.act(self.bn(torch.cat([m(x) for m in self.m], 1)))


#Ensemble类继承自nn.ModuleList，它表示一个模型的集合。在初始化过程中，可以将多个模型添加到集合中。在前向传播过程中，
# 会对输入 x 应用每个模型，并将它们的输出进行拼接。这里的 forward 方法支持 augment（数据增强）、profile（性能分析）和 visualize（可视化）参数。
class Ensemble(nn.ModuleList):
    """Ensemble of models."""

    def __init__(self):
        """Initializes an ensemble of models to be used for aggregated predictions."""
        super().__init__()

    def forward(self, x, augment=False, profile=False, visualize=False):
        """Performs forward pass aggregating outputs from an ensemble of models.."""
        y = [module(x, augment, profile, visualize)[0] for module in self]
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output

#
#attempt_load 函数用于加载和融合一个或多个 YOLOv5 模型的权重。
#它支持从单个权重文件或一个权重文件列表中加载模型。加载模型后，会根据需要进行一些模型兼容性更新，然后将模型添加到 Ensemble 类的实例中。
def attempt_load(weights, device=None, inplace=True, fuse=True):
    """
    Loads and fuses an ensemble or single YOLOv5 model from weights, handling device placement and model adjustments.

    Example inputs: weights=[a,b,c] or a single model weights=[a] or weights=a.
    """
    from models.yolo import Detect, Model

    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        ckpt = torch.load(attempt_download(w), map_location="cpu")  # load
        ckpt = (ckpt.get("ema") or ckpt["model"]).to(device).float()  # FP32 model

        # Model compatibility updates
        if not hasattr(ckpt, "stride"):
            ckpt.stride = torch.tensor([32.0])
        if hasattr(ckpt, "names") and isinstance(ckpt.names, (list, tuple)):
            ckpt.names = dict(enumerate(ckpt.names))  # convert to dict

        model.append(ckpt.fuse().eval() if fuse and hasattr(ckpt, "fuse") else ckpt.eval())  # model in eval mode

    # Module updates
#此代码段用于对模型中的特定模块进行更新。它会遍历模型的所有模块，如果发现特定类型的模块（如激活函数、检测器等），则会更新其 inplace 属性。此外，对于 nn.Upsample 模块，
    # 它还会添加一个 recompute_scale_factor 属性，以保持与 Torch 1.11.0 的兼容性。
    for m in model.modules():
        t = type(m)
        if t in (nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU, Detect, Model):
            m.inplace = inplace
            if t is Detect and not isinstance(m.anchor_grid, list):
                delattr(m, "anchor_grid")
                setattr(m, "anchor_grid", [torch.zeros(1)] * m.nl)
        elif t is nn.Upsample and not hasattr(m, "recompute_scale_factor"):
            m.recompute_scale_factor = None  # torch 1.11.0 compatibility

    # Return model
    if len(model) == 1:
        return model[-1]

    # Return detection ensemble
    print(f"Ensemble created with {weights}\n")
    for k in "names", "nc", "yaml":
        setattr(model, k, getattr(model[0], k))
    model.stride = model[torch.argmax(torch.tensor([m.stride.max() for m in model])).int()].stride  # max stride
    assert all(model[0].nc == m.nc for m in model), f"Models have different class counts: {[m.nc for m in model]}"
    return model
#这段代码用于根据加载的模型权重返回模型或检测集合。如果加载的权重只对应一个模型，则直接返回该模型。如果加载的权重对应多个模型，则创建一个检测集合并返回。
# 在返回检测集合之前，还进行了一些属性的设置，包括设置集合的 names、nc 和 yaml 属性，以及确定集合中模型的最大步长，并将其赋给集合的 stride 属性。
# 如果集合中的模型有不同的类别数量，则会触发断言错误。