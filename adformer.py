import torch
import torch.nn as nn
import torch.nn.functional as F

class AdaptiveMultiReceptiveField(nn.Module):
    """
    自适应多感受野 (AMRF) 模块
    """
    def __init__(self, in_channels, out_channels, dilation_rates=[1, 2, 4]):
        super().__init__()
        self.dilation_rates = dilation_rates
        
        # 创建多条不同扩张率的卷积路径
        self.conv_paths = nn.ModuleList()
        for d in dilation_rates:
            self.conv_paths.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              padding=d, dilation=d, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
            )
        
        # 权重预测器: 根据输入特征生成各路径的融合权重
        self.weight_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, len(dilation_rates)),
            nn.Softmax(dim=1)  # 权重归一化
        )
        
        # 融合后的卷积
        self.fusion_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 1. 计算各路径的融合权重
        weights = self.weight_predictor(x)  # [B, num_paths]
        weights = weights.view(B, -1, 1, 1)  # 扩维以用于广播乘
        
        # 2. 并行计算各路径特征并按权重融合
        path_outputs = []
        for i, conv in enumerate(self.conv_paths):
            feat = conv(x)
            # 使用对应的权重加权
            weighted_feat = feat * weights[:, i:i+1, :, :]
            path_outputs.append(weighted_feat)
        
        # 3. 求和并融合
        fused = sum(path_outputs)
        return self.fusion_conv(fused)


class AMRFEnhancedPositionEmbedding(nn.Module):
    """
    使用AMRF特征增强的位置编码
    """
    def __init__(self, hidden_dim, amrf_channels):
        super().__init__()
        self.hidden_dim = hidden_dim
        # 将AMRF特征投影到与位置编码相同的维度
        self.amrf_projection = nn.Conv2d(amrf_channels, hidden_dim, kernel_size=1)
        # 标准的可学习位置编码 (例如，对应图像patch序列)
        self.standard_pos_embed = nn.Parameter(torch.randn(1, 196, hidden_dim) * 0.02)  # 假设14x14=196个patch
        
    def forward(self, x, amrf_feature_map):
        """
        x: Transformer的输入token序列 [B, N, C]
        amrf_feature_map: AMRF模块输出的特征图 [B, C_amrf, H, W]
        """
        B, N, C = x.shape
        
        # 1. 投影AMRF特征图
        amrf_proj = self.amrf_projection(amrf_feature_map)  # [B, C, H, W]
        # 2. 将特征图的空间维度展平为序列
        amrf_proj = amrf_proj.flatten(2).transpose(1, 2)  # [B, H*W, C]
        # 3. 与标准位置编码相加，作为增强版的位置信息
        # 注意：这里需要确保H*W等于N (patch数量)，可通过调整AMRF步长实现
        enhanced_pos = self.standard_pos_embed + amrf_proj
        
        return x + enhanced_pos


# 简易的ADformer编码器块示例
class ADformerEncoderLayer(nn.Module):
    def __init__(self, hidden_dim, num_heads, amrf_channels):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        
        # 此处集成了AMRF增强的位置编码
        self.pos_embed = AMRFEnhancedPositionEmbedding(hidden_dim, amrf_channels)
        
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.GELU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
    def forward(self, x, amrf_feature_map):
        # 应用增强的位置编码
        x = self.pos_embed(x, amrf_feature_map)
        
        # 自注意力
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = residual + x
        
        # FFN
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = residual + x
        
        return x


# 简易的ADformer前向流程示例
class SimpleADformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, hidden_dim=768, num_layers=6, num_heads=8):
        super().__init__()
        # 1. Patch Embedding
        self.patch_embed = nn.Conv2d(in_chans, hidden_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (img_size // patch_size) ** 2
        
        # 2. AMRF 前置特征提取模块 (你可以将其放在patch embedding之前或之后)
        self.amrf_module = AdaptiveMultiReceptiveField(in_channels=in_chans, out_channels=hidden_dim)
        
        # 3. ADformer 编码器堆叠
        self.encoder_layers = nn.ModuleList([
            ADformerEncoderLayer(hidden_dim, num_heads, amrf_channels=hidden_dim)
            for _ in range(num_layers)
        ])
        
        # 4. 分类头或解码器...
        
    def forward(self, x):
        B = x.shape[0]
        # 提取AMRF特征图
        amrf_feat = self.amrf_module(x)  # [B, C, H', W']
        
        # 创建图像块序列 (tokens)
        x = self.patch_embed(x)  # [B, C, H_p, W_p]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, C]
        
        # 通过ADformer编码器层
        for layer in self.encoder_layers:
            x = layer(x, amrf_feat)  # 将amrf特征图传入每一层
        
        return x