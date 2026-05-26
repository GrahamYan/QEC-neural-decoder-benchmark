"""Neural decoder architectures used in the paper."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import brevitas.nn as qnn

from .data import (
    StimTo3DMapper,
    StimToTemporalGridMapper,
    StimToGraphMapper,
    SyndromeConsistencyLoss,
)


def _quant_enabled(quant_config) -> bool:
    return quant_config is not None and getattr(quant_config, "weight_quant", None) is not None


def _conv1d(in_ch, out_ch, kernel_size, quant_config=None, **kwargs):
    if _quant_enabled(quant_config):
        return qnn.QuantConv1d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            weight_quant=quant_config.weight_quant,
            input_quant=quant_config.act_quant,
            return_quant_tensor=False,
            **kwargs,
        )
    return nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, **kwargs)


def _conv2d(in_ch, out_ch, kernel_size, quant_config=None, **kwargs):
    if _quant_enabled(quant_config):
        return qnn.QuantConv2d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            weight_quant=quant_config.weight_quant,
            input_quant=quant_config.act_quant,
            return_quant_tensor=False,
            **kwargs,
        )
    return nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, **kwargs)


def _conv3d(in_ch, out_ch, kernel_size, quant_config=None, **kwargs):
    if _quant_enabled(quant_config):
        return qnn.QuantConv3d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            weight_quant=quant_config.weight_quant,
            input_quant=quant_config.act_quant,
            return_quant_tensor=False,
            **kwargs,
        )
    return nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, **kwargs)


def _linear(in_features, out_features, quant_config=None, **kwargs):
    if _quant_enabled(quant_config):
        return qnn.QuantLinear(
            in_features,
            out_features,
            weight_quant=quant_config.weight_quant,
            input_quant=quant_config.act_quant,
            return_quant_tensor=False,
            **kwargs,
        )
    return nn.Linear(in_features, out_features, **kwargs)


class ResidualBlock(nn.Module):
    """Pre-activation residual MLP block."""

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class MLPDecoder(nn.Module):
    """Structure-agnostic residual MLP decoder."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        ratio: float = 1.0,
        num_blocks: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_dim = int(1024 * ratio)
        self.ratio = ratio
        self.num_blocks = num_blocks
        self.dropout_rate = dropout
        self.hidden_dim = hidden_dim
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.res_blocks = nn.Sequential(*[ResidualBlock(hidden_dim, dropout) for _ in range(num_blocks)])
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        nn.init.constant_(self.output_layer.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_layer(self.res_blocks(self.input_layer(x)))

    def get_arch_info(self) -> dict:
        return {
            "hidden_dim": self.hidden_dim,
            "num_blocks": self.num_blocks,
            "ratio": self.ratio,
            "dropout": self.dropout_rate,
        }


class ResidualBlock3D(nn.Module):
    """3D convolutional residual block used by CNN3D and Transformer STE."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride=1,
        padding=1,
        dilation=1,
        dropout: float = 0.1,
        quant_config=None,
    ):
        super().__init__()
        self.conv1 = _conv3d(
            in_channels,
            out_channels,
            kernel_size,
            quant_config,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.conv2 = _conv3d(
            out_channels,
            out_channels,
            kernel_size,
            quant_config,
            stride=1,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                _conv3d(in_channels, out_channels, 1, quant_config, stride=stride, padding=0, bias=False),
                nn.BatchNorm3d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

        self.add = None
        if _quant_enabled(quant_config) and quant_config.act_quant is not None:
            self.add = qnn.QuantEltwiseAdd(input_quant=quant_config.act_quant, return_quant_tensor=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.act1 = nn.ReLU()
        self.act2 = nn.ReLU()
        self.dropout = nn.Dropout3d(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout(self.act1(self.bn1(self.conv1(x))))
        out = self.bn2(self.conv2(out))
        shortcut = self.shortcut(x)
        out = self.add(out, shortcut) if self.add is not None else out + shortcut
        return self.act2(out)


class CNN3DDecoder(nn.Module):
    """Dilated 3D-CNN decoder."""

    BASE_CHANNELS = {"c1": 32, "c2": 64, "bottle": 8}

    def __init__(
        self,
        output_dim: int,
        d: int,
        r: int,
        circuit,
        ratio: float = 1.0,
        dropout: float = 0.5,
        quant_config=None,
    ):
        super().__init__()
        self.d = d
        self.r = r
        self.ratio = ratio
        self.dropout_rate = dropout
        c1 = int(self.BASE_CHANNELS["c1"] * ratio)
        c2 = int(self.BASE_CHANNELS["c2"] * ratio)
        c_bottle = max(int(self.BASE_CHANNELS["bottle"] * ratio), 4)

        self.mapper = StimTo3DMapper(circuit, d, r)
        flat_features = c_bottle * self.mapper.max_t * self.mapper.max_h * self.mapper.max_w
        c_fc = flat_features // 4
        self.channels = {"c1": c1, "c2": c2, "bottle": c_bottle, "fc": c_fc}

        self.stem_conv = _conv3d(2, c1, 3, quant_config, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm3d(c1)
        self.stem_act = nn.ReLU()
        self.res_block1 = ResidualBlock3D(c1, c1, dilation=1, padding=1, dropout=0.1, quant_config=quant_config)
        self.res_block2 = ResidualBlock3D(c1, c2, dilation=2, padding=2, dropout=0.1, quant_config=quant_config)
        self.res_block3 = ResidualBlock3D(c2, c2, dilation=3, padding=3, dropout=0.1, quant_config=quant_config)
        self.bottle_conv = _conv3d(c2, c_bottle, 1, quant_config, bias=False)
        self.bottle_bn = nn.BatchNorm3d(c_bottle)
        self.bottle_act = nn.ReLU()
        self.fc1 = _linear(flat_features, c_fc, quant_config, bias=True)
        self.fc1_bn = nn.BatchNorm1d(c_fc)
        self.fc1_act = nn.ReLU()
        self.fc1_dropout = nn.Dropout(dropout)
        self.fc2 = _linear(c_fc, output_dim, quant_config, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mapper(x)
        x = self.stem_act(self.stem_bn(self.stem_conv(x)))
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.bottle_act(self.bottle_bn(self.bottle_conv(x)))
        x = torch.flatten(x, start_dim=1)
        x = self.fc1_dropout(self.fc1_act(self.fc1_bn(self.fc1(x))))
        return self.fc2(x)

    def get_arch_info(self) -> dict:
        return {
            "channels": self.channels,
            "input_shape": {"T": self.mapper.max_t, "H": self.mapper.max_h, "W": self.mapper.max_w},
            "res_blocks": 3,
            "dilations": [1, 2, 3],
        }


class ResConv2D(nn.Module):
    """Residual 2D convolution block used by the TCN spatial encoder."""

    def __init__(self, channels: int, dropout: float = 0.1, quant_config=None):
        super().__init__()
        self.conv = _conv2d(channels, channels, 3, quant_config, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)
        self.dropout = nn.Dropout2d(dropout)
        self.add = None
        if _quant_enabled(quant_config) and quant_config.act_quant is not None:
            self.add = qnn.QuantEltwiseAdd(input_quant=quant_config.act_quant, return_quant_tensor=False)
        self.act = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.dropout(self.bn(self.conv(x)))
        out = self.add(out, x) if self.add is not None else out + x
        return self.act(out)


class ConvBNReLU2D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, quant_config=None):
        super().__init__()
        self.block = nn.Sequential(
            _conv2d(in_ch, out_ch, 3, quant_config, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConvBNReLU1D(nn.Module):
    def __init__(self, channels: int, quant_config=None):
        super().__init__()
        self.block = nn.Sequential(
            _conv1d(channels, channels, 3, quant_config, padding=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TCNDecoder(nn.Module):
    """Spatially decoupled TCN decoder."""

    BASE_C1 = 64
    BASE_C2 = 128

    def __init__(
        self,
        output_dim: int,
        d: int,
        r: int,
        circuit,
        ratio: float = 1.0,
        dropout: float = 0.1,
        quant_config=None,
        embedding_seed: int = 0,
    ):
        super().__init__()
        self.d = d
        self.r = r
        self.ratio = ratio
        c1 = max(int(self.BASE_C1 * ratio), 8)
        c2 = max(int(self.BASE_C2 * ratio), 8)
        self.c1 = c1
        self.c2 = c2

        self.mapper = StimToTemporalGridMapper(circuit, d, r, c1, seed=embedding_seed)
        self.res_block = ResConv2D(c1, dropout=dropout, quant_config=quant_config)
        self.spatial = nn.Sequential(
            ConvBNReLU2D(c1, c1, quant_config),
            ConvBNReLU2D(c1, c2, quant_config),
            ConvBNReLU2D(c2, c2, quant_config),
        )
        self.temporal = nn.Sequential(
            ConvBNReLU1D(c2, quant_config),
            ConvBNReLU1D(c2, quant_config),
        )
        self.position_fc = _linear(c2, c2, quant_config, bias=True)
        self.position_act = nn.ReLU()
        self.classifier_norm = nn.LayerNorm(c2)
        self.classifier_fc = _linear(c2, output_dim, quant_config, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mapper(x)
        b, t, c, h, w = x.shape
        x = x.reshape(b * t, c, h, w)
        x = self.spatial(self.res_block(x))
        _, c2, h, w = x.shape
        x = x.view(b, t, c2, h, w).permute(0, 1, 3, 4, 2).reshape(b, t * h * w, c2)
        x = self.temporal(x.transpose(1, 2)).transpose(1, 2)
        x = self.position_act(self.position_fc(x))
        x = x.mean(dim=1)
        return self.classifier_fc(self.classifier_norm(x))

    def get_arch_info(self) -> dict:
        return {
            "c1": self.c1,
            "c2": self.c2,
            "input_shape": {"T": self.mapper.max_t, "H": self.mapper.max_h, "W": self.mapper.max_w},
            "ratio": self.ratio,
        }


class SinusoidalPositionalEmbedding3D(nn.Module):
    """3D sinusoidal positional encoding."""

    def __init__(self, embed_dim: int, max_len: int = 100):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, t, h, w, c = x.shape
        return (
            x
            + self.pe[:t, :].view(1, t, 1, 1, c)
            + self.pe[:h, :].view(1, 1, h, 1, c)
            + self.pe[:w, :].view(1, 1, 1, w, c)
        )


class SpatioTemporalEncoder(nn.Module):
    """3D convolutional tokenizer used by the Transformer decoder."""

    BASE_CHANNELS = [16, 32, 64]

    def __init__(
        self,
        d: int,
        r: int,
        embed_dim: int = 256,
        ratio: float = 1.0,
        dropout: float = 0.1,
        quant_config=None,
        use_pos_encoding: bool = True,
    ):
        super().__init__()
        c1 = max(int(self.BASE_CHANNELS[0] * ratio), 8)
        c2 = max(int(self.BASE_CHANNELS[1] * ratio), 8)
        c3 = max(int(self.BASE_CHANNELS[2] * ratio), 8)
        self.stem_conv = _conv3d(2, c1, 3, quant_config, padding=1, bias=False)
        self.stem_bn = nn.BatchNorm3d(c1)
        self.stem_act = nn.ReLU()
        self.layer1 = ResidualBlock3D(c1, c2, stride=1, padding=1, dilation=1, dropout=dropout, quant_config=quant_config)
        self.layer2 = ResidualBlock3D(
            c2,
            c2,
            stride=(2, 1, 1),
            padding=(1, 2, 2),
            dilation=(1, 2, 2),
            dropout=dropout,
            quant_config=quant_config,
        )
        self.layer3 = ResidualBlock3D(
            c2,
            c3,
            stride=1,
            padding=(1, 3, 3),
            dilation=(1, 3, 3),
            dropout=dropout,
            quant_config=quant_config,
        )
        self.proj = _conv3d(c3, embed_dim, 1, quant_config, bias=False)
        self.pos_encoder = SinusoidalPositionalEmbedding3D(embed_dim) if use_pos_encoding else None

    def forward(self, x: torch.Tensor):
        x = self.stem_act(self.stem_bn(self.stem_conv(x)))
        x = self.layer3(self.layer2(self.layer1(x)))
        x = self.proj(x).permute(0, 2, 3, 4, 1).contiguous()
        if self.pos_encoder is not None:
            x = self.pos_encoder(x)
        b, t, h, w, c = x.shape
        return x.view(b, -1, c), (t, h, w)


class QuantTransformerEncoderLayer(nn.Module):
    """Pre-LN Transformer encoder layer with Brevitas quantized attention/MLP."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int = None, dropout: float = 0.1, quant_config=None):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = 4 * d_model
        weight_quant = quant_config.weight_quant if quant_config else None
        act_quant = quant_config.act_quant if quant_config else None
        self.self_attn = qnn.QuantMultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=0.0,
            batch_first=True,
            bias=True,
            in_proj_weight_quant=weight_quant,
            in_proj_input_quant=act_quant,
            in_proj_bias_quant=None,
            out_proj_weight_quant=weight_quant,
            out_proj_input_quant=act_quant,
            out_proj_bias_quant=None,
        )
        self.linear1 = _linear(d_model, dim_feedforward, quant_config, bias=True)
        self.linear2 = _linear(dim_feedforward, d_model, quant_config, bias=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.activation = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        qkv = self.norm1(src)
        src = src + self.dropout1(self.self_attn(qkv, qkv, qkv, need_weights=False)[0])
        x = self.linear2(self.dropout(self.activation(self.linear1(self.norm2(src)))))
        return src + self.dropout2(x)


class TransformerDecoder(nn.Module):
    """Transformer decoder with a convolutional spatiotemporal tokenizer."""

    BASE_EMBED_DIM = 128

    def __init__(
        self,
        output_dim: int,
        d: int,
        r: int,
        circuit,
        ratio: float = 1.0,
        num_heads: int = 8,
        num_layers: int = 3,
        dropout: float = 0.1,
        quant_config=None,
    ):
        super().__init__()
        self.d = d
        self.r = r
        self.ratio = ratio
        self.num_layers = num_layers
        embed_dim = int(self.BASE_EMBED_DIM * ratio)
        actual_num_heads = 4 if ratio <= 0.5 else num_heads
        while embed_dim % actual_num_heads != 0 and actual_num_heads > 2:
            actual_num_heads -= 1
        if embed_dim % actual_num_heads != 0:
            embed_dim = (embed_dim // actual_num_heads) * actual_num_heads
        self.embed_dim = embed_dim
        self.num_heads = actual_num_heads

        self.mapper = StimTo3DMapper(circuit, d, r)
        self.encoder = SpatioTemporalEncoder(d, r, embed_dim=embed_dim, ratio=ratio, dropout=0.1, quant_config=quant_config)
        if _quant_enabled(quant_config):
            self.transformer_layers = nn.ModuleList(
                [
                    QuantTransformerEncoderLayer(
                        d_model=embed_dim,
                        nhead=actual_num_heads,
                        dim_feedforward=embed_dim * 4,
                        dropout=dropout,
                        quant_config=quant_config,
                    )
                    for _ in range(num_layers)
                ]
            )
        else:
            self.transformer_layers = nn.ModuleList(
                [
                    nn.TransformerEncoderLayer(
                        d_model=embed_dim,
                        nhead=actual_num_heads,
                        dim_feedforward=embed_dim * 4,
                        dropout=dropout,
                        batch_first=True,
                        norm_first=True,
                        activation="relu",
                    )
                    for _ in range(num_layers)
                ]
            )
        self.classifier_norm = nn.LayerNorm(embed_dim)
        self.classifier_fc = _linear(embed_dim, output_dim, quant_config, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.mapper(x)
        x, _ = self.encoder(x)
        for layer in self.transformer_layers:
            x = layer(x)
        x = x.mean(dim=1)
        return self.classifier_fc(self.classifier_norm(x))

    def get_arch_info(self) -> dict:
        return {
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "num_layers": self.num_layers,
            "ffn_dim": self.embed_dim * 4,
            "input_shape": {"T": self.mapper.max_t, "H": self.mapper.max_h, "W": self.mapper.max_w},
        }


class ScaledTanh(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.tanh(x * torch.clamp(self.scale, min=0.1, max=5.0))


class NeuralBPLayer(nn.Module):
    """Shared-weight neural belief propagation layer."""

    def __init__(self, hidden_dim: int = 64, dropout: float = 0.0):
        super().__init__()
        self.c2v_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            ScaledTanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.v2c_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            ScaledTanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.check_gru = nn.GRUCell(hidden_dim, hidden_dim)
        self.var_gru = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, h_v, h_c, h_c_initial, edge_index_v2c, edge_index_c2v):
        aggr_v = torch.zeros_like(h_c)
        aggr_v.index_add_(0, edge_index_v2c[1], h_v[edge_index_v2c[0]])
        h_c_new = self.check_gru(self.v2c_mlp(torch.cat([aggr_v, h_c_initial], dim=1)), h_c)

        aggr_c = torch.zeros_like(h_v)
        aggr_c.index_add_(0, edge_index_c2v[1], h_c_new[edge_index_c2v[0]])
        h_v_new = self.var_gru(self.c2v_mlp(aggr_c), h_v)
        return h_v_new, h_c_new


class GNNDecoder(nn.Module):
    """Neural belief-propagation GNN decoder on a Tanner graph."""

    BASE_HIDDEN_DIM = 64

    def __init__(self, dem, ratio: float = 1.0, num_iterations: int = 20, dropout: float = 0.0):
        super().__init__()
        self.ratio = ratio
        self.num_iterations = num_iterations
        self.dropout_rate = dropout
        hidden_dim = max(int(self.BASE_HIDDEN_DIM * ratio), 16)
        self.hidden_dim = hidden_dim
        self.mapper = StimToGraphMapper(dem)
        self.check_encoder = nn.Linear(1, hidden_dim)
        self.var_encoder = nn.Linear(1, hidden_dim)
        self.processor = NeuralBPLayer(hidden_dim, dropout=dropout)
        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.constant_(self.readout[-1].bias, -2.0)
        self.criterion_syn = SyndromeConsistencyLoss()

    def forward(self, x: torch.Tensor, compute_syn_loss: bool = True):
        x_c_flat, x_v_flat, edge_index_v2c, edge_index_c2v, _ = self.mapper(x)
        h_c = self.check_encoder(1.0 - 2.0 * x_c_flat)
        h_v = self.var_encoder(x_v_flat)
        h_c_initial = h_c.clone()
        for _ in range(self.num_iterations):
            h_v, h_c = self.processor(h_v, h_c, h_c_initial, edge_index_v2c, edge_index_c2v)
        output_logits = self.readout(h_v)
        output = output_logits.view(x.shape[0], -1)
        if compute_syn_loss:
            loss_syn = self.criterion_syn(output_logits, x, edge_index_v2c)
        else:
            loss_syn = torch.tensor(0.0, device=x.device)
        return output, loss_syn

    def get_arch_info(self) -> dict:
        return {
            "hidden_dim": self.hidden_dim,
            "num_iterations": self.num_iterations,
            "num_vars": self.mapper.num_vars,
            "num_checks": self.mapper.num_checks,
            "ratio": self.ratio,
            "dropout": self.dropout_rate,
        }
