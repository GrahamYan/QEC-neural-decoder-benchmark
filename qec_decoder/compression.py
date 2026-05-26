"""Quantization, PTQ calibration, QAT helpers, and pruning utilities."""

from __future__ import annotations

from typing import Iterable, Optional

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import brevitas.nn as qnn
from brevitas.quant import Int8ActPerTensorFloat, Int8WeightPerChannelFloat


class Int4WeightPerChannelFloat(Int8WeightPerChannelFloat):
    bit_width = 4


class Int4ActPerTensorFloat(Int8ActPerTensorFloat):
    bit_width = 4


class FP32Config:
    weight_quant = None
    act_quant = None
    name = "FP32"


class W4A4Config:
    weight_quant = Int4WeightPerChannelFloat
    act_quant = Int4ActPerTensorFloat
    name = "W4A4"


class W8A8Config:
    weight_quant = Int8WeightPerChannelFloat
    act_quant = Int8ActPerTensorFloat
    name = "W8A8"


_QUANT_CONFIGS = {
    "FP32": FP32Config,
    "W4A4": W4A4Config,
    "W8A8": W8A8Config,
}


def get_quant_config(name: Optional[str]):
    if name is None:
        return FP32Config
    name = name.upper()
    if name not in _QUANT_CONFIGS:
        raise ValueError(f"Unknown quantization config {name!r}; available={sorted(_QUANT_CONFIGS)}")
    return _QUANT_CONFIGS[name]


def get_quant_name(quant_config) -> str:
    return "FP32" if quant_config is None else quant_config.name


def convert_transformer_state_dict(state_dict: dict, num_layers: int = 3) -> dict:
    """Maps native PyTorch MHA keys to Brevitas QuantMultiheadAttention keys."""

    mapped = {}
    for key, value in state_dict.items():
        new_key = key
        for i in range(num_layers):
            prefix = f"transformer_layers.{i}."
            if key == f"{prefix}self_attn.in_proj_weight":
                new_key = f"{prefix}self_attn.in_proj.weight"
            elif key == f"{prefix}self_attn.in_proj_bias":
                new_key = f"{prefix}self_attn.in_proj.bias"
        mapped[new_key] = value
    return mapped


def load_pretrained_weights(model: nn.Module, checkpoint, transformer_layers: Optional[int] = None, strict: bool = False):
    """Loads FP32 weights into an FP32 or quantized model."""

    state_dict = torch.load(checkpoint, map_location="cpu") if isinstance(checkpoint, str) else checkpoint
    if transformer_layers is not None:
        state_dict = convert_transformer_state_dict(state_dict, transformer_layers)
    model_state = model.state_dict()
    compatible = {
        key: value
        for key, value in state_dict.items()
        if key in model_state and tuple(value.shape) == tuple(model_state[key].shape)
    }
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    if strict and (missing or unexpected):
        raise RuntimeError(f"Checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    return {"loaded": len(compatible), "total": len(model_state), "missing": missing, "unexpected": unexpected}


def _fold_bn_into_layer(layer, bn):
    std = torch.sqrt(bn.running_var + bn.eps)
    scale = (bn.weight / std).view(-1, *([1] * (layer.weight.dim() - 1)))
    layer.weight.data *= scale
    if layer.bias is not None:
        layer.bias.data = (layer.bias.data - bn.running_mean) * bn.weight / std + bn.bias
    else:
        layer.bias = nn.Parameter(bn.bias - bn.weight * bn.running_mean / std)

    bn.weight.data.fill_(1.0)
    bn.bias.data.zero_()
    bn.running_mean.zero_()
    bn.running_var.fill_(1.0)


def apply_bn_folding(model: nn.Module) -> int:
    """Folds explicitly paired BatchNorm layers into Conv/Linear layers for PTQ."""

    foldable = (
        qnn.QuantConv3d,
        qnn.QuantConv2d,
        qnn.QuantConv1d,
        qnn.QuantLinear,
        nn.Conv3d,
        nn.Conv2d,
        nn.Conv1d,
        nn.Linear,
    )
    batchnorm = (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d)
    folded_bns = set()
    folded = 0

    def maybe_fold(parent_name: str, parent: nn.Module, layer_name: str, bn_name: str) -> None:
        nonlocal folded
        layer = getattr(parent, layer_name, None)
        bn = getattr(parent, bn_name, None)
        full_bn_name = f"{parent_name}.{bn_name}" if parent_name else bn_name
        if full_bn_name in folded_bns:
            return
        if not isinstance(layer, foldable) or not isinstance(bn, batchnorm):
            return
        out_ch = getattr(layer, "out_channels", getattr(layer, "out_features", None))
        if out_ch != bn.num_features:
            return
        _fold_bn_into_layer(layer, bn)
        folded_bns.add(full_bn_name)
        folded += 1

    for parent_name, parent in model.named_modules():
        for layer_name, bn_name in (
            ("conv", "bn"),
            ("conv1", "bn1"),
            ("conv2", "bn2"),
            ("stem_conv", "stem_bn"),
            ("bottle_conv", "bottle_bn"),
            ("fc1", "fc1_bn"),
        ):
            maybe_fold(parent_name, parent, layer_name, bn_name)

        children = list(parent._modules.items())
        for (layer_name, layer), (bn_name, bn) in zip(children, children[1:]):
            full_bn_name = f"{parent_name}.{bn_name}" if parent_name else bn_name
            if full_bn_name in folded_bns:
                continue
            if not isinstance(layer, foldable) or not isinstance(bn, batchnorm):
                continue
            out_ch = getattr(layer, "out_channels", getattr(layer, "out_features", None))
            if out_ch != bn.num_features:
                continue
            _fold_bn_into_layer(layer, bn)
            folded_bns.add(full_bn_name)
            folded += 1
    return folded


def calibrate_ptq(model: nn.Module, dataloader, num_batches: int = 50, device: Optional[torch.device] = None) -> None:
    """Collects activation statistics for Brevitas PTQ calibration."""

    from brevitas.graph.calibrate import calibration_mode

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    with calibration_mode(model):
        with torch.no_grad():
            for i, (x, _) in enumerate(dataloader):
                if i >= num_batches:
                    break
                model(x.to(device))


def prepare_ptq(model: nn.Module, dataloader, num_batches: int = 50, fold_bn: bool = True, device: Optional[torch.device] = None):
    """Runs the PTQ preparation steps described in the appendix."""

    folded = apply_bn_folding(model) if fold_bn else 0
    calibrate_ptq(model, dataloader, num_batches=num_batches, device=device)
    return {"folded_bn": folded, "calibration_batches": num_batches}


def get_prunable_modules(model: nn.Module, protect_last_layer: bool = True):
    prunable_types = (
        nn.Linear,
        nn.Conv1d,
        nn.Conv2d,
        nn.Conv3d,
        qnn.QuantLinear,
        qnn.QuantConv1d,
        qnn.QuantConv2d,
        qnn.QuantConv3d,
    )
    modules = [(name, module) for name, module in model.named_modules() if isinstance(module, prunable_types)]
    return modules[:-1] if protect_last_layer and modules else modules


def apply_global_l1_pruning(model: nn.Module, sparsity: float, protect_last_layer: bool = True) -> None:
    """Applies global unstructured L1 pruning across Conv/Linear weights."""

    if not 0.0 < sparsity < 1.0:
        raise ValueError(f"sparsity must be in (0, 1), got {sparsity}")
    modules = get_prunable_modules(model, protect_last_layer=protect_last_layer)
    if not modules:
        return
    prune.global_unstructured(
        [(module, "weight") for _, module in modules],
        pruning_method=prune.L1Unstructured,
        amount=sparsity,
    )


def remove_pruning(model: nn.Module) -> int:
    """Permanently applies pruning masks by removing re-parameterizations."""

    removed = 0
    for module in model.modules():
        if hasattr(module, "weight_mask"):
            prune.remove(module, "weight")
            removed += 1
    return removed


def pruning_stats(model: nn.Module) -> dict:
    total = 0
    pruned = 0
    layer_stats = []
    for name, module in model.named_modules():
        if not hasattr(module, "weight_mask"):
            continue
        mask = module.weight_mask
        layer_total = mask.numel()
        layer_remaining = int(mask.sum().item())
        layer_pruned = layer_total - layer_remaining
        total += layer_total
        pruned += layer_pruned
        layer_stats.append(
            {
                "name": name,
                "total": layer_total,
                "remaining": layer_remaining,
                "pruned": layer_pruned,
                "sparsity": layer_pruned / layer_total,
            }
        )
    return {
        "total_params": total,
        "pruned_params": pruned,
        "remaining_params": total - pruned,
        "global_sparsity": pruned / total if total else 0.0,
        "layer_stats": layer_stats,
    }


def check_model_quantization(model: nn.Module) -> dict:
    """Returns a compact summary of active Brevitas quantized layers."""

    quant_layers = (qnn.QuantConv3d, qnn.QuantConv2d, qnn.QuantConv1d, qnn.QuantLinear)
    active = 0
    disabled = 0
    weight_bits = []
    act_bits = []
    for module in model.modules():
        if not isinstance(module, quant_layers):
            continue
        if getattr(module, "weight_quant", None) is None:
            disabled += 1
            continue
        try:
            q_weight = module.quant_weight()
            if getattr(q_weight, "bit_width", None) is not None:
                active += 1
                weight_bits.append(int(q_weight.bit_width.item()))
            else:
                disabled += 1
        except Exception:
            disabled += 1
        input_quant = getattr(module, "input_quant", None)
        if input_quant is not None and hasattr(input_quant, "bit_width"):
            try:
                bit_width = input_quant.bit_width()
                if bit_width is not None:
                    act_bits.append(int(bit_width.item() if hasattr(bit_width, "item") else bit_width))
            except Exception:
                pass

    weight_bw = weight_bits[0] if weight_bits else None
    act_bw = act_bits[0] if act_bits else None
    return {
        "active_quant_layers": active,
        "disabled_quant_layers": disabled,
        "weight_bit_width": weight_bw,
        "activation_bit_width": act_bw,
        "quant_mode": f"W{weight_bw}A{act_bw}" if weight_bw and act_bw else "FP32",
    }


# Backward-compatible names.
calibrate = calibrate_ptq
apply_pruning = apply_global_l1_pruning
get_pruning_stats = pruning_stats
