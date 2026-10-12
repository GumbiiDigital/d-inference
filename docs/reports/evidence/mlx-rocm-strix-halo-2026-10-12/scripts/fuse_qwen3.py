"""Fuse Qwen3 Q/K/V and gate/up QuantizedLinear layers for fewer GPU kernels."""

from __future__ import annotations

from typing import Sequence

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.activations import swiglu
from mlx_lm.models.base import scaled_dot_product_attention
from mlx_lm.models.qwen3 import Attention, MLP


def _require_quantized(name: str, layer: nn.Module) -> nn.QuantizedLinear:
    if not isinstance(layer, nn.QuantizedLinear):
        raise ValueError(f"{name} must be nn.QuantizedLinear, got {type(layer)}")
    return layer


def _qlinear_input_dims(layer: nn.QuantizedLinear) -> int:
    _, packed = layer.weight.shape
    return (packed * 32) // layer.bits


def concat_qlinear(layers: Sequence[nn.QuantizedLinear]) -> nn.QuantizedLinear:
    if not layers:
        raise ValueError("concat_qlinear: empty layer list")
    first = _require_quantized("layers[0]", layers[0])
    in_dims = _qlinear_input_dims(first)
    mode = getattr(first, "mode", "affine")
    total_out = 0
    for i, layer in enumerate(layers):
        layer = _require_quantized(f"layers[{i}]", layer)
        if "bias" in layer:
            raise ValueError(
                "concat_qlinear: input layers must not have Linear bias (bias=False)"
            )
        if layer.group_size != first.group_size or layer.bits != first.bits:
            raise ValueError("concat_qlinear: mismatched group_size or bits")
        if getattr(layer, "mode", "affine") != mode:
            raise ValueError("concat_qlinear: mismatched quantization mode")
        if _qlinear_input_dims(layer) != in_dims:
            raise ValueError("concat_qlinear: mismatched input_dims")
        total_out += layer.weight.shape[0]

    fused = nn.QuantizedLinear(
        in_dims,
        total_out,
        bias=False,
        group_size=first.group_size,
        bits=first.bits,
        mode=mode,
    )
    fused.weight = mx.concatenate([layer.weight for layer in layers], axis=0)
    fused.scales = mx.concatenate([layer.scales for layer in layers], axis=0)
    if first.biases is not None:
        fused.biases = mx.concatenate([layer.biases for layer in layers], axis=0)
    else:
        fused.biases = None
    return fused


class FusedAttention(Attention):
    def __call__(self, x, mask=None, cache=None):
        B, L, D = x.shape

        qkv = self.qkv_proj(x)
        queries, keys, values = mx.split(qkv, [self._n_q, self._n_q + self._n_kv], axis=-1)

        queries = self.q_norm(queries.reshape(B, L, self.n_heads, -1)).transpose(
            0, 2, 1, 3
        )
        keys = self.k_norm(keys.reshape(B, L, self.n_kv_heads, -1)).transpose(
            0, 2, 1, 3
        )
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class FusedMLP(MLP):
    def __call__(self, x) -> mx.array:
        g, u = mx.split(self.gate_up_proj(x), 2, axis=-1)
        return self.down_proj(swiglu(g, u))


def _fuse_attention(attn: Attention) -> None:
    q = _require_quantized("q_proj", attn.q_proj)
    k = _require_quantized("k_proj", attn.k_proj)
    v = _require_quantized("v_proj", attn.v_proj)
    attn.qkv_proj = concat_qlinear([q, k, v])
    attn._n_q = q.weight.shape[0]
    attn._n_kv = k.weight.shape[0]
    del attn.q_proj, attn.k_proj, attn.v_proj
    attn.__class__ = FusedAttention


def _fuse_mlp(mlp: MLP) -> None:
    gate = _require_quantized("gate_proj", mlp.gate_proj)
    up = _require_quantized("up_proj", mlp.up_proj)
    mlp.gate_up_proj = concat_qlinear([gate, up])
    del mlp.gate_proj, mlp.up_proj
    mlp.__class__ = FusedMLP


def _model_layers(model):
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "layers"):
        return inner.layers
    return model.layers


def fuse(model) -> int:
    layers = _model_layers(model)
    for layer in layers:
        _fuse_attention(layer.self_attn)
        _fuse_mlp(layer.mlp)
    return len(layers)
