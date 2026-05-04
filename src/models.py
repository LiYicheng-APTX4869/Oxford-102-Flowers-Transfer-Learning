from __future__ import annotations

import types
from typing import Any

import torch
import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, channels, _, _ = x.shape
        weights = self.pool(x).view(batch, channels)
        weights = self.fc(weights).view(batch, channels, 1, 1)
        return x * weights


class ChannelAttention(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = self.mlp(self.avg_pool(x))
        max_out = self.mlp(self.max_pool(x))
        return self.activation(avg_out + max_out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size: int = 7) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        merged = torch.cat([avg_out, max_out], dim=1)
        return self.activation(self.conv(merged))


class CBAMBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel_size: int = 7) -> None:
        super().__init__()
        self.channel_attention = ChannelAttention(channels, reduction=reduction)
        self.spatial_attention = SpatialAttention(kernel_size=spatial_kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x * self.channel_attention(x)
        x = x * self.spatial_attention(x)
        return x


def _make_block_forward_with_attention():
    def block_forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if hasattr(self, "conv3"):
            out = self.relu(out)
            out = self.conv3(out)
            out = self.bn3(out)

        if hasattr(self, "attention"):
            out = self.attention(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out

    return block_forward


def inject_attention_into_resnet(model: nn.Module, attention_type: str | None) -> nn.Module:
    if not attention_type or attention_type.lower() == "none":
        return model

    attention_type = attention_type.lower()
    block_forward = _make_block_forward_with_attention()

    for module in model.modules():
        if not hasattr(module, "conv1") or not hasattr(module, "conv2") or not hasattr(module, "relu"):
            continue
        channels = None
        if hasattr(module, "bn3"):
            channels = module.bn3.num_features
        elif hasattr(module, "bn2"):
            channels = module.bn2.num_features
        if channels is None:
            continue

        if attention_type == "se":
            module.attention = SEBlock(channels)
        elif attention_type == "cbam":
            module.attention = CBAMBlock(channels)
        else:
            raise ValueError(f"Unsupported attention type: {attention_type}")

        module.forward = types.MethodType(block_forward, module)
    return model


def _resolve_torchvision_weights(model_name: str, pretrained: bool):
    from torchvision import models

    weight_map = {
        "resnet18": getattr(models, "ResNet18_Weights", None),
        "resnet34": getattr(models, "ResNet34_Weights", None),
    }
    if not pretrained:
        return None
    weight_enum = weight_map.get(model_name)
    if weight_enum is None:
        return None
    return weight_enum.DEFAULT


def build_vit_tiny(num_classes: int, pretrained: bool) -> nn.Module:
    try:
        from torchvision.models.vision_transformer import VisionTransformer
    except Exception as exc:
        raise ImportError(
            "torchvision VisionTransformer is required for vit_tiny."
        ) from exc

    model = VisionTransformer(
        image_size=224,
        patch_size=16,
        num_layers=12,
        num_heads=3,
        hidden_dim=192,
        mlp_dim=768,
        num_classes=num_classes,
        dropout=0.0,
        attention_dropout=0.0,
    )
    if pretrained:
        raise ValueError(
            "vit_tiny pretrained weights are not bundled in this project. "
            "Set pretrained=false for vit_tiny or extend the builder to load custom weights."
        )
    return model


def build_model(config: dict[str, Any]) -> nn.Module:
    import torchvision.models as models

    model_name = config["model_name"].lower()
    num_classes = int(config["num_classes"])
    pretrained = bool(config.get("pretrained", False))
    attention_type = config.get("attention_type")

    if model_name in {"resnet18", "resnet34"}:
        builder = getattr(models, model_name)
        weights = _resolve_torchvision_weights(model_name, pretrained)
        model = builder(weights=weights)
        model = inject_attention_into_resnet(model, attention_type)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return model

    if model_name == "vit_tiny":
        return build_vit_tiny(num_classes=num_classes, pretrained=pretrained)

    raise ValueError(f"Unsupported model_name: {model_name}")


def make_parameter_groups(model: nn.Module, config: dict[str, Any]) -> list[dict[str, Any]]:
    backbone_lr = float(config["backbone_lr"])
    head_lr = float(config["head_lr"])
    weight_decay = float(config.get("weight_decay", 0.0))
    groups: list[dict[str, Any]] = []

    if hasattr(model, "fc"):
        head_params = list(model.fc.parameters())
        head_ids = {id(param) for param in head_params}
        backbone_params = [param for param in model.parameters() if id(param) not in head_ids]
        groups.append({"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay})
        groups.append({"params": head_params, "lr": head_lr, "weight_decay": weight_decay})
        return groups

    if hasattr(model, "heads"):
        head_params = list(model.heads.parameters())
        head_ids = {id(param) for param in head_params}
        backbone_params = [param for param in model.parameters() if id(param) not in head_ids]
        groups.append({"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay})
        groups.append({"params": head_params, "lr": head_lr, "weight_decay": weight_decay})
        return groups

    groups.append({"params": model.parameters(), "lr": head_lr, "weight_decay": weight_decay})
    return groups
