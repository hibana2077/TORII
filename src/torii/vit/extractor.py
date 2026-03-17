from dataclasses import dataclass
from typing import List, Optional, Tuple

import timm
import torch
from PIL import Image
from timm.data import create_transform, resolve_model_data_config


@dataclass
class ViTExtractorConfig:
    model_name: str = "vit_base_patch16_224"
    pretrained: bool = False
    image_size: int = 224
    num_attn_layers: int = 2
    device: str = "cpu"


class ViTFeatureExtractor:
    def __init__(self, config: ViTExtractorConfig):
        self.config = config
        self.device = torch.device(config.device)

        self.model = timm.create_model(
            config.model_name,
            pretrained=config.pretrained,
            num_classes=0,
            img_size=config.image_size,
        )
        self.model.eval().to(self.device)

        data_config = resolve_model_data_config(self.model)
        self.transform = create_transform(**data_config, is_training=False)

        self._attention_captures: List[torch.Tensor] = []
        self._hook_handles: List[torch.utils.hooks.RemovableHandle] = []
        self._register_attention_hooks(config.num_attn_layers)

    def _register_attention_hooks(self, num_attn_layers: int) -> None:
        if not hasattr(self.model, "blocks"):
            return

        n_blocks = len(self.model.blocks)
        use_layers = min(max(1, num_attn_layers), n_blocks)
        target_blocks = self.model.blocks[-use_layers:]

        for block in target_blocks:
            if hasattr(block.attn, "fused_attn"):
                block.attn.fused_attn = False

            handle = block.attn.attn_drop.register_forward_hook(self._capture_attention_hook)
            self._hook_handles.append(handle)

    def _capture_attention_hook(self, module, inputs, output):
        if isinstance(output, torch.Tensor):
            self._attention_captures.append(output.detach())

    def extract_from_image_path(self, image_path: str) -> Tuple[torch.Tensor, Optional[torch.Tensor], Tuple[int, int]]:
        image = Image.open(image_path).convert("RGB")
        x = self.transform(image).unsqueeze(0).to(self.device)

        self._attention_captures.clear()
        with torch.no_grad():
            tokens = self.model.forward_features(x)

        if tokens.dim() != 3:
            raise RuntimeError(f"Unexpected token tensor shape: {tokens.shape}")

        patch_tokens = tokens[:, 1:, :].squeeze(0)
        grid_size = self._infer_grid_size(patch_tokens.shape[0])

        attention = self._aggregate_attention()
        if attention is not None:
            attention = attention[1:, 1:]

        return patch_tokens, attention, grid_size

    def _aggregate_attention(self) -> Optional[torch.Tensor]:
        if not self._attention_captures:
            return None

        per_layer = []
        for attn in self._attention_captures:
            # attn shape: [B, heads, N, N]
            attn_mean = attn.mean(dim=1).squeeze(0)
            per_layer.append(attn_mean)

        stacked = torch.stack(per_layer, dim=0)
        return stacked.mean(dim=0)

    def _infer_grid_size(self, n_patches: int) -> Tuple[int, int]:
        if hasattr(self.model, "patch_embed") and hasattr(self.model.patch_embed, "grid_size"):
            gs = self.model.patch_embed.grid_size
            if isinstance(gs, tuple) and gs[0] * gs[1] == n_patches:
                return gs

        side = int(n_patches ** 0.5)
        if side * side != n_patches:
            raise RuntimeError(f"Cannot infer square patch grid from {n_patches} patches")
        return (side, side)
