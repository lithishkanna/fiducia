"""Grad-CAM saliency map generator for 2D convolutional spatial heatmap branches.

Provides spatial attribution for:
- Model C: HeatmapCNN (conv3 layer)
- Model D: DualBranchFusion (spatial_branch.conv2 layer)

Highlights which relative price bands and historical time windows in the
(4, P, T) tensor most strongly influenced the model's UP, DOWN, or NEUTRAL logits.
All outputs are strictly framed as internal network diagnostics.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class GradCAM:
    """Gradient-weighted Class Activation Mapping (Grad-CAM) for 2D CNNs."""

    def __init__(
        self,
        model: nn.Module,
        target_layer: Optional[nn.Module] = None,
    ) -> None:
        """Args:
            model: PyTorch model (e.g. HeatmapCNN or DualBranchFusion).
            target_layer: Specific Conv2d module. If None, automatically detects
                the last Conv2d module in the network.
        """
        self.model = model
        self.model.eval()

        if target_layer is None:
            self.target_layer = self._find_last_conv_layer(model)
        else:
            self.target_layer = target_layer

        self.activations: Optional[torch.Tensor] = None
        self.gradients: Optional[torch.Tensor] = None
        self._hook_handle = None

        self._register_hooks()

    def _find_last_conv_layer(self, module: nn.Module) -> nn.Module:
        """Find the final Conv2d module in module hierarchy."""
        last_conv = None
        for mod in module.modules():
            if isinstance(mod, nn.Conv2d):
                last_conv = mod
        if last_conv is None:
            raise ValueError("No Conv2d layer found in the provided model.")
        return last_conv

    def _register_hooks(self) -> None:
        """Register forward hook to capture activations and backward hook on the tensor."""
        def forward_hook(module: nn.Module, inp: Any, out: torch.Tensor):
            self.activations = out

            def backward_hook(grad: torch.Tensor):
                self.gradients = grad

            out.register_hook(backward_hook)
            return None

        self._hook_handle = self.target_layer.register_forward_hook(forward_hook)

    def remove_hooks(self) -> None:
        """Remove registered forward hooks."""
        if self._hook_handle is not None:
            self._hook_handle.remove()
            self._hook_handle = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.remove_hooks()

    def generate_cam(
        self,
        inputs: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """Generate normalized Grad-CAM saliency map.

        Args:
            inputs: Heatmap tensor (4, P, T) or (B, 4, P, T), or tuple of
                (tensor, tabular) for dual-branch models.
            target_class: Target class index (0: DOWN, 1: NEUTRAL, 2: UP).
                If None, uses the class with the highest predicted logit.

        Returns:
            Normalized saliency map in [0, 1] of shape (P, T) or (B, P, T).
        """
        is_dual = isinstance(inputs, (tuple, list))
        if is_dual:
            x_tensor, x_tabular = inputs
            if x_tensor.ndim == 3:
                x_tensor = x_tensor.unsqueeze(0)
                single_sample = True
            else:
                single_sample = False
            if x_tabular.ndim == 1:
                x_tabular = x_tabular.unsqueeze(0)

            batch_size = x_tensor.size(0)
            target_spatial_shape = (x_tensor.size(2), x_tensor.size(3))  # (P, T)
        else:
            x_tensor = inputs
            if x_tensor.ndim == 3:
                x_tensor = x_tensor.unsqueeze(0)
                single_sample = True
            else:
                single_sample = False

            batch_size = x_tensor.size(0)
            target_spatial_shape = (x_tensor.size(2), x_tensor.size(3))  # (P, T)

        # Clear existing gradients
        self.model.zero_grad()
        self.activations = None
        self.gradients = None

        # Forward pass
        if is_dual:
            logits = self.model(x_tensor, x_tabular)
        else:
            logits = self.model(x_tensor)

        if target_class is None:
            if single_sample:
                target_class = int(logits.argmax(dim=-1).item())
            else:
                target_class = int(logits[0].argmax(dim=-1).item())

        # Backward pass on target class score
        score = logits[:, target_class].sum()
        score.backward(retain_graph=True)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Failed to capture activations or gradients from the target layer.")

        # Channel-wise pooling of gradients
        # activations & gradients shape: (B, C, H, W)
        weights = torch.mean(self.gradients, dim=(2, 3), keepdim=True)  # (B, C, 1, 1)

        # Linear combination weighted by alpha
        cam = torch.sum(weights * self.activations, dim=1, keepdim=True)  # (B, 1, H, W)

        # Rectify: only positive contributions to the chosen class
        cam = F.relu(cam)

        # Upsample back to input spatial dimension (P, T)
        cam = F.interpolate(
            cam,
            size=target_spatial_shape,
            mode="bilinear",
            align_corners=False,
        )  # (B, 1, P, T)

        # Normalize each sample in batch to [0, 1]
        cams_np = []
        for b in range(batch_size):
            sample_cam = cam[b, 0].detach().cpu().numpy()
            c_min = float(sample_cam.min())
            c_max = float(sample_cam.max())
            if c_max > c_min + 1e-8:
                sample_cam = (sample_cam - c_min) / (c_max - c_min)
            else:
                sample_cam = np.zeros_like(sample_cam)
            cams_np.append(sample_cam)

        result = np.stack(cams_np, axis=0)  # (B, P, T)
        if single_sample:
            return result[0]  # (P, T)
        return result


def identify_salient_price_bands(
    saliency_map: np.ndarray,
    num_bands: int = 5,
    p_span: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Identify price levels relative to mid that exhibit highest Grad-CAM saliency.

    Args:
        saliency_map: Normalized 2D array of shape (P, T).
        num_bands: Number of top salient bands to return.
        p_span: Price grid span in ticks (if None, inferred from P = 2 * p_span).

    Returns:
        List of dicts with price_offset_ticks, relative_position, and saliency score.
    """
    if saliency_map.ndim != 2:
        raise ValueError(f"Expected 2D saliency map (P, T), got shape {saliency_map.shape}")

    P, T = saliency_map.shape
    if p_span is None:
        p_span = P // 2

    # Average saliency over time dimension T to obtain 1D profile across price levels P
    price_profile = np.mean(saliency_map, axis=1)

    top_indices = np.argsort(price_profile)[::-1][:num_bands]

    results = []
    for idx in top_indices:
        offset_tick = int(idx) - p_span
        score = float(price_profile[idx])
        side = "bid" if offset_tick < 0 else ("ask" if offset_tick > 0 else "mid")
        results.append(
            {
                "grid_index": int(idx),
                "price_offset_ticks": offset_tick,
                "side": side,
                "mean_saliency": score,
            }
        )

    return results
