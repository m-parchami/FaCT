"""
The content of this file comes from B-cos-v2 codebase, with minor potential changes.
For the latest version, please use their original repo:
https://github.com/B-cos/B-cos-v2
"""

import torch
import torch.nn.functional as F
class AddInverse(torch.nn.Module):
    """To a [B, C, H, W] input add the inverse channels of the given one to it.
    Results in a [B, 2C, H, W] output. Single image [C, H, W] is also accepted.

    Args:
        dim (int): where to add channels to. Default: -3
    """

    def __init__(self, dim: int = -3):
        super().__init__()
        self.dim = dim

    def forward(self, in_tensor: torch.Tensor) -> torch.Tensor:
        return torch.cat([in_tensor, 1 - in_tensor], dim=self.dim)

def gradient_to_image(image, linear_mapping, smooth=0, alpha_percentile=99.5, rescale=None):
    """
    From https://github.com/moboehle/B-cos/blob/0023500ce/interpretability/utils.py#L41.
    Computing color image from dynamic linear mapping of B-cos models.

    Parameters
    ----------
    image: Tensor
        Original input image (encoded with 6 color channels)
        Shape: [C, H, W] with C=6
    linear_mapping: Tensor
        Linear mapping W_{1\rightarrow l} of the B-cos model
        Shape: [C, H, W] same as image
    smooth: int
        Kernel size for smoothing the alpha values
    alpha_percentile: float
        Cut-off percentile for the alpha value. In range [0, 100].

    Returns
    -------
    np.ndarray
        image explanation of the B-cos model.
        Shape: [H, W, C] (C=4 ie RGBA)
    """
    # shape of img and linmap is [C, H, W], summing over first dimension gives the contribution map per location
    contribs = (image * linear_mapping).sum(0, keepdim=True)  # [H, W]
    # Normalise each pixel vector (r, g, b, 1-r, 1-g, 1-b) s.t. max entry is 1, maintaining direction
    rgb_grad = linear_mapping / (
        linear_mapping.abs().max(0, keepdim=True).values + 1e-12
    )
    # clip off values below 0 (i.e., set negatively weighted channels to 0 weighting)
    rgb_grad = rgb_grad.clamp(min=0)
    # normalise s.t. each pair (e.g., r and 1-r) sums to 1 and only use resulting rgb values
    rgb_grad = rgb_grad[:3] / (rgb_grad[:3] + rgb_grad[3:] + 1e-12)  # [3, H, W]

    # Set alpha value to the strength (L2 norm) of each location's gradient
    alpha = linear_mapping.norm(p=2, dim=0, keepdim=True)
    # Only show positive contributions
    alpha = torch.where(contribs < 0, 1e-12, alpha)
    if smooth:
        alpha = F.avg_pool2d(alpha, smooth, stride=1, padding=(smooth - 1) // 2)
    if rescale == 'none':
        pass
    elif rescale == 'default':
        alpha = (alpha / torch.quantile(alpha, q=alpha_percentile / 100)).clip(0, 1)
    else:
        assert isinstance(rescale, float)
        alpha = (alpha / rescale).clip(0, 1)

    rgb_grad = torch.concatenate([rgb_grad, alpha], dim=0)  # [4, H, W]
    # Reshaping to [H, W, C]
    grad_image = rgb_grad.permute(1, 2, 0)
    return grad_image.detach().cpu().numpy()
