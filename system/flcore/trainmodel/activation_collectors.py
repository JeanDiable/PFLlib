from typing import Dict, Optional, Tuple

import torch
from torch import nn


def _flatten_linear_input(x: torch.Tensor) -> torch.Tensor:
    # x shape: [N, d] or [N, seq, d] or more; flatten leading dims except last
    if x.ndim == 2:
        return x
    new_shape = (-1, x.shape[-1])
    return x.reshape(new_shape)


def collect_activations_for_model(
    model: nn.Module,
    batch_x: torch.Tensor,
    device: torch.device,
    orth_set: Optional[Dict[str, torch.Tensor]] = None,
    proj_width_factor: float = 5.0,
) -> Dict[str, Tuple[torch.Tensor, float, int]]:
    """
    Returns dict: layer_name -> (Y, r, b)
      - Y: torch.Tensor of shape (d, m)
      - r: residual ratio (float)
      - b: local sample count contributing (int)
    Targets Conv2d and Linear layers.
    """
    model = model.to(device)
    model.eval()
    activations: Dict[str, torch.Tensor] = {}

    hooks = []

    def register_hooks():
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):

                def pre_hook_conv(mod, inp, layer_name=name):
                    x_in = inp[0].detach()
                    activations[layer_name + '.weight'] = x_in

                hooks.append(module.register_forward_pre_hook(pre_hook_conv))
            elif isinstance(module, nn.Linear):

                def pre_hook_linear(mod, inp, layer_name=name):
                    x_in = inp[0].detach()
                    activations[layer_name + '.weight'] = x_in

                hooks.append(module.register_forward_pre_hook(pre_hook_linear))

    register_hooks()

    with torch.no_grad():
        if isinstance(batch_x, list):
            bx = batch_x[0].to(device)
        else:
            bx = batch_x.to(device)
        _ = model(bx)

    # remove hooks
    for h in hooks:
        h.remove()

    results: Dict[str, Tuple[torch.Tensor, float, int]] = {}

    for name, x_in in activations.items():
        # find corresponding module by name
        module: Optional[nn.Module] = None
        try:
            module = dict(model.named_modules())[name.rsplit('.weight', 1)[0]]
        except Exception:
            module = None

        if isinstance(module, nn.Conv2d):
            # build A via unfold
            unfold = nn.Unfold(
                kernel_size=module.kernel_size,
                dilation=module.dilation,
                padding=module.padding,
                stride=module.stride,
            )
            X_unf = unfold(x_in.to(device))  # [N, C*k*k, L]
            N = X_unf.shape[0]
            d = X_unf.shape[1]
            A = X_unf.permute(1, 0, 2).reshape(d, -1)  # (d, N*L)
        elif isinstance(module, nn.Linear):
            X = _flatten_linear_input(x_in.to(device))  # [N*, d]
            N = X.shape[0]
            d = X.shape[1]
            A = X.t().contiguous()  # (d, N*)
        else:
            continue

        # residualize if basis provided
        U = None if orth_set is None else orth_set.get(name, None)
        if U is not None:
            U = U.to(device)
            # R = (I - U U^T) A
            UA = U.t().matmul(A)
            R = A - U.matmul(UA)
        else:
            R = A

        # residual ratio
        a_norm = torch.linalg.norm(A, ord='fro') + 1e-12
        r_norm = torch.linalg.norm(R, ord='fro')
        r_ratio = float((r_norm / a_norm).item())

        # random projection
        n = R.shape[1]
        m = max(1, int(proj_width_factor * d))
        G = torch.randn(n, m, device=device) / (n**0.5)
        Y = R.matmul(G)  # (d, m)

        results[name] = (Y.detach().cpu(), r_ratio, int(N))

    return results





