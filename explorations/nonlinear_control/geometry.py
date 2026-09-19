"""Local control geometry; no nonlinear random-matrix equivalence assumed."""
import numpy as np
import torch


def pc_gradient_power(scores, inputs, indices, channel_std=None):
    """Exact VJPs for one image, without materializing the full Jacobian.

    inputs are normalized model inputs. Dividing by channel_std changes the
    derivative to RGB coordinates on the already resized/cropped input grid.
    """
    if inputs.shape[0] != 1 or scores.shape[0] != 1:
        raise ValueError("One image at a time avoids batch-coupling ambiguity")
    powers = []
    for k in indices:
        grad, = torch.autograd.grad(scores[0, k], inputs, retain_graph=True)
        if channel_std is not None:
            std = torch.as_tensor(channel_std, device=grad.device, dtype=grad.dtype)
            if std.numel() != grad.shape[1]:
                raise ValueError('channel_std must contain one value per input channel')
            grad = grad / std.reshape(1, -1, 1, 1)
        powers.append(grad.double().square().sum().item())
    return np.asarray(powers)


def trace_contributions(spectrum, powers, kappas):
    spectrum, powers, kappas = map(np.asarray, (spectrum, powers, kappas))
    if np.any(spectrum < 0) or np.any(kappas <= 0) or np.any(powers < 0):
        raise ValueError("Require nonnegative spectrum/powers and positive kappa")
    return powers[None, :] * spectrum[None, :] / (spectrum[None, :] + kappas[:, None])**2


def batched_pc_gradient_power(scores, inputs, indices, channel_std=None):
    """Vectorize exact VJPs over a bounded block of PC output directions."""
    if inputs.shape[0] != 1 or scores.shape[0] != 1:
        raise ValueError('Expected one image')
    seeds = torch.zeros(len(indices), scores.shape[1], device=scores.device, dtype=scores.dtype)
    seeds[torch.arange(len(indices), device=scores.device), list(indices)] = 1
    grad, = torch.autograd.grad(scores[0], inputs, grad_outputs=seeds,
                               is_grads_batched=True, retain_graph=True)
    if channel_std is not None:
        std = torch.as_tensor(channel_std, device=grad.device, dtype=grad.dtype)
        if std.numel() != grad.shape[2]:
            raise ValueError('channel_std must contain one value per input channel')
        grad = grad / std.reshape(1, 1, -1, 1, 1)
    return grad.double().flatten(1).square().sum(1).detach().cpu().numpy()
