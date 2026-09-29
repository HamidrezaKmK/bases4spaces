from typing import Iterable

import torch


ParameterSource = Iterable[torch.nn.Parameter] | torch.nn.Module


def _iter_parameters(params: ParameterSource) -> Iterable[torch.nn.Parameter]:
    return params.parameters() if isinstance(params, torch.nn.Module) else params


def get_grad_norm(params: ParameterSource) -> float:
    """Global L2 norm of gradients from a module or iterable of parameters.

    Pass ``model.parameters()`` for a single module, or a concatenated list when
    several modules are optimized together.
    """
    all_grads = [p.grad.view(-1) for p in _iter_parameters(params) if p.grad is not None]
    if not all_grads:
        return 0.0
    return torch.norm(torch.cat(all_grads)).item()


def get_param_norm(params: ParameterSource) -> float:
    """Global L2 norm from a module or iterable of parameters."""
    all_params = [p.view(-1) for p in _iter_parameters(params)]
    if not all_params:
        return 0.0
    return torch.norm(torch.cat(all_params)).item()


def get_avg_lr(optimizer: torch.optim.Optimizer) -> float:
    return sum(g['lr'] for g in optimizer.param_groups) / len(optimizer.param_groups)


def step_scheduler(
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    metric: float,
) -> None:
    if scheduler is None:
        return
    if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
        scheduler.step(metric)
    else:
        scheduler.step()
