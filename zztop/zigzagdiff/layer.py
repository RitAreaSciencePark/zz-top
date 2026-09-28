"""``torch.nn.Module`` wrapper tying a front-end to the differentiable core."""

from __future__ import annotations

from typing import Sequence, Union

import torch

from .core import ZigzagDiffResult, zigzag_diagrams
from .frontends import lower_star, passthrough

__all__ = ["ZigzagDiffLayer"]

_FRONTENDS = {"lower_star": lower_star, "passthrough": passthrough}


class ZigzagDiffLayer(torch.nn.Module):
    """Differentiable zigzag persistence of an evolving filtering function.

    Parameters
    ----------
    complex_ : SimplicialComplex
        The fixed complex ``K``.
    dims : sequence of int
        Homological degrees to return.
    eps : float or Tensor
        Threshold.  Present iff value ``> eps``.
    learnable_eps : bool
        Register ``eps`` as an ``nn.Parameter`` so gradient reaches it via
        ``d t*/d eps = 1 / (v_{k+1} - v_k)``.
    frontend : {'lower_star', 'passthrough'}
        ``lower_star`` expects ``theta`` of shape ``(n_frames, n_vertices)``;
        ``passthrough`` expects ``V`` of shape ``(n_frames, n_cells)``.
    backend : {'auto', 'python'}
        Kept for compatibility; both select the pure-Python engine.
    guard : float
        Magnitude clamp for near-degenerate interpolation denominators.
    validate : bool
        Replay-check closure on every forward (off by default; it is O(n) with
        a Python loop).
    """

    def __init__(
        self,
        complex_,
        dims: Sequence[int] = (1,),
        eps: Union[float, torch.Tensor] = 0.0,
        learnable_eps: bool = False,
        frontend: str = "lower_star",
        backend: str = "auto",
        guard: float = 1e-12,
        validate: bool = False,
    ) -> None:
        super().__init__()
        if frontend not in _FRONTENDS:
            raise ValueError(
                f"unknown frontend {frontend!r}; use one of {sorted(_FRONTENDS)}"
            )
        self.complex_ = complex_
        self.dims = tuple(dims)
        self.frontend = frontend
        self.backend = backend
        self.guard = guard
        self.validate = validate

        eps_tensor = torch.as_tensor(eps, dtype=torch.get_default_dtype())
        if learnable_eps:
            self.eps = torch.nn.Parameter(eps_tensor.clone())
        else:
            self.register_buffer("eps", eps_tensor.clone())

    def forward(self, x: torch.Tensor) -> ZigzagDiffResult:
        """Map parameters to differentiable diagrams."""
        if self.frontend == "lower_star":
            values = lower_star(x, self.complex_)
        else:
            values = passthrough(x, self.complex_, validate=self.validate)
        return zigzag_diagrams(
            self.complex_,
            values,
            eps=self.eps,
            dims=self.dims,
            backend=self.backend,
            guard=self.guard,
            validate=self.validate,
        )

    def extra_repr(self) -> str:
        return (
            f"dims={self.dims}, frontend={self.frontend!r}, "
            f"eps={float(self.eps.detach())!r}, backend={self.backend!r}"
        )
