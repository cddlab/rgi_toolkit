"""PyTorch L-BFGS with device-resident coefficients for CUDA vector updates.

The step follows torch.optim.LBFGS (PyTorch 2.8.0, BSD-3-Clause; see LICENSE).
History, line search, stopping rules and scalar precision are unchanged.
"""

import logging
from dataclasses import dataclass

import torch
from torch.optim.lbfgs import _strong_wolfe

from rgi_toolkit.optim._torch_cg_gpu import _COMPILE_DISABLED

logger = logging.getLogger(__name__)
_ADD = None
_ADD_FAILED = False
_DIRECTION_FAILED = False


def add_device_scalar(x, other, alpha):
    """Preserve native CUDA add(alpha)'s fused multiply-add and in-place storage."""
    global _ADD, _ADD_FAILED
    if (
        isinstance(alpha, torch.Tensor)
        and alpha.is_cuda
        and x.is_contiguous()
        and other.is_contiguous()
        and not _COMPILE_DISABLED
        and not _ADD_FAILED
    ):
        try:
            if _ADD is None:
                from rgi_toolkit.optim._torch_lbfgs_cuda import add

                _ADD = add
            return _ADD(x, other, alpha)
        except Exception as exc:
            logger.warning("compiled L-BFGS scalar update failed (%s); eager", exc)
            _ADD_FAILED = True
    return x.add_(other, alpha=alpha)


@dataclass
class _SearchGradient:
    tensor: torch.Tensor
    slope: torch.Tensor | None = None

    def clone(self, **kwargs):
        return _SearchGradient(self.tensor.clone(**kwargs), self.slope)

    def dot(self, _direction):
        return self.slope


@dataclass
class _SearchDirection:
    norm: torch.Tensor

    def abs(self):
        return self.norm


class CudaLBFGS(torch.optim.LBFGS):
    """Retain PyTorch's optimizer while avoiding scalar uploads and downloads."""

    def _add_grad(self, step_size, update):
        offset = 0
        for p in self._params:
            numel = p.numel()
            add_device_scalar(p, update[offset : offset + numel].view_as(p), step_size)
            offset += numel
        assert offset == self._numel()

    @torch.no_grad()
    def step(self, closure):
        global _DIRECTION_FAILED
        assert len(self.param_groups) == 1
        group = self.param_groups[0]
        if group["line_search_fn"] != "strong_wolfe":
            return super().step(closure)
        closure = torch.enable_grad()(closure)
        lr = group["lr"]
        if isinstance(lr, torch.Tensor):
            lr = lr.item()
        max_iter, max_eval = group["max_iter"], group["max_eval"]
        tolerance_grad = group["tolerance_grad"]
        tolerance_change = group["tolerance_change"]
        history_size = group["history_size"]
        state = self.state[self._params[0]]
        state.setdefault("func_evals", 0)
        state.setdefault("n_iter", 0)
        orig_loss = closure()
        loss = float(orig_loss)
        current_evals = 1
        state["func_evals"] += 1
        flat_grad = self._gather_flat_grad()
        if flat_grad.abs().max() <= tolerance_grad:
            return orig_loss
        d, t = state.get("d"), state.get("t")
        old_dirs, old_stps = state.get("old_dirs"), state.get("old_stps")
        ro, h_diag = state.get("ro"), state.get("H_diag")
        prev_flat_grad, prev_loss = state.get("prev_flat_grad"), state.get("prev_loss")
        n_iter = 0
        while n_iter < max_iter:
            n_iter += 1
            state["n_iter"] += 1
            if state["n_iter"] == 1:
                d = flat_grad.neg()
                old_dirs, old_stps, ro = [], [], []
                h_diag = 1
            else:
                y = flat_grad.sub(prev_flat_grad)
                s = d.mul(t)
                ys = y.dot(s)
                if ys > 1e-10:
                    if len(old_dirs) == history_size:
                        old_dirs.pop(0)
                        old_stps.pop(0)
                        ro.pop(0)
                    old_dirs.append(y)
                    old_stps.append(s)
                    ro.append(1.0 / ys)
                    h_diag = ys / y.dot(y)
                num_old = len(old_dirs)
                if "al" not in state:
                    state["al"] = [None] * history_size
                al = state["al"]
                used_fused = False
                if (
                    not _COMPILE_DISABLED
                    and not _DIRECTION_FAILED
                    and flat_grad.numel() <= 8192
                ):
                    try:
                        from rgi_toolkit.optim._torch_lbfgs_cuda import direction

                        d, coefficients = direction(
                            flat_grad, old_dirs, old_stps, ro, h_diag
                        )
                        al[:num_old] = coefficients
                        used_fused = True
                    except Exception as exc:
                        logger.warning(
                            "compiled L-BFGS direction failed (%s); eager", exc
                        )
                        _DIRECTION_FAILED = True
                if not used_fused:
                    q = flat_grad.neg()
                    for i in range(num_old - 1, -1, -1):
                        al[i] = old_stps[i].dot(q) * ro[i]
                        add_device_scalar(q, old_dirs[i], -al[i])
                    d = r = torch.mul(q, h_diag)
                    for i in range(num_old):
                        be_i = old_dirs[i].dot(r) * ro[i]
                        add_device_scalar(r, old_stps[i], al[i] - be_i)
            if prev_flat_grad is None:
                prev_flat_grad = flat_grad.clone(memory_format=torch.contiguous_format)
            else:
                prev_flat_grad.copy_(flat_grad)
            prev_loss = loss
            t = (
                min(1.0, 1.0 / flat_grad.abs().sum()) * lr
                if state["n_iter"] == 1
                else lr
            )
            scalars = torch.stack((flat_grad.dot(d), d.abs().max())).cpu()
            gtd, direction_norm = scalars.unbind()
            if gtd > -tolerance_change:
                break
            x_init = self._clone_param()

            def obj_func(x, t, _direction):
                self._add_grad(t, d)
                value = closure()
                gradient = self._gather_flat_grad()
                values = torch.stack((value, gradient.dot(d))).cpu()
                self._set_param(x)
                return float(values[0]), _SearchGradient(
                    gradient, values[1].to(gradient.dtype)
                )

            loss, gradient, t, ls_func_evals = _strong_wolfe(
                obj_func,
                x_init,
                t.cpu() if isinstance(t, torch.Tensor) else t,
                _SearchDirection(direction_norm),
                loss,
                _SearchGradient(flat_grad),
                gtd,
            )
            flat_grad = gradient.tensor
            self._add_grad(t, d)
            opt_cond = flat_grad.abs().max() <= tolerance_grad
            current_evals += ls_func_evals
            state["func_evals"] += ls_func_evals
            if n_iter == max_iter or current_evals >= max_eval:
                break
            if opt_cond:
                break
            if d.mul(t).abs().max() <= tolerance_change:
                break
            if abs(loss - prev_loss) < tolerance_change:
                break
        state.update(
            d=d,
            t=t,
            old_dirs=old_dirs,
            old_stps=old_stps,
            ro=ro,
            H_diag=h_diag,
            prev_flat_grad=prev_flat_grad,
            prev_loss=prev_loss,
        )
        return orig_loss
