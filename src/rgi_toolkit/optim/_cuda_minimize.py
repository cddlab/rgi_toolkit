"""Bounded CUDA graph adapter for repeatedly solved restraint objectives.

Use complete fixed-background pairs, including topology and chemistry, so every
trial sees the same objective as the neighbor-list path. Bound allocation size
and retain the existing solver for larger problems and unsupported runtimes.
All registered terms and custom closures use their existing autodiff energies. Compilation never changes defaults
for the energy, line search, convergence, or iteration limit.
"""

from __future__ import annotations

import logging
import os

import torch

from rgi_toolkit.energy import torch_energy

logger = logging.getLogger(__name__)
MAX_PAIRS = 65536
MAX_VARIABLES = 8192


class DecompositionFailure(RuntimeError):
    """A captured fit failed; rerun with the original library recovery path."""


def eligible(optimizer, coords, max_iter, conformer_in_window):
    """Decide eligibility without reading a CUDA scalar or importing CUDA bindings."""
    if os.environ.get("RGI_DISABLE_COMPILE", "") not in ("", "0", "false"):
        return False
    if not coords.is_cuda or max_iter < 1:
        return False
    if coords.dtype not in (torch.float16, torch.bfloat16, torch.float32):
        return False
    batch = coords.numel() // (coords.shape[-2] * 3)
    n_active = len(optimizer.spec.active_sites)
    if n_active * 3 * batch > MAX_VARIABLES:
        return False
    pairs = 0
    fixed = optimizer._vdw if conformer_in_window else None
    if fixed is not None:
        pairs += fixed["lig_local"].numel() * fixed["bg_global"].numel()
    if optimizer._active_vdw is not None and conformer_in_window:
        pairs += n_active * n_active
    if pairs * batch > MAX_PAIRS:
        return False
    # Private FX/Inductor APIs and CUDA IF/ELSE capture are validated for Torch 2.8.
    # Other versions keep the existing optimizer instead of assuming compatibility.
    version = torch.__version__.split("+")[0].split(".")[:2]
    cuda = (torch.version.cuda or "0.0").split(".")[:2]
    return version == ["2", "8"] and tuple(map(int, cuda)) >= (12, 8)


class GraphSolve:
    def __init__(
        self,
        optimizer,
        coords,
        prepared,
        max_iter,
        sigma,
        step,
        conformer_in_window,
        active_terms,
    ):
        from rgi_toolkit.optim import _torch_cg_gpu as gpu
        from rgi_toolkit.optim._cg import run_cg
        from rgi_toolkit.optim._cuda_graph import FusedConditionalGraph
        from rgi_toolkit.optim._cuda_graph_ops import DeviceCG, DeviceScalars

        self.graph = None
        self.active = coords[..., optimizer._active_idx, :].float().clone().detach()
        self.background = None
        arguments = ()
        fixed = optimizer._vdw if conformer_in_window else None
        if fixed is not None:
            self.background = (
                coords[..., fixed["bg_global"], :].float().clone().detach()
            )
            count = self.background.shape[-2]
            neighbors = (
                torch.arange(count, device=coords.device)
                .reshape(1, 1, -1)
                .expand(1, fixed["lig_local"].numel(), count)
                .contiguous()
            )
            mask = (fixed["lig_r"].reshape(1, -1, 1) > 0) & (
                fixed["bg_r"][neighbors] > 0
            )
            arguments = (
                self.background,
                fixed["lig_local"],
                neighbors,
                mask.to(self.active.dtype),
                fixed["lig_r"],
                fixed["bg_r"],
                fixed["scale"],
                fixed["weight"],
                fixed["chemistry"],
            )

        moving_arguments = ()
        moving = optimizer._active_vdw if conformer_in_window else None
        if moving is not None:
            n = self.active.shape[-2]
            source = torch.arange(n, device=coords.device).reshape(1, n, 1)
            neighbors = (
                torch.arange(n, device=coords.device)
                .reshape(1, 1, n)
                .expand(1, n, n)
                .contiguous()
            )
            valid = source != neighbors
            if moving["chemistry"] is None:
                valid = valid & (
                    moving["polymer_mask"][source] | moving["polymer_mask"][neighbors]
                )
                valid = (
                    valid
                    & (moving["radii"][source] > 0)
                    & (moving["radii"][neighbors] > 0)
                )
                excluded = moving["excluded_codes"]
                if excluded.numel():
                    codes = torch.minimum(source, neighbors) * n + torch.maximum(
                        source, neighbors
                    )
                    places = torch.searchsorted(excluded, codes).clamp(
                        max=excluded.numel() - 1
                    )
                    valid = valid & (excluded[places] != codes)
            moving_arguments = (
                neighbors,
                valid.to(self.active.dtype) * 0.5,
                moving["radii"],
                moving["scale"],
                moving["weight"],
                moving["chemistry"],
            )
        mapping = (
            optimizer._coordinates.bind(
                "torch", self.active, sigma, step, enabled=conformer_in_window
            )
            if optimizer._is_cg()
            else None
        )

        def physical(x):
            return x if mapping is None else mapping(x, self.active)

        graph = FusedConditionalGraph()
        self.graph = graph
        try:
            graph.begin()
            # Peptide alternatives are rebound from the new input at the start of
            # every replay, then frozen throughout that invocation's line search.
            bound = torch_energy.bind_peptide_states(self.active, prepared)

            def energy(u):
                x = physical(u)
                value = gpu._energy(x, bound)
                if arguments:
                    value = value + gpu._vdw_pair_energy(x, *arguments)
                if moving_arguments:
                    value = value + gpu.active_vdw_pair_energy(x, *moving_arguments)
                for index in active_terms:
                    value = value + optimizer._custom_terms[index][-1](x)
                return value

            raw_vg = torch.func.grad_and_value(energy)

            def vg(u):
                return graph.call(raw_vg, u)

            if optimizer._is_cg():
                result, state = run_cg(
                    DeviceCG(graph, self.active),
                    vg,
                    self.active,
                    max_iter,
                    line_search=optimizer.line_search,
                    gtol=optimizer.gtol,
                )
                info = state.info
            else:
                from rgi_toolkit.optim._cuda_lbfgs import run_lbfgs

                state = run_lbfgs(
                    DeviceScalars(graph, self.active),
                    vg,
                    self.active,
                    max_iter,
                    optimizer.gtol,
                )
                result, info = state.x, None
            # Preserve the caller's coordinates on nonfinite solver output without
            # a host decision on every denoising step. CG still exposes its status.
            result = physical(result)
            result = torch.where(torch.isfinite(result).all(), result, self.active)
            graph.finish((result, info))
        except Exception:
            graph.close()
            raise

    def solve(self, optimizer, coords, return_info):
        with torch.no_grad():
            self.active.copy_(coords[..., optimizer._active_idx, :])
            if self.background is not None:
                self.background.copy_(coords[..., optimizer._vdw["bg_global"], :])
            result, info = self.graph.replay()
            if self.graph.uses_linalg and bool(self.graph.library_failure):
                raise DecompositionFailure("cuSOLVER reported a failed geometry fit")
            # Return independent storage: the graph will overwrite its buffers on
            # its next replay. The original batch dimensions are retained.
            result = result.clone()
            if return_info:
                from rgi_toolkit.optim.info import CGInfo, CGStatus

                values = torch.stack([v.to(torch.float64) for v in info]).tolist()
                info = CGInfo(
                    CGStatus(int(values[0])), *map(int, values[1:4]), *values[4:]
                )
            return result, info

    def close(self):
        if self.graph is not None:
            self.graph.close()
            self.graph = None


def try_minimize(
    optimizer, coords, sigma, step, max_iter, conformer_in_window, return_info
):
    """Return an update or None when the ordinary optimizer should handle it."""
    if not eligible(optimizer, coords, max_iter, conformer_in_window):
        return None
    with torch.inference_mode(False):
        prepared = optimizer._gated_prepared(sigma, step)
    active_terms = tuple(
        i
        for i, (_name, start, stop, start_step, stop_step, _closure) in enumerate(
            optimizer._custom_terms
        )
        if (sigma is None or stop <= sigma <= start)
        and (step is None or start_step <= step <= stop_step)
    )
    key = (
        id(prepared),
        tuple(coords.shape),
        max_iter,
        optimizer.method,
        optimizer.line_search,
        optimizer.gtol,
        conformer_in_window,
        active_terms,
    )
    if key in optimizer._device_graph_failures:
        return None
    with (
        torch.cuda.device(coords.device),
        torch.inference_mode(False),
        torch.enable_grad(),
        torch.autocast("cuda", enabled=False),
    ):
        cached = optimizer._device_graph_cache
        if cached is None or cached[0] != key:
            if cached is not None:
                cached[1].close()
                optimizer._device_graph_cache = None
            try:
                solve = GraphSolve(
                    optimizer,
                    coords,
                    prepared,
                    max_iter,
                    sigma,
                    step,
                    conformer_in_window,
                    active_terms,
                )
            except Exception as exc:
                logger.warning(
                    "CUDA solver graph unavailable (%s); using existing optimizer", exc
                )
                optimizer._device_graph_failures.add(key)
                return None
            optimizer._device_graph_cache = (key, solve)
        try:
            return optimizer._device_graph_cache[1].solve(
                optimizer, coords, return_info
            )
        except DecompositionFailure as exc:
            optimizer._device_graph_cache[1].close()
            optimizer._device_graph_cache = None
            optimizer._device_graph_failures.add(key)
            logger.warning("%s; retrying the existing optimizer", exc)
            return None
