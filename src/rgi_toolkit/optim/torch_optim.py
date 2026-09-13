"""GPU restraint optimizer for integrated PyTorch predictors.

Minimizes the restraint energy on active-site coordinates using autograd for
gradients. ``method`` selects the solver: ``"CG"`` (default) -> a nonlinear
conjugate-gradient solver following SciPy 1.17.1 (PR+, DCSRCH/Wolfe2, strict strong
Wolfe), shared with the JAX backend through ``optim/_cg.py``; ``"l-bfgs"`` ->
``torch.optim.LBFGS`` (strong-Wolfe). Operates in-place on the coordinate tensor
and stays on whatever device the coordinates live on, so ``gpu: true`` runs
entirely on GPU.

Dynamic VdW caches are validated before every trial evaluation by the shared
``_vdw_runtime``. Fixed partners receive the full Verlet skin displacement budget;
two moving partners each receive half. Overflow rows use complete pair sums in
bounded chunks. Rebuilding preserves the objective and CG history.

"""

from __future__ import annotations

import logging
import os

import torch

from rgi_toolkit.energy import torch_energy
from rgi_toolkit.energy._terms import CONF_KEYS, PER_ENTRY_KEYS, TERM_BY_KEY
from rgi_toolkit.optim._cg_config import GTOL

logger = logging.getLogger(__name__)


def _max_disp(current, reference) -> float:
    """Largest per-atom Euclidean displacement between two coordinate sets.

    The Euclidean norm, NOT a component-wise max: the latter underestimates the true
    displacement by up to sqrt(3) and would let a neighbour list go stale unnoticed. The
    jax optimizer computes the same quantity — keep the two in step.
    """
    return float(torch.linalg.norm(current - reference, dim=-1).max())


class TorchRestraintOptimizer:
    def __init__(self, spec, max_iter: int = 100, method: str = "CG"):
        self.spec = spec
        self.max_iter = max_iter
        self.method = method
        self._prepared = None
        self._prepared_g = {}  # cache {gate-state -> stable pre-gated prepared} (GPU CG)
        self._active_idx = None
        self._device = None
        self._dtype = None
        self._vdw = None  # dict of device tensors for the fixed-background VdW term
        self._active_vdw = None  # dynamic active-active polymer neighbour metadata
        # Build custom closures outside inference mode so their index tensors support
        # autograd, including under boltz/Lightning inference.
        self._custom_terms = None
        # Compiled custom energy/gradient artifacts, keyed by VdW mode and active
        # custom subset. A failed artifact disables only its own key.
        self._custom_cvg = {}
        from rgi_toolkit.optim._coordinates import CentroidCoordinates

        self._coordinates = CentroidCoordinates(spec)

    def _custom_energy(self, active, sigma, step):
        """Per-entry gated sum of the custom-restraint closure energies at ``active``
        (local coords). Returns ``None`` when no custom term is active so the caller can
        skip adding it. Gate: the active sigma window (``stop_sigma <= sigma <=
        start_sigma``) AND the active step window (``start_step <= step <= stop_step``);
        a restraint uses one or the other (mutually exclusive). Python-float compares,
        since the eager CG has python-float sigma/step."""
        if not self._custom_terms:
            return None
        total = None
        for _name, start, stop, start_step, stop_step, closure in self._custom_terms:
            if sigma is not None and not (sigma <= start and sigma >= stop):
                continue
            if step is not None and not (start_step <= step <= stop_step):
                continue
            e = closure(active)
            total = e if total is None else total + e
        return total

    def _get_custom_cvg(self, mode=0, active_terms=None):
        """Compile the mode's base energy plus the active custom closures.

        Cache per optimizer, VdW mode and active subset. Omitting inactive closures
        prevents undefined values in a disabled formula from poisoning the objective.
        Device/dtype changes invalidate this cache in _ensure. None means compilation
        is disabled or failed, so the caller uses eager evaluation.
        """
        if active_terms is None:
            active_terms = tuple(range(len(self._custom_terms)))
        key = (mode, tuple(active_terms))
        cached = self._custom_cvg.get(key)
        if cached is False:
            return None
        if cached is not None:
            return cached
        if os.environ.get("RGI_DISABLE_COMPILE", "") not in ("", "0", "false"):
            self._custom_cvg[key] = False
            return None
        from rgi_toolkit.optim._torch_cg_gpu import _ENERGY_BY_MODE

        base = _ENERGY_BY_MODE[mode]
        closures = [self._custom_terms[i][-1] for i in active_terms]

        def energy(a, prepared, *vdw_args):
            e = base(a, prepared, *vdw_args)
            for closure in closures:
                e = e + closure(a)
            return e

        try:
            self._custom_cvg[key] = torch.compile(
                torch.func.grad_and_value(energy, argnums=0),
                fullgraph=False,
                dynamic=False,  # specs and neighbor capacities have static shapes
            )
        except Exception as exc:
            logger.warning(
                "torch.compile of the custom GPU energy failed (%s); eager", exc
            )
            self._custom_cvg[key] = False
            return None
        return self._custom_cvg[key]

    def _ensure(self, device, dtype) -> None:
        if (
            self._prepared is not None
            and self._device == device
            and self._dtype == dtype
        ):
            return
        # Build constant tensors outside inference mode so they are normal (not
        # inference) tensors and can participate in autograd ops with the leaf.
        with torch.inference_mode(False):
            self._active_idx = torch.as_tensor(
                self.spec.active_sites, dtype=torch.long, device=device
            )
            self._prepared = torch_energy.prepare_spec(
                self.spec, device=device, dtype=dtype
            )
            self._prepared_g = {}  # rebuilt lazily for the new device/dtype
            self._setup_vdw(device, dtype)
            from rgi_toolkit.custom.closure import build_terms

            self._custom_terms = build_terms(self.spec.custom, "torch", device=device)
            # Compiled closures capture the device/dtype-specific terms rebuilt above.
            self._custom_cvg = {}
        self._device = device
        self._dtype = dtype
        if self._custom_terms:
            logger.info(
                "%d custom restraint(s): on CUDA they run inside a per-optimizer "
                "torch.compile'd energy+grad (eager on CPU / on a compile fallback)",
                len(self._custom_terms),
            )

    def _gated_prepared(self, sigma, step=None):
        """Stable pre-gated ``prepared`` for the compiled GPU CG, cached by the discrete
        GATE STATE — the conformer gate plus the per-restraint gate of every per-entry
        term (distance, rmsd, group_angle, group_dihedral — ``PER_ENTRY_KEYS``). Each gate is the
        active sigma window (``stop_sigma <= sigma <= start_sigma``) AND the active step
        window (``start_step <= step <= stop_step``) — a restraint uses one or the other
        (mutually exclusive at config), the unused axis always-on so the AND is correct.
        The gate is folded into the masks here so the compiled energy is called with
        ``sigma=None``; scalar leaves (e.g. ``conf_start_sigma``) are DROPPED because a
        python-float in the compiled energy's pytree makes dynamo guard on its value and
        recompile per distinct value. Stable object identity per gate state lets
        ``torch.compile`` reuse its artifact; sigma decreases / step increases
        monotonically so each gate flips at most once -> a few states -> a few compiles,
        then reuse. distance is now a per-entry term (in ``PER_ENTRY_KEYS``), so it is gated
        and folded here like rmsd/group. The conformer-gated key set (``CONF_KEYS``) and the
        per-entry-gated set (``PER_ENTRY_KEYS``) both come from ``TERM_DEFS``, so adding a term
        can't silently leave it ungated on the compiled path."""
        p = self._prepared
        cg_on = sigma is None or (
            (sigma <= float(self.spec.conf_start_sigma))
            and (sigma >= float(self.spec.conf_stop_sigma))
        )
        if step is not None:
            cg_on = cg_on and (
                step >= float(self.spec.conf_start_step)
                and step <= float(self.spec.conf_stop_step)
            )
        cg_key = 1.0 if (sigma is None and step is None) else float(cg_on)
        # Per-restraint on/off for every per-entry term present, evaluated against the
        # host-side spec arrays. Reading the prepared tensors here would force one
        # device-to-host synchronization per term on CUDA via ``on.tolist()``.
        gates: dict[str, tuple] = {}
        if sigma is not None or step is not None:
            for gk in PER_ENTRY_KEYS:
                if gk in p:
                    array = getattr(self.spec, TERM_BY_KEY[gk].spec_attr)
                    on = None
                    if sigma is not None:
                        on = (sigma <= array.start_sigma) & (sigma >= array.stop_sigma)
                    if step is not None:
                        step_on = (step >= array.start_step) & (step <= array.stop_step)
                        on = step_on if on is None else (on & step_on)
                    gates[gk] = tuple(bool(b) for b in on.tolist())
        # Cache every gate state, including each per-entry term.
        key = (cg_key, tuple(sorted(gates.items())))
        cache = self._prepared_g
        if key not in cache:
            pg = {}
            for k, v in p.items():
                if not isinstance(v, dict):
                    continue  # drop scalar leaves (conf_start_sigma): a python-float in
                    # the compiled energy's pytree forces a per-value dynamo recompile,
                    # and total_energy(sigma=None) never reads it
                elif k in CONF_KEYS:
                    pg[k] = {**v, "mask": v["mask"] * cg_key}
                elif k in gates:
                    rg = torch.tensor(
                        gates[k], dtype=v["mask"].dtype, device=v["mask"].device
                    )
                    pg[k] = {**v, "mask": v["mask"] * rg}
                else:
                    pg[k] = v
            cache[key] = pg
        return cache[key]

    def _setup_vdw(self, device, dtype) -> None:
        from rgi_toolkit._array_ops import get_ops
        from rgi_toolkit.energy._nonbonded import prepare_chemistry

        ops = get_ops("torch", device)
        like = torch.empty((), device=device, dtype=dtype)
        vc = getattr(self.spec, "vdw_config", None)
        if vc is None or vc.weight <= 0:
            self._vdw = None
        else:
            self._vdw = {
                "lig_local": torch.as_tensor(
                    vc.ligand_local, dtype=torch.long, device=device
                ),
                "lig_r": torch.as_tensor(vc.ligand_radii, dtype=dtype, device=device),
                "bg_global": torch.as_tensor(
                    vc.background_global, dtype=torch.long, device=device
                ),
                "bg_r": torch.as_tensor(
                    vc.background_radii, dtype=dtype, device=device
                ),
                "weight": torch.as_tensor(float(vc.weight), dtype=dtype, device=device),
                "scale": torch.as_tensor(float(vc.scale), dtype=dtype, device=device),
                "dmax": torch.as_tensor(vc.search_radius, dtype=dtype, device=device),
                "max_neighbors": int(vc.max_neighbors),
                "contact": torch.as_tensor(vc.max_contact, dtype=dtype, device=device),
                "chemistry": prepare_chemistry(ops, vc.chemistry, like),
            }

        ac = getattr(self.spec, "active_vdw_config", None)
        if ac is None or ac.weight <= 0:
            self._active_vdw = None
        else:
            self._active_vdw = {
                "radii": torch.as_tensor(ac.radii, dtype=dtype, device=device),
                "polymer_mask": torch.as_tensor(
                    ac.polymer_mask, dtype=torch.bool, device=device
                ),
                "excluded_codes": torch.as_tensor(
                    ac.excluded_codes, dtype=torch.long, device=device
                ),
                "weight": torch.as_tensor(float(ac.weight), dtype=dtype, device=device),
                "scale": torch.as_tensor(float(ac.scale), dtype=dtype, device=device),
                "dmax": torch.as_tensor(ac.search_radius, dtype=dtype, device=device),
                "max_neighbors": int(ac.max_neighbors),
                "contact": torch.as_tensor(ac.max_contact, dtype=dtype, device=device),
                "chemistry": prepare_chemistry(ops, ac.chemistry, like),
            }

    def _fixed_vdw_pairs(self, active, bg_pos, dmax=None):
        """Build the fixed-background neighbour list at the supplied coordinates."""
        from rgi_toolkit.optim._torch_cg_gpu import build_fixed_vdw_pairs

        v = self._vdw
        return build_fixed_vdw_pairs(
            active,
            bg_pos,
            v["lig_local"],
            v["dmax"] if dmax is None else dmax,
            v["max_neighbors"],
            v["lig_r"],
            v["bg_r"],
            v["scale"],
            v["chemistry"],
        )

    def _vdw_energy(self, active, bg_pos, pairs=None):
        """Fixed-background VdW repulsion over a fixed-width neighbour list."""
        from rgi_toolkit.optim._torch_cg_gpu import _vdw_pair_energy

        v = self._vdw
        if pairs is None:
            pairs = self._fixed_vdw_pairs(active, bg_pos)
        return _vdw_pair_energy(
            active,
            bg_pos,
            v["lig_local"],
            pairs[0],
            pairs[1],
            v["lig_r"],
            v["bg_r"],
            v["scale"],
            v["weight"],
            v["chemistry"],
        )

    def minimize(
        self,
        coords,
        sigma=None,
        step=None,
        start_sigma=None,
        max_iter=None,
        *,
        return_info=False,
    ):
        """Optimize ``coords`` (..., n_atom, 3) in-place. Each restraint is gated on its
        active sigma window AND its active step window (``step`` = diffusion step index)
        inside the energy. If every window is inactive, no objective is evaluated.
        ``return_info=True`` returns ``(coords, CGInfo)`` for CG only."""
        from rgi_toolkit.optim._gates import active_windows, window_on
        from rgi_toolkit.optim.info import inactive_info

        if return_info and not self._is_cg():
            raise ValueError("return_info is supported only for method='cg'")
        info = inactive_info()
        if not self.spec.is_active():
            return (coords, info) if return_info else coords
        # Host-side gates require scalar sigma and step values.
        if sigma is not None:
            sigma = float(sigma)
        if step is not None:
            step = int(step)
        if not window_on(active_windows(self.spec), sigma, step):
            return (coords, info) if return_info else coords
        # Use at least fp32 for CG and its gradients; fp16/bf16 are too coarse
        # for the line search. Restore the input dtype after optimization.
        out_dtype = coords.dtype
        work_dtype = (
            torch.float32
            if coords.dtype in (torch.float16, torch.bfloat16)
            else coords.dtype
        )
        self._ensure(coords.device, work_dtype)
        mi = max_iter if max_iter is not None else self.max_iter
        # Dynamic and static VdW share the conformer gate and CG safety controls.
        conformer_in_window = (
            sigma is None
            or (
                sigma <= float(self.spec.conf_start_sigma)
                and sigma >= float(self.spec.conf_stop_sigma)
            )
        ) and (
            step is None
            or (
                step >= float(self.spec.conf_start_step)
                and step <= float(self.spec.conf_stop_step)
            )
        )
        vdw_active = self._vdw is not None and conformer_in_window
        active_vdw_active = self._active_vdw is not None and conformer_in_window
        from rgi_toolkit.optim import _torch_cg_gpu as gpu
        from rgi_toolkit.optim._cg import torch_cg
        from rgi_toolkit.optim._vdw_runtime import VdwRuntime

        with (
            torch.inference_mode(False),
            torch.enable_grad(),
            torch.autocast(device_type=coords.device.type, enabled=False),
        ):
            active = coords[..., self._active_idx, :].to(work_dtype).clone().detach()
            prepared = torch_energy.bind_peptide_states(
                active, self._gated_prepared(sigma, step)
            )
            bg_pos = (
                coords[..., self._vdw["bg_global"], :].to(work_dtype).clone().detach()
                if vdw_active
                else None
            )
            runtime = VdwRuntime(
                "torch",
                active,
                fixed=self._vdw if vdw_active else None,
                moving=self._active_vdw if active_vdw_active else None,
                background=bg_pos,
                skin=self.spec.vdw_neighbor_skin,
            )
            active_terms = tuple(
                i
                for i, (_n, start, stop, sstep, estep, _c) in enumerate(
                    self._custom_terms
                )
                if (sigma is None or stop <= sigma <= start)
                and (step is None or sstep <= step <= estep)
            )
            base = gpu._ENERGY_BY_MODE[runtime.mode]

            def sparse_energy(a, prepared, *args):
                e = base(a, prepared, *args)
                for i in active_terms:
                    e = e + self._custom_terms[i][-1](a)
                return e

            eager = torch.func.grad_and_value(sparse_energy)
            compiled = None
            if active.is_cuda:
                compiled = (
                    self._get_custom_cvg(runtime.mode, active_terms)
                    if active_terms
                    else gpu._get_cvg(runtime.mode)
                )

            def value_grad(a, cache):
                nonlocal compiled
                args = runtime.args(cache)
                if compiled is not None:
                    try:
                        g, f = compiled(a, prepared, *args)
                    except Exception as exc:
                        logger.warning("compiled CG objective failed (%s); eager", exc)
                        if active_terms:
                            self._custom_cvg[(runtime.mode, active_terms)] = False
                        else:
                            gpu._compile_failed[runtime.mode] = True
                        compiled = None
                        g, f = eager(a, prepared, *args)
                else:
                    g, f = eager(a, prepared, *args)
                dg, df = runtime.dense_value_grad(a, cache)
                return g + dg, f + df

            cache = runtime.empty(active)
            if self._is_cg():
                mapping = self._coordinates.bind(
                    "torch", active, sigma, step, enabled=conformer_in_window
                )
                origin = active

                def physical(u):
                    return u if mapping is None else mapping(u, origin)

                def mapped_value_grad(u, cache):
                    g, f = value_grad(physical(u), cache)
                    return (g if mapping is None else mapping(g)), f

                active, state = torch_cg(
                    mapped_value_grad,
                    active,
                    mi,
                    cache=cache,
                    prepare=lambda u, c: runtime.prepare(physical(u), c),
                )
                active = physical(active)
                info = state.info
            else:
                active.requires_grad_(True)
                opt = torch.optim.LBFGS(
                    [active], max_iter=mi, line_search_fn="strong_wolfe"
                )

                def closure():
                    nonlocal cache
                    cache = runtime.prepare(active.detach(), cache)
                    g, f = value_grad(active.detach(), cache)
                    active.grad = g.detach()
                    return f.detach()

                opt.step(closure)
            new_active = active.detach()

        # Retain input coordinates if optimization produces non-finite values.
        if not torch.isfinite(new_active).all():
            # Distinguish pre-existing non-finite inputs from optimizer failure.
            input_finite = bool(torch.isfinite(coords[..., self._active_idx, :]).all())
            logger.warning(
                "restraint step produced non-finite coords; skipping update "
                "(input_finite=%s)",
                input_finite,
            )
            return (coords, info) if return_info else coords
        # The optimizer is an in-place correction, including for autograd leaf inputs.
        with torch.no_grad():
            coords[..., self._active_idx, :] = new_active.to(out_dtype)
        return (coords, info) if return_info else coords

    def _is_cg(self) -> bool:
        return (self.method or "cg").lower() in (
            "cg",
            "ncg",
            "nonlinear-cg",
            "nonlinearcg",
        )

    def _minimize_cg(
        self,
        active,
        energy_fn,
        max_iter,
        gtol=GTOL,
        state=None,
        **search_options,
    ):
        """Adapt an in-place eager objective to the shared strict-Wolfe solver."""
        from rgi_toolkit.optim._cg import torch_cg

        def value_grad(x):
            with torch.no_grad():
                active.copy_(x)
            active.grad = None
            energy = energy_fn()
            if energy.requires_grad:
                energy.backward()
                gradient = active.grad.detach().clone()
            else:
                gradient = torch.zeros_like(active)
            return gradient, energy.detach()

        out, result = torch_cg(
            value_grad,
            active.detach(),
            max_iter,
            gtol=gtol,
            state=state,
            **search_options,
        )
        with torch.no_grad():
            active.copy_(out)
        return result

    def energy(self, coords) -> float:
        """Current restraint energy (for verbose stats / finalize)."""
        if not self.spec.is_active():
            return 0.0
        self._ensure(coords.device, coords.dtype)
        with torch.no_grad():
            active = coords[..., self._active_idx, :]
            e = torch_energy.total_energy(active, self._prepared)
            custom = self._custom_energy(active, None, None)
            if custom is not None:
                e = e + custom
            return float(e) + self.dynamic_vdw_energy(coords)

    def dynamic_vdw_energy(self, coords) -> float:
        """The dynamic optimizer-only VdW terms (>= 0); for finalize stats.

        Covers both the fixed-background and active-active polymer neighbour-list
        halves. Neither is included in the static array energy breakdown.

        Computed directly (not as energy - static_total) so the reported value is
        exact and non-negative, with no float32/float64 cancellation error. It is
        intentionally ungated: this reports the residual at the final coordinates.
        """
        if not self.spec.is_active():
            return 0.0
        if not isinstance(coords, torch.Tensor):
            coords = torch.as_tensor(coords, dtype=torch.float64)
        # _ensure builds both dynamic halves; call it before checking them so a fresh
        # optimizer reports the true residual rather than a false zero.
        self._ensure(coords.device, coords.dtype)
        if self._vdw is None and self._active_vdw is None:
            return 0.0
        from rgi_toolkit.optim._vdw_runtime import VdwRuntime

        with torch.no_grad():
            active = coords[..., self._active_idx, :]
            runtime = VdwRuntime(
                "torch",
                active,
                fixed=self._vdw,
                moving=self._active_vdw,
                background=coords[..., self._vdw["bg_global"], :]
                if self._vdw
                else None,
                skin=self.spec.vdw_neighbor_skin,
            )
            cache = runtime.prepare(active, runtime.empty(active))
            return float(
                runtime.sparse_energy(active, cache)
                + runtime.dense_value_grad(active, cache, gradient=False)[1]
            )
