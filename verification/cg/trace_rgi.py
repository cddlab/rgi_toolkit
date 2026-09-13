"""Capture public-API activity, timing and replay inputs outside production code."""

from __future__ import annotations

import atexit
import hashlib
import importlib
import inspect
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np


def install():
    folder = Path(os.environ["RGI_VERIFY_TRACE"])
    folder.mkdir(parents=True, exist_ok=True)
    if os.environ.get("RGI_VERIFY_LEGACY"):
        package = importlib.import_module("rgi_utils")
        sys.modules["rgi_toolkit"] = package
        sys.modules["rgi_toolkit.combined"] = importlib.import_module(
            "rgi_utils.combined"
        )
        # New predictor parsers accept external config files; the historical engine
        # predates that loader. All baseline inputs here are fully expanded mappings.
        config = importlib.import_module("rgi_utils.config")
        if not hasattr(config, "resolve_restraints_config"):
            config.resolve_restraints_config = lambda config, **_: config
        sys.modules["rgi_toolkit.config"] = config
        adapter = importlib.import_module("rgi_utils.openfold3.adapter")
        constructor = adapter.Openfold3Adapter.__init__

        def legacy_openfold(self, *args, smiles_by_chain=None, **kwargs):
            if smiles_by_chain:
                raise ValueError("The historical baseline only supports protein inputs")
            constructor(self, *args, **kwargs)

        adapter.Openfold3Adapter.__init__ = legacy_openfold
        sys.modules["rgi_toolkit.openfold3.adapter"] = adapter
    package = importlib.import_module("rgi_toolkit")
    source = Path(package.__file__).parents[1]
    digest = hashlib.sha256(
        b"".join(
            str(Path("src") / p.relative_to(source)).encode() + p.read_bytes()
            for p in sorted(source.rglob("*.py"))
        )
    ).hexdigest()
    provenance = folder / f".source-{os.getpid()}.json"
    provenance.write_text(
        json.dumps({"module": package.__file__, "source_sha256": digest}, indent=2)
        + "\n"
    )
    provenance.replace(folder / "source.json")
    if os.environ.get("RGI_VERIFY_INSTRUMENT") == "0":
        return
    import rgi_toolkit

    cls = rgi_toolkit.CombinedRestraints
    setup, minimize, finalize = cls.setup, cls.minimize, cls.finalize
    supports_info = "return_info" in inspect.signature(minimize).parameters
    records = []
    origin = time.perf_counter()
    first = None
    last = None

    def host(array):
        if hasattr(array, "detach"):
            return array.detach().cpu().numpy()
        return np.asarray(array)

    def sync(array):
        if hasattr(array, "is_cuda") and array.is_cuda:
            import torch

            torch.cuda.synchronize(array.device)
        elif hasattr(array, "block_until_ready"):
            array.block_until_ready()

    def wrapped_setup(self, adapter, *args, **kwargs):
        begin = time.perf_counter()
        result = setup(self, adapter, *args, **kwargs)
        records.append(
            {
                "stage": "setup",
                "seconds": time.perf_counter() - begin,
                "n_active": self.spec.n_active,
                "source": rgi_toolkit.__file__,
            }
        )
        (folder / "atoms.json").write_text(
            json.dumps([vars(atom) for atom in adapter.iter_atoms()], default=str)
        )
        try:
            with (folder / "spec.pkl").open("wb") as stream:
                pickle.dump(self.spec, stream)
        except (pickle.PicklingError, AttributeError, TypeError) as error:
            records.append({"stage": "capture_error", "error": str(error)})
        return result

    def wrapped_minimize(self, coords, istep=0, sigma=None, **kwargs):
        nonlocal first, last
        sync(coords)
        begin = time.perf_counter()
        if first is None:
            first = begin
        capture = istep in (0, 100) or float(sigma) < 0.01
        before = host(coords).copy() if capture else None
        want_info = kwargs.pop("return_info", False)
        result = minimize(
            self,
            coords,
            istep,
            sigma,
            **kwargs,
            **({"return_info": True} if supports_info else {}),
        )
        out, info = result if supports_info else (result, None)
        sync(out)
        last = time.perf_counter()
        record = {
            "stage": "minimize",
            "step": int(istep),
            "sigma": float(sigma),
            "seconds": last - begin,
        }
        if info is not None:
            record.update(
                zip(
                    ("status", "nit", "nfev", "njev", "fun", "grad_norm"),
                    map(float, info),
                )
            )
        records.append(record)
        if capture:
            np.savez_compressed(
                folder
                / ("step_last.npz" if float(sigma) < 0.01 else f"step_{istep}.npz"),
                before=before,
                after=host(out),
                sigma=sigma,
                step=istep,
            )
        return (out, info) if want_info else out

    def wrapped_finalize(self, coords, *args, **kwargs):
        sync(coords)
        np.save(folder / "final.npy", host(coords))
        return finalize(self, coords, *args, **kwargs)

    cls.setup, cls.minimize, cls.finalize = (
        wrapped_setup,
        wrapped_minimize,
        wrapped_finalize,
    )

    if os.environ.get("RGI_VERIFY_JAX_CAPTURE"):
        import jax

        get_minimizer = cls.get_minimizer
        has_info = "return_info" in inspect.signature(get_minimizer).parameters
        last_captured_step = -1

        def capture(before, after, sigma, step, info):
            nonlocal last_captured_step
            late = float(sigma) < 0.01
            # An outer vmap may lower the surrounding cond to select.
            if not late and int(step) not in (0, 100):
                return
            if late and int(step) < last_captured_step:
                return
            if late:
                last_captured_step = int(step)
            np.savez_compressed(
                folder / ("step_last.npz" if late else f"step_{int(step)}.npz"),
                before=before,
                after=after,
                sigma=sigma,
                step=step,
            )
            record = {
                "stage": "jax_minimize_capture",
                "step": int(step),
                "sigma": float(sigma),
            }
            if info is not None:
                record.update(
                    zip(
                        ("status", "nit", "nfev", "njev", "fun", "grad_norm"),
                        map(float, info),
                    )
                )
            records.append(record)

        def wrapped_get(self, **kwargs):
            want_info = kwargs.pop("return_info", False)
            fn = get_minimizer(
                self, **kwargs, **({"return_info": True} if has_info else {})
            )

            def call(coords, sigma, step=0):
                with jax.named_scope("rgi_minimize"):
                    result = fn(coords, sigma, step)
                after, info = result if has_info else (result, None)

                def save_trial(_):
                    jax.debug.callback(capture, coords, after, sigma, step, info)

                jax.lax.cond(
                    (step == 0) | (step == 100) | (sigma < 0.01),
                    save_trial,
                    lambda _: None,
                    None,
                )
                return (after, info) if want_info else after

            return call

        cls.get_minimizer = wrapped_get

    def save():
        if not records:
            return
        if os.environ.get("RGI_VERIFY_JAX_CAPTURE"):
            import jax

            jax.effects_barrier()
        summary = {
            "seconds": time.perf_counter() - origin,
            "first_minimize": None if first is None else first - origin,
            "diffusion_seconds": None if first is None else last - first,
            "events": records,
        }
        (folder / "trace.json").write_text(json.dumps(summary, indent=2) + "\n")

    atexit.register(save)
