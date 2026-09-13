"""Replay real denoising coordinates against SciPy and a complete-pair objective."""

from __future__ import annotations

import argparse
import contextlib
import inspect
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
from scipy.optimize import minimize
from scipy.optimize._optimize import _line_search_wolfe12, _LineSearchError

from rgi_toolkit.energy import torch_energy
from rgi_toolkit.optim._vdw_runtime import VdwRuntime
from rgi_toolkit.optim.info import CGStatus
from rgi_toolkit.optim.torch_optim import TorchRestraintOptimizer


@contextlib.contextmanager
def capture_objective(record):
    """Retain the exact production floating-point callback for a numerical control."""
    from rgi_toolkit.optim import _cg

    original = _cg.torch_cg

    def capture(vg, x0, max_iter, **kwargs):
        record.update(vg=vg, x0=x0.clone(), max_iter=max_iter, **kwargs)
        return original(vg, x0, max_iter, **kwargs)

    _cg.torch_cg = capture
    try:
        yield
    finally:
        _cg.torch_cg = original


def production_control(record):
    """Run SciPy with the identical sparse summation and coordinate transformation."""
    cache = record["cache"]
    origin = record["x0"]

    def objective(x):
        nonlocal cache
        a = torch.as_tensor(x.reshape(origin.shape), dtype=origin.dtype)
        cache = record["prepare"](a, cache)
        g, f = record["vg"](a, cache)
        return float(f), g.numpy().reshape(-1).copy()

    return minimize(
        objective,
        origin.numpy().reshape(-1),
        jac=True,
        method="CG",
        options={"maxiter": record["max_iter"], "gtol": 1e-7, "c1": 1e-4, "c2": 0.4},
    )


@contextlib.contextmanager
def audit_searches(records):
    """Compare each actual search with SciPy from the identical direction and state."""
    from rgi_toolkit.optim import _cg

    original = _cg.strong_wolfe

    def checked(s, evaluate, extra, trial, f0, slope, previous, amin, amax, **options):
        result = original(
            s, evaluate, extra, trial, f0, slope, previous, amin, amax, **options
        )
        state = inspect.getclosurevars(evaluate).nonlocals["st"]
        old_g = state.g.numpy().reshape(-1)
        direction = state.d.numpy().reshape(-1)
        cached = trial

        def at(alpha):
            nonlocal cached
            cached = evaluate(float(np.asarray(alpha).reshape(-1)[0]), cached)
            return cached

        def descent(alpha, *_):
            candidate = at(alpha)
            g = candidate.g.numpy().reshape(-1)
            if np.max(abs(g)) <= 1e-7:
                return True
            beta = max(0.0, np.dot(g, g - old_g) / np.dot(old_g, old_g))
            return np.dot(g, -g + beta * direction) <= -0.01 * np.dot(g, g)

        expected = None
        try:
            reference = _line_search_wolfe12(
                lambda alpha: float(at(alpha).f),
                lambda alpha: np.array([float(at(alpha).slope)]),
                np.zeros(1),
                np.ones(1),
                np.array([float(slope)]),
                float(f0),
                float(previous),
                c1=1e-4,
                c2=0.4,
                amin=float(amin),
                amax=float(amax),
                extra_condition=descent,
            )
            expected = reference[0]
        except _LineSearchError:
            pass
        actual, accepted, _phase = result
        verified = False
        if expected is not None:
            candidate = at(expected)
            verified = bool(
                candidate.finite
                and candidate.moved
                and candidate.f <= f0 + 1e-4 * expected * slope
                and abs(candidate.slope) <= -0.4 * slope
                and descent(expected)
            )
        matched = bool(accepted) == verified
        if matched and verified:
            matched = bool(
                np.isclose(float(actual.alpha), expected, rtol=1e-8, atol=1e-12)
            )
        records.append(
            dict(
                alpha=float(actual.alpha) if accepted else None,
                scipy_alpha=None if expected is None else float(expected),
                accepted=bool(accepted),
                scipy_verified=verified,
                matched=matched,
            )
        )
        return result

    _cg.strong_wolfe = checked
    try:
        yield
    finally:
        _cg.strong_wolfe = original


def complete_pairs(runtime, active):
    """Enumerate chemistry-eligible pairs once, with no coordinate or capacity cutoff."""
    evaluators = []
    for moving, parameters in ((False, runtime.fixed), (True, runtime.moving)):
        if parameters is None:
            continue
        chemistry = {
            key: value.numpy() for key, value in parameters["chemistry"].items()
        }
        nq, nt = len(chemistry["query_types"]), len(chemistry["target_types"])
        sources, targets, contacts, inverses = [], [], [], []
        for i in range(nq):
            js = (
                np.arange(nt)
                if chemistry["query_moving"][i]
                else np.flatnonzero(chemistry["target_moving"])
            )
            code = i * nt + js
            valid = ~np.isin(code, chemistry["excluded"])
            valid &= ~(
                bool(chemistry["query_static"][i])
                & chemistry["target_static"][js].astype(bool)
            )
            same = chemistry["query_molecules"][i] == chemistry["target_molecules"][js]
            mode = int(chemistry["mode"])
            valid &= (mode == 0) | ((mode == 1) & same) | ((mode == 2) & ~same)
            one_four = np.isin(code, chemistry["one_four"])
            first, second = chemistry["query_types"][i], chemistry["target_types"][js]
            contact = np.where(
                one_four,
                chemistry["one_four_contacts"][first, second],
                chemistry["contacts"][first, second],
            )
            inverse = np.where(
                one_four,
                chemistry["one_four_inv_variances"][first, second],
                chemistry["inv_variances"][first, second],
            )
            valid &= contact > 0
            sources.extend([i] * int(valid.sum()))
            targets.extend(js[valid])
            contacts.extend(contact[valid])
            inverses.extend(inverse[valid])
        source = torch.tensor(sources, dtype=torch.int64)
        target = torch.tensor(targets, dtype=torch.int64)
        if not moving:
            source = parameters["lig_local"][source]
        threshold = parameters["scale"] * torch.tensor(contacts, dtype=active.dtype)
        coefficient = parameters["weight"] * torch.tensor(inverses, dtype=active.dtype)
        if moving:
            coefficient = coefficient * 0.5
        # Reproduce the deterministic escape direction only for degenerate overlaps.
        lo = torch.minimum(source, target) if moving else source
        hi = torch.maximum(source, target) if moving else target
        code = lo * 31 + hi
        sign = torch.where((code // 3) % 2 == 0, 1.0, -1.0)
        if moving:
            sign = sign * torch.where(source <= target, 1.0, -1.0)
        fallback = torch.nn.functional.one_hot(code % 3, num_classes=3) * (
            1e-3 * sign[..., None]
        )
        evaluators.append((moving, source, target, threshold, coefficient, fallback))

    def energy(a):
        total = a.sum() * 0.0
        for moving, source, target, threshold, coefficient, fallback in evaluators:
            other = a if moving else runtime.background
            delta = a[..., source, :] - other[..., target, :]
            delta = torch.where(
                (delta.square().sum(-1) < 1e-6)[..., None],
                delta + (fallback - delta).detach(),
                delta,
            )
            distance = (delta.square().sum(-1) + 1e-12).sqrt()
            total = (
                total
                + (coefficient * (distance - threshold).clamp(max=0).square()).sum()
            )
        return total

    return energy


def reference_coordinates(spec, origin, sigma, step):
    """Independent dense-row construction of the fixed affine CG coordinates."""
    terms = spec.distance
    rows, factors = [], []
    if (
        terms is not None
        and spec.has_conformer()
        and spec.conf_stop_sigma <= sigma <= spec.conf_start_sigma
        and spec.conf_start_step <= step <= spec.conf_stop_step
    ):
        for i in np.flatnonzero((terms.mask > 0) & (terms.weight > 0)):
            if not (
                terms.stop_sigma[i] <= sigma <= terms.start_sigma[i]
                and terms.start_step[i] <= step <= terms.stop_step[i]
            ):
                continue
            w = np.zeros(spec.n_active)
            for k in (1, 2):
                if terms.move_mode[i] not in (0, k):
                    continue
                mask = getattr(terms, f"grp{k}_mask")[i]
                np.add.at(
                    w,
                    getattr(terms, f"grp{k}_idx")[i],
                    (1 if k == 1 else -1) * mask / mask.sum(),
                )
            norm = np.linalg.norm(w)
            if 0 < norm < 1:
                rows.append(w / norm)
                factors.append(1 / norm - 1)
    if not rows:
        return lambda u, gradient=False: u
    rows, factors = np.stack(rows), np.asarray(factors)

    def transform(u, gradient=False):
        delta = u if gradient else u - origin
        projected = np.einsum("rn,...nc->...rc", rows, delta) * factors[:, None]
        return u + np.einsum("rn,...rc->...nc", rows, projected)

    return transform


def compare(spec, before, actual, sigma, step, max_iter):
    optimizer = TorchRestraintOptimizer(spec, max_iter=max_iter)
    optimizer._ensure(torch.device("cpu"), torch.float64)
    coords = torch.as_tensor(before, dtype=torch.float64)
    active = coords[..., spec.active_sites, :].clone()
    prepared = torch_energy.bind_peptide_states(
        active, optimizer._gated_prepared(sigma, step)
    )
    in_window = (
        spec.conf_stop_sigma <= sigma <= spec.conf_start_sigma
        and spec.conf_start_step <= step <= spec.conf_stop_step
    )
    fixed = optimizer._vdw if in_window else None
    moving = optimizer._active_vdw if in_window else None
    runtime = VdwRuntime(
        "torch",
        active,
        fixed=fixed,
        moving=moving,
        background=coords[..., fixed["bg_global"], :] if fixed else None,
        skin=spec.vdw_neighbor_skin,
    )
    complete_vdw = complete_pairs(runtime, active)

    def base(a):
        energy = torch_energy.total_energy(a, prepared) + complete_vdw(a)
        custom = optimizer._custom_energy(a, sigma, step)
        return energy if custom is None else energy + custom

    vg = torch.func.grad_and_value(base)

    def oracle(x):
        a = torch.as_tensor(np.asarray(x).reshape(active.shape), dtype=torch.float64)
        g, f = vg(a)
        return float(f), g.numpy().reshape(-1).copy()

    transform = reference_coordinates(spec, active.numpy(), sigma, step)

    def parameter_oracle(u):
        physical = transform(u.reshape(active.shape))
        f, g = oracle(physical)
        return f, transform(g.reshape(active.shape), gradient=True).reshape(-1)

    initial_f, _ = oracle(active.numpy())
    start = time.perf_counter()
    reference = minimize(
        parameter_oracle,
        active.numpy().reshape(-1),
        jac=True,
        method="CG",
        options={"maxiter": max_iter, "gtol": 1e-7, "c1": 1e-4, "c2": 0.4},
    )
    reference_point = transform(reference.x.reshape(active.shape))
    scipy_seconds = time.perf_counter() - start
    start = time.perf_counter()
    searches = []
    objective = {}
    with audit_searches(searches), capture_objective(objective):
        candidate, info = optimizer.minimize(
            coords.clone(), sigma=sigma, step=step, return_info=True
        )
    candidate_seconds = time.perf_counter() - start
    point = candidate[..., spec.active_sites, :].numpy()
    f, g = oracle(point)
    actual_f, actual_g = oracle(np.asarray(actual)[..., spec.active_sites, :])
    reference_f, reference_g = oracle(reference_point)
    energy_error = abs(f - reference_f)
    # Finite trajectories amplify roundoff; their individual searches are audited
    # from identical states instead of treating two unfinished endpoints as minima.
    tolerance = 1e-5 * (1 + abs(reference_f))
    stationary_match = energy_error <= tolerance and np.max(abs(g)) <= max(
        1e-5, 10 * np.max(abs(reference_g))
    )
    optimization_ok = stationary_match if reference.success else f <= initial_f
    roundoff_control = None
    if (
        reference.success
        and not stationary_match
        and int(info.status) == CGStatus.MAX_ITER
        and sigma > 1
    ):
        control = production_control(objective)
        # This extra invocation diagnoses the endpoint; it does not replace the
        # recorded default-budget prediction or change any production setting.
        refined, refinement = optimizer.minimize(
            candidate.clone(), sigma=sigma, step=step, return_info=True
        )
        refined_f, refined_g = oracle(refined[..., spec.active_sites, :].numpy())
        roundoff_control = dict(
            scipy_success=bool(control.success),
            scipy_status=int(control.status),
            scipy_nit=int(control.nit),
            scipy_energy=float(control.fun),
            scipy_gradient=float(np.max(abs(control.jac))),
            refinement_status=int(refinement.status),
            refinement_nit=int(refinement.nit),
            refinement_energy=refined_f,
            refinement_gradient=float(np.max(abs(refined_g))),
            passed=bool(
                control.status == 1
                and control.nit == max_iter
                and f <= float(control.fun) + tolerance
                and int(refinement.status) == CGStatus.CONVERGED
                and abs(refined_f - reference_f) <= tolerance
                and np.max(abs(refined_g)) <= max(1e-5, 10 * np.max(abs(reference_g)))
            ),
        )
        optimization_ok = roundoff_control["passed"]
    parity = []
    for name, coordinates in (
        ("initial", active.numpy()),
        ("scipy", reference_point),
        ("candidate", point),
        ("actual", np.asarray(actual)[..., spec.active_sites, :]),
    ):
        a = torch.as_tensor(coordinates, dtype=torch.float64)
        cache = runtime.prepare(a, runtime.empty(a))

        def sparse(x):
            custom = optimizer._custom_energy(x, sigma, step)
            e = torch_energy.total_energy(x, prepared) + runtime.sparse_energy(x, cache)
            return e if custom is None else e + custom

        pg, pf = torch.func.grad_and_value(sparse)(a)
        dg, df = runtime.dense_value_grad(a, cache)
        expected_f, expected_g = oracle(coordinates)
        df = abs(float(pf + df) - expected_f)
        dg = float(np.max(abs((pg + dg).numpy().reshape(-1) - expected_g)))
        parity.append(
            dict(
                point=name,
                energy_error=df,
                gradient_error=dg,
                passed=bool(
                    df <= 1e-11 * (1 + abs(expected_f))
                    and dg <= 1e-10 * (1 + np.max(abs(expected_g)))
                ),
            )
        )
    distance_difference = []
    if spec.distance is not None:
        terms = spec.distance
        for i in np.flatnonzero(terms.mask):
            values = []
            for coordinates in (point, reference_point):
                centroids = [
                    np.mean(
                        coordinates[
                            ...,
                            getattr(terms, f"grp{k}_idx")[i][
                                getattr(terms, f"grp{k}_mask")[i].astype(bool)
                            ],
                            :,
                        ],
                        axis=-2,
                    )
                    for k in (1, 2)
                ]
                values.append(np.linalg.norm(centroids[0] - centroids[1], axis=-1))
            distance_difference.append(float(np.max(abs(values[0] - values[1]))))
    passed = (
        optimization_ok
        and all(search["matched"] for search in searches)
        and all(p["passed"] for p in parity)
        and (
            (not reference.success and sigma > 1)
            or all(d <= 0.1 for d in distance_difference)
        )
    )
    return dict(
        passed=bool(passed),
        sigma=sigma,
        step=step,
        max_iter=max_iter,
        scipy_success=bool(reference.success),
        scipy_message=str(reference.message),
        scipy_energy=reference_f,
        scipy_gradient=float(np.max(abs(reference_g))),
        scipy_nit=int(reference.nit),
        initial_energy=initial_f,
        candidate_energy=f,
        candidate_gradient=float(np.max(abs(g))),
        candidate_status=int(info.status),
        candidate_nit=int(info.nit),
        energy_error=energy_error,
        energy_tolerance=tolerance,
        comparison=(
            "finite_budget_roundoff"
            if roundoff_control is not None
            else "stationary"
            if reference.success
            else "finite_iteration_budget"
        ),
        stationary_match=bool(stationary_match),
        roundoff_control=roundoff_control,
        search_audit=searches,
        objective_parity=parity,
        distance_difference=distance_difference,
        actual_energy=actual_f,
        actual_gradient=float(np.max(abs(actual_g))),
        scipy_seconds=scipy_seconds,
        candidate_seconds=candidate_seconds,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--captures", default="0,100,last")
    parser.add_argument(
        "--cases", default="distance,distance_conformer,distance_vdw,angle,rmsd,custom"
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    manifest = json.loads(args.manifest.read_text())
    output = Path(manifest["output"])
    summary = []
    requested = {f"step_{name}.npz" for name in args.captures.split(",")}
    for case in args.cases.split(","):
        trace = output / "results" / args.model / case / "0" / "new" / "trace"
        if not (trace / "spec.pkl").exists():
            summary.append(dict(case=case, state="pending"))
            continue
        result_path = trace.parent / "result.json"
        missing = [name for name in sorted(requested) if not (trace / name).exists()]
        if (
            not result_path.exists()
            or json.loads(result_path.read_text())["returncode"] != 0
            or missing
        ):
            summary.append(
                dict(case=case, state="incomplete", missing_captures=missing)
            )
            continue
        with (trace / "spec.pkl").open("rb") as stream:
            spec = pickle.load(stream)
        run = next(
            r
            for r in manifest["runs"]
            if r["model"] == args.model and r["case"] == case and r["seed"] == 0
        )
        config = json.loads(Path(run["input"]).with_name("restraints.json").read_text())
        for capture in sorted(trace.glob("step_*.npz")):
            if capture.name not in requested:
                continue
            data = np.load(capture)
            result = dict(
                case=case,
                capture=capture.name,
                **compare(
                    spec,
                    data["before"],
                    data["after"],
                    float(data["sigma"]),
                    int(data["step"]),
                    config.get("max_iter", 100),
                ),
            )
            summary.append(result)
            print(json.dumps(result), flush=True)
    suffix = (
        "" if args.captures == "0,100,last" else "-" + args.captures.replace(",", "-")
    )
    target = args.output or output / "replay" / (args.model + suffix + ".json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, indent=2) + "\n")
    if any(not r.get("passed", False) for r in summary):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
