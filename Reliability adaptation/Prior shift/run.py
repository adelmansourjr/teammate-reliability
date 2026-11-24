"""Hold support-specific lapse marginals fixed while reversing their dependence."""

import argparse
import importlib.util
import json
from pathlib import Path
import resource
import sys

import numpy as np

PARENT = Path(__file__).resolve().parent.parent / "run.py"
spec = importlib.util.spec_from_file_location("coordination_reliability", PARENT)
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


def matched_scenarios():
    rng = np.random.default_rng(273269)
    n = study.PARTNERS * study.ROUTES
    easy = rng.uniform(0.04, 0.40, study.PARTNERS)
    hard = np.sort(rng.uniform(0.09, 0.47, study.PARTNERS))
    rank = np.argsort(np.argsort(easy))
    both_hard = (hard[rank], hard[::-1][rank])
    assert np.array_equal(np.sort(both_hard[0]), np.sort(both_hard[1]))

    preferred = rng.random((study.PARTNERS, 3)) < (0.52, 0.34, 0.19)
    preferred = np.repeat(preferred, study.ROUTES, axis=0)
    route_mix = rng.dirichlet([0.9] * 3, size=n)
    modes = (rng.random((n, study.CONTACTS, 1)) >= route_mix.cumsum(axis=1)[:, None, :-1]).sum(axis=2)
    supports = (rng.random((n, study.CONTACTS)) < rng.uniform(0.25, 0.75, (n, 1))).astype(int)
    costs = rng.uniform(0.7, 4.6, (n, study.CONTACTS))
    sides = np.take_along_axis(preferred, modes, axis=1)
    cues = sides ^ (rng.random((n, study.CONTACTS)) < study.CUE_ERROR)
    probe_modes = (modes[:, :, None] == np.arange(3)).sum(axis=1).argmax(axis=1)
    probes = preferred[np.arange(n), probe_modes] ^ (rng.random(n) < study.PROBE_ERROR)
    motors = rng.random((n, study.CONTACTS))
    panels = {}
    for name, hard_i in zip(("aligned", "reversed"), both_hard):
        rates_i = np.repeat(np.column_stack((easy, hard_i)), study.ROUTES, axis=0)
        rates = np.take_along_axis(rates_i, supports, axis=1)
        forces = sides ^ (motors < rates)
        scenarios = (modes, supports, costs, preferred, rates, cues, forces, probe_modes, probes)
        panels[name] = scenarios
    return panels


def fit_regimes():
    trained = study.fit_population()
    preference, dependence, pooled, support = trained
    independent = np.outer(dependence.sum(axis=1), dependence.sum(axis=0))
    mixture = (dependence + independent) * 0.5
    assert np.allclose(dependence.sum(axis=0), mixture.sum(axis=0))
    assert np.allclose(dependence.sum(axis=1), mixture.sum(axis=1))
    mixed = (preference, mixture, pooled, support)
    return {
        "fixed_support": ("fixed_support", trained),
        "adaptive_pooled": ("adaptive_pooled", trained),
        "joint": ("joint", trained),
        "factorized": ("factorized", trained),
        "mixture": ("joint", mixed),
        "dual_oracle": ("dual_oracle", trained),
    }


def evaluate_bank(scenario, models):
    partners = {name: [] for name in models}
    for start in range(0, study.PARTNERS, study.BATCH):
        route_slice = slice(start * study.ROUTES, (start + study.BATCH) * study.ROUTES)
        batch = tuple(v[route_slice] for v in scenario)
        for name, (method, fitted) in models.items():
            diary = study.evaluate(method, fitted, batch)
            partners[name].append(diary.reshape(-1, study.ROUTES, len(study.METRICS)).mean(axis=1))
    return {name: np.concatenate(ledger) for name, ledger in partners.items()}


def score(ledgers):
    rng = np.random.default_rng(776312)
    indices = rng.integers(study.PARTNERS, size=(study.BOOT, study.PARTNERS))

    def paired(values):
        return study.interval(values, indices)[0]

    contrasts = {
        name: values["joint"][:, 0] - values["factorized"][:, 0]
        for name, values in ledgers.items()
    }
    interaction = paired(contrasts["aligned"] - contrasts["reversed"])

    # For each resample, compare the same arm against the better component by panel.
    uncertainty = {}
    for candidate in ("joint", "factorized", "mixture"):
        points, bootstrap = [], []
        for values in ledgers.values():
            gap_to_joint = values["joint"][:, 0] - values[candidate][:, 0]
            gap_to_factorized = values["factorized"][:, 0] - values[candidate][:, 0]
            points.append(max(gap_to_joint.mean(), gap_to_factorized.mean()))
            bootstrap.append(np.maximum(gap_to_joint[indices].mean(axis=1), gap_to_factorized[indices].mean(axis=1)))
        largest = np.max(bootstrap, axis=0)
        uncertainty[candidate] = [float(max(points)), *np.quantile(largest, (0.025, 0.975)).tolist()]

    return {
        "members_per_panel": study.PARTNERS,
        "routes_per_member": study.ROUTES,
        "aligned_minus_reversed_prior_interaction": interaction,
        "primary_pass": interaction[0] >= 0.20 and interaction[1] > 0,
        "joint_minus_factorized_by_panel": {name: paired(c) for name, c in contrasts.items()},
        "worst_panel_gap_from_best_component": uncertainty,
        "panels": {
            panel: {
                "means": {name: dict(zip(study.METRICS, rows.mean(axis=0).tolist())) for name, rows in values.items()},
                "partner_means": {name: rows.tolist() for name, rows in values.items()},
            } for panel, values in ledgers.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    models = fit_regimes()
    result = score({name: evaluate_bank(data, models) for name, data in matched_scenarios().items()})
    args.output.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"Peak RSS {peak / (2**20 if sys.platform == 'darwin' else 1024):.1f} MiB")


if __name__ == "__main__":
    main()
