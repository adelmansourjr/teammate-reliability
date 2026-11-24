"""Paired evaluation of learned partner reliability in a finite coordination task."""

import argparse
import itertools
import json
from pathlib import Path
import resource
import sys

import numpy as np


TRAIN = 2048
CALIBRATION_N = 64
PARTNERS = 256
ROUTES = 20
CONTACTS = 24
BATCH = 16
BOOT = 10000
PROBE_COST = 0.08
CUE_ERROR = 0.28
PROBE_ERROR = 0.15
GRID = np.arange(0.04, 0.47, 0.06)
SIGNS = np.array(list(itertools.product((0, 1), repeat=3)), dtype=float)
ALL_METHODS = (
    "fixed_pooled", "fixed_support", "adaptive_pooled", "mode_local",
    "factorized", "joint", "known_preference", "known_reliability", "dual_oracle",
)
PRIMARY = ("fixed_support", "adaptive_pooled", "mode_local")
PANELS = ("heterogeneous", "reversed_support", "within_route_shift")
METRICS = ("reward", "mismatches", "matches", "abstentions", "brier",
           "first_half_reward", "second_half_reward", "first_half_brier", "second_half_brier")


def normalize(weights, axis=-1):
    return weights / weights.sum(axis=axis, keepdims=True)


def fit_population():
    rng = np.random.default_rng(884323)
    preferences = rng.random((TRAIN, 3)) < (0.50, 0.30, 0.16)
    latent = rng.beta(1.6, 2.2, TRAIN)
    rates = np.column_stack((
        0.035 + 0.39 * latent + rng.normal(0, 0.046, TRAIN),
        0.095 + 0.37 * latent + rng.normal(0, 0.060, TRAIN),
    )).clip(0.025, 0.475)
    flips = rng.binomial(CALIBRATION_N, rates)
    observed = (flips + 0.5) / (CALIBRATION_N + 1)
    grid_labels = np.abs(observed[:, :, None] - GRID).argmin(axis=2)
    n_grid = len(GRID)
    joint = np.bincount(grid_labels[:, 0] * n_grid + grid_labels[:, 1], minlength=n_grid**2)
    joint = normalize(joint.reshape(n_grid, n_grid) + 0.5, axis=None)
    pooled = np.abs(observed.mean(axis=1, keepdims=True) - GRID).argmin(axis=1)
    pooled = normalize(np.bincount(pooled, minlength=n_grid) + 0.5)
    convention = np.bincount(preferences @ np.array((4, 2, 1)), minlength=8)
    return normalize(convention + 0.5), joint, pooled, observed.mean(axis=0)


def bank(name):
    rng = np.random.default_rng(dict(zip(PANELS, (740310, 909351, 511732)))[name])
    if name == "heterogeneous":
        preferences = rng.random((PARTNERS, 3)) < (0.47, 0.36, 0.23)
        pair = rng.random(PARTNERS) < 0.30
        preferences[pair, 1] = preferences[pair, 0]
        frail = (rng.random(PARTNERS) < 0.46)[:, None]
        robust = rng.uniform((0.025, 0.08), (0.11, 0.20), size=(PARTNERS, 2))
        difficult = rng.uniform((0.09, 0.33), (0.22, 0.47), size=(PARTNERS, 2))
        initial = np.where(frail, difficult, robust)
    else:
        preferences = rng.random((PARTNERS, 3)) < (0.66, 0.42, 0.28)
        pair = rng.random(PARTNERS) < 0.60
        preferences[pair, 2] = ~preferences[pair, 0]
        initial = np.column_stack((
            rng.triangular(0.055, 0.11, 0.23, PARTNERS),
            rng.triangular(0.252, 0.39, 0.475, PARTNERS),
        ))
        invert = rng.random(PARTNERS) < 0.42
        initial[invert] = initial[invert, ::-1]
    later = initial.copy()
    if name == "within_route_shift":
        switches = rng.random(PARTNERS) < 0.70
        later[switches] = (later[switches, ::-1] + (0.015, 0.05)).clip(max=0.475)
    n = PARTNERS * ROUTES
    route_mix = rng.dirichlet(np.ones(3) * 0.9, size=n)
    mode = (rng.random((n, CONTACTS, 1)) >= route_mix.cumsum(axis=1)[:, None, :-1]).sum(axis=2)
    difficult = rng.random((n, CONTACTS)) < rng.uniform(0.25, 0.75, size=(n, 1))
    penalty = rng.uniform(0.7, 4.6, size=(n, CONTACTS))
    side = np.repeat(preferences, ROUTES, axis=0)
    preferred = np.take_along_axis(side, mode, axis=1)
    initial_r = np.repeat(initial, ROUTES, axis=0)
    later_r = np.repeat(later, ROUTES, axis=0)
    r = np.where(
        np.arange(CONTACTS)[None, :] < CONTACTS // 2,
        np.take_along_axis(initial_r, difficult.astype(int), axis=1),
        np.take_along_axis(later_r, difficult.astype(int), axis=1),
    )
    cue = preferred ^ (rng.random((n, CONTACTS)) < CUE_ERROR)
    action = preferred ^ (rng.random((n, CONTACTS)) < r)
    probe_mode = (mode[:, :, None] == np.arange(3)).sum(axis=1).argmax(axis=1)
    idx = np.arange(n)
    probe = side[idx, probe_mode] ^ (rng.random(n) < PROBE_ERROR)
    return mode, difficult.astype(int), penalty, side, r, cue, action, probe_mode, probe


def types(method, fit):
    convention, joint, pooled, means = fit
    if method == "fixed_pooled":
        support = np.full((1, 2), means.mean())
        slip_prior = np.ones(1)
    elif method in ("fixed_support", "known_reliability", "dual_oracle"):
        support = means[None, :]
        slip_prior = np.ones(1)
    elif method == "adaptive_pooled":
        support = np.repeat(GRID[:, None], 2, axis=1)
        slip_prior = pooled
    else:
        support = np.array(list(itertools.product(GRID, repeat=2)))
        slip_prior = joint.ravel()
        if method == "factorized":
            slip_prior = np.outer(joint.sum(axis=1), joint.sum(axis=0)).ravel()
    return (
        np.repeat(SIGNS, len(support), axis=0),
        np.tile(support, (len(SIGNS), 1)),
        np.outer(convention, slip_prior).ravel(),
    )


def update(belief, types_observed, observations, error):
    match = types_observed == observations[:, None]
    return normalize(belief * np.where(match, 1 - error, error))


def evaluate(method, fit, scenario):
    modes, supports, costs, prefers, actual_rates, cues, actions, probe_modes, probes = scenario
    rows = np.arange(len(modes))
    side_types, rate_types, prior = types(method, fit)
    known_side = method in ("known_preference", "dual_oracle")
    known_rate = method in ("known_reliability", "dual_oracle")
    local_only = method == "mode_local"
    belief = np.broadcast_to(prior, (len(rows), len(prior))).copy()
    if known_side:
        belief = normalize(belief * np.all(prefers[:, None, :] == side_types[None, :, :], axis=2))
    prior_after_probe = update(belief, side_types.T[probe_modes], probes, PROBE_ERROR)
    if local_only:
        local = np.repeat(belief[:, None, :], 3, axis=1)
        local[rows, probe_modes] = prior_after_probe
    else:
        belief = prior_after_probe

    totals = np.zeros((len(rows), len(METRICS)))
    totals[:, 0] = -PROBE_COST
    totals[:, 5] = -PROBE_COST
    for t in range(CONTACTS):
        mode = modes[:, t]
        support = supports[:, t]
        actual = actions[:, t]
        hypothesized_side = side_types.T[mode]
        if local_only:
            belief = local[rows, mode]
        belief = update(belief, hypothesized_side, cues[:, t], CUE_ERROR)
        lapse = actual_rates[:, t, None] if known_rate else rate_types.T[support]
        likelihood_one = np.where(hypothesized_side, 1 - lapse, lapse)
        prediction = np.einsum("ij,ij->i", belief, likelihood_one)
        agree_one = prediction >= 0.5
        certainty = np.maximum(prediction, 1 - prediction)
        push = certainty > costs[:, t] / (1 + costs[:, t])
        match = agree_one == actual
        payoff = np.where(push, np.where(match, 1, -costs[:, t]), 0)
        brier = (prediction - actual) ** 2
        totals[:, 0] += payoff
        totals[:, 1] += push & ~match
        totals[:, 2] += push & match
        totals[:, 3] += ~push
        totals[:, 4] += brier / CONTACTS
        half = int(t >= CONTACTS // 2)
        totals[:, 5 + half] += payoff
        totals[:, 7 + half] += brier / (CONTACTS // 2)
        likelihood = np.where(actual[:, None], likelihood_one, 1 - likelihood_one)
        belief = normalize(belief * np.where(push[:, None], likelihood, 1))
        if local_only:
            local[rows, mode] = belief
    assert np.isfinite(totals).all()
    assert np.all(totals[:, 1:4].sum(axis=1) == CONTACTS)
    assert np.allclose(totals[:, 0], totals[:, 5:7].sum(axis=1))
    return totals


def run_panel(name, fit):
    data = bank(name)
    participants = {method: [] for method in ALL_METHODS}
    for start in range(0, PARTNERS, BATCH):
        routes = slice(start * ROUTES, (start + BATCH) * ROUTES)
        case = tuple(d[routes] for d in data)
        for method in ALL_METHODS:
            scores = evaluate(method, fit, case)
            participants[method].append(scores.reshape(-1, ROUTES, len(METRICS)).mean(axis=1))
    return {method: np.concatenate(scores) for method, scores in participants.items()}


def interval(values, bootstrap):
    samples = values[bootstrap].mean(axis=1)
    low, high = np.quantile(samples, [0.025, 0.975])
    return [float(values.mean()), float(low), float(high)], float(np.quantile(samples, 0.05 / 6))


def summarize(scenarios, seed):
    rng = np.random.default_rng(seed)
    indices = rng.integers(len(scenarios["joint"]), size=(BOOT, len(scenarios["joint"])))
    joint = scenarios["joint"]
    details = {}
    for method, scores in scenarios.items():
        if method == "joint":
            continue
        differences = {}
        lower = {}
        for field, metric in enumerate(METRICS):
            estimate, low = interval(joint[:, field] - scores[:, field], indices)
            differences[metric] = estimate
            if metric == "reward" and method in PRIMARY:
                lower[metric] = low
        details[method] = {"joint_minus_baseline": differences, "primary_lower_bound": lower}
    return {
        "means": {method: dict(zip(METRICS, s.mean(axis=0).tolist())) for method, s in scenarios.items()},
        "contrasts": details,
        "partner_means": {method: s.tolist() for method, s in scenarios.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    fit = fit_population()
    panels = {name: summarize(run_panel(name, fit), 12102 + i) for i, name in enumerate(PANELS)}
    criteria = {
        name: {
            method: panel["contrasts"][method]["joint_minus_baseline"]["reward"][0] >= 0.20
            and panel["contrasts"][method]["primary_lower_bound"]["reward"] > 0
            for method in PRIMARY
        }
        for name, panel in panels.items() if name != "within_route_shift"
    }
    result = {
        "training_partners": TRAIN, "training_trials_per_support": CALIBRATION_N,
        "partners_per_panel": PARTNERS, "routes_per_partner": ROUTES, "steps_per_route": CONTACTS,
        "grid": GRID.tolist(), "fitted_preferences": fit[0].tolist(),
        "fitted_joint_grid_prior": fit[1].tolist(), "fitted_pooled_grid_prior": fit[2].tolist(),
        "fitted_support_means": fit[3].tolist(), "passed_practical_and_statistical": criteria,
        "panels": panels,
    }
    args.output.write_text(json.dumps(result, separators=(",", ":"), allow_nan=False) + "\n")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    print(f"Peak RSS {peak / (2**20 if sys.platform == 'darwin' else 1024):.1f} MiB")


if __name__ == "__main__":
    main()
