"""Compare equal-cost partner probes in a finite cooperative load task."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


MODES = 3
PARTNERS = 128
ROUTES = 32
CONTACTS = 5
CONVENTIONS = np.array(list(itertools.product((0, 1), repeat=MODES)))
TRAIN_ONE = np.array([0.50, 0.30, 0.16])
PROBE_ERROR = 0.15
CUE_ERROR = 0.28
MOTOR_ERROR = 0.12
PROBE_COST = 0.08
METHODS = ("no_probe", "random", "entropy", "route_frequency", "route_load", "decision_value", "oracle")


def likelihood(conventions, observed, error):
    return np.where(conventions == observed, 1 - error, error)


def normalize(belief):
    return belief / belief.sum(axis=1, keepdims=True)


def predictive_value(belief, mode, cost):
    """Expected local payoff after seeing the passive cue at the contact."""
    positions = CONVENTIONS[:, mode].T
    result = np.zeros(len(belief))
    for signal in (0, 1):
        joint = belief * likelihood(positions, signal, CUE_ERROR)
        mass = joint.sum(axis=1)
        p_one = MOTOR_ERROR + (1 - 2 * MOTOR_ERROR) * ((joint * positions).sum(axis=1) / mass)
        best = np.maximum(p_one, 1 - p_one)
        result += mass * np.maximum(0, best - (1 - best) * cost)
    return result


def choose_probes(prior, mode, cost, random_choice):
    counts = (mode[:, :, None] == np.arange(MODES)).sum(axis=1)
    loads = ((mode[:, :, None] == np.arange(MODES)) * cost[:, :, None]).sum(axis=1)
    p_one = PROBE_ERROR + (1 - 2 * PROBE_ERROR) * (prior @ CONVENTIONS)
    entropy = -p_one * np.log(p_one) - (1 - p_one) * np.log(1 - p_one)
    planning = np.empty((len(mode), MODES))
    for probe in range(MODES):
        value = np.zeros(len(mode))
        for observed in (0, 1):
            unnormalized = prior * likelihood(CONVENTIONS[:, probe], observed, PROBE_ERROR)
            mass = unnormalized.sum()
            beliefs = np.broadcast_to(unnormalized / mass, (len(mode), len(prior)))
            for step in range(CONTACTS):
                value += mass * predictive_value(beliefs, mode[:, step], cost[:, step])
        planning[:, probe] = value
    return {
        "random": random_choice, "entropy": np.full(len(mode), entropy.argmax()),
        "route_frequency": counts.argmax(axis=1), "route_load": loads.argmax(axis=1),
        "decision_value": planning.argmax(axis=1),
    }


def evaluate(method, probe_modes, prior, hidden, motor_error, mode, cost, cues, probe_signals, force):
    n = len(mode)
    rows = np.arange(n)
    belief = np.broadcast_to(prior, (n, len(prior))).copy()
    reward = np.zeros(n)
    mismatches = np.zeros(n)
    passes = np.zeros(n)
    if method in probe_modes:
        probe = probe_modes[method]
        belief = normalize(belief * likelihood(CONVENTIONS[:, probe].T, probe_signals[rows, probe, None], PROBE_ERROR))
        reward -= PROBE_COST
        task_relevant = (mode == probe[:, None]).any(axis=1)
    else:
        task_relevant = np.zeros(n, dtype=bool)
    for i in range(CONTACTS):
        contact = mode[:, i]
        signs = CONVENTIONS[:, contact].T
        if method == "oracle":
            chance_one = motor_error + (1 - 2 * motor_error) * hidden[rows, contact]
        else:
            belief = normalize(belief * likelihood(signs, cues[:, i, None], CUE_ERROR))
            chance_one = MOTOR_ERROR + (1 - 2 * MOTOR_ERROR) * (belief * signs).sum(axis=1)
        push_one = chance_one >= 0.5
        success_rate = np.maximum(chance_one, 1 - chance_one)
        push = success_rate - (1 - success_rate) * cost[:, i] > 0
        success = push_one == force[:, i]
        reward += np.where(push, np.where(success, 1, -cost[:, i]), 0)
        mismatches += push & ~success
        passes += ~push
        if method != "oracle":
            update = likelihood(signs, force[:, i, None], MOTOR_ERROR)
            belief = normalize(belief * np.where(push[:, None], update, 1))
    return np.stack((reward, mismatches, passes, task_relevant), axis=1).reshape(PARTNERS, ROUTES, 4)


def run_panel(prior, seed, shifted):
    rng = np.random.default_rng(seed)
    bank = rng.random((PARTNERS, MODES)) < (np.array([0.65, 0.48, 0.35]) if shifted else TRAIN_ONE)
    if shifted:
        # Introduce a correlation absent from training without changing the action space.
        linked = rng.random(PARTNERS) < 0.55
        bank[linked, 2] = ~bank[linked, 0]
    error = rng.uniform(*( (0.19, 0.33) if shifted else (0.07, 0.17) ), size=PARTNERS)
    hidden = np.repeat(bank, ROUTES, axis=0)
    errors = np.repeat(error, ROUTES)
    n = len(hidden)
    route_mix = rng.dirichlet(np.ones(MODES) * 0.85, size=n)
    mode = (rng.random((n, CONTACTS, 1)) >= route_mix.cumsum(axis=1)[:, None, :-1]).sum(axis=2)
    costs = rng.uniform(0.65, 4.25, size=(n, CONTACTS))
    target = np.take_along_axis(hidden, mode, axis=1)
    cues = target ^ (rng.random((n, CONTACTS)) < CUE_ERROR)
    probes = hidden ^ (rng.random((n, MODES)) < PROBE_ERROR)
    force = target ^ (rng.random((n, CONTACTS)) < errors[:, None])
    probe_modes = choose_probes(prior, mode, costs, rng.integers(MODES, size=n))
    results = {
        method: evaluate(method, probe_modes, prior, hidden, errors, mode, costs, cues, probes, force).mean(axis=1)
        for method in METHODS
    }
    return results


def summarize(results, seed):
    rng = np.random.default_rng(seed)
    metrics = ("net_reward", "mismatches", "abstentions", "probe_used_configuration")
    indices = rng.integers(PARTNERS, size=(5000, PARTNERS))
    differences = {}
    for method, result in results.items():
        if method == "decision_value":
            continue
        paired = results["decision_value"][:, 0] - result[:, 0]
        low, high = np.quantile(paired[indices].mean(axis=1), (0.025, 0.975))
        differences[method] = [float(paired.mean()), float(low), float(high)]
    return {
        "means": {name: dict(zip(metrics, x.mean(axis=0).tolist())) for name, x in results.items()},
        "decision_value_improvement_vs": differences,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    train = np.random.default_rng(287114).random((4096, MODES)) < TRAIN_ONE
    indices = train @ np.array([4, 2, 1])
    prior = (np.bincount(indices, minlength=len(CONVENTIONS)) + 1) / (len(train) + len(CONVENTIONS))
    assert np.isclose(prior.sum(), 1)
    result = {
        "training_partners": len(train), "fit_prior": prior.tolist(),
        "partners_per_panel": PARTNERS, "routes_per_partner": ROUTES,
        "contact_steps": CONTACTS, "probe_cost": PROBE_COST,
        "matched": summarize(run_panel(prior, 16669, False), 101),
        "shifted": summarize(run_panel(prior, 90871, True), 102),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
