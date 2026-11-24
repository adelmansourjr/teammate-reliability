"""Train two decentralized thrust policies and evaluate held-out cross-play."""

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import resource
import sys
import time

import numpy as np


WORLDS = 12
PAIRS = 8
STEPS = 14_400
BLOCK = 24
GATES = 5
HORIZON = 22
REPEATS = 16
BOOT_DRAWS = 10_000
ACTIONS = np.array((-1, 0, 1), dtype=np.int8)
ARMS = {
    "fixed_short": (False, False), "fixed_history": (False, True),
    "rotating_short": (True, False), "rotating_history": (True, True),
}
METHODS = (*ARMS, "bayesian_guide")
METRICS = ("completion", "jam", "bends", "contacts", "reward", "clashes")


@dataclass(frozen=True)
class Conditions:
    flip: float
    rough: float
    cue_noise: tuple[float, float]
    slip: tuple[float, float]
    jam: float = 2.20


TRAIN = Conditions(0.50, 0.33, (0.22, 0.22), (0.03, 0.10))
BANK = (
    Conditions(0.50, 0.52, (0.28, 0.25), (0.035, 0.13)),
    Conditions(0.25, 0.40, (0.23, 0.23), (0.035, 0.14), 3.00),
)
TEST = (
    Conditions(0.82, 0.52, (0.24, 0.27), (0.045, 0.14)),
    Conditions(0.22, 0.60, (0.33, 0.29), (0.055, 0.19)),
)
PANEL_NAMES = ("alternating_bends", "clustered_bends")


def layouts(rng, count, conditions):
    flips = np.where(rng.random((count, GATES)) < conditions.flip, -1, 1)
    flips[:, 0] = np.where(rng.random(count) < 0.5, -1, 1)
    turns = np.cumprod(flips, axis=1, dtype=np.int8)
    rough = rng.random((count, GATES)) < conditions.rough
    return turns, rough


class Payload:
    def __init__(self, routes):
        self.turns, self.rough = routes
        self.n = n = len(self.turns)
        self.rows = np.arange(n)
        self.gate = np.zeros(n, dtype=int)
        self.clock = np.zeros(n, dtype=int)
        self.twist = np.zeros(n, dtype=np.int8)
        self.known = np.zeros(n, dtype=np.int8)
        self.last_cue = np.zeros((2, n), dtype=np.int8)
        self.last_force = np.full((2, n), 2, dtype=np.int8)

    def reset(self, select, routes):
        self.turns[select] = routes[0][select]
        self.rough[select] = routes[1][select]
        for x in (self.gate, self.clock, self.twist, self.known):
            x[select] = 0
        self.last_cue[:, select] = 0
        self.last_force[:, select] = 2

    def terrain(self):
        gate = np.minimum(self.gate, GATES - 1)
        return self.turns[self.rows, gate], self.rough[self.rows, gate]

    def sense(self, draws, conditions):
        goal, _ = self.terrain()
        noise = np.asarray(conditions.cue_noise)[:, None]
        return np.where(draws < noise, -goal, goal)

    def obs(self, cues, history, role):
        _, rough = self.terrain()
        local_twist = self.twist * (1 - 2 * role) * cues[role]
        coarse = (local_twist + 2) * 2 + rough
        if not history:
            return coarse
        partner = self.last_force[role]
        rel_force = np.where(partner == 2, 3, partner * cues[role] + 1)
        rel_hint = self.known * cues[role] + 1
        rel_cue = self.last_cue[role] * cues[role] + 1
        return ((coarse * 4 + rel_force) * 3 + rel_hint) * 3 + rel_cue

    def advance(self, sent, cues, uniforms, conditions, active):
        goal, rough = self.terrain()
        flip = uniforms < np.asarray(conditions.slip)[rough.astype(int)]
        realised = np.where(flip, -sent, sent)
        torque = realised[0] - realised[1]
        relaxed = np.sign(self.twist) * np.maximum(0, np.abs(self.twist) - 1)
        proposed = np.where(torque, self.twist + torque, relaxed)
        jam = active & (np.abs(proposed) > 2)
        safe = active & (np.abs(proposed) <= 1)
        two = (realised[0] == realised[1]) & (realised[0] != 0)
        move = safe & two & (realised[0] == goal)
        wrong = safe & two & ~move
        clash = active & (realised[0] * realised[1] < 0)
        self.gate += move
        self.clock += active
        self.twist[active] = np.clip(proposed[active], -2, 2)
        self.known[wrong] = -realised[0, wrong]
        self.last_force[:, active] = realised[::-1, active]
        self.last_cue[:, active] = cues[:, active]
        self.last_force[:, move] = 2
        self.last_cue[:, move] = 0
        self.known[move] = 0
        complete = active & (self.gate == GATES)
        reward = (-0.06 - 0.08 * np.abs(self.twist) - 0.70 * wrong + move
                  + 1.50 * complete - conditions.jam * jam) * active
        return reward, complete, jam, active & (self.clock >= HORIZON), clash, move


def train(seed, rotating, history, conditions):
    rng = np.random.default_rng(seed)
    count = WORLDS * PAIRS
    tables = np.zeros((2, count, 360 if history else 10, 3), dtype=np.float32)
    world = np.arange(WORLDS)[:, None] * PAIRS
    own = np.arange(count)
    partners = own
    payload = Payload(layouts(rng, count, conditions))
    active = np.ones(count, dtype=bool)
    successes = np.zeros(WORLDS, dtype=int)
    jams = np.zeros(WORLDS, dtype=int)
    ended = np.zeros(WORLDS, dtype=int)
    fit_return = np.zeros(WORLDS)
    cue_draw = rng.random((2, count))
    cues = payload.sense(cue_draw, conditions)
    for t in range(STEPS):
        if t % BLOCK == 0:
            perm = rng.random((WORLDS, PAIRS)).argsort(axis=1)
            partners = (world + perm).ravel() if rotating else own
            payload.reset(active, layouts(rng, count, conditions))
            cues = payload.sense(cue_draw, conditions)
        ids = (own, partners)
        obs = tuple(payload.obs(cues, history, side) for side in range(2))
        values = np.stack([tables[side, ids[side], obs[side]] for side in range(2)])
        greedy = np.argmax(values + (values == values.max(axis=2, keepdims=True))
                           * rng.uniform(0, 1e-6, values.shape), axis=2)
        exploration = 0.045 + 0.235 * (1 - t / STEPS)
        act = np.where(rng.random((2, count)) < exploration, rng.integers(3, size=(2, count)), greedy)
        chosen = ACTIONS[act] * cues
        reward, success, jam, timeout, _, _ = payload.advance(chosen, cues, rng.random((2, count)), conditions, active)
        terminal = success | jam | timeout
        cue_draw = rng.random((2, count))
        cues = payload.sense(cue_draw, conditions)
        latest = tuple(payload.obs(cues, history, side) for side in range(2))
        learning_terminal = terminal | (t % BLOCK == BLOCK - 1)
        for side in range(2):
            target = reward + 0.96 * ~learning_terminal * tables[side, ids[side], latest[side]].max(axis=1)
            index = (side, ids[side], obs[side], act[side])
            tables[index] += 0.105 * (target - tables[index])
        payload.reset(terminal, layouts(rng, count, conditions))
        cues = payload.sense(cue_draw, conditions)
        if t >= STEPS * 3 // 4:
            successes += success.reshape(WORLDS, PAIRS).sum(axis=1)
            jams += jam.reshape(WORLDS, PAIRS).sum(axis=1)
            ended += terminal.reshape(WORLDS, PAIRS).sum(axis=1)
            fit_return += reward.reshape(WORLDS, PAIRS).sum(axis=1)
    assert np.isfinite(tables).all()
    return tables, {
        "last_quarter_finished_routes": ended.tolist(),
        "last_quarter_completed_routes": successes.tolist(),
        "last_quarter_jams": jams.tolist(),
        "last_quarter_reward_per_transition": (fit_return / (STEPS // 4 * PAIRS)).tolist(),
    }


def sample_test(seed, conditions):
    rng = np.random.default_rng(seed)
    count = WORLDS * PAIRS * REPEATS
    return (
        layouts(rng, count, conditions),
        rng.random((HORIZON, 2, count)),
        rng.random((HORIZON, 2, count)),
        (np.arange(WORLDS)[:, None, None] * PAIRS
         + rng.integers(PAIRS, size=(WORLDS, PAIRS, REPEATS))).ravel(),
        rng.integers(3, size=(HORIZON, 2, count)),
    )


def choose(table, ids, state, tie):
    values = table[ids, state]
    # Tie order is a common draw, not an extra cue.
    cyclic = (np.arange(3) - tie[:, None]) % 3
    return np.argmax(values + (values == values.max(axis=1, keepdims=True)) * (2 - cyclic) * 1e-7, axis=1)


def crossplay(focal_name, focal, partner, conditions, if_route, focal_role, partner_history=True):
    routes, cue_draws, motors, partner_ids, ties = if_route
    payload = Payload(tuple(x.copy() for x in routes))
    n = payload.n
    focal_ids = np.repeat(np.arange(WORLDS * PAIRS), REPEATS)
    tally = np.zeros((n, len(METRICS)))
    active = np.ones(n, dtype=bool)
    long_obs = ARMS.get(focal_name, (False, False))[1]
    guide_logodds = np.zeros(n)
    teammate = 1 - focal_role
    for t in range(HORIZON):
        cues = payload.sense(cue_draws[t], conditions)
        action = np.empty((2, n), dtype=np.int8)
        state = payload.obs(cues, partner_history, teammate)
        action[teammate] = ACTIONS[choose(partner[teammate], partner_ids, state, ties[t, teammate])] * cues[teammate]
        if focal_name == "bayesian_guide":
            guide_logodds += cues[focal_role] * np.log((1 - 0.22) / 0.22)
            direction = np.where(payload.known, payload.known, np.where(guide_logodds >= 0, 1, -1))
            reliable = (np.abs(guide_logodds) > 0.2) | (payload.known != 0)
            action[focal_role] = np.where(reliable & (np.abs(payload.twist) < 2), direction, 0)
        else:
            state = payload.obs(cues, long_obs, focal_role)
            action[focal_role] = ACTIONS[choose(focal[focal_role], focal_ids, state, ties[t, focal_role])] * cues[focal_role]
        tally[:, 3] += active
        reward, success, jam, timeout, clash, move = payload.advance(action, cues, motors[t], conditions, active)
        tally[:, 0] += success
        tally[:, 1] += jam
        tally[:, 4] += reward
        tally[:, 5] += clash
        guide_logodds[move] = 0
        active &= ~(success | jam | timeout)
    tally[:, 2] = payload.gate
    assert not active.any()
    assert np.all(tally[:, 3] <= HORIZON)
    assert np.all(tally[:, 0] + tally[:, 1] <= 1)
    assert np.array_equal(tally[:, 0] > 0, tally[:, 2] == GATES)
    return tally.reshape(WORLDS, PAIRS * REPEATS, -1).mean(axis=1)


def report_panel(outcomes, seed):
    bootstrap = np.random.default_rng(seed).integers(WORLDS, size=(BOOT_DRAWS, WORLDS))
    comparisons = (
        ("rotating_history", "fixed_history"), ("rotating_short", "fixed_short"),
        ("rotating_history", "rotating_short"), ("fixed_history", "fixed_short"),
        ("rotating_history", "bayesian_guide"),
    )
    contrasts = {}
    for treatment, control in comparisons:
        difference = outcomes[treatment] - outcomes[control]
        draws = difference[bootstrap].mean(axis=1)
        mean = difference.mean(axis=0)
        limits = np.quantile(draws, (0.025, 0.975), axis=0)
        contrasts[f"{treatment}_minus_{control}"] = {
            metric: [float(mean[i]), *limits[:, i].tolist()] for i, metric in enumerate(METRICS)
        }
    key = "rotating_history_minus_fixed_history"
    primary = contrasts[key]["completion"]
    return {
        "means": {key: dict(zip(METRICS, value.mean(axis=0).tolist())) for key, value in outcomes.items()},
        "contrasts": contrasts,
        "primary_pass": primary[0] >= 0.03 and primary[1] > 0,
        "block_means": {key: value.tolist() for key, value in outcomes.items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    start = time.monotonic()
    focal = {}
    diagnostics = {}
    for name, (rotate, history) in ARMS.items():
        focal[name], diagnostics[name] = train(357132, rotate, history, TRAIN)
        print(f"Fit {name} at {time.monotonic() - start:.1f}s", flush=True)
    opponents = (
        train(717551, False, True, BANK[0])[0],
        train(188993, True, True, BANK[1])[0],
    )
    panels = {}
    for p, (name, conditions) in enumerate(zip(PANEL_NAMES, TEST)):
        routes = (sample_test(840329 + p * 113, conditions), sample_test(611633 + p * 211, conditions))
        outcomes = {
            name: np.mean([crossplay(name, focal.get(name), opponents[p], conditions, routes[role], role)
                           for role in range(2)], axis=0)
            for name in METHODS
        }
        panels[name] = report_panel(outcomes, 23117 + p * 123)
    own_pop_test = sample_test(458067, TRAIN)
    # Identity-matched pairs quantify whether fixed learners coordinate with their training partner.
    ids = np.repeat(np.arange(WORLDS * PAIRS), REPEATS)
    own_pop_test = (*own_pop_test[:3], ids, own_pop_test[4])
    same_pop = {
        name: dict(zip(METRICS, crossplay(name, focal[name], focal[name], TRAIN, own_pop_test, 0,
                                        partner_history=history).mean(axis=0).tolist()))
        for name, (_, history) in ARMS.items()
    }
    output = {
        "per_policy_transitions": STEPS,
        "joint_transitions_per_arm": STEPS * WORLDS * PAIRS,
        "independent_blocks": WORLDS,
        "policies_trained_per_role_per_block": PAIRS,
        "routes_per_role_per_block": PAIRS * REPEATS,
        "routes_per_panel_per_method": 2 * WORLDS * PAIRS * REPEATS,
        "train_diagnostics": diagnostics,
        "same_population": same_pop,
        "panels": panels,
        "primary_pass_both_panels": all(panel["primary_pass"] for panel in panels.values()),
    }
    args.output.write_text(json.dumps(output, indent=2, allow_nan=False) + "\n")
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (2**20 if sys.platform == "darwin" else 1024)
    print(f"Elapsed {time.monotonic() - start:.1f}s, peak RSS {peak:.1f} MiB")


if __name__ == "__main__":
    main()
