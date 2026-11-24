# Adapting to unfamiliar teammates

A robot can learn which way a teammate intends to push and still have the wrong expectations about whether that teammate will push as intended. The two CPU studies below test online inference about those two quantities. They use a finite decentralized coordination simulator with a shared reward. The predictors learn calibration priors from offline examples and update their beliefs on the test routes. The partners are simulated policies. This is a mechanism study for an embodied MARL benchmark, with robot dynamics and trained partner policies left to the next stage. [Chu’s cooperative-agents advert](https://www.findaphd.com/phds/project/multi-agent-reinforcement-learning-for-embodied-cooperative-agents/?p197574) · [Chu and colleagues’ distributed collaboration study](https://www.repository.cam.ac.uk/items/985b7368-c075-4ac2-8e24-73cc94e31189)

## Main finding

**Reversing dependence between teammate lapse rates reversed which coordination prior performed best, even with the same lapse-rate marginals.** A separate paired confirmation used 10,240 simulated shared-load routes across two panels, with 256 paired partner slots and 20 routes per slot in each panel. The intervals resample slots conditional on the fixed lapse-rate banks. It held each rate’s marginal distribution, preference vectors, routes, motor random numbers and probe budget fixed. Only the rank alignment between the two support-specific lapse rates changed.

| Fresh panel | Joint prior | Factorized prior | Fixed mixture | Joint minus factorized, 95% interval |
|---|---:|---:|---:|---:|
| Aligned lapse rates | 3.803 | 3.671 | 3.779 | +0.132 [0.076, 0.186] |
| Reversed lapse rates | 3.372 | 3.632 | 3.538 | −0.260 [−0.322, −0.201] |

Values are net reward per 24-contact route, including the same cost for a single frequency-based probe. The frozen primary interaction was **+0.392 [0.328, 0.457]**, exceeding the +0.20 practical criterion. The fixed mixture had the smallest observed worst-panel gap relative to the stronger prior in each panel, at **0.094** versus **0.132** for the factorized prior and **0.260** for the joint prior. This is a secondary result for two deliberately strong dependence shifts. The log-likelihood bound in the protocol is conditional on a common action and observation history. The empirical reward comparison allows each agent to choose its own actions and thus elicit different feedback. [Confirmation protocol](Prior%20shift/Protocol.md) · [Confirmation ledger](Prior%20shift/results.json)

## What led to the confirmation

In three independently seeded panels of 5,120 routes each, all methods got the same route and one noisy probe. On the two stationary shifted panels the joint support-specific learner improved over the adaptive one-rate learner by **+0.474 [0.378, 0.571]** and **+0.397 [0.285, 0.509]** reward per route. It missed the frozen full success rule. Its gains over learning independently in each contact mode were below the +0.20 practical threshold, and its gain over fixed support in the second panel was +0.167, also below threshold. The factorized model beat the joint model by **+0.306 [0.248, 0.365]** under reversed support rates. This first experiment shifted multiple characteristics. The confirmation above kept the lapse marginals and cues fixed to isolate dependence.

| First study panel | Fixed support | Personal pooled rate | Mode-local support | Joint support | Factorized support | Dual oracle |
|---|---:|---:|---:|---:|---:|---:|
| Heterogeneous stationary rates | 6.353 | 6.628 | 6.921 | 7.102 | 7.034 | 11.027 |
| Reversed stationary support | 2.812 | 2.582 | 2.876 | 2.978 | 3.285 | 6.816 |
| Within-route change | 2.097 | 2.045 | 2.094 | 2.096 | 2.161 | 6.191 |

Within-route change defeated the stationary learner. The fixed pooled control scored 2.213 there and beat the joint learner by +0.117 [0.021, 0.210]. Giving the learner the true current execution rates alone raised its score to 4.358, compared with 3.134 if given only the preferred directions. Those are greedy privileged-information controls and do not measure the gain of an implementable change detector. This points to a precise next comparison: can a robust prior and an online change detector adapt beyond a fixed-prior controller with the same probe, training and feedback budgets? The next embodied version should hold out entire teammate-training families and include force mismatch, time, abstention and payload completion. [First study protocol](Protocol.md) · [First study ledger](results.json)

## Reproduce

Python 3.11 or newer and NumPy. No downloaded data or model weights. From this directory:

```sh
shasum -a 256 -c frozen-source.sha256
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python run.py --output /tmp/coordination-replay.json
cmp results.json /tmp/coordination-replay.json
python 'Prior shift/run.py' --output /tmp/prior-shift-replay.json
cmp 'Prior shift/results.json' /tmp/prior-shift-replay.json
```

On Apple Silicon, peak process RSS stayed below 190 MiB across the first-study run and replay. Both saved result ledgers preserve all paired partner means.
