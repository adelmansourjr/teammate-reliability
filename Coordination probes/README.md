# Learning what matters about an unfamiliar teammate

**Research problem.** Can an agent use one cheap interaction to learn the behaviour that matters most for a shared task, then adapt its willingness to act to an unfamiliar teammate's reliability?

The first pilot uses a finite shared-load task with local cues, coupled success and one observation at a fixed price. It evaluates on two disjoint samples of 128 partners and 32 routes per partner. The focal policy estimates its partner's convention and uses the same fitted population prior and actions as the control policies.

| Evaluation | Value-based probe vs entropy probe | Value-based probe vs route frequency | 95% paired intervals |
|---|---:|---:|---|
| Familiar distribution, new partners | +0.155 | +0.003 | [0.102, 0.209] and [−0.019, 0.025] |
| Shifted conventions and reliability | +0.116 | −0.016 | [0.061, 0.174] and [−0.044, 0.011] |

The sophisticated probe did not improve on asking about the most frequent upcoming contact configuration. Under the shifted panel, every deployable policy averaged negative net reward, while the informed oracle earned +0.994. The oracle knows both the convention and the lapse rate, so its gap alone cannot identify which source of information matters.

The next problem is to **learn teammate reliability separately from teammate intent**. Add two oracle ablations that reveal only one latent factor, then fit a posterior or a recurrent estimator over both using local feedback. If probe value still tracks route frequency, keep the simple probe and allocate the research effort to safer action thresholds. Test against recurrent population play, frequency probing, and diverse independently trained partners using equal communication and interaction budgets. Hold out entire partner-training families, layouts and friction regimes. Evaluate transport completion, mismatched force, unnecessary abstention and adaptation time.

This is a decision-theoretic simulator. The project becomes embodied MARL when both teammate behaviour and the focal policy are learned in a simulator with physical interaction. This first experiment gives the next policy a baseline to beat. [Other-Play](https://proceedings.mlr.press/v119/hu20a.html) · [Behavior Diversity Play for embodied rearrangement](https://proceedings.mlr.press/v202/szot23a.html)

## Reproduce

```sh
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python run.py --output replay.json
cmp results.json replay.json
```

The saved replay reproduced byte for byte. Peak process RSS was about 59 MiB on the replay.

[Protocol](Protocol.md) · [All outcomes](results.json)
