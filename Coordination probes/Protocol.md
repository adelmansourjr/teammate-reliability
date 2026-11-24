# What should one agent learn about a new teammate before moving a shared load?

When agents can obtain one extra observation of an unfamiliar teammate before a joint task, should it concern the teammate's most uncertain behaviour, its most frequently needed behaviour, or the behaviour that carries the highest expected decision value?

## Controlled model

Two agents must supply matching lateral forces at five contacts with a shared load. Each contact has one of three local configurations and a visible mismatch penalty. At a contact the focal agent sees a noisy local cue about the teammate's preferred direction. It can push one of two ways or abstain. Matched forces earn +1, mismatches lose the visible penalty, and abstention earns zero. After a push, the shared success or mismatch reveals the realised direction. The teammate has one fixed convention for each of the three configurations and has independent lapses at each contact. The focal agent sees observations and its own payoff. It receives no partner identifier, latent convention or exact lapse rate.

All active policies may buy exactly one probe before the route at a fixed cost of 0.08 payoff units. Each chooses a contact configuration and observes an equally noisy indication of the partner's direction convention there. The upcoming contact sequence and costs are known. Use matched partner, route, cue, probe and motor randomness across policies.

## Learning and held-out evaluation

Fit the prior over the eight possible three-bit conventions from 4,096 independently sampled training teammate conventions. Use identical fitted priors, noise assumptions, posterior updates and action rules across all non-oracle policies. Test with two sets of 128 independently drawn held-out partners, 32 routes per partner. The first has conventions from the training distribution with individually unseen motor lapse rates. The second has unseen joint convention frequencies and higher motor lapse rates. The same eight convention vectors are possible in both. These tests do not establish systematic transfer to novel action structures.

Compare no probe, random probe, largest expected entropy reduction, most frequent route configuration, largest total route mismatch penalty, and highest expected route reward. The last policy uses a one-step reward calculation at each future contact and sums those terms before incorporating future outcome feedback. It need not be optimal for the complete sequential problem. An oracle has access to the actual latent convention and motor lapse rate. Every other agent observes the same sensory channels at the same decision times.

## Decision criterion

Use mean net reward per route including probe cost. For the decision-value probe to pass, require at least +0.10 reward versus each active baseline and a positive lower endpoint of the paired 95% interval against each active baseline on both held-out panels. Compare against no probe to test absolute value. Report scheduled and actually useful probes, mismatches, abstentions, and the oracle. Resample partners, keeping all routes for a partner together, for 5,000 paired bootstrap draws. No policy or parameter selection uses either test panel.

The finite model isolates the value of information under a coupled reward. It has a fitted prior and a Bayesian action rule. The embodied MARL experiment would replace this abstraction with two independently trained policies in a simulator, randomise physical coupling and friction, and reserve entire teammate-training families for cross-play. Other-Play motivates the partner split. A learned opponent model and recurrent policy would then be compared under a fixed training-interaction and communication budget.

[Chu's project advert](https://www.findaphd.com/phds/project/multi-agent-reinforcement-learning-for-embodied-cooperative-agents/?p197574) · [Other-Play](https://proceedings.mlr.press/v119/hu20a.html) · [Ad hoc teamwork](https://ojs.aaai.org/index.php/AAAI/article/view/7529)
