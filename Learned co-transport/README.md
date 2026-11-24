# Learning to carry a payload with unfamiliar teammates

**Does rotating teammates during training improve transfer?** Two simulated robots learn from the same route reward but act and update independently. They each choose left thrust, coast or right thrust. Balanced pushes can advance a payload. Opposing forces accumulate twist and can jam it. Hidden bend directions, separate noisy guide cues, nonzero motor slip and a deadline create imperfect information, coordination and recovery requirements. This discrete planar model tests joint learning and transfer. Fine contact dynamics and physical robots remain open.

We trained twelve independently randomised populations per arm, with eight policies per role. Every learned policy received exactly **14,400 transition updates**. The primary comparison holds the observation state and learning algorithm fixed and changes which learning teammate is paired every 24 steps. A separately trained frozen teammate bank is used for evaluation, and each arm is matched on route, role, cues, slip draws and teammate ID. Cluster intervals resample the independent training blocks.

## Main result

| Fresh evaluation | Fixed teammates | Rotating teammates | Paired completion change |
|---|---:|---:|---:|
| Alternating bends and held-out bank 1 | 52.6% | 59.0% | +6.4 percentage points [3.0, 9.8] |
| Clustered bends and held-out bank 2 | 44.7% | 50.7% | +6.1 percentage points [0.2, 11.9] |
| Independent retraining with both banks crossed with both layouts | 47.6% | 53.7% | +6.1 percentage points [3.4, 9.0] |

The initial two-panel study cleared its frozen positive-interval and +3-point criterion in both panels. The lower bound in the clustered panel was close to zero. The crossed replication used **12,288 fresh routes per method** from new seeds, including new training for all focal and teammate policies. It cleared the separate frozen pooled criterion. All four crossed cell estimates were positive, ranging from **+2.2 to +9.6 points**. The weakest cell’s interval crossed zero. The numbers are percentage-point differences with 95% paired block-bootstrap intervals. [Initial protocol](Protocol.md) · [Initial outcomes](results.json) · [Crossed protocol](Crossed%20replication/Protocol.md) · [Crossed outcomes](Crossed%20replication/results.json)

## What held up against stronger controls

The result concerns rotating partner exposure with longer history held fixed. In the crossed replication, rotation with short history also improved over its fixed-pair control by **+2.1 points [0.6, 3.8]**. Giving rotating learners the longer history yielded an uncertain completion difference, **−0.7 points [−2.9, 1.4]**, and increased jam probability, **+2.3 points [1.3, 3.3]**. This history contrast changes both the information and table size. A Bayesian guide that integrated repeated cues and coasted at high twist was weaker against the learned partners. It estimated the bend rather than learning how the other policy coordinated. The learned population baseline is more informative for deciding which representation to use next.

On fresh training-like routes, completion with matching policies from the same population ranged from 85.4% to 91.1%, compared with substantially lower cross-play scores. The combined evidence supports partner diversity as a training intervention in this testbed and shows that simply adding local history leaves difficult coordination and jam control unresolved. A useful next comparison is to hold the route fixed while combining independent partner learning algorithms and testing a small, structured reliability state. Contact robotics validation would need a separate simulator and physical trial. [Chu’s advert](https://www.findaphd.com/phds/project/multi-agent-reinforcement-learning-for-embodied-cooperative-agents/?p197574) · [Adaptation to unfamiliar partners in embodied rearrangement](https://proceedings.mlr.press/v202/szot23a.html)

## Reproduce

Python 3.11 or newer and NumPy. The experiment uses neither downloaded data nor model weights. From this directory:

```sh
shasum -a 256 -c frozen-source.sha256
shasum -a 256 -c crossed-freeze.sha256
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
python run.py --output /tmp/co-transport.json
cmp results.json /tmp/co-transport.json
python 'Crossed replication/run.py' --output /tmp/co-transport-crossed.json
cmp 'Crossed replication/results.json' /tmp/co-transport-crossed.json
```

Peak process RSS stayed under 90 MiB on the measured Apple Silicon runs. The ledgers include block means and all prespecified contrasts.
