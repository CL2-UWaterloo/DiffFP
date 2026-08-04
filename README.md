# DiffFP: Learning Behaviors from Scratch via Diffusion-based Fictitious Play

This repository contains the MPE code and supplementary videos for the paper:

> @misc{karthikeyan2025difffplearningbehaviorsscratch,
      title={DiffFP: Learning Behaviors from Scratch via Diffusion-based Fictitious Play}, 
      author={Akash Karthikeyan and Yash Vardhan Pant},
      year={2025},
      eprint={2511.13186},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2511.13186}, 
}
## Code (MPE Experiments)

Fictitious play with diffusion-policy best responses on the MPE games.

### Setup

```bash
pip install -r requirements.txt
# Custom PettingZoo fork (required — modified simple_adversary / simple_tag)
git clone https://github.com/Aku02/PettingZoo.git
pip install -e ./PettingZoo
```

### Run

```bash
python mpe_multi_adv.py    # MPE-Adversary: 2 ego agents + 1 adversary, 3 landmarks
python mpe_multi_tag.py    # MPE-Tag: 1 ego agent vs 3 adversaries
```

Both trainers accept `--algo {dipo, sac, td3, qsm}` to select the
best-response learner inside the fictitious-play loop (default: `dipo`,
the diffusion policy):

```bash
python mpe_multi_adv.py --algo dipo   # DiffFP (diffusion best response)
python mpe_multi_adv.py --algo sac    # SACFP baseline
python mpe_multi_adv.py --algo td3    # TD3FP baseline
python mpe_multi_adv.py --algo qsm    # QSMFP baseline
```

`agent/` contains the diffusion best-response agent (`DiPo.py`, `diffusion.py`)
with the double-Q critic and Q-guided action refinement, the SAC / TD3 / QSM
baseline agents, and the shared networks in `networks.py`.

## 🎥 Supplementary Videos

### MPE - Adversary

<p align="center">
  <img src="assets/epi1.gif" width="30%" />
  <img src="assets/epi2.gif" width="30%" />
  <img src="assets/epi3.gif" width="30%" />
</p>

<p align="center"><em>MPE - Adversary: Model Stochasticity. We fix the seed and run the evaluation three times to demonstrate the inherent stochasticity in the model. We observe that using DiffFP leads to learning more diverse strategies</em></p>

---

### MPE - Tag

<p align="center">
  <img src="assets/episode_105.gif" width="30%" />
  <img src="assets/episode_106.gif" width="30%" />
  <img src="assets/episode_108.gif" width="30%" />
</p>

<p align="center"><em>MPE - Tag: Qualitative Results (Predator Prey). We observe competitive gameplay, even only when using sparse reward setup.</em></p>

---
### RaceTrack (Robustness to Unseen Opponents)

<p align="center">
  <img src="assets/blockpass_1.gif" width="30%" />
  <img src="assets/blockpass_2.gif" width="30%" />
  <img src="assets/blockpass_3.gif" width="30%" />
</p>

<p align="center"><em>Racetrack: Trained agents are shown in yellow, while unseen agents are in blue. We deploy the agents in a more complex setting where they must perform multiple overtakes. Overall, we observe that the agents learn to navigate corners effectively before executing overtakes. In particular, some agents exhibit a block pass behavior—deliberately taking an inside line at a corner to prevent the opponent from passing</em></p>

---
### RaceTrack (Robustness to Unseen Opponents - Failure Modes)

<p align="center">
  <img src="assets/QSM_fail.gif" width="45%" />
  <img src="assets/DiffFP_violation.gif" width="45%" />
</p>

<p align="center">
<em>
Racetrack: Left (QSMFP) fails to perform a lane change and instead rear-ends the opponent. The agents make decisions based solely on local observations and do not have access to the full state of all agents. Right: DiffFP infers the presence of agents ahead and chooses to violate track boundaries in order to overtake them.
</em>
</p>


---

### RaceTrack (Overtake - 1)

![Overtake 1](assets/overtake_1.gif)

*The attacking agent car overtakes the defending agent at a turn and then performs a lane change to block any attempt at re-overtaking.*

---

### RaceTrack (Overtake - 2)

![Overtake 2](assets/overtake_2.gif)

*Another instance where the attacking agent executes a strategic overtake at a curve and immediately transitions to a blocking maneuver.*

---

### RaceTrack (Overtake - 2)

<p align="left">
  <img src="assets/1v1_overtake.gif" width="45%" />
</p>

*Another instance where the attacking agent executes a strategic overtake at a curve and immediately transitions to a blocking maneuver.*

---

### RaceTrack (Block - 1)

![Blocking](assets/blocking.gif)

*The defending agent performs defensive blocking to prevent the attacjing agent from overtaking, maintaining lane control throughout.*

---

### RaceTrack (Defensive Driving - 1)

![Defensive Driving](assets/defencive_driving.gif)

*The attacking agent maintains a safe distance and matches the defending car's speed, occasionally performing a shoulder check without attempting to overtake.*

---

### RaceTrack (Overtake Fail - 1)

![Overtake Fail](assets/fail_rear_end.gif)

*The attacking agent fails to complete an overtake, braking at the last moment to avoid a rear-end collision.*

---

### RaceTrack (Brake Check Follower - 1)

![Brake Check](assets/brakecheck_follower.gif)

*The defending agent executes a brake check, forcing the attacking agent to react defensively to avoid a collision.*



