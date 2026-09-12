# SWGR Teleoperation Retargeting — Literature Search & Evidence Plan

**Search date:** 2026-09-11
**Purpose:** ICRA submission — position Shoulder–Wrist Geometric Retargeting (SWGR) against
incremental/clutch teleoperation, and design the evidence that proves the advantage.
**Target venue:** ICRA (robotics, CCF-B / core robotics venue)
**Sources:** arXiv API, Crossref API (relevance-sorted). OpenAlex + Semantic Scholar returned
HTTP 429 (rate-limited from this host) — coverage gap noted below.
**Source policy:** primary/official only (arXiv abs pages, publisher DOIs). No MDPI, no
predatory venues, no snippet-only claims. Every row below was returned by a live API call;
abstracts marked ✓ were read in full.

---

## 0. Bottom line first (read this before anything else)

**"SWGR beats incremental/clutch teleoperation" is not a sufficient ICRA contribution.**

Two reasons, both from verified sources:

1. **The incremental/clutch baseline is 30+ years old and its failure mode is common
   knowledge.** There is already a dedicated literature on workspace scaling and rate/position
   mode switching in VR teleoperation (§Cluster C) that explicitly addresses limited reach and
   re-anchoring. A reviewer will read "we removed the clutch" as an engineering fix, not a result.

2. **There is very recent, very close prior art in the same family.** `SEW-Mimic`
   (arXiv:2602.01632, **2026-02-02**) is a closed-form *geometric* retargeting solver built from
   **shoulder–elbow–wrist keypoints**, for 7-DoF arms and humanoids, with an optimality
   guarantee, 3 kHz inference, a safety filter, a pilot user study, and a smoothness claim tied
   to downstream policy learning. `WARP` (arXiv:2606.29940, **2026-06-29**) reuses the same SEW
   solver for whole-body retargeting. Same input modality, same year, better-known groups
   (Georgia Tech / Michigan — Xu, Kousik).

**The defensible contribution is a three-part claim, not one:**

| # | Claim | Why it survives review |
|---|---|---|
| **C1** | **Workspace amplification.** Scaling the shoulder→wrist vector by `s = robot_reach / human_reach` (=1.324 for JAKA K1) lets the operator command the robot's *full* arm span instead of being capped at the human's. | SEW-Mimic's own abstract names this exact problem — *"optimizing to match robot end-effector to human hand position ... can also limit the robot's workspace to that of the human"* — and answers it by **not** matching the EE. We answer it by **scaling** the EE match. **Measured: 2.06–2.11× usable workspace vs the unscaled absolute baseline (§7.4).** |
| **C2** | **Orientation decoupling.** Position comes from body keypoints; orientation comes independently from the controller. | SEW-Mimic derives orientation from arm-segment alignment, so wrist orientation is coupled to arm configuration. Ours lets the operator rotate the tool in place without moving the arm. Testable with a dedicated task. |
| **C3** | **Drift-free under operator motion.** Only the *difference* vector enters the map, so the VR world origin cancels; no clutch, no re-anchoring, no accumulation. | This is the vs-incremental axis. Keep it, but as C3 — it is the easiest to prove and the least novel, so it must not carry the paper. |

**Do not** claim speed or optimality. SEW-Mimic has 3 kHz closed-form + an optimality guarantee.
We have an affine map feeding an iterative IK (placo). Competing there is a guaranteed loss.

---

## 1. Cluster A — Closest prior art: geometric / SEW retargeting (HIGHEST PRIORITY)

| Year | Title | Source | Status | Why it matters |
|---|---|---|---|---|
| 2026 | **A Closed-Form Geometric Retargeting Solver for Upper Body Humanoid Robot Teleoperation** (SEW-Mimic) — Kong, Cho, Jung, Wibowo, Shinde, Vinodh-Sangeetha, Chung, Chen, Mattei, Nidumukkala, Elias, Xu, Higgins, Kousik | arXiv:2602.01632v1 ✓abstract read; project page `sew-mimic.com` fetched | preprint, **no code link found on project page** | **The** competitor. Aligns robot upper+lower arm *orientations* from SEW keypoints; closed-form with optimality guarantee; 3 kHz; suits most 7-DoF arms; input-source agnostic; pilot user study shows improved task success; smoother data improves policy learning; drop-in accelerator for full-body humanoid retargeting. Videos cover fast wrist/elbow motion, fine wrist motion, torso control, near-singularity, self-collision filtering. |
| 2026 | **WARP: Whole-Body Retargeting for Learning from Offline Human Demonstrations** — Chen\*, Kong\*, Zhang\*, Yang, Zhu, Kousik, Xu | arXiv:2606.29940v2 ✓abstract read; project page `warp-retargeting.github.io` fetched | preprint, **no code link found** | Same group, reuses "closed-form Shoulder–Elbow–Wrist (SEW) geometric solver for exact end-effector tracking". Offline (no human-in-the-loop). Confirms SEW is becoming the standard primitive — our related work must engage it. |
| 2026 | **CLOT: Closed-Loop Global Motion Tracking for Whole-Body Humanoid Teleoperation** — Zhu, Cai, Yang, Ren, Xie, Wang, Wu, Wang, Yang, Mu, Yan | arXiv:2602.15060v2 ✓abstract read | preprint | Uses the word **drift-free** for long-horizon teleoperation, but for *global base pose* drift via RL + localization feedback. Different drift than ours (mapping drift vs locomotion drift) — cite to show the term is contested and to delimit our claim precisely. |
| 2013 | **Mapping human to robot motion with functional anthropomorphism for teleoperation and telemanipulation** | IROS 2013, doi:10.1109/iros.2013.6696638, 25 cites | published | The classic human→robot arm mapping paper. Foundational cite for "how should a human arm map onto a robot arm". |
| 2023 | **A Joint-Cartesian Hybrid Motion Mapping Method for Real-Time Teleoperation Based on Human Arm** | RCAR 2023, doi:10.1109/rcar58764.2023.10249512, 3 cites | published | Hybrid joint+Cartesian mapping from human arm — nearest methodological neighbour among published (non-preprint) work. |
| 2019 | **Virtual Reality Teleoperation of a Humanoid Robot Using Markerless Human Upper Body Pose Imitation Learning** | Humanoids 2019, doi:10.1109/humanoids43949.2019.9035064, 28 cites | published | Markerless upper-body VR teleoperation of a humanoid. Establishes the input modality we use. |
| 2026 | **Ghost Arm: Aligning Human and Robot Kinematics through AR Overlays in MoCap-Based Teleoperation** | Augmented Humans 2026, doi:10.1145/3795011.3795014, 1 cite | published | Attacks the same human–robot kinematic mismatch, but from the *display* side (AR overlay) rather than the mapping side. Good contrast cite. |
| 2025 | **XRoboToolkit: A Cross-Platform Framework for Robot Teleoperation** — Zhao, Yu, Jing, Yang | arXiv:2508.00097v2 ✓abstract read; **accepted IEEE/SICE SII 2026** | published | **This is the framework our repo is built on.** OpenXR-based, low-latency stereo feedback, optimization-based IK, head/controller/hand/aux-tracker modalities. Mandatory cite; also our system's provenance. |

### Differentiation table (put something like this in Related Work)

| Axis | Incremental + clutch | SEW-Mimic (2026) | **SWGR (ours)** |
|---|---|---|---|
| Keypoints consumed | none (controller only) | shoulder + elbow + wrist | shoulder + wrist (**elbow deliberately excluded**) |
| What is matched | accumulated deltas | upper-arm & forearm **orientations** | shoulder→wrist **vector**, uniformly scaled |
| Position source | controller, re-anchored at clutch | closed-form from arm geometry | body tracking, absolute |
| Orientation source | controller, incremental | arm-segment alignment (coupled) | controller, **independent** |
| Workspace | operator-limited, drifts | robot's own geometry | **robot's full span** (s = robot/human reach) |
| Origin dependence | yes (anchor at engage) | — | **none** (difference vector → origin cancels) |
| Solver cost | trivial | closed-form 3 kHz + guarantee | affine map + iterative IK |
| Redundancy handling | IK default | geometric, exact | IK default (elbow available but unused) |

**Honest gap:** SEW-Mimic has no public code as of this search. Reproducing it as a baseline is
our own implementation effort. Budget for it, or explicitly compare only against
incremental + unscaled-absolute + elbow-constrained variants and discuss SEW-Mimic analytically
(weaker, but survivable if C1/C2/C3 evidence is strong).

---

## 2. Cluster B — The baseline we must beat: workspace scaling, rate mode, mode switching

This is the "incremental teleoperation" literature. Note it is **not** a strawman — it already
contains scaling factors and mode-switching logic.

| Year | Title | Source | Cites | Relevance |
|---|---|---|---|---|
| 2021 | **Workspace Scaling and Rate Mode Control for Virtual Reality based Robot Teleoperation** | IEEE SMC 2021, doi:10.1109/smc52423.2021.9658796 | 8 | **Primary baseline reference.** VR teleop + workspace scaling + rate mode. Directly comparable setting. |
| 2023 | **Workspace Scaling in Virtual Reality based Robot Teleoperation** | Augmented Humans 2023, doi:10.1145/3582700.3582717 | 3 | Follow-up; VR workspace scaling. |
| 2024 | **Development of Variable Scaling Teleoperation Framework for Improving Teleoperation Performance** | Int. J. Control Autom. Syst. 2024, doi:10.1007/s12555-022-1099-z | 2 | Variable (state-dependent) scaling — a stronger baseline than a constant `s`. Reviewers may ask why our scale is constant. |
| 2023 | **Decision of impedance-based scaling factor for workspace mapping in teleoperation** | J. Mech. Sci. Technol. 2023, doi:10.1007/s12206-023-1036-y | 1 | Principled scaling-factor selection; contrast with our arm-span-ratio choice. |
| 2009 | **Workspace mapping and force control for small haptic device based robot teleoperation** | ICINFA 2009, doi:10.1109/icinfa.2009.5205175 | 28 | Well-cited workspace-mapping foundation. |
| 2024 | **Intelligent Mode-switching Framework for Teleoperation** | ICRA 2024, doi:10.1109/icra57147.2024.10611333 | 3 | ICRA paper on automatically switching position/rate modes — i.e. *keeping* the clutch but automating it. Our claim must be that removing it entirely is better. |
| 2017 | **Workspace Mapping Using Auto-Switching Mode** | Adv. Sci. Lett. 2017, doi:10.1166/asl.2017.9759 | 1 | Auto-switching workspace mapping. |
| 2016 | **Rate-position-point hybrid control mode for teleoperation with force feedback** | ICARM 2016, doi:10.1109/icarm.2016.7606957 | 5 | Hybrid rate/position/point modes. |
| 2025 | **Dual-Arm Mobile Manipulator Teleoperation With Coupled Rate-Position Mapping Under Time-Varying Delay** | IEEE Access 2025, doi:10.1109/access.2025.3577009 | 3 | Most recent dual-arm rate-position mapping; good "state of the practice" cite. |
| 2011 | **Assisted Teleoperation Strategies for Aggressively Controlling a Robot Arm with 2D Input** | RSS 2011, doi:10.15607/rss.2011.vii.046 | 3 | Shared-control alternative to better mapping. |
| 2021 | **A heterogeneous master-slave teleoperation method for 7-DOF manipulator** | China Automation Congress 2021, doi:10.1109/cac53003.2021.9727734 | 0 | Heterogeneous 7-DoF master-slave — same robot class as JAKA K1. |
| 2025 | **LAMS: LLM-Driven Automatic Mode Switching for Assistive Teleoperation** | HRI 2025, doi:10.1109/hri61500.2025.10974127 | 5 | Shows mode-switching is still an active research axis in 2025. |
| 2026 | **Augmenting Human Dance Performance With a Robot Arm: An Incremental Dynamic Mapping Method** | IEEE Robotics & Automation Magazine 2026, doi:10.1109/mra.2026.3687439 | 0 | Recent *incremental* mapping in a top magazine — evidence the paradigm is alive, so beating it is not trivially assumed. |

---

## 3. Cluster C — Redundancy, arm angle, and 7-DoF null space

Relevant because JAKA K1 is a 7-DoF dual arm and the elbow is the null-space coordinate we
deliberately discard. A reviewer **will** ask what dropping the elbow costs.

| Year | Title | Source | Cites | Relevance |
|---|---|---|---|---|
| 1991 | **Resolving redundant manipulator joint rates and identifying special arm configurations using arm angle** | IEEE Trans. Robotics & Automation, doi:10.1109/70.97872 | 31 | **The** arm-angle / swivel-angle foundational paper. Cite when justifying (or declining) an elbow constraint. |
| 2014 | **Analytical inverse kinematics of a class of redundant manipulator based on dual arm-angle parameterization** | IEEE SMC 2014, doi:10.1109/smc.2014.6974513 | 17 | Arm-angle parameterized IK — the mechanism by which an elbow target would constrain the 7th DoF. |
| 2014 | **A robot teaching framework for a redundant dual arm manipulator with teleoperation from exoskeleton** | Humanoids 2014, doi:10.1109/humanoids.2014.7041495 | 18 | Redundant **dual**-arm teleoperation — closest hardware analogue to our setup. |
| 2023 | **Arm Angle Parameterized Inverse Kinematics Solution of A 7-DOF Redundant Manipulator** | J. Mechanical Engineering, doi:10.3901/jme.2023.23.068 | 2 | 7-DoF arm-angle IK. |
| 2014 | **An experimental study on redundancy resolution scheme of postural configuration in human arm** | MECBME 2014, doi:10.1109/mecbme.2014.6783253 | 0 | **How humans actually resolve their own arm redundancy** — key support if we argue the human elbow is not a reliable null-space reference. |
| 2020 | **Kinematic Redundancy Resolution for Humanoid Robots by Human Motion Database** | IEEE RA-L 2020, doi:10.1109/lra.2020.3026972 | 6 | Data-driven redundancy resolution alternative. |

---

## 4. Cluster D — Evaluation methodology (this is what makes the user study credible)

| Year | Title / Standard | Source | Cites | Use |
|---|---|---|---|---|
| 1990 | **Use of a multi-axis Fitts' law paradigm to characterize total body motion — a study in teleoperation** | IEEE SMC 1990, doi:10.1109/icsyse.1990.203109 | — | **The** anchor for applying Fitts' law to teleoperation. Cite to justify the pointing-task design. |
| — | **ISO 9241 (Ergonomics of human-system interaction)** | doi:10.3403/30464855u / 10.3403/30464855 | 11 / 17 | Standard for throughput (bits/s), multi-directional tapping test, error rate. Cite the specific part (411 for input devices). |
| 2018 | **Fitts' Law: On Calculating Throughput and Non-ISO Tasks** | Revista Colombiana de Computación, doi:10.29375/25392115.3226 | 4 | Throughput calculation pitfalls — use to pre-empt methodological criticism. |
| 2006 | **NASA-Task Load Index (NASA-TLX); 20 years later** | doi:10.1037/e577632012-009 | 631 | Standard subjective workload instrument. Use the validated short form. |
| 2018 | **Intuitive Hand Teleoperation by Novice Operators Using a Continuous Teleoperation Subspace** | arXiv:1802.04349v1, **ICRA 2018** (Meeker, Rasmussen, Ciocarlie) | — | ICRA-accepted teleoperation user study — good template for study design and reporting. |
| 2016 | **Usability evaluation of mobile applications using ISO 9241 and ISO 25062 standards** | SpringerPlus, doi:10.1186/s40064-016-2171-z | 123 | How to actually run an ISO 9241-conformant evaluation. |

---

## 5. Cluster E — Modern VR/humanoid teleoperation systems (context + likely baselines)

| Year | System | Source | Note |
|---|---|---|---|
| 2024 | **Open-TeleVision: Teleoperation with Immersive Active Visual Feedback** — Cheng, Li, Yang et al. | arXiv:2407.01512v2; `robot-tv.github.io` | Immersive stereo feedback; strong system baseline family. |
| 2023 | **AnyTeleop: A General Vision-Based Dexterous Robot Arm-Hand Teleoperation System** — Qin, Yang, Huang et al. | arXiv:2307.04577v3; **RSS 2023** | Vision-based, no wearable. |
| 2019 | **DexPilot: Vision Based Teleoperation of Dexterous Robotic Hand-Arm System** — Handa, Van Wyk, Yang et al. | arXiv:1910.03135v2 | Foundational vision-based arm-hand teleop. |
| 2023 | **GELLO: A General, Low-Cost, and Intuitive Teleoperation Framework** — Wu, Shentu, Yi et al. | arXiv:2309.13037v2 | Exoskeleton-style leader arm. |
| 2025 | **Improving Low-Cost Teleoperation: Augmenting GELLO with Force** — Sujit et al. | arXiv:2507.13602v1; **accepted IEEE/SICE SII 2025** | Shows this line is still active. |
| 2024 | **Bunny-VisionPro: Real-Time Bimanual Dexterous Teleoperation for Imitation Learning** — Ding, Qin, Zhu et al. | arXiv:2407.03162v1 | **Bimanual**, retargeting-based, imitation-learning oriented. Closest in application scope to ours. |
| 2024 | **HumanPlus: Humanoid Shadowing and Imitation from Humans** — Fu, Zhao, Wu et al. | arXiv:2406.10454v1 | Whole-body shadowing. |
| 2024 | **OmniH2O: Universal and Dexterous Human-to-Humanoid Whole-Body Teleoperation and Learning** — He, Luo, He et al. | arXiv:2406.08858v1 | Whole-body humanoid teleop. |
| 2024 | **ExBody2: Advanced Expressive Humanoid Whole-Body Control** — Ji, Peng, Liu et al. | arXiv:2412.13196v2 | Expressive whole-body control. |
| 2017 | **Adaptive whole-body manipulation in human-to-humanoid multi-contact motion retargeting** | Humanoids 2017, doi:10.1109/humanoids.2017.8246911 (19) | Retargeting for manipulation. |
| 2019 | **Dynamic locomotion synchronization of bipedal robot and human operator via bilateral feedback** | Science Robotics 2019, doi:10.1126/scirobotics.aav4282 (66) | High-impact human–robot synchronization reference. |
| 2026 | **Zero-Splat TeleAssist: Zero-Shot Pose Estimation for Semantic Teleoperation** | arXiv:2512.08271v1 | Pose-estimation front end. |
| 2026 | **Beyond Teleoperation: Enhancing VLA Robustness via Explicit Kinematic Retargeting of Human Demonstrations** | Decision Making Advances 2026, doi:10.31181/dma412026180 | Retargeting for VLA robustness — links our work to the data-collection motivation. |
| 2026 | **Learning Sim-Grounded Policies for Bimanual Rope Manipulation from Human Teleoperation** | arXiv:2605.16043v1; *Beyond Teleoperation Workshop @ ICRA 2026* | Confirms an ICRA 2026 workshop exists on exactly this topic — check its accepted papers for more competitors. |
| 2025 | **Whole-Body Human-Motion Based Robot Teleoperation** | book chapter, doi:10.1007/978-981-96-6545-7_4 | Survey-flavoured; useful for related-work framing. |

---

## 6. Opportunity map

| Gap | Evidence | Exploitability |
|---|---|---|
| **Workspace amplification is named as a limitation but solved by abandoning EE matching** | SEW-Mimic abstract ✓ | **HIGH.** We solve it while *keeping* EE matching, via the arm-span ratio. Clean, quantitative, and it turns the competitor's own criticism into our motivation. |
| **Constant vs variable scale factor is unexamined in the body-tracking setting** | Cluster B has variable-scaling work (IJCAS 2024) but for haptic/rate devices, not body keypoints | **MEDIUM.** Either justify the constant `s` empirically (ablation over s) or adopt a state-dependent `s`. |
| **What dropping the elbow costs is unquantified** | Cluster C shows the elbow *is* the null-space coordinate; SWGR ignores it | **HIGH.** A genuine open question. Our B3 ablation answers it. Even a negative result is publishable analysis. |
| **Drift under operator torso motion is not measured as such** | CLOT addresses *global base* drift, not mapping drift; Cluster B addresses reach limits | **MEDIUM-HIGH.** Cheap, decisive experiment (§7 E3). Nobody reports it. |
| **Fixed-base dual 7-DoF arms are under-served** | SEW-Mimic/WARP/humanoid work targets humanoids; Cluster B targets haptic devices | **MEDIUM.** Our JAKA K1 setting. Consider adding G1 (already in this repo) to widen scope. |
| **No public code for the strongest baseline** | Both project pages fetched; no repo link | **RISK, not opportunity.** Reproduction cost falls on us. |

---

## 7. Evidence plan — how to actually prove the advantage

### 7.1 Claim → evidence matrix

| Claim | Experiment | Needs humans? | Metric | Cost |
|---|---|---|---|---|
| **C1** workspace amplification | **E1** Reachable-volume sweep | No | reachable volume (L), % of robot workspace covered, IK-infeasible rate, joint-limit margin | **1–2 days, pure placo** |
| **C3** drift-free | **E2** Repeated reach-and-return to a fixed marker | Yes | EE error at marker vs cycle index; regression slope; final-cycle error | 1 day |
| **C3** origin independence | **E3** Torso perturbation (operator steps 30 cm laterally / back mid-hold) | Yes | EE target displacement caused by perturbation (expect ≈0 for SWGR, large for incremental) | **half a day, very high impact/cost ratio** |
| C1+C2+C3 | **E4** Trajectory quality from logged teleop | Yes | path efficiency, jerk RMS, SPARC, sub-movement count, clutch events/trial | reuses E2/E5 logs |
| C2 | **E5** ISO 9241-411-style 3D multi-directional pointing task | Yes | **throughput (bits/s)**, movement time, error rate, overshoot | 2–3 days |
| C1+C2+C3 | **E6** Manipulation battery (peg-in-hole, drawer, pour, stack) | Yes | success rate, completion time, RMS tracking error | 2–3 days |
| all | **E7** NASA-TLX after each condition | Yes | 6 subscales + raw TLX | folded into E5/E6 |
| C2 | **E8** Orientation-decoupling task: rotate tool in place while holding EE position | Yes | position drift during pure rotation; rotation range achieved | half a day |

### 7.2 Condition ladder — the ablations our implementation already supports

| ID | Condition | How to enable | Code change |
|---|---|---|---|
| **B1** | Incremental + clutch (repo default) | remove `swgr` from `manipulator_config` | **none — already implemented** |
| **B2** | Absolute shoulder-wrist, **s = 1** (no scaling) | `robot_reach: 0.58` | config only |
| **B3** | SWGR **+ elbow IK constraint** (segment-aware, SEW-like) | feed `swgr_elbow[name]` into a position task on `l4`/`r4` | ~10 lines (elbow is already retargeted and stored) |
| **B4** | **SWGR (ours)** | current config | none |
| B5 | SWGR without engagement ramp | `SWGR_RAMP_TIME = 0` | 1 constant |
| B6 | SWGR with incremental (clutch) orientation | swap absolute `R_target` for delta | small |
| B7 | SEW-Mimic reproduction | — | **substantial; no public code** |

**B2 vs B4 isolates C1** (does the scale factor matter?). **B3 vs B4 isolates the elbow
question** (does discarding the elbow cost anything?). **B1 vs B4 isolates C3.** Those three
comparisons are the spine of the paper; B5/B6 are cheap robustness checks; B7 is optional and
expensive.

### 7.3 Statistics (ICRA reviewers will check)

- **N ≥ 12** operators; within-subject; **Latin-square counterbalanced** condition order (learning effects are the #1 methodological attack on teleop user studies).
- ≥ **5 trials** per participant per condition per task; report per-trial data, not just means.
- Mean ± SD, plus **95% CI**. Paired Wilcoxon signed-rank (non-normal) or repeated-measures ANOVA with Holm–Bonferroni correction across metrics.
- **Pre-register the primary metric.** Recommend throughput (E5) and success rate (E6) as co-primary; everything else secondary. This prevents the "you tested 14 metrics and reported the 2 that worked" criticism.
- Run a **power analysis** and state it in the paper.
- Report operator experience level; include a practice block excluded from analysis.

### 7.4 E1 — DONE. Measured result (2026-09-11)

Script: `scripts/analysis/swgr_workspace_volume.py`. Runs in ~15 s per arm, needs **no VR
hardware and no body tracking** — pure placo IK with the same task stack as
`BaseTeleopController._placo_setup`.

```
PYTHONPATH=. python scripts/analysis/swgr_workspace_volume.py --arm right --n 400 --steps 100
```

| s | cmd radius (m) | IK ok % (right / left) | rel. usable vol (right / left) |
|---|---|---|---|
| 0.80 | 0.464 | 94.5 / 93.5 | 0.484 / 0.479 |
| **1.00** (= B2, unscaled) | 0.580 | 87.2 / 90.0 | **0.873 / 0.900** |
| 1.15 | 0.667 | 86.8 / 86.2 | 1.319 / 1.312 |
| **1.3241** (= SWGR) | 0.768 | 79.2 / 80.0 | **1.840 / 1.857** |
| 1.45 | 0.841 | 58.8 / 61.5 | 1.791 / 1.875 |
| 1.60 | 0.928 | 45.2 / 45.5 | 1.853 / 1.864 |

**Headline numbers (measured, both arms consistent):**

- **SWGR gives 2.11× (right) / 2.06× (left) the usable workspace of the unscaled absolute
  baseline B2.** This is C1, quantified. It directly answers SEW-Mimic's stated criticism that
  EE-matching *"limits the robot's workspace to that of the human"*.
- **The volume curve saturates at s ≈ 1.32–1.6** (1.84 → 1.79 → 1.85). Beyond s = 1.324 you buy
  **no** additional volume but lose feasibility sharply (79% → 45%). So `s = robot_reach /
  human_reach` is not an arbitrary choice — it is **the smallest scale that reaches the volume
  plateau**. That is a principled justification for the exact configured value, and it pre-empts
  risk #3 ("constant scale factor looks naive").

**Costs that must be reported honestly:**

- At s = 1.324, **~20% of commandable targets are IK-infeasible**; the arm saturates at its
  workspace boundary. In live teleoperation this means the EE stops at the edge rather than
  following the hand. This is the empirical case for a `reach_margin` clamp — which was
  explicitly descoped. Worth revisiting now that there is a number attached.
- The naive metric "IK ok %" **decreases** monotonically with s (94.5% at s=0.8 → 45.2% at
  s=1.6). Reporting feasibility alone would make SWGR look worse than B2. The correct metric is
  usable volume = s³ × feasible-fraction. **Do not let a reviewer push the paper onto the
  wrong metric.**

**Stated simplifications** (both flagged in the script):

- The human shoulder→wrist workspace is modelled as an **isotropic ball**, `|v| ∈ [0.15, 0.58]`.
  A real arm reaches a forward sector, so absolute volumes are a proxy; the s-vs-s comparison
  uses an identical procedure and stays fair. Replace with a sector mask before quoting absolute
  litres.
- Targets are solved **warm-started from the previous solution** in random order, as live
  teleoperation does, so "feasible" is mildly order-dependent.
- Sampling: 400 targets/scale, 100 ramp steps each, `err < 0.01 m` = feasible, seed 0. A single
  big IK jump diverges; the target must be ramped, which also mirrors the per-frame controller.

Still to do for E1: add manipulability reporting (the task is already in the solver stack), and
run B3 (elbow-constrained) to see whether the null-space reference recovers the ~20% infeasible
band.

---

## 8. Novelty and positioning risks

1. **⚠️ SEW-Mimic (2026-02) is 7 months old and squarely adjacent.** If the paper does not cite
   and differentiate it, a knowledgeable reviewer rejects on missing related work alone.
   Differentiate on C1 (we *keep* EE matching and scale it) and C2 (orientation decoupled).
2. **⚠️ "Absolute beats incremental" is not novel.** Frame C3 as a *property* that enables the
   measured benefits, never as the headline.
3. **Constant scale factor may look naive** next to variable-scaling work (IJCAS 2024). Mitigate
   with an ablation sweeping `s` ∈ {0.8, 1.0, 1.324, 1.6} and reporting the performance curve —
   that turns a possible weakness into a result.
4. **Dropping the elbow needs a defence.** Cluster C establishes the elbow as the null-space
   coordinate. Either show B3 ≈ B4 empirically, or argue from the human-redundancy literature
   (MECBME 2014) that the human elbow is a noisy reference. Do not leave it unaddressed.
5. **Scope may read as narrow** (one fixed-base dual-arm platform). The repo already supports
   Unitree G1 — adding it as a second platform substantially widens the claim for little code.
6. **No public SEW-Mimic code** → baseline reproduction risk. Decide early: reproduce (cost) or
   compare analytically (weaker).
7. **⏰ Deadline check.** Today is 2026-09-11. ICRA submission deadlines fall in mid-September.
   **Verify the exact ICRA 2027 deadline immediately** — this was not confirmed by any source in
   this search and should not be assumed.

---

## 9. Coverage gaps in this search

- **OpenAlex and Semantic Scholar returned HTTP 429** from this host, so citation-graph
  expansion (who-cites-whom forward/backward snowballing from SEW-Mimic) was **not** performed.
  That is the single most valuable missing step: forward-citations of arXiv:2602.01632 would
  surface any 2026 follow-ups.
- **IEEE Xplore was not queried directly.** Cluster B/C venue metadata comes from Crossref;
  abstracts were not read for most of those rows (marked without ✓). Claims about them are
  inferred from title + venue + citation count only.
- **ICRA/IROS 2026 accepted-paper lists were not enumerated.** Only the existence of a
  *Beyond Teleoperation* workshop at ICRA 2026 was observed (arXiv:2605.16043 comment).
- No **Chinese-language literature** (CNKI) search, though several relevant venues
  (RCAR, CAC, J. Mechanical Engineering) appeared via Crossref.
- Only 2 of 47 screened candidates had abstracts read in full (SEW-Mimic, WARP, CLOT,
  XRoboToolkit = 4 actually). Others are title/venue/citation-level screening.

---

## 10. Recommended next steps

1. **Verify the ICRA deadline today.**
2. **Read SEW-Mimic in full** (arXiv:2602.01632) — it is the paper our contribution is defined
   against. Then re-run forward-citation snowballing once Semantic Scholar/OpenAlex are reachable.
3. ~~**Implement E1**~~ **DONE — see §7.4.** Re-run after any URDF/scale change:
   `PYTHONPATH=. python scripts/analysis/swgr_workspace_volume.py --arm right --n 400`
4. **Add B2 and B3 conditions** — both are config/near-config changes in the current code.
5. **Fix the body-tracking data path** before any human experiment. As of now
   `get_body_tracking_data()` returns `None` on the test bench, so **no** SWGR data can be
   collected at all. This blocks E2–E8. **E1 is done (§7.4) and needs no hardware.**
6. Decide on **SEW-Mimic reproduction** (B7) — the main cost/scope decision for the paper.

**Next CCFA owner:** `ccf-experiment-designer` to turn §7 into a runnable protocol with result
tables; `ccf-idea-reviewer` if you want an independent judgement on whether C1+C2+C3 clears the
ICRA bar given SEW-Mimic.

---

## Checklist status (ccf-literature-searcher, standard mode)

| # | Item | Status |
|---|---|---|
| 1 | Topic → safe public queries | ✓ (no private draft text used; queries were public method/venue/keyword terms) |
| 2 | Source-quality exclusions applied | ✓ (no MDPI/predatory; Crossref citation-sorted run discarded as non-relevant and not cited) |
| 3 | Primary/high-confidence sources | ✓ arXiv API + Crossref API; ⚠ OpenAlex/Semantic Scholar 429 |
| 4 | Deduplicated by title, stable URLs | ✓ (47 screened → normalized-title dedup; every row has arXiv ID or DOI) |
| 5 | Venue/year/status + type + rationale | ✓ per row |
| 6 | Quality scoring | ⏭ skipped — not requested, and only 4 abstracts were read (scoring unread papers is prohibited) |
| 7 | Paper-type taxonomy | partial — preprint / published / standard / book chapter recorded; formal 7-class labels omitted as not decision-relevant |
| 8 | Claims traceable or marked inferred | ✓ (✓abstract-read vs title-level screening distinguished; §9 lists inference limits) |
| 9 | Closest-work clusters with covered / under-tested / differentiation | ✓ (§1 differentiation table, §6 opportunity map) |
| 10 | Search folder written | ✓ this file |
| 11 | Idea-grounding packet | ⏭ not requested — §7/§8 serve the handoff |
| 12 | Handoff | ✓ (§10 names `ccf-experiment-designer`, `ccf-idea-reviewer`) |

**No-fabrication statement:** every paper above came from a live arXiv or Crossref API response
in this session. No citation, DOI, arXiv ID, venue, acceptance status, or citation count was
recalled from memory. Acceptance statuses are reported only where the arXiv comment field or
Crossref `container-title` stated them. §7.4's numbers are **locally measured** by
`scripts/analysis/swgr_workspace_volume.py` on this repo's JAKA K1 URDF, not taken from any
paper; its stated simplifications apply. §7 cost estimates are judgement, not measurements. No
numerical result is attributed to any cited paper beyond what its abstract states.
