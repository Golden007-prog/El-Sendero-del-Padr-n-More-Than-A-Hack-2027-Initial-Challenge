<p align="center">
  <img src="docs/figures/trace-cover.svg" alt="TRACE — recover whole possessions. El Sendero del Padrón, More Than A Hack 2027." width="1200">
</p>

<p align="center">
  <strong>Initial Challenge · Frame-level Ball Possession</strong><br>
  <a href="#the-approach">Approach</a> ·
  <a href="#reproduce-the-possession-file">Reproduce</a> ·
  <a href="#measured-evidence">Evidence</a> ·
  <a href="docs/MODEL_CARD.md">Model &amp; data</a> ·
  <a href="docs/technical-report.pdf">Technical report</a>
</p>

TRACE turns automatic tactical tracking into a complete **home / away / dead-ball / unknown** timeline. It combines learned ownership evidence with a constrained sequence decoder, aiming to recover whole possessions while retaining credible handovers.

The football distinction matters: a pass, loose ball, deflection or contested touch does not automatically transfer possession. The controlling team remains the owner until the opposition establishes control. The challenge evaluates whether those complete intervals are recovered—not simply whether nearby frames look similar.

| Input | Output | Primary metric | Runtime |
|---|---|---|---|
| Automatic tactical tracking TXT + XML metadata | `frame,label` CSV · 199,319 rows | Official **spellF1** | Offline CPU inference · Python 3.11.9 |

## The approach

<img src="docs/figures/pipeline.svg" alt="Four stages: live/dead support, learned ownership, turnover evidence, and a constrained possession path." width="1200">

1. **Establish live/dead support.** Player motion and geometric control evidence form a status timeline. Learned timing and interval-quality models refine its stoppages.
2. **Estimate team ownership.** A 330-feature model combines ball proximity, control evidence, temporal summaries, motion, visibility and missingness. A sequence decoder produces an initial H/A path with switch cost `0.6`.
3. **Estimate turnover evidence.** A separate model scores incoming H/A boundaries. Strong boundary evidence lowers the cost of a switch.
4. **Constrain the final path.** The decoder preserves the order and direction of initial handovers within ±`0.20 s` windows. Additional handovers must form complete return excursions under fixed `0.40 s` duration rules. Dead-ball and unknown support remain unchanged.
5. **Require minimum control.** The organizers' [definition](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/CHALLENGE.md) says possession changes only on *controlled* possession: a simple touch, a deflection or an aerial duel won without subsequent control does not count, and a proxy for control is that a player keeps the ball for a few seconds. An opponent run shorter than `3.0 s` that is immediately preceded and followed by the same team, with no dead ball between, is therefore not a change of possession. Such runs are absorbed shortest first. The threshold is a fixed reading of "a few seconds", not a fitted value.
6. **Apply one documented dead-ball window.** The organizers' [FAQ](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/FAQ.md) states that the second half is graded from frame `110,808` but does not kick off until `113,742`, and that those graded frames are dead ball. The final file labels that window `D`. The constant is the organizers' published one; no frame was inspected or labelled by the team.

This composition couples learned local evidence with explicit temporal structure. The constraints limit arbitrary edits and make the final path traceable; they do not guarantee that an initial predicted handover is correct. All parameters and five fitted models are fixed during inference. The method is offline and may use future observations.

<img src="docs/figures/possession-semantics.svg" alt="Invented illustration: home possession continues through a pass and loose ball until away control, followed by a dead-ball stoppage." width="1200">

| Label | Meaning |
|---|---|
| **H** | Home team possession |
| **A** | Away team possession |
| **D** | Dead ball; separates possession spells |
| **X** | Unknown / abstention |

The official scorer matches maximal H/A spells one-to-one by temporal overlap and averages F1 at IoU thresholds **0.3, 0.4, 0.5, 0.6 and 0.7**. Frame-level macroF1 is also reported and is the first tie-break. See the organizers’ [challenge definition](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/CHALLENGE.md) and [evaluation specification](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/EVALUATION.md).

## Reproduce the possession file

Use **64-bit CPython 3.11.9** and the pinned dependencies. The checked environment is Windows; a second machine and other operating systems have not been validated. No GPU, external service or training dataset is required for inference.

Clone this repository, then run the following from its root in PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-research-lock.txt

.\.venv\Scripts\python.exe -I -B infer_candidate.py `
  --data-dir "<authorized-bundle-root>" `
  --output "outputs/El Sendero del Padrón.csv"

.\.venv\Scripts\python.exe -I -B evaluation/validate_submission.py `
  "outputs/El Sendero del Padrón.csv"
```

The organizer-provided bundle must have this layout; [DATA.md](DATA.md) explains authorized access and local placement:

```text
<authorized-bundle-root>/
└── tactical/
    ├── BAR-ATH_tactical_atd.xml
    └── BAR-ATH_tactical_atd.txt
```

The command checks packaged source/model hashes and the exact runtime, processes the raw tracking pair, and writes the CSV plus a local verification receipt. It refuses to overwrite an existing output. Generated outputs and match inputs are excluded from this repository.

The retained output is identified by:

| Check | Expected value |
|---|---|
| Header | `frame,label` |
| Data rows | **199,319** |
| Frames | **0–199318**, once each, in order |
| Labels | **H / A / D / X** |
| File size | **1,682,773 bytes** |
| SHA-256 | `5ac039ea0d6b017689b440f0cd88008921e0d31082d2193b32148afc2632e791` |

Two runs from raw inputs regenerated these identical bytes, with **51.76 s** and **50.13 s** of inference on the checked host. A clean installation of all 12 pinned distributions passed for the first edition, and no dependency has changed since. These checks establish reproducibility and output format; they do not establish accuracy against the private challenge labels.

## Measured evidence

<img src="docs/figures/development-results.svg" alt="On the same three exposed development matches, official spellF1 is 0.439622 for the status/geometry anchor, 0.512489 after learned ownership, and 0.537786 after constrained refinement." width="1200">

| Same external development cohort | Status / geometry anchor | Learned owner | TRACE constrained |
|---|---:|---:|---:|
| Official spellF1, equal-match mean | 0.439622 | 0.512489 | **0.537786** |
| Original owner-cohort events retained | — | 408 / 408 | **405 / 408** |

The constrained stage adds **0.025297** absolute spellF1 over the immediate owner comparator. Three original matched events are lost; timing analysis retains the full original cohort and penalizes those losses rather than reporting only surviving events.

The released pipeline adds a minimum-control rule after this stage (step 5). On this same cohort it **lowers** the mean to **0.475109**; why it is applied anyway is explained under *Checks on the supplied match* below.

**Scope of the evidence.** These results use the same three repeatedly exposed Sportec development matches: J03WPY, J03WQQ and J03WR9. They are neither private challenge scores nor an independent generalization estimate. All seven external matches have been exposed during development. The organizer’s baseline was measured on a different match and is not a valid before/after comparator here.

The method can miss isolated stoppages, carry ownership through a real takeover, or preserve an incorrect initial transition. Cleaner external tracking and differences in provider control/restart conventions also limit transfer. The [technical report](docs/technical-report.pdf) explains the architecture, safeguards and observed trade-offs.

### Checks on the supplied match

No model was trained or fitted on the supplied match. The official evaluator was run **once** on the public five-minute window as a pipeline check, on the file before the minimum-control rule existed. It confirmed frame alignment and label meaning: macroF1 `0.8504`, alive/dead F1 `0.9687`, matched hand-overs a median 2 frames late. On those 17 possessions spellF1 was `0.4182`, with 27 predicted possessions against 17 in the reference. Seventeen possessions cannot rank methods and this is not a full-match estimate.

**What followed from that check, stated plainly.** It showed the file splitting possession more often than the reference. The organizers' definition asks teams to treat the public ground truth as the arbiter of how "controlled possession" is read, so the minimum-control rule (step 5) was added afterwards. Its `3.0 s` threshold was fixed in advance from the written definition and from the three-second control rule of an external annotation scheme. No threshold was compared on the window, and the released file was **not** scored on it. The decision to add the rule was prompted by the check, and that is a use of the window beyond pure pipeline checking.

**The rule lowers the external development score.** On the three exposed Sportec matches official spellF1 falls from `0.537786` to `0.475109` with the rule, because 22.6% of possessions in that provider's labels (269 of 1,192) are opponent runs under three seconds flanked by the same team. That convention is the one the challenge definition excludes. On the supplied match the rule absorbs 66 such runs and the file goes from 415 to 283 possessions; the effect on the official score is unmeasured and could be negative.

Smart Tagging and the second-camera tracking were analysed and not used for the file. Tagged set pieces already coincide with predicted restarts in 53 of 58 cases, and the broadcast ball fills only 5–9% of frames where the tactical ball is missing. The [technical report](docs/technical-report.pdf) lists every asset tried and dropped.

## Models and external data

Five bundled models cover **status**, **timing**, **interval quality**, **ownership** and **turnover boundaries**. All were fitted using four external Sportec Open DFL matches: J03WOH, J03WMX, J03WN1 and J03WOY. Supplied challenge possession labels and video are not training inputs.

The targets come from provider `BallStatus` and `BallPossession` fields, mechanically mapped into H/A/D/X. They are not new manual annotations by this team. Their complete frame-level annotation/correction workflow is not established by the available source documentation. See the [model and data card](docs/MODEL_CARD.md) for the exact mapping, training roles and limitations, and [external asset identities](sportec-required-assets.json) for versioned source references.

The external dataset is [Bassek et al. (2025), *An integrated dataset of spatiotemporal and event data in elite soccer*](https://doi.org/10.1038/s41597-025-04505-y), [dataset v1](https://doi.org/10.6084/m9.figshare.28196177.v1), licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). It is not redistributed here.

## Repository map

```text
infer_candidate.py           Raw tracking → possession CSV
candidate_pipeline.py        Fixed inference composition and integrity checks
owner_model.py               Ownership features, estimator type and decoder
adaptive_turnovers.py        Turnover evidence model
constrained_turnovers.py     Constrained temporal decoder
controlled_possession.py     Minimum-control rule from the written definition
documented_structure.py      Organizer-documented second-half kick-off window
research_possession/         Tracking, status and timing components
models/                     Five fitted artifacts and boundary provenance
evaluation/                 Official metrics and CSV validator
docs/                       Technical report, model card and figures
artifacts.json              Package source/model integrity manifest
```

## Team and attribution

**El Sendero del Padrón** — Oikantik Basu · Aniket Pal · Sayan Chakraborty

Prepared for the [More Than A Hack 2027 Initial Challenge](https://github.com/metrica-sports/more-than-a-hack-2027), organized by Metrica Sports and Barça Innovation Hub.

TRACE is distributed under the repository’s [MIT licence](LICENSE). Included organizer code retains its [Metrica Sports licence](licenses/METRICA_LICENSE). The separate external-data attribution and rights boundaries are recorded in [NOTICE.md](NOTICE.md). Match assets and generated submission predictions are not distributed in this public repository.
