# TRACE model and data card

TRACE estimates frame-level team possession from automatic tactical tracking. The released composition is `constrained_full330`: status/timing/interval quality, learned owner with switch cost 0.6, full330 turnover evidence, and constrained refinement. It produces H/A/D/X labels without training or model selection during inference.

## Intended use and boundaries

The supplied entry point reproduces the possession file for the organizer-defined match and frame/period layout. It uses automatic tactical tracking TXT and matching XML metadata. Video, Smart Tagging, homography, corrected tracking and possession references are not inference inputs. Five fitted models and their supporting code are bundled.

This is offline sequence inference: centered temporal features may use later observations. It is not a real-time event detector or a certification of physical ball control. Missing balls, tracking errors, crowded play, stoppages and provider-definition differences can affect performance.

## External source and split

All five components were fitted using **Sportec Open DFL**, dataset version 1, [DOI 10.6084/m9.figshare.28196177.v1](https://doi.org/10.6084/m9.figshare.28196177.v1), described by Bassek, Rein, Weber and Memmert (2025), [DOI 10.1038/s41597-025-04505-y](https://doi.org/10.1038/s41597-025-04505-y). The saved release identifies official DFL/Sportec data and TRACAB optical tracking. Its data licence is [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

| Role | Match identities | Use |
|---|---|---|
| Training | J03WOH, J03WMX, J03WN1, J03WOY | Fitted component training |
| Development | J03WPY, J03WQQ, J03WR9 | Repeated selection and evaluation |

All seven games have been exposed during development; no fresh-test claim is made. The [asset manifest](../sportec-required-assets.json) identifies 14 tracking/match-information files with versioned download URLs and checksums. Event XML and external video are not required by the retained route.

## Targets and annotation provenance

TRACE did not manually annotate these targets. It mechanically maps the provider’s frame-level `BallStatus` and `BallPossession` attributes:

| Provider fields | Four-state target |
|---|---|
| `BallStatus="0"` | D, regardless of ownership |
| `BallStatus="1"`, `BallPossession="1"` | H |
| `BallStatus="1"`, `BallPossession="2"` | A |
| Missing/unrecognized status, or live status without recognized ownership | X |

Separate live/dead supervision maps inactive to 1, active to 0, and unavailable/unrecognized to −1. A frame can therefore have an unknown four-state owner (X) while retaining a known live target. Proximity does not fill missing ownership targets.

Source documentation describes tracking quality control and trained event operators, but does not establish a named annotator or a complete per-frame correction workflow for these two attributes. Event-annotation provenance is not substituted for possession/status provenance. Provider control and restart conventions are not proven identical to the challenge definitions.

Target attributes are excluded from predictor features. Predictors use observations, geometry, speed, visibility, missingness and temporal summaries. Training ingestion reads XML that contains both observations and targets; separation is semantic rather than a claim that target bytes never occur in the source files. Supplied challenge labels and video are not used for fitting or tuning.

## Fitted components

| Artifact | Supervision / purpose |
|---|---|
| `models/status-model.joblib` | External live/dead status |
| `models/timing-model.joblib` | Derived live/dead boundary timing |
| `models/quality-model.joblib` | Derived interval quality using leave-one-match-out status predictions |
| `models/owner-model.joblib` | Known H/A ownership; D/X excluded |
| `models/full330-boundary.joblib` | Direct known H↔A changes with ±0.20 s target radius; D/X gaps excluded |

The owner and boundary heads use histogram gradient boosting: 160 iterations, learning rate 0.07, at most 15 leaves, minimum 60 samples per leaf, L2 regularization 2, seed 2027, and no early stopping. They use 5 Hz training sampling, match/class balancing and team-swap augmentation. Their scores are not calibrated probabilities.

Thirty-three raw observation channels plus centered 0.5/2-second means and contrasts, with missingness indicators, produce 330 features. The final boundary fit used **60,389 original sampled rows / 120,778 augmented rows**. This is the boundary head’s count, not the number of independent examples for every component. Its detailed aggregate settings and counts are in [full330-boundary-training.json](../models/full330-boundary-training.json).

The status stage uses fixed interval-quality threshold 0.9 and timing strength 0.5. Owner switch cost is 0.6. Boundary evidence changes the switch cost to `0.6 × (1 − 0.8b)`. The constrained decoder retains original H/A transitions within ±0.20 s, preserves their direction/order, and permits complete return excursions under the fixed 0.40 s rules. Original shorter principal runs remain feasible. D/X masks and period endpoints stay fixed through ownership refinement. These are engineering constraints, not physiological minima.

After decoding, a minimum-control rule from the organizers' written definition is applied: an opponent run shorter than 3.0 s that is immediately preceded and followed by the same team, with no dead ball between, is not a change of possession and takes the surrounding team's label. The threshold is a fixed reading of "a few seconds", not a fitted value. On the three exposed external development matches this rule lowers official spellF1 from 0.537786 to 0.475109, because the provider's labels do record such brief opponent runs; it is applied because the challenge definition excludes them. Its effect on the supplied match is unmeasured.

Then one organizer-documented window is labelled dead ball: the second half is graded from frame 110,808 but does not kick off until 113,742 (organizers' FAQ). This uses the organizers' published constant, not a fitted parameter or a team annotation.

## Evaluation and limitations

Official spellF1 averages one-to-one whole-spell F1 over IoU 0.3/0.4/0.5/0.6/0.7. On the same three exposed development games, the equal-match mean is 0.439621838 for the anchor, 0.512489088 for learned ownership, and 0.537786282 for the final constrained path. The final path retains 405/408 original owner-cohort matched events. Losses retain a 2-second penalty and the original denominator in timing comparisons.

These results are development evidence. They do not estimate the private challenge score, an unseen-match gain or a leaderboard rank. Cleaner external tracking, seven repeatedly exposed matches, and uncertain provider-control semantics constrain generalization. The status stage can miss isolated stoppages; fixed D/X support prevents later ownership stages from repairing them. Retaining an initial handover can also preserve a mistake.

Two raw-input runs reproduced the exact 199,319-row CSV, and a clean pinned installation passed on Windows/Python 3.11.9. Other hosts and operating systems remain unverified. Runtime/model identities are checked through [artifacts.json](../artifacts.json); the expected output hash is recorded in the [README](../README.md#reproduce-the-possession-file).
