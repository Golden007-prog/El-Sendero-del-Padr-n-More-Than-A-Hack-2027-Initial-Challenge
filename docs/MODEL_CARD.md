# TRACE model and data card

TRACE estimates frame-level team possession from automatic tactical tracking. The released composition is `constrained_full330`: status/timing/interval quality, learned owner with switch cost 0.9, full330 turnover evidence, and constrained refinement. It produces H/A/D/X labels without training or model selection during inference.

## Intended use and boundaries

The supplied entry point reproduces the possession file for the organizer-defined match and frame/period layout. It uses automatic tactical tracking TXT and matching XML metadata. Video, Smart Tagging, homography, corrected tracking and possession references are not inference inputs. Seven fitted models and their supporting code are bundled.

This is offline sequence inference: centered temporal features may use later observations. It is not a real-time event detector or a certification of physical ball control. Missing balls, tracking errors, crowded play, stoppages and provider-definition differences can affect performance.

## External source and split

All fitted components use **Sportec Open DFL**, dataset version 1, [DOI 10.6084/m9.figshare.28196177.v1](https://doi.org/10.6084/m9.figshare.28196177.v1), described by Bassek, Rein, Weber and Memmert (2025), [DOI 10.1038/s41597-025-04505-y](https://doi.org/10.1038/s41597-025-04505-y). The saved release identifies official DFL/Sportec data and TRACAB optical tracking. Its data licence is [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

| Role | Match identities | Use |
|---|---|---|
| Training | J03WOH, J03WMX, J03WN1, J03WOY | All fitted components |
| Training | Metrica Sports Sample Games 1, 2, 3 | Ownership and boundary heads only |
| Development | J03WPY, J03WQQ, J03WR9 | Repeated selection and evaluation |

All seven games have been exposed during development; no fresh-test claim is made. The [asset manifest](../sportec-required-assets.json) identifies 14 tracking/match-information files with versioned download URLs and checksums. Event XML and external video are not required by the retained route.

The ownership and boundary heads are additionally fitted on the three public sample matches of [metrica-sports/sample-data](https://github.com/metrica-sports/sample-data) (source acknowledged; file identities in [metrica-sample-assets.json](../metrica-sample-assets.json)). Their per-frame targets are derived from Metrica's human-coded event data by a fixed rule in [fit_heads_seven_matches.py](../training/fit_heads_seven_matches.py): possession changes only at a RECOVERY or SET PIECE of the other team; the ball is dead from the end of the event before each set piece to that set piece; frames outside the two periods are unknown. The events are Metrica's annotations. The frame-level derivation is ours and is not an official possession label.

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
| `models/owner-model-smoothed.joblib` | As `owner-model.joblib`, fitted with smoothed ball tracks added; used when the ball track is smoothed |
| `models/full330-boundary-smoothed.joblib` | As `full330-boundary.joblib`, fitted with smoothed ball tracks added; used when the ball track is smoothed |

The owner and boundary heads use histogram gradient boosting: 800 iterations, learning rate 0.07, at most 31 leaves, minimum 60 samples per leaf, L2 regularization 2, seed 2027, and no early stopping. They use 5 Hz training sampling, match/class balancing and team-swap augmentation. Their scores are not calibrated probabilities.

Thirty-three raw observation channels plus centered 0.5/2-second means and contrasts, with missingness indicators, produce 330 features. The ownership and boundary heads each use **120,438 original sampled rows** (60,389 from the Sportec matches and 60,049 from the Metrica sample matches), doubled to 240,876 by team-swap augmentation. Its detailed aggregate settings and counts are in [full330-boundary-training.json](../models/full330-boundary-training.json) and, for the second pair, [full330-boundary-smoothed-training.json](../models/full330-boundary-smoothed-training.json).

The supplied tracking carries a ball track smoothed over about half a second; the training matches do not. A second pair of heads (`owner-model-smoothed.joblib`, `full330-boundary-smoothed.joblib`) is fitted with every training match added once more, its ball track smoothed with a Gaussian of 6 frames (240,876 original sampled rows, 481,752 after team-swap augmentation). At inference the pipeline computes the 99th percentile of the change in ball velocity over 0.2 s and uses the second pair when it is below 12 m/s; external matches give 16.6 to 20.5 as recorded, the supplied match 7.4. With the ball smoothed at 6 frames the second pair changes held-out spellF1 by +0.002 (Sportec) and +0.009 (Metrica, provider dead ball) against the first pair; on tracks as recorded it would lose 0.009 and 0.011, which is why it is not used there. The switch was designed after those results; the README says what was registered in advance. The smoothing width and the threshold were set from statistics of the supplied tracking that need no label. The smoothing itself costs 0.050 and 0.067 and is mostly not recoverable. The Gaussian was the emulation in use when the pair was chosen. The supplied filter has since been identified from its spectrum as a sharp zero-phase low-pass near 0.6 Hz; under it the second pair changes held-out spellF1 by +0.001 and +0.011, a pair fitted with that filter did no better, and undoing the filter was tested and not released (README, Ball smoothing).

The status stage uses fixed interval-quality threshold 0.9 and timing strength 0.5. Owner switch cost is 0.9. Boundary evidence changes the switch cost to `0.9 × (1 − 0.8b)`. The constrained decoder retains original H/A transitions within ±0.20 s, preserves their direction/order, and permits complete return excursions under the fixed 0.40 s rules. Original shorter principal runs remain feasible. D/X masks and period endpoints stay fixed through ownership refinement. These are engineering constraints, not physiological minima.

After decoding, stoppage onsets are moved to the moment the ball leaves the pitch ([ball_exit.py](../ball_exit.py)): if, within 3 s before a predicted stoppage, the ball is tracked outside the pitch, or was last tracked within 2 m of a boundary line and is not tracked again, the stoppage starts 2 frames after the last in-pitch sighting. The three constants are fixed by reasoning, not fitted, and only live frames directly before a predicted stoppage can change.

Then one organizer-documented window is labelled dead ball: the second half is graded from frame 110,808 but does not kick off until 113,742 (organizers' FAQ). This uses the organizers' published constant, not a fitted parameter or a team annotation.

## Evaluation and limitations

Official spellF1 averages one-to-one whole-spell F1 over IoU 0.3/0.4/0.5/0.6/0.7. On the same three exposed development games, the equal-match mean is 0.439621838 for the anchor, 0.512489088 for learned ownership, and 0.537786282 for the final constrained path. The final path retains 405/408 original owner-cohort matched events. Losses retain a 2-second penalty and the original denominator in timing comparisons.

Those figures are for heads fitted on the four Sportec matches. The released heads add the three Metrica sample matches. With each sample match held out in turn, official spellF1 against the event-derived reference rises from 0.610233 to 0.643132 (every match improves), and the Sportec development mean moves from 0.537786 to 0.531989. The acceptance gate was fixed before the fits. A minimum-control rule that absorbed opponent runs under three seconds was published in one edition and withdrawn after it lowered the score on both external benchmarks; it is not part of this release.

The ball-exit onset adds to these figures: Sportec development 0.531989 to 0.536550, Metrica sample matches 0.643132 to 0.652335, and 0.546375 to 0.551132 on the four Sportec fitting matches used as a replication. Every one of the ten matches improves. The rule missed the size set in advance for the first six matches (+0.008 on Sportec) and was adopted after the replication; both facts are stated in the README and the technical report.

The released heads are larger than the ones behind those figures: 800 iterations and 31 leaves instead of 160 and 15, chosen in two nested cross-validation steps on the Metrica sample matches with gates fixed in advance. Held-out means with the ball-exit rule: Sportec development 0.536550 to 0.557311; Metrica sample matches 0.652335 to 0.682533 against the event-derived reference and 0.611160 to 0.638169 when dead ball is taken where the provider's tracking has no ball. The second step (400 to 800 iterations) passed narrowly: nested estimate +0.009 against a gate of +0.008, with two of six matches slightly lower. Larger dead-ball models were tested in the same way and not released. The same six matches chose the size, so these are development scores.

The released switching cost is 0.9 (0.6 in earlier editions). It was chosen by the team outside the rule written in advance, the second such exception after the ball-exit rule. Tested by nested cross-validation it raised held-out spellF1 (Sportec 0.557311 to 0.565928; Metrica 0.682533 to 0.701647 against the event-derived reference and 0.638169 to 0.652915 with provider dead ball), but the registered rule selected 1.2, which failed a robustness check, so by that rule the cost would have stayed at 0.6. The choice of 0.9 was made after seeing the results, on the same six matches; the README states the evidence and the caveats.

These results are development evidence. They do not estimate the private challenge score, an unseen-match gain or a leaderboard rank. Cleaner external tracking, seven repeatedly exposed matches, and uncertain provider-control semantics constrain generalization. The status stage can miss isolated stoppages; fixed D/X support prevents later ownership stages from repairing them. Retaining an initial handover can also preserve a mistake.

Raw-input runs reproduce the exact 199,319-row CSV on Windows and in a Linux container (official Python 3.11.9 image, fresh installation of the pinned packages). macOS and a second machine remain unverified. Runtime/model identities are checked through [artifacts.json](../artifacts.json); the expected output hash is recorded in the [README](../README.md#reproduce-the-possession-file).
