"""OOF interval-level evidence for removing wholly false DEAD predictions.

Only complete interior DEAD runs are scored. The caller supplies predictions
from a model that excluded each training match; this module never fits a status
head. Scores from class-weighted training are uncalibrated detection strengths.
No threshold, owner filling, boundary retiming, or live-hole filling occurs here.
"""

from collections.abc import Mapping
from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from threadpoolctl import threadpool_limits

from .motion import PLAYER_SUMMARY_NAMES, _base_summary
from .timing_model import _blocks


CHANNELS = (
    'learned_dead_score', 'fused_dead_score', 'boundary_strength',
    'speed_q50_mps', 'stationary_fraction', 'visible_location_count',
    'status_quality', 'ball_quality', 'abs_control_evidence',
)
SUMMARIES = ('inside_q10', 'inside_q50', 'inside_q90',
             'flank_q50_min', 'flank_q50_max', 'known_fraction')
EVIDENCE_KEYS = ('learned_dead_score', 'fused_dead_score', 'boundary_strength',
                 'status_quality', 'ball_quality', 'control_evidence')
FEATURE_NAMES = ('duration_s',) + tuple(f'{channel}_{stat}' for channel in CHANNELS
                                       for stat in SUMMARIES)
TARGET_POLICY = {
    'false': '1 only if the interval and full two-second margins are known LIVE in one period',
    'true': '0 if at least one interval frame is GT DEAD in one period',
    'excluded': '-1 for nearby nonoverlapping DEAD, unknown context, or censored negative margins',
    'margin_s': 2.,
    'training_predictions': 'Caller-supplied OOF: the status/evidence fit must exclude this match',
    'weighting': 'Equal usable match weight; equal present-class weight within each match',
    'score': 'Uncalibrated false-interval detection strength; zero fallback if fewer than two classes',
}


def _status_and_periods(status, blocks, fps):
    status = np.asarray(status)
    if status.ndim != 1 or not np.isin(status, [-1, 0, 1]).all():
        raise ValueError('Status must be a vector with -1 unknown, 0 live, 1 dead')
    if not np.isscalar(fps) or not np.isfinite(fps) or fps <= 0:
        raise ValueError('fps must be finite and positive')
    return status, _blocks(blocks, len(status))


def _bounds(candidate, n):
    if isinstance(candidate, Mapping):
        if 'start' not in candidate or 'end' not in candidate:
            raise ValueError('Interval records need start and end')
        start, end = candidate['start'], candidate['end']
    else:
        if len(candidate) != 2:
            raise ValueError('Intervals must be half-open (start, end) pairs')
        start, end = candidate
    if (not isinstance(start, (int, np.integer)) or isinstance(start, (bool, np.bool_)) or
            not isinstance(end, (int, np.integer)) or isinstance(end, (bool, np.bool_)) or
            start < 0 or end <= start or end > n):
        raise ValueError('Interval bounds must be integer slices within the status rows')
    return int(start), int(end)


def interval_candidates(status, blocks, fps):
    """Return eligible DEAD run records, with half-open ``start``/``end``.

    Both immediately adjacent runs must be LIVE for at least one second in the
    same period. Unknown runs and period seams are never bridged. DEAD duration
    may be at most ten seconds. First/final censored runs are excluded.
    ``before_start`` and ``after_end`` delimit the complete adjacent LIVE runs;
    these bounds and block bounds are audit metadata, never model inputs.
    """
    status, periods = _status_and_periods(status, blocks, fps)
    result = []
    for lo, hi in periods:
        edges = np.r_[lo, np.flatnonzero(np.diff(status[lo:hi])) + lo + 1, hi]
        for index in range(1, len(edges) - 2):
            before, start, end, after = map(int, edges[index - 1:index + 3])
            if (status[start] == 1 and status[before] == 0 and status[end] == 0 and
                    end - start <= 10 * fps + 1e-9 and
                    start - before >= fps - 1e-9 and after - end >= fps - 1e-9):
                result.append(dict(start=start, end=end, block_start=lo, block_end=hi,
                                   before_start=before, after_end=after))
    return result


def _evidence_arrays(evidence, n):
    if not isinstance(evidence, Mapping):
        raise ValueError('Evidence must provide the frozen decoder score vectors')
    result = {}
    for name in EVIDENCE_KEYS:
        if name not in evidence:
            raise ValueError(f'Missing interval evidence: {name}')
        values = np.asarray(evidence[name], dtype=float)
        if values.shape != (n,) or not np.isfinite(values).all():
            raise ValueError(f'Interval evidence {name} must be a finite vector of length N')
        if name != 'control_evidence' and ((values < 0) | (values > 1)).any():
            raise ValueError(f'Interval evidence {name} must lie in [0,1]')
        result[name] = values
    return result


def _quantiles(values, probabilities):
    observed = np.isfinite(values)
    if not observed.any():
        return np.zeros(len(probabilities), dtype=float)
    limit = np.finfo(np.float32).max
    return np.quantile(np.clip(values[observed], -limit, limit), probabilities)


def interval_design(features, status, evidence, blocks, candidates=None):
    """Return ``(records, float32[N_intervals,55], ordered_names)``.

    Every channel contributes interval q10/q50/q90, the minimum/maximum of
    separate one-second before/after medians, and the observed fraction across
    the interval and these flanks. Quantiles use observed values only; missing
    values impute zero accompanied by the channel's observed-fraction feature.
    Flank min/max and absolute control make the design invariant to reversing
    time and to exchanging teams. No coordinates, GT, IDs or frame indices enter
    X. Temporal flanks stay inside their immediately adjacent LIVE runs.

    Explicit candidates must be a duplicate-free subset of the eligible runs.
    Extra record fields survive for auditing; derived bounds are authoritative.
    All six EVIDENCE_KEYS are required finite frame vectors even for zero rows.
    """
    n = len(features)
    status, periods = _status_and_periods(status, blocks, features.fps)
    if len(status) != n:
        raise ValueError('Status length must match the feature rows')
    supplied = _evidence_arrays(evidence, n)
    eligible = interval_candidates(status, periods, features.fps)
    by_bounds = {(item['start'], item['end']): item for item in eligible}
    records = []
    seen = set()
    for item in eligible if candidates is None else candidates:
        bounds = _bounds(item, n)
        if bounds not in by_bounds or bounds in seen:
            raise ValueError('Explicit candidates must be distinct complete eligible DEAD runs')
        metadata = dict(item) if isinstance(item, Mapping) else {}
        records.append({**metadata, **by_bounds[bounds]})
        seen.add(bounds)
    summary = _base_summary(features)
    channels = {
        **{name: supplied[name] for name in EVIDENCE_KEYS if name != 'control_evidence'},
        **{name: summary[:, PLAYER_SUMMARY_NAMES.index(name)] for name in
           ('speed_q50_mps', 'stationary_fraction', 'visible_location_count')},
        'abs_control_evidence': np.abs(supplied['control_evidence']),
    }
    design = np.empty((len(records), len(FEATURE_NAMES)), dtype=np.float32)
    flank = max(1, int(np.ceil(float(features.fps) - 1e-9)))
    for row, record in enumerate(records):
        start, end = record['start'], record['end']
        left, right = start - flank, end + flank
        values = [(end - start) / float(features.fps)]
        for channel in CHANNELS:
            observations = channels[channel]
            inside = _quantiles(observations[start:end], [.1, .5, .9])
            before = _quantiles(observations[left:start], [.5])[0]
            after = _quantiles(observations[end:right], [.5])[0]
            known = float(np.isfinite(observations[left:right]).mean())
            values.extend((*inside, min(before, after), max(before, after), known))
        limit = np.finfo(np.float32).max
        design[row] = np.clip(values, -limit, limit)
    return records, design, list(FEATURE_NAMES)


def quality_targets(candidates, gt_status, blocks, fps):
    """Return int8 false=1/true=0/ambiguous=-1 labels for training only.

    A predicted DEAD interval overlapping even one GT DEAD frame is true. A
    nonoverlapping nearby genuine event is ambiguous, never a false example.
    A false label requires the full interval plus ceil(2*fps)-frame margins to
    fit within one period and contain only known LIVE GT. Margins are not
    clipped at periods or annotation gaps. No target is an inference feature.
    """
    truth, periods = _status_and_periods(gt_status, blocks, fps)
    candidates = list(candidates)
    result = np.full(len(candidates), -1, dtype=np.int8)
    margin = max(1, int(np.ceil(2 * float(fps) - 1e-9)))
    for index, candidate in enumerate(candidates):
        start, end = _bounds(candidate, len(truth))
        period = next(((lo, hi) for lo, hi in periods if lo <= start < end <= hi), None)
        if period is None:
            continue
        if np.any(truth[start:end] == 1):
            result[index] = 0
        elif (start - margin >= period[0] and end + margin <= period[1] and
              np.all(truth[start - margin:end + margin] == 0)):
            result[index] = 1
    return result


@dataclass
class IntervalQualityModel:
    classifier: object
    feature_names: list[str]
    training_ids: list[str]
    sample_counts: dict
    target_policy: dict = field(default_factory=lambda: dict(TARGET_POLICY))
    constant_reason: str | None = None

    def predict_false(self, design):
        """Return an uncalibrated strength for false DEAD, conservatively zero if inactive."""
        design = np.asarray(design, dtype=np.float32)
        if (design.ndim != 2 or design.shape[1] != len(self.feature_names) or
                not np.isfinite(design).all()):
            raise ValueError('Interval design differs from the trained finite feature schema')
        if self.classifier is None or not len(design):
            return np.zeros(len(design), dtype=float)
        with threadpool_limits(limits=4):
            classes = list(self.classifier.classes_)
            return self.classifier.predict_proba(design)[:, classes.index(1)]


def fit_interval_quality(training_matches):
    """Fit a fixed CPU-feasible classifier on caller-supplied OOF predictions.

    Each dict supplies match_id, features, status, evidence, blocks and truth
    (binary -1/0/1 GT). Each usable match has equal total weight, with its present
    classes weighted equally within that match. No threshold or hyperparameter
    search, random frame validation, calibration, or internal early stopping is
    performed. A single class (including only false examples) leaves the model
    conservatively inactive with score zero and a recorded reason.
    """
    ids, counts, xs, ys, weights = [], {}, [], [], []
    for match in training_matches:
        match_id = match['match_id']
        if not isinstance(match_id, str) or not match_id or match_id in ids:
            raise ValueError('Need distinct nonempty training match identities')
        ids.append(match_id)
        features = match['features']
        records, design, names = interval_design(features, match['status'], match['evidence'], match['blocks'])
        truth = np.asarray(match['truth'])
        if truth.shape != (len(features),):
            raise ValueError('Training truth must match the feature rows')
        target = quality_targets(records, truth, match['blocks'], features.fps)
        usable = target >= 0
        counts[match_id] = dict(eligible=len(records), labeled=int(usable.sum()),
                               excluded=int((~usable).sum()), false=int((target == 1).sum()),
                               true=int((target == 0).sum()))
        if not usable.any():
            continue
        target = target[usable]
        classes, class_counts = np.unique(target, return_counts=True)
        weight = np.empty(len(target), dtype=float)
        for value, count in zip(classes, class_counts):
            weight[target == value] = 1. / (len(classes) * count)
        xs.append(design[usable])
        ys.append(target)
        weights.append(weight)
    if not ids:
        raise ValueError('Need at least one OOF training match')
    common = dict(feature_names=list(FEATURE_NAMES), training_ids=ids, sample_counts=counts)
    if not xs:
        return IntervalQualityModel(None, **common, constant_reason='No labeled eligible intervals')
    target = np.concatenate(ys)
    if len(np.unique(target)) < 2:
        return IntervalQualityModel(None, **common, constant_reason='Only one labeled interval class')
    weight = np.concatenate(weights)
    weight *= len(weight) / weight.sum()
    classifier = HistGradientBoostingClassifier(max_iter=80, max_leaf_nodes=7,
        min_samples_leaf=10, l2_regularization=5., learning_rate=.05,
        early_stopping=False, random_state=2027)
    with threadpool_limits(limits=4):
        classifier.fit(np.concatenate(xs), target, sample_weight=weight)
    return IntervalQualityModel(classifier, **common)
