"""Period-aware research diagnostics. Never a replacement for the official scorer.

Unknown predictions are not filled into dead/live. Opposite known states across
an unknown gap define an uncertain boundary interval, matched conservatively by
its worst endpoint. Selection uses label-derived status identically for every
model; the v3 pre-ownership status sidecar is reported separately.
"""
import numpy as np

TOLERANCES = (.2, .5, 1., 2.)


def label_status(labels):
    labels = np.asarray(labels)
    if labels.ndim != 1 or not np.isin(labels, ['H', 'A', 'D', 'X']).all():
        raise ValueError('Expected one-dimensional H/A/D/X labels')
    return np.where(labels == 'D', 1, np.where(np.isin(labels, ['H', 'A']), 0, -1)).astype(np.int8)


def _segments(mask):
    edges = np.diff(np.r_[False, mask, False].astype(np.int8))
    return list(zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)))


def _validate(pred, truth, blocks, fps):
    pred, truth = np.asarray(pred), np.asarray(truth)
    if pred.ndim != 1 or pred.shape != truth.shape or not np.isfinite(fps) or fps <= 0:
        raise ValueError('Status shape or fps invalid')
    if not np.isin(pred, [-1, 0, 1]).all() or not np.isin(truth, [-1, 0, 1]).all():
        raise ValueError('Statuses must be -1/0/1')
    last = 0
    for start, end in blocks:
        if start < last or end <= start or end > len(pred):
            raise ValueError('Invalid half-open period blocks')
        last = end
    return pred, truth


def valid_blocks(truth, blocks):
    for period, (start, end) in enumerate(blocks):
        for run, (lo, hi) in enumerate(_segments(truth[start:end] >= 0)):
            yield (period, run), int(start + lo), int(start + hi)


def events(status, start, end, direction=None):
    index = np.arange(start + 1, end)
    keep = (status[index - 1] >= 0) & (status[index] >= 0) & (status[index - 1] != status[index])
    if direction is not None:
        keep &= status[index] == direction
    return index[keep]


def boundary_intervals(status, start, end, direction=None):
    known = np.flatnonzero(status[start:end] >= 0) + start
    before, after = known[:-1], known[1:]
    keep = status[before] != status[after]
    if direction is not None: keep &= status[after] == direction
    return np.column_stack((before[keep]+1, after[keep])).astype(np.int64)


def ordered_match(reference, predicted, max_frames, predicted_lower=None):
    """Maximize match count, then minimize total displacement; deterministic ties."""
    reference, predicted = np.asarray(reference), np.asarray(predicted)
    lower = predicted if predicted_lower is None else np.asarray(predicted_lower)
    if lower.shape != predicted.shape or np.any(lower > predicted):
        raise ValueError('Invalid prediction boundary intervals')
    m, n = len(reference), len(predicted)
    count = np.zeros((m + 1, n + 1), dtype=np.int32)
    cost = np.zeros((m + 1, n + 1), dtype=float)
    back = np.zeros((m + 1, n + 1), dtype=np.int8)
    back[1:, 0], back[0, 1:] = 1, 2
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            choices = [(count[i - 1, j], cost[i - 1, j], 1),
                       (count[i, j - 1], cost[i, j - 1], 2)]
            delta = max(abs(int(reference[i - 1]) - int(predicted[j - 1])),
                        abs(int(reference[i - 1]) - int(lower[j - 1])))
            if delta <= max_frames + 1e-10:
                choices.append((count[i - 1, j - 1] + 1, cost[i - 1, j - 1] + delta, 3))
            best = min(choices, key=lambda x: (-x[0], x[1], -x[2]))
            count[i, j], cost[i, j], back[i, j] = best
    pairs, i, j = [], m, n
    while i or j:
        move = back[i, j]
        if move == 3:
            pairs.append((i - 1, j - 1)); i -= 1; j -= 1
        elif move == 1:
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


def associations(pred, truth, blocks, fps, tolerance, directions=(0, 1)):
    matched, ntrue, npred, ambiguous = {}, 0, 0, 0
    for key, start, end in valid_blocks(truth, blocks):
        for direction in directions:
            gt = events(truth, start, end, direction)
            intervals = boundary_intervals(pred, start, end, direction)
            lo, pr = intervals[:, 0], intervals[:, 1]
            ntrue += len(gt); npred += len(pr)
            if len(gt) and len(pr):
                feasible = np.maximum(abs(gt[:, None] - pr[None, :]), abs(gt[:, None] - lo[None, :])) <= tolerance * fps + 1e-10
                ambiguous += int(np.sum(feasible.sum(axis=0) > 1) + np.sum(feasible.sum(axis=1) > 1))
            for i, j in ordered_match(gt, pr, tolerance * fps, lo):
                matched[(*key, int(gt[i]), direction)] = (int(lo[j]), int(pr[j]))
    return matched, ntrue, npred, ambiguous


def _rates(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.
    recall = tp / (tp + fn) if tp + fn else 0.
    return dict(precision=precision, recall=recall,
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.,
                f0_5=1.25 * tp / (1.25 * tp + fp + .25 * fn) if 1.25 * tp + fp + .25 * fn else 0.)


def _error_summary(errors):
    errors = np.asarray(errors, float)
    return {'n': len(errors), **{key: float(fun(errors)) if len(errors) else None for key, fun in {
        'signed_mean_s': np.mean, 'signed_median_s': np.median,
        'mae_s': lambda x: np.mean(abs(x)), 'median_absolute_s': lambda x: np.median(abs(x)),
        'p90_absolute_s': lambda x: np.quantile(abs(x), .9)}.items()}}


def _worst_error(interval, reference, fps):
    lo, hi = interval
    return (lo-reference if abs(lo-reference) >= abs(hi-reference) else hi-reference) / fps


def binary_report(pred, truth, blocks, fps):
    pred, truth = _validate(pred, truth, blocks, fps)
    valid = np.zeros(len(truth), bool)
    for _, start, end in valid_blocks(truth, blocks): valid[start:end] = True
    tp = int(np.sum(valid & (truth == 1) & (pred == 1)))
    fp = int(np.sum(valid & (truth == 0) & (pred == 1)))
    fn = int(np.sum(valid & (truth == 1) & (pred != 1)))
    hours = float(valid.sum() / fps / 3600)
    curve = {}
    for tolerance in TOLERANCES:
        matched, nt, np_, ambiguous = associations(pred, truth, blocks, fps, tolerance)
        errors = [_worst_error(interval, key[2], fps) for key, interval in matched.items()]
        n = len(matched)
        curve[f'{tolerance:g}'] = {
            'tp': n, 'fp': np_ - n, 'fn': nt - n, 'reference_count': nt, 'predicted_count': np_,
            **_rates(n, np_ - n, nt - n), 'unpaired_matched_timing': _error_summary(errors),
            'false_boundaries_per_valid_hour': (np_ - n) / hours if hours else None,
            'ambiguous_feasible_nodes': ambiguous,
            'matched_uncertain_intervals': sum(lo != hi for lo,hi in matched.values()),
            'penalized_timing_loss_s': (sum(abs(x) for x in errors) + 2 * (nt + np_ - 2*n)) / max(1, nt)
                if tolerance == 2 else None}
        curve[f'{tolerance:g}']['directions'] = {}
        for direction, name in ((1, 'live_to_dead'), (0, 'dead_to_live')):
            dm = {key: value for key, value in matched.items() if key[3] == direction}
            dt = sum(len(events(truth, a, b, direction)) for _,a,b in valid_blocks(truth, blocks))
            dp = sum(len(boundary_intervals(pred, a, b, direction)) for _,a,b in valid_blocks(truth, blocks))
            errors_direction = [_worst_error(interval,key[2],fps) for key,interval in dm.items()]
            curve[f'{tolerance:g}']['directions'][name] = {
                'tp': len(dm), 'fp': dp-len(dm), 'fn': dt-len(dm),
                **_rates(len(dm), dp-len(dm), dt-len(dm)),
                'unpaired_matched_timing': _error_summary(errors_direction)}
    paired, _, _, _ = associations(pred, truth, blocks, fps, 2.)
    names, limits = ('le_0.5', '0.5_to_1', '1_to_2', '2_to_5', 'gt_5'), (.5, 1., 2., 5.)
    bins = {name: {'reference': 0, 'paired_detected': 0} for name in names}
    censored, false_dead_splits, false_live_splits, merged_dead = 0, 0, 0, 0
    for key, start, end in valid_blocks(truth, blocks):
        gruns = [(start+a, start+b) for a, b in _segments(truth[start:end] == 1)]
        pruns = [(start+a, start+b) for a, b in _segments(pred[start:end] == 1)]
        prunset = set(pruns)
        for a, b in gruns:
            if a == start or b == end:
                censored += 1; continue
            bucket = names[int(np.searchsorted(limits, (b-a)/fps, side='left'))]
            bins[bucket]['reference'] += 1
            pa, pb = paired.get((*key, a, 1)), paired.get((*key, b, 0))
            if pa is not None and pb is not None and (pa[1], pb[0]) in prunset:
                bins[bucket]['paired_detected'] += 1
        for a, b in pruns:
            if a > start and b < end and np.all(truth[a-1:b+1] == 0): false_dead_splits += 1
            merged_dead += max(0, sum(a < gb and b > ga for ga, gb in gruns) - 1)
        for a, b in _segments(pred[start:end] == 0):
            a, b = start+a, start+b
            if a > start and b < end and np.all(truth[a-1:b+1] == 1): false_live_splits += 1
    for item in bins.values():
        item['recall'] = item['paired_detected'] / item['reference'] if item['reference'] else None
    short_n = sum(bins[name]['reference'] for name in names[:3])
    short_tp = sum(bins[name]['paired_detected'] for name in names[:3])
    return {'status_representation': 'explicit -1 unknown / 0 live / 1 dead; no fill',
            'boundary_uncertainty': 'Opposite known states across X define [last_known+1,next_known]; worst endpoint matching/cost',
            'valid_hours': hours, 'valid_frames': int(valid.sum()),
            'known_prediction_fraction': float(np.mean(pred[valid] >= 0)) if valid.any() else None,
            'status_accuracy_unknown_wrong': float(np.mean(pred[valid] == truth[valid])) if valid.any() else None,
            'dead': {'tp': tp, 'fp': fp, 'fn': fn, **_rates(tp, fp, fn)}, 'curve': curve,
            'mean_curve_f0_5': float(np.mean([x['f0_5'] for x in curve.values()])),
            'complete_dead_spells': bins, 'censored_dead_spells': censored,
            'short_dead_recall_le_2s': short_tp / short_n if short_n else None,
            'short_dead_reference_count': short_n, 'false_dead_splits_in_live': false_dead_splits,
            'false_live_splits_in_dead': false_live_splits, 'merged_dead_spells': int(merged_dead)}


def paired_timing(old, new, truth, blocks, fps):
    old, truth = _validate(old, truth, blocks, fps)
    new, _ = _validate(new, truth, blocks, fps)
    om, total, _, _ = associations(old, truth, blocks, fps, 2.)
    nm, _, _, _ = associations(new, truth, blocks, fps, 2.)
    shared = om.keys() & nm.keys()
    common = sorted(k for k in shared if om[k][0] == om[k][1] and nm[k][0] == nm[k][1])
    oe = np.array([(om[k][0] - k[2]) / fps for k in common])
    ne = np.array([(nm[k][0] - k[2]) / fps for k in common])
    delta = abs(ne) - abs(oe)
    return {'cohort': 'same GT boundary IDs matched by both at 2 seconds, exact (non-X-gap) boundaries only', 'tolerance_s': 2.,
            'shared_matches_excluded_for_uncertain_prediction_interval': len(shared)-len(common),
            'common_count': len(common), 'old_only_count': len(om.keys() - nm.keys()),
            'new_only_count': len(nm.keys() - om.keys()), 'missed_by_both_count': total-len(om.keys() | nm.keys()),
            'old': _error_summary(oe), 'new': _error_summary(ne),
            'mean_absolute_error_change_s': float(delta.mean()) if len(delta) else None,
            'improved': int(np.sum(delta < -1e-9)), 'equal': int(np.sum(abs(delta) <= 1e-9)),
            'worse': int(np.sum(delta > 1e-9))}
