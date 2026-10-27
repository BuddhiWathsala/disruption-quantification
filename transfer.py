import csv
import os

import numpy as np

import configs
from data_gen import RUNS_DIR

# Order of the impact vector y = (peak capacity, cumulative capacity, peak lead time,
# cumulative lead time degradation), i.e. the columns of B
OUTCOME_FIELDS = ("peak_loss", "integrated_loss", "peak_lead_time", "cum_lead_time")


def parse_symsplit_mapping(stdout):
    """
    Return the matching pi as [(source vertex, target vertex)], 0-based, from symsplit's
    stdout. The last two comma-terminated lines are the matched pattern and target vertices.
    """
    lists = [line for line in stdout.splitlines() if line.rstrip().endswith(",")]
    if len(lists) < 2:
        return []
    source, target = ([int(x) for x in line.split(",") if x.strip()] for line in lists[-2:])
    return list(zip(source, target))


def load_outcomes(run, directory=RUNS_DIR):
    """
    Read {firm: y} from the outcomes CSV of one simulation run (data_gen.run_id), y ordered
    as OUTCOME_FIELDS. Firms not in the file were not affected.
    """
    outcomes = {}
    with open(os.path.join(directory, f"{run}_outcomes.csv"), newline="",
              encoding="utf-8") as f:
        for r in csv.DictReader(f):
            outcomes[r["firm"]] = np.array([float(r[k]) for k in OUTCOME_FIELDS])
    return outcomes


def attribute_vector(features, names=configs.TRANSFER_ATTRIBUTES):
    """x = (inventory cover in days, substitute share) for one firm, or None if any is unknown."""
    if features is None or any(features.get(k) is None for k in names):
        return None
    return np.array([float(features[k]) for k in names])


def shortage(x, sigma, duration):
    """
    Shortage depth f = sigma (1 - s) and length D = max(duration - c, 0) of a firm with
    x = (c, s). Under this model y = (f, f D, f D, f D^2 / 2).
    """
    c, s = x
    return sigma * (1.0 - s), max(duration - c, 0.0)


def adaptation_matrix(x_s, duration):
    """
    Option 1 (Table adaptation-matrix): B = d log y / d x at the source node, with
    D = duration - c_S and substitute share s_S. Rows (cover per day, substitute share).
    A node whose cover outlasts the disruption has no shortage, so cover has no local slope.
    """
    _, D = shortage(x_s, 1.0, duration)
    _, s = x_s
    cover = [0.0, -1.0 / D, -1.0 / D, -2.0 / D] if D > 0 else [0.0] * len(OUTCOME_FIELDS)
    substitute = [-1.0 / max(1.0 - s, 1e-6)] * len(OUTCOME_FIELDS)
    return np.array([cover, substitute])


def transfer(y_s, x_s, x_t, B, sigma_s, sigma_t):
    """Equation (transfer): y_T = sigma_T / sigma_S * y_S * exp(B^T (x_T - x_S)), peak capped at 1."""
    y_t = (sigma_t / sigma_s) * y_s * np.exp(B.T @ (x_t - x_s))
    y_t[0] = min(y_t[0], 1.0)
    return y_t


def transfer_exact(y_s, x_s, x_t, sigma_s, sigma_t, duration_s, duration_t):
    """
    Exact Option 1 ratios: y_T = y_S * (f_T/f_S, f_T D_T/(f_S D_S), same, f_T D_T^2/(f_S D_S^2)).
    Unlike transfer(), this also accounts for a change in disruption duration. If the source
    had no shortage (D_S = 0) the duration ratio is unknown and left at 1.
    """
    f_s, D_s = shortage(x_s, sigma_s, duration_s)
    f_t, D_t = shortage(x_t, sigma_t, duration_t)
    depth = f_t / f_s if f_s > 0 else sigma_t / sigma_s
    length = D_t / D_s if D_s > 0 else 1.0
    y_t = y_s * depth * np.array([1.0, length, length, length ** 2])
    y_t[0] = min(y_t[0], 1.0)
    return y_t


def predict_outcomes(pi, source, target, idx_to_name, source_outcomes, sigma_s, sigma_t,
                     duration_s, duration_t, method="option1", B=None):
    """
    Predict y for every matched target firm. source/target are the Company graphs symsplit
    compared and pi comes from parse_symsplit_mapping. Returns {target firm: y_hat}.
    method: "option1" derives B per pair from the source node (adaptation_matrix),
    "exact" uses transfer_exact, "fixed" uses the given B (e.g. from fit_adaptation_matrix).
    A source firm missing from source_outcomes was never reached by the disruption, so y_S = 0.
    A firm with unknown attributes gets a zero attribute gap (severity scaling only).
    """
    predictions = {}
    for v, pv in pi:
        y_s = source_outcomes.get(idx_to_name[source.companies[v]], np.zeros(len(OUTCOME_FIELDS)))
        x_s = attribute_vector(source.features[v])
        x_t = attribute_vector(target.features[pv])
        if x_s is None or x_t is None:
            y_t = (sigma_t / sigma_s) * y_s
            y_t[0] = min(y_t[0], 1.0)
        elif method == "exact":
            y_t = transfer_exact(y_s, x_s, x_t, sigma_s, sigma_t, duration_s, duration_t)
        else:
            B_v = adaptation_matrix(x_s, duration_s) if method == "option1" else B
            y_t = transfer(y_s, x_s, x_t, B_v, sigma_s, sigma_t)
        predictions[idx_to_name[target.companies[pv]]] = y_t
    return predictions


def fit_adaptation_matrix(y_s, y_t, gaps, severity_ratios):
    """
    Least-squares estimate of B from matched pairs with known outcomes on both sides:
    log(y_T / (sigma_T / sigma_S * y_S)) = B^T (x_T - x_S), fitted per impact component.
    y_s, y_t: (m, 4); gaps: (m, d); severity_ratios: (m,). Pairs where either outcome
    component is 0 carry no information about that component and are skipped for it.
    """
    y_s, y_t, gaps = np.asarray(y_s), np.asarray(y_t), np.asarray(gaps)
    ratios = np.asarray(severity_ratios)
    B = np.zeros((gaps.shape[1], y_s.shape[1]))
    for j in range(y_s.shape[1]):
        ok = (y_s[:, j] > 0) & (y_t[:, j] > 0)
        if ok.sum() >= gaps.shape[1]:
            log_ratio = np.log(y_t[ok, j] / (ratios[ok] * y_s[ok, j]))
            B[:, j] = np.linalg.lstsq(gaps[ok], log_ratio, rcond=None)[0]
    return B
