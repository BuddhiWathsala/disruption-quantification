import csv
import json
import os
import sys
import time
from collections import defaultdict, deque
from itertools import product
import numpy as np

import configs


def load_kg(path):
    suppliers, customers, industry = defaultdict(set), defaultdict(set), {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            u, v = r["Supplier"], r["Customer"]
            industry[u], industry[v] = r["Supplier Industry"], r["Customer Industry"]
            if u != v:
                suppliers[v].add(u)
                customers[u].add(v)
    return suppliers, customers, industry


def downstream(customers, origins):
    seen, q = set(origins), deque(origins)
    while q:
        for v in customers[q.popleft()]:
            if v not in seen:
                seen.add(v)
                q.append(v)
    return seen


def sample_firm_attributes(firms, params):
    """
    Draw x = (inventory cover in days, substitute share) for every firm, uniformly within
    PARAMS ranges. Firm-level values are unobserved, so they are sampled once with a fixed
    seed and shared by every run, as a firm's protection does not depend on the disruption.
    """
    rng = np.random.default_rng(params["seed"])
    firms = sorted(firms)
    cover = rng.uniform(*params["inventory_cover"], len(firms))
    share = rng.uniform(*params["substitute_share"], len(firms))
    return {v: {"inventory_cover": c, "substitute_share": s}
            for v, c, s in zip(firms, cover, share)}


def simulate(suppliers, customers, industry, origins, severity, t0, duration, attributes,
             T=180, recovery_days=30, headroom=0.1):
    firms = sorted(downstream(customers, origins))   # only these can be affected
    idx = {v: i for i, v in enumerate(firms)}
    n = len(firms)
    cover = np.array([attributes[v]["inventory_cover"] for v in firms])
    share = np.array([attributes[v]["substitute_share"] for v in firms])

    groups = []                                      # per firm: input groups by industry
    for v in firms:
        g = defaultdict(list)
        for u in suppliers[v]:
            g[industry[u]].append(idx.get(u, -1))    # -1 = supplier never affected
        groups.append(list(g.values()))
    inv = [np.full(len(g), cover[i]) for i, g in enumerate(groups)]

    def capacity(t):                                 # shock, then linear recovery
        if t < t0:
            return 1.0
        if t < t0 + duration:
            return 1.0 - severity
        ramp = (t - t0 - duration + 1) / max(1, recovery_days)
        return min(1.0, 1.0 - severity + severity * ramp)

    origin_idx = [idx[o] for o in origins]
    P = np.ones((n, T))
    for t in range(1, T):
        prev = P[:, t - 1]                           # 1-day delivery lag
        cap = np.ones(n)
        cap[origin_idx] = capacity(t)
        for i in range(n):
            limit, deliveries = cap[i], []
            for k, members in enumerate(groups[i]):
                flow = np.array([prev[j] if j >= 0 else 1.0 for j in members])
                # a share s of each input comes from an unaffected alternative supplier
                d = (1 - share[i]) * flow.mean() + share[i]
                deliveries.append(d)
                limit = min(limit, d + inv[i][k])    # stock covers shortfalls
            P[i, t] = min(1.0, limit)
            for k, d in enumerate(deliveries):       # consume; headroom restocks
                inv[i][k] = np.clip(inv[i][k] + d * (1 + headroom) - P[i, t],
                                    0.0, cover[i])
    return firms, P


def outcomes(firms, P, t0, eps=0.01, theta=0.05):
    g = 1.0 - P                                      # gap, Eq. 1
    out = {}
    for i, v in enumerate(firms):
        above = np.nonzero(g[i, t0:] > eps)[0]
        ttr = 0 if above.size == 0 else above[-1] + 1
        # Unmet orders in days of normal output = extra wait of an order placed on day t
        backlog = np.cumsum(g[i, t0:])[:ttr]
        out[v] = {"peak_loss": g[i, t0:].max(),                    # Eq. 2
                  "integrated_loss": g[i, t0:].sum(),              # Eq. 3
                  "ttr": ttr,                                      # Eq. 4
                  "peak_lead_time": backlog.max() if ttr else 0.0,
                  "cum_lead_time": backlog.sum()}
    footprint = {v for v, y in out.items() if y["peak_loss"] > theta}  # Eq. 5
    return out, footprint


# ---------------------------------------------------------------------------
# Run all scenarios over the severity x duration grid and save the results
# ---------------------------------------------------------------------------

KG_CSV = "data/supplychainKG.csv"
SCENARIOS_JSON = "data/multi_company_scenarios.json"
OUT_DIR = "data/simulated"
RUNS_DIR = os.path.join(OUT_DIR, "runs")
ATTRIBUTES_CSV = os.path.join(OUT_DIR, "firm_attributes.csv")
PAIRS_CSV = os.path.join(OUT_DIR, "pairs.csv")

# The scenario file gives no severity or duration, so every scenario is simulated at every
# grid point. Firm attributes are drawn uniformly from the given ranges. All recorded in
# params.json.
PARAMS = {"severities": [0.2, 0.4, 0.6], "durations": [20, 40, 60],
          "inventory_cover": [0.0, 30.0], "substitute_share": [0.0, 0.5], "seed": 0,
          "t0": 10, "T": 180, "recovery_days": 30, "headroom": 0.1,
          "eps": 0.01, "theta": 0.05}


def run_id(scenario_id, severity, duration):
    """Name of one simulation: a scenario at one grid point, e.g. TSLA_SC002_s0.40_d60."""
    return f"{scenario_id}_s{severity:.2f}_d{int(duration)}"


def save_firm_attributes(attributes, path=ATTRIBUTES_CSV):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["firm", "inventory_cover", "substitute_share"])
        for v, x in sorted(attributes.items()):
            w.writerow([v, f"{x['inventory_cover']:.4f}", f"{x['substitute_share']:.4f}"])


def load_firm_attributes(path=ATTRIBUTES_CSV):
    """Read {firm: {"inventory_cover": c, "substitute_share": s}}."""
    with open(path, newline="", encoding="utf-8") as f:
        return {r.pop("firm"): {k: float(x) for k, x in r.items()} for r in csv.DictReader(f)}


def save_run(rid, firms, P, out, footprint):
    # Daily production p(v, t), compressed (mostly 1.0, so it stays small).
    np.savez_compressed(os.path.join(RUNS_DIR, f"{rid}_production.npz"),
                        firms=np.array(firms), P=P)
    # Per-firm outcomes, Eqs. 2-5. Only firms that lost output are written; a firm missing
    # from the file was not affected (all outcomes 0).
    with open(os.path.join(RUNS_DIR, f"{rid}_outcomes.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["firm", "peak_loss", "integrated_loss", "ttr", "peak_lead_time",
                    "cum_lead_time", "in_footprint"])
        for v in firms:
            y = out[v]
            if y["peak_loss"] > 0:
                w.writerow([v, f"{y['peak_loss']:.6f}", f"{y['integrated_loss']:.6f}",
                            int(y["ttr"]), f"{y['peak_lead_time']:.6f}",
                            f"{y['cum_lead_time']:.6f}", v in footprint])


def save_pairs(ok_scenarios, path=PAIRS_CSV, mapping_file=configs.MAPPING_FILE):
    """
    Source/target pairs for evaluating the transfer: for each row of the GT mapping, the
    similar (historical) disruption is the source and the case is the target, at every
    combination of source and target grid points. Pairs with a skipped scenario are left out.
    """
    grid = list(product(PARAMS["severities"], PARAMS["durations"]))
    rows = []
    with open(mapping_file, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            src, tgt = r["Similar Disruption"], r["Case"]
            if src not in ok_scenarios or tgt not in ok_scenarios:
                continue
            for (s_sev, s_dur), (t_sev, t_dur) in product(grid, grid):
                rows.append({"pair_id": len(rows),
                             "source_scenario": src, "source_severity": s_sev,
                             "source_duration": s_dur,
                             "source_run": run_id(src, s_sev, s_dur),
                             "target_scenario": tgt, "target_severity": t_sev,
                             "target_duration": t_dur,
                             "target_run": run_id(tgt, t_sev, t_dur)})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return len(rows)


def rebuild_outcomes():
    """Recompute every outcomes CSV from the saved production files, without re-simulating."""
    for name in sorted(os.listdir(RUNS_DIR)):
        if name.endswith("_production.npz"):
            rid = name[:-len("_production.npz")]
            data = np.load(os.path.join(RUNS_DIR, name))
            firms, P = list(data["firms"]), data["P"]
            out, footprint = outcomes(firms, P, PARAMS["t0"], PARAMS["eps"], PARAMS["theta"])
            save_run(rid, firms, P, out, footprint)
            print(f"{rid}: rebuilt outcomes for {len(firms)} firms")


def main():
    os.makedirs(RUNS_DIR, exist_ok=True)
    sup, cus, ind = load_kg(KG_CSV)
    with open(SCENARIOS_JSON, encoding="utf-8") as f:
        scenarios = json.load(f)
    attributes = sample_firm_attributes(ind, PARAMS)
    save_firm_attributes(attributes)

    sim = {k: PARAMS[k] for k in ("t0", "T", "recovery_days", "headroom")}
    grid = list(product(PARAMS["severities"], PARAMS["durations"]))
    # The simulation depends only on the origins, not the focal firm, so scenarios that
    # share origins are simulated once per grid point
    by_origins = defaultdict(list)
    manifest, ok_scenarios = [], set()
    for sc in scenarios:
        sid = sc["scenario_id"]
        origins = sorted(set(sc["affected_companies"]))
        missing = [o for o in origins if o not in ind]
        known = [o for o in origins if o in ind]
        base = {"scenario_id": sid, "monitored_company": sc["monitored_company"],
                "origins": "; ".join(origins), "missing_origins": "; ".join(missing)}
        if not known:
            manifest.append(dict(base, status="skipped: no origin in KG"))
            print(f"{sid}: skipped, origin(s) not in KG: {missing}")
            continue
        ok_scenarios.add(sid)
        by_origins[tuple(known)].append((sc, base))

    for known, group in by_origins.items():
        for severity, duration in grid:
            t = time.time()
            firms, P = simulate(sup, cus, ind, list(known), severity, duration=duration,
                                attributes=attributes, **sim)
            out, footprint = outcomes(firms, P, PARAMS["t0"], PARAMS["eps"], PARAMS["theta"])
            print(f"{'; '.join(known)} s={severity} d={duration}: simulated {len(firms)} "
                  f"firms in {time.time() - t:.1f}s, footprint {len(footprint)}")
            for sc, base in group:
                rid = run_id(sc["scenario_id"], severity, duration)
                save_run(rid, firms, P, out, footprint)
                peak = out.get(sc["monitored_company"], {"peak_loss": 0.0})["peak_loss"]
                manifest.append(dict(base, run_id=rid, severity=severity, duration=duration,
                                     status="ok", simulated_firms=len(firms),
                                     footprint_size=len(footprint),
                                     monitored_company_peak_loss=round(peak, 6)))

    fields = ["scenario_id", "run_id", "severity", "duration", "monitored_company", "origins",
              "missing_origins", "status", "simulated_firms", "footprint_size",
              "monitored_company_peak_loss"]
    with open(os.path.join(OUT_DIR, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(manifest)
    with open(os.path.join(OUT_DIR, "params.json"), "w", encoding="utf-8") as f:
        json.dump(PARAMS, f, indent=2)
    n_pairs = save_pairs(ok_scenarios)
    print(f"Wrote {sum(r['status'] == 'ok' for r in manifest)} runs to {RUNS_DIR}, "
          f"{n_pairs} source/target pairs to {PAIRS_CSV}")


if __name__ == "__main__":
    if "--outcomes-only" in sys.argv:
        rebuild_outcomes()
    else:
        main()
