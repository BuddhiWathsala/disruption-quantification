import csv
import json
import os
import time
from collections import defaultdict, deque
import numpy as np


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


def simulate(suppliers, customers, industry, origins, severity, t0, duration,
             T=180, recovery_days=30, inv_days=15.0, substitution=0.0, headroom=0.1):
    firms = sorted(downstream(customers, origins))   # only these can be affected
    idx = {v: i for i, v in enumerate(firms)}
    n = len(firms)

    groups = []                                      # per firm: input groups by industry
    for v in firms:
        g = defaultdict(list)
        for u in suppliers[v]:
            g[industry[u]].append(idx.get(u, -1))    # -1 = supplier never affected
        groups.append(list(g.values()))
    inv = [np.full(len(g), float(inv_days)) for g in groups]

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
                d = (1 - substitution) * flow.mean() + substitution * flow.max()
                deliveries.append(d)
                limit = min(limit, d + inv[i][k])    # stock covers shortfalls
            P[i, t] = min(1.0, limit)
            for k, d in enumerate(deliveries):       # consume; headroom restocks
                inv[i][k] = np.clip(inv[i][k] + d * (1 + headroom) - P[i, t],
                                    0.0, inv_days)
    return firms, P


def outcomes(firms, P, t0, eps=0.01, theta=0.05):
    g = 1.0 - P                                      # gap, Eq. 1
    out = {}
    for i, v in enumerate(firms):
        above = np.nonzero(g[i, t0:] > eps)[0]
        out[v] = {"peak_loss": g[i, t0:].max(),                    # Eq. 2
                  "integrated_loss": g[i, t0:].sum(),              # Eq. 3
                  "ttr": 0 if above.size == 0 else above[-1] + 1}  # Eq. 4
    footprint = {v for v, y in out.items() if y["peak_loss"] > theta}  # Eq. 5
    return out, footprint


# ---------------------------------------------------------------------------
# Run all scenarios and save the results
# ---------------------------------------------------------------------------

KG_CSV = "data/supplychainKG.csv"
SCENARIOS_JSON = "data/multi_company_scenarios.json"
OUT_DIR = "data/simulated"

# The scenario file gives no severity or duration, so these are assumptions,
# applied to every scenario and recorded in the manifest.
PARAMS = {"severity": 0.4, "t0": 10, "duration": 60, "T": 180,
          "recovery_days": 30, "inv_days": 15.0, "substitution": 0.0,
          "headroom": 0.1, "eps": 0.01, "theta": 0.05}


def save_scenario(sid, firms, P, out, footprint):
    # Daily production p(v, t), compressed (mostly 1.0, so it stays small).
    np.savez_compressed(os.path.join(OUT_DIR, f"{sid}_production.npz"),
                        firms=np.array(firms), P=P)
    # Per-firm outcomes, Eqs. 2-5.
    with open(os.path.join(OUT_DIR, f"{sid}_outcomes.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["firm", "peak_loss", "integrated_loss", "ttr", "in_footprint"])
        for v in firms:
            y = out[v]
            w.writerow([v, f"{y['peak_loss']:.6f}", f"{y['integrated_loss']:.6f}",
                        int(y["ttr"]), v in footprint])


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    sup, cus, ind = load_kg(KG_CSV)
    with open(SCENARIOS_JSON, encoding="utf-8") as f:
        scenarios = json.load(f)

    sim = {k: PARAMS[k] for k in ("severity", "t0", "duration", "T", "recovery_days",
                                  "inv_days", "substitution", "headroom")}
    cache = {}       # the simulation depends only on the origins, not the focal firm
    manifest = []
    for sc in scenarios:
        sid = sc["scenario_id"]
        origins = sorted(set(sc["affected_companies"]))
        missing = [o for o in origins if o not in ind]
        known = [o for o in origins if o in ind]
        row = {"scenario_id": sid, "monitored_company": sc["monitored_company"],
               "origins": "; ".join(origins), "missing_origins": "; ".join(missing)}
        if not known:
            row["status"] = "skipped: no origin in KG"
            print(f"{sid}: skipped, origin(s) not in KG: {missing}")
            manifest.append(row)
            continue

        key = tuple(known)
        if key not in cache:
            t = time.time()
            firms, P = simulate(sup, cus, ind, list(key), **sim)
            out, footprint = outcomes(firms, P, PARAMS["t0"], PARAMS["eps"], PARAMS["theta"])
            cache[key] = (firms, P, out, footprint)
            print(f"{sid}: simulated {len(firms)} firms in {time.time() - t:.1f}s, "
                  f"footprint {len(footprint)}")
        else:
            print(f"{sid}: same origins as an earlier scenario, reusing its simulation")
        firms, P, out, footprint = cache[key]
        save_scenario(sid, firms, P, out, footprint)
        row.update(status="ok", simulated_firms=len(firms), footprint_size=len(footprint),
                   monitored_company_peak_loss=round(out.get(sc["monitored_company"],
                                                             {"peak_loss": 0.0})["peak_loss"], 6))
        manifest.append(row)

    fields = ["scenario_id", "monitored_company", "origins", "missing_origins", "status",
              "simulated_firms", "footprint_size", "monitored_company_peak_loss"]
    with open(os.path.join(OUT_DIR, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(manifest)
    with open(os.path.join(OUT_DIR, "params.json"), "w", encoding="utf-8") as f:
        json.dump(PARAMS, f, indent=2)
    print(f"Wrote {sum(r['status'] == 'ok' for r in manifest)} scenarios to {OUT_DIR}")


if __name__ == "__main__":
    main()

