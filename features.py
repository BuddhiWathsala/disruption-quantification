import csv
from collections import defaultdict

import numpy as np

import configs
from data_gen import load_firm_attributes

KG_CSV = "data/supplychainKG.csv"
FEATURES_CSV = "data/company_features.csv"

FIELDS = ["company", "industry", "country", "majority_tier",
          "n_suppliers", "n_customers", "n_supplier_industries", "n_single_sourced_inputs",
          "n_supplier_countries", "supplier_country_hhi", "pagerank",
          "inventory_cover", "substitute_share"]
INT_FIELDS = {"majority_tier", "n_suppliers", "n_customers", "n_supplier_industries",
              "n_single_sourced_inputs", "n_supplier_countries"}
FLOAT_FIELDS = {"supplier_country_hhi", "pagerank", "inventory_cover", "substitute_share"}


def _pagerank(nodes, edges, damping=0.85, iters=100, tol=1e-10):
    """PageRank over directed edges (u, v); dangling mass is spread uniformly."""
    idx = {v: i for i, v in enumerate(nodes)}
    n = len(nodes)
    src = np.array([idx[u] for u, _ in edges])
    dst = np.array([idx[v] for _, v in edges])
    out_deg = np.bincount(src, minlength=n).astype(float)
    r = np.full(n, 1.0 / n)
    for _ in range(iters):
        contrib = np.zeros(n)
        np.add.at(contrib, dst, r[src] / out_deg[src])
        dangling = r[out_deg == 0].sum()
        new = (1 - damping) / n + damping * (contrib + dangling / n)
        if np.abs(new - r).sum() < tol:
            return dict(zip(nodes, new))
        r = new
    return dict(zip(nodes, r))


def build_company_features(kg_path=KG_CSV, tier_path=configs.TIER_FILE, out_path=FEATURES_CSV):
    """Compute per-company graph features from the supply chain KG and write them to a CSV."""
    suppliers, customers = defaultdict(set), defaultdict(set)
    industry, country = {}, {}
    with open(kg_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            u, v = r[configs.SUPPLIER_ID], r[configs.CUSTOMER_ID]
            industry[u], country[u] = r[configs.SUPPLIER_INDUSTRY_ID], r[configs.SUPPLIER_COUNTRY_ID]
            industry[v], country[v] = r[configs.CUSTOMER_INDUSTRY_ID], r[configs.CUSTOMER_COUNTRY_ID]
            if u != v:
                suppliers[v].add(u)
                customers[u].add(v)

    tiers = {}
    with open(tier_path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            tiers[r["Company"]] = r["Majority Tier"]

    # Reverse edges (customer -> supplier): a supplier ranks high when it feeds important firms
    companies = sorted(industry)
    edges = [(v, u) for v in companies for u in suppliers[v]]
    pagerank = _pagerank(companies, edges)
    attributes = load_firm_attributes()   # sampled by data_gen.py; run it first

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        for c in companies:
            sups = suppliers[c]
            by_industry = defaultdict(int)
            by_country = defaultdict(int)
            for s in sups:
                by_industry[industry[s]] += 1
                by_country[country[s]] += 1
            hhi = sum((k / len(sups)) ** 2 for k in by_country.values()) if sups else 0.0
            w.writerow({
                "company": c,
                "industry": industry[c],
                "country": country[c],
                "majority_tier": tiers.get(c, ""),
                "n_suppliers": len(sups),
                "n_customers": len(customers[c]),
                "n_supplier_industries": len(by_industry),
                "n_single_sourced_inputs": sum(1 for k in by_industry.values() if k == 1),
                "n_supplier_countries": len(by_country),
                "supplier_country_hhi": f"{hhi:.6f}",
                "pagerank": f"{pagerank[c]:.8e}",
                # No firm-level data: use the values data/simulated was generated with
                "inventory_cover": f"{attributes[c]['inventory_cover']:.4f}",
                "substitute_share": f"{attributes[c]['substitute_share']:.4f}",
            })
    return out_path


def load_company_features(path=FEATURES_CSV):
    """Read the features CSV and return {company name: {feature: value}}."""
    features = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            name = r.pop("company")
            for k in INT_FIELDS:
                r[k] = int(r[k]) if r[k] != "" else None   # majority_tier is blank for untiered firms
            for k in FLOAT_FIELDS:
                r[k] = float(r[k])
            features[name] = r
    return features


if __name__ == "__main__":
    print(f"Wrote {build_company_features()}")
