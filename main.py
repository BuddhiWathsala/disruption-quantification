import csv
import json
import os
import pandas as pd

import configs
from configs import *
from features import load_company_features
from mcs import write_dimacs, run_symsplit
from transfer import parse_symsplit_mapping, load_outcomes, predict_outcomes
import numpy as np


class Company(object):
    def __init__(self, adjmat, companies, labels, indices, features):
        self.adjmat = adjmat
        self.companies = companies
        self.labels = labels
        self.indices = indices
        self.features = features  # features[i]: feature dict of vertex i, or None if unknown

class SupplyChain(object):
    def __init__(self, adjmat, companies, countries, industries, company_labels, company_features):
        self.adjmat = adjmat
        self.companies = companies
        self.countries = countries
        self.industries = industries
        self.company_labels = company_labels
        self.company_features = company_features  # company index -> feature dict


# Read GT Mapping
def read_mapping(filename):
    mapping = {}
    with open(filename) as f:
        for row in csv.DictReader(f):
            mapping[row["Case"]] = row["Similar Disruption"]
    return mapping


# Read and return the content of a disruption JSON file for a given filename.
def read_scenario(filename, directory="data/gt"):
    scenario = {}
    file_path = os.path.join(directory, filename + ".json")
    with open(file_path) as f:
        scenario = json.load(f)
    return scenario


def load_entire_supply_chain(filename) -> SupplyChain:
    dataset = []
    with open(filename) as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            dataset.append(row[0:6])
    extracted_cols = [SUPPLIER_ID, SUPPLIER_INDUSTRY_ID, SUPPLIER_COUNTRY_ID,
                      CUSTOMER_ID, CUSTOMER_INDUSTRY_ID, CUSTOMER_COUNTRY_ID]
    df = pd.DataFrame(dataset, columns=extracted_cols)
    countries = list(set(df[CUSTOMER_COUNTRY_ID].unique().tolist() + df[SUPPLIER_COUNTRY_ID].unique().tolist()))
    countries.sort()
    countries_map = {}
    for i in range(len(countries)):
        countries_map[countries[i]] = i

    industries = list(set(df[CUSTOMER_INDUSTRY_ID].unique().tolist() + df[SUPPLIER_INDUSTRY_ID].unique().tolist()))
    industries.sort()
    industries_map = {}
    for i in range(len(industries)):
        industries_map[industries[i]] = i+1

    companies = list(set(df[SUPPLIER_ID].unique().tolist() + df[CUSTOMER_ID].unique().tolist()))
    companies.sort()
    companies_map = {}
    company_labels = {}
    for i in range(len(companies)):
        companies_map[companies[i]] = i

    companies = pd.concat([
        df[[CUSTOMER_ID, CUSTOMER_INDUSTRY_ID]].set_axis(["company", "industry"], axis=1),
        df[[SUPPLIER_ID, SUPPLIER_INDUSTRY_ID]].set_axis(["company", "industry"], axis=1),
    ]).drop_duplicates()

    company_labels.update(
        zip(companies["company"].map(companies_map), companies["industry"].map(industries_map))
    )

    adjmat = [[] for i in range(len(companies))]
    for row in df.iterrows():
        adjmat[companies_map[row[1][SUPPLIER_ID]]].append(companies_map[row[1][CUSTOMER_ID]])
        adjmat[companies_map[row[1][CUSTOMER_ID]]].append(companies_map[row[1][SUPPLIER_ID]])
    features_by_name = load_company_features(configs.FEATURES_FILE)
    company_features = {i: features_by_name[name] for name, i in companies_map.items()
                        if name in features_by_name}
    return SupplyChain(adjmat, companies_map, countries_map, industries_map, company_labels,
                       company_features)

def origin_company_id(companies, name):
    if "BMW" in name:
        return companies["Bayerische Motoren Werke AG"]
    elif "MERC" in name:
        return companies["Mercedes-Benz Group AG"]
    return companies["Tesla Inc"]

def load_company_graph(supply_chain: SupplyChain, origin_company) -> Company:

    company_id = origin_company_id(supply_chain.companies, origin_company)
    # adjmat[i][j] = 1 means i supplies j
    n = len(supply_chain.adjmat)
    suppliers = [[] for _ in range(n)]
    for i in range(n):
        row = supply_chain.adjmat[i]
        for j in row:
            suppliers[j].append(i)


    # Collect every company reachable from company_id by following directed edges
    def reachable(_neighbours):
        r_visited = {company_id}
        queue = [company_id]
        while queue:
            node = queue.pop()
            for nb in _neighbours[node]:
                if nb not in r_visited:
                    r_visited.add(nb)
                    queue.append(nb)
        return r_visited

    visited = {company_id}
    visited |= reachable(suppliers)

    # Build the sub adjacency matrix, preserving the original index order
    sub_companies = sorted(visited)
    sub_companies_map = {}
    for i, j in enumerate(sub_companies):
        sub_companies_map[j] = i
    sub_adjmat = [[] for _ in range(len(sub_companies))]
    for i in range(len(sub_companies)):
        neighbours = supply_chain.adjmat[sub_companies[i]]
        intersection = list(set(neighbours) & set(sub_companies))
        sub_adjmat[i] = [sub_companies_map[item] for item in intersection]

    sub_labels = [supply_chain.company_labels[i] for i in sub_companies]
    sub_indices = {old: new for new, old in enumerate(sub_companies)}
    sub_features = [supply_chain.company_features.get(i) for i in sub_companies]
    return Company(sub_adjmat, sub_companies, sub_labels, sub_indices, sub_features)

def load_company_tiers(supply_chain: SupplyChain, company: Company, origin_name):
    visited = set()
    tier1 = set()
    tier2 = set()
    tier3 = set()
    tier4 = set()
    n = len(company.adjmat)
    origin_id = origin_company_id(supply_chain.companies, origin_name)
    origin_id = company.companies.index(origin_id)
    # Tier 1
    tier1 |= set(company.adjmat[origin_id])
    visited |= set(company.adjmat[origin_id])

    # Tier 2
    elements = set()
    for i in tier1:
        elements |= set(company.adjmat[i])
    tier2 = elements - visited
    visited |= tier2

    # Tier 3
    elements.clear()
    for i in tier2:
        elements |= set(company.adjmat[i])
    tier3 = elements - visited
    visited |= tier3

    # Tier 4
    elements.clear()
    for i in tier3:
        elements |= set(company.adjmat[i])
    tier4 = elements - visited
    visited |= tier4

    idx_to_company = {}
    for name, idx in supply_chain.companies.items():
        idx_to_company[idx] = name

    def get_companies_from_tiers(_supply_chain: SupplyChain, _company: Company, t_indices, _idx_to_company):
        companies = [_idx_to_company[o] for o in t_indices]
        return get_subgraph_for_companies(_supply_chain, companies)

    t1_subgraph = get_companies_from_tiers(supply_chain, company, tier1, idx_to_company)
    t2_subgraph = get_companies_from_tiers(supply_chain, company, tier2, idx_to_company)
    t3_subgraph = get_companies_from_tiers(supply_chain, company, tier3, idx_to_company)
    t4_subgraph = get_companies_from_tiers(supply_chain, company, tier4, idx_to_company)

    return [t1_subgraph, t2_subgraph, t3_subgraph, t4_subgraph]


def get_subgraph_for_companies(supply_chain: SupplyChain, selected_companies):
    selected_company_indices = []
    selected_labels = []
    selected_features = []
    for company in selected_companies:
        c_id = supply_chain.companies[company]
        selected_company_indices.append(c_id)
        selected_labels.append(supply_chain.company_labels[c_id])
        selected_features.append(supply_chain.company_features.get(c_id))

    def extract_subgraph(adjmat, vertices):
        sub_n = len(vertices)
        _subgraph_adjmat = [[] for _ in range(sub_n)]
        _o2c_index = {}
        for i in range(sub_n):
            _o2c_index[vertices[i]] = i
        for i in range(sub_n):
            neighbours = list(set(adjmat[vertices[i]]) & set(vertices))
            _subgraph_adjmat[i] = [vertices.index(o) for o in neighbours]
        return _subgraph_adjmat, _o2c_index

    subgraph_adjmat, o2c_index = extract_subgraph(supply_chain.adjmat, selected_company_indices)
    return Company(subgraph_adjmat, selected_company_indices, selected_labels, o2c_index,
                   selected_features)


def get_tier_subgraphs_from_json(supply_chain: SupplyChain, scenario):

    def find_tier_companies(sc, tier_no):
        tier_name = ""
        companies = set()
        if tier_no == 1:
            tier_name = "tier_1"
        elif tier_no == 2:
            tier_name = "tier_2"
        elif tier_no == 3:
            tier_name = "tier_3"
        elif tier_no == 4:
            tier_name = "tier_4"
        else:
            return companies
        paths = []
        for i in range(N_TIERS):
            paths += scenario["kg_results"][f"tier_{i+1}"]
        for path in paths:
            if len(path) > tier_no:
                companies.add(path[tier_no]["company"])
        return list(companies)
    tier1_companies = find_tier_companies(scenario, tier_no=1)
    tier2_companies = find_tier_companies(scenario, tier_no=2)
    tier3_companies = find_tier_companies(scenario, tier_no=3)
    tier4_companies = find_tier_companies(scenario, tier_no=4)

    t1_subgraph = get_subgraph_for_companies(supply_chain, tier1_companies)
    t2_subgraph = get_subgraph_for_companies(supply_chain, tier2_companies)
    t3_subgraph = get_subgraph_for_companies(supply_chain, tier3_companies)
    t4_subgraph = get_subgraph_for_companies(supply_chain, tier4_companies)

    return [t1_subgraph, t2_subgraph, t3_subgraph, t4_subgraph]




def __main__():
    mapping = read_mapping(configs.MAPPING_FILE)
    test_file = "TSLA_SC002"
    company_name = configs.COMPANY_TESLA
    scenario = read_scenario(test_file, directory="data\\gt")
    supply_chain = load_entire_supply_chain("data/supplychainKG.csv")
    # Get tier graphs from history
    history_tier_subgraphs = get_tier_subgraphs_from_json(supply_chain, scenario)


    # Get tier graphs from company graph
    company = load_company_graph(supply_chain, origin_company=test_file)
    origin_id = origin_company_id(supply_chain.companies, test_file)
    current_tier_subgraphs = load_company_tiers(supply_chain, company, company_name)

    idx_to_name = {idx: name for name, idx in supply_chain.companies.items()}
    source_outcomes = load_outcomes(test_file)
    B = np.array(configs.ADAPTATION_MATRIX)

    # Compare one-by-one history_tier_subgraphs vs current_tier_subgraphs
    n = min(len(current_tier_subgraphs), len(history_tier_subgraphs))
    for i in range(n):
        p_company = history_tier_subgraphs[i]
        t_company = current_tier_subgraphs[i]
        p_file = write_dimacs(p_company)
        t_file = write_dimacs(t_company)
        _, stdout, stderr = run_symsplit(p_file, t_file)
        print(stdout)
        print(stderr)
        # Transfer the historical outcomes along the matching
        pi = parse_symsplit_mapping(stdout)
        predictions = predict_outcomes(pi, p_company, t_company, idx_to_name, source_outcomes,
                                       configs.SOURCE_SEVERITY, configs.TARGET_SEVERITY,
                                       configs.SOURCE_DURATION, configs.TARGET_DURATION,
                                       configs.TRANSFER_METHOD, B)
        print(f"Tier {i + 1}: {len(predictions)} predicted firms")
        for firm, y in sorted(predictions.items(), key=lambda kv: -kv[1][0])[:5]:
            print(f"  {firm}: peak cap.={y[0]:.3f} cum. cap.={y[1]:.3f} "
                  f"peak lead time={y[2]:.1f} cum. lead time={y[3]:.1f}")
    print()
__main__()