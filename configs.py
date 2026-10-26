MAPPING_FILE = "data/gt_mapping.csv"
TIER_FILE = "data/companies_with_tiers.csv"
COMPANY_TESLA = "Tesla Inc"
COMPANY_BMW = "Bayerische Motoren Werke AG"
COMPANY_MERC = "Mercedes-Benz Group AG"

SUPPLIER_ID = "Supplier"
SUPPLIER_INDUSTRY_ID = "Supplier Industry"
SUPPLIER_COUNTRY_ID = "Supplier Country"
CUSTOMER_ID = "Customer"
CUSTOMER_INDUSTRY_ID = "Customer Industry"
CUSTOMER_COUNTRY_ID = "Customer Country"
N_TIERS = 4
FEATURES_FILE = "data/company_features.csv"
# Column of FEATURES_FILE written as the "f <vertex> <value>" lines in DIMACS files
DIMACS_FEATURE = "pagerank"

# Knowledge transfer (eq:transfer)
TRANSFER_ATTRIBUTES = ["inventory_cover", "substitute_share"]   # columns of FEATURES_FILE, rows of B
# "option1": B from each source node (Table adaptation-matrix); "exact": exact Option 1 ratios;
# "fixed": ADAPTATION_MATRIX below (e.g. replaced with fit_adaptation_matrix estimates)
TRANSFER_METHOD = "option1"
# Fixed B (d x 4; columns peak cap., cum. cap., peak lead time, cum. lead time). These are the
# paper's example values for a source node with D = 15 days and s = 0.
ADAPTATION_MATRIX = [[0.0, -1 / 15, -1 / 15, -2 / 15],
                     [-1.0, -1.0, -1.0, -1.0]]
SOURCE_SEVERITY = 0.4   # severity data/simulated was generated with (data_gen.PARAMS)
TARGET_SEVERITY = 0.4   # severity of the new disruption
SOURCE_DURATION = 60    # duration (days) data/simulated was generated with (data_gen.PARAMS)
TARGET_DURATION = 60    # duration of the new disruption (only "exact" uses a change)
