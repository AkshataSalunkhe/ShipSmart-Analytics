# %% [markdown]
# # ShipSmart Logistics -- Synthetic Data Generation
#
# This notebook builds the synthetic dataset that the rest of the project
# runs on. "ShipSmart Logistics" is a fictional B2B freight/parcel shipper
# serving e-commerce, SMB, mid-market and enterprise accounts.
#
# We generate three linked tables:
#
# 1. **customers** -- one row per account, with a hidden ("true") segment
#    and price-sensitivity that no downstream model gets to see directly.
# 2. **quotes** -- one row per price quote (accepted or not), the atomic
#    unit for the willingness-to-pay model.
# 3. **monthly_revenue** -- accepted quotes rolled up to customer-segment x
#    month, for the forecasting notebook.
#
# A churn label is derived afterwards using a leakage-safe cutoff design
# (features only from data before a cutoff date, label from the 6 months
# after it) -- see the "Churn labels" section below.

# %%
import numpy as np
import pandas as pd
import sys, os

sys.path.append(os.path.dirname(os.path.abspath("__file__")))
from simulate import (
    SEGMENTS, SEGMENT_PARAMS, INDUSTRIES, REGIONS,
    sample_segment, sample_customer_beta, sample_base_cost,
    sample_competitor_price, sample_price_ratio, true_accept_prob,
)
from config import DATA_DIR, RANDOM_SEED

rng = np.random.default_rng(RANDOM_SEED)
TODAY = pd.Timestamp("2026-08-20")
N_CUSTOMERS = 3000

print(f"Simulating as of {TODAY.date()}, N_CUSTOMERS={N_CUSTOMERS}")

# %% [markdown]
# ## 1. Customers
#
# Each customer gets a segment, a signup date, an industry/region, a
# contract type, and a latent price-sensitivity (`true_beta`) drawn around
# their segment's mean. `true_beta` and `true_segment` are the "ground
# truth" -- in the segmentation notebook we pretend not to know them and
# try to recover similar groups from behavior alone.

# %%
customer_id = np.arange(1, N_CUSTOMERS + 1)
true_segment = sample_segment(rng, N_CUSTOMERS)
true_beta = sample_customer_beta(rng, true_segment)

signup_start = pd.Timestamp("2019-01-01")
signup_end = pd.Timestamp("2025-09-01")
signup_offset_days = rng.integers(0, (signup_end - signup_start).days, size=N_CUSTOMERS)
signup_date = signup_start + pd.to_timedelta(signup_offset_days, unit="D")

industry = rng.choice(INDUSTRIES, size=N_CUSTOMERS)
region = rng.choice(REGIONS, size=N_CUSTOMERS, p=[0.24, 0.22, 0.20, 0.24, 0.10])

# Enterprise/Mid-Market customers are more likely to be on negotiated
# contracts; SMB/Occasional are mostly spot-priced.
contract_prob = pd.Series(true_segment).map({
    "Enterprise": 0.85, "Mid-Market": 0.55, "SMB": 0.20, "Occasional": 0.04,
}).values
contract_type = np.where(rng.random(N_CUSTOMERS) < contract_prob, "Contract", "Spot")

customers = pd.DataFrame({
    "customer_id": customer_id,
    "true_segment": true_segment,
    "true_beta": true_beta.round(3),
    "signup_date": signup_date,
    "industry": industry,
    "region": region,
    "contract_type": contract_type,
})
customers.head()

# %% [markdown]
# ## 2. Quotes
#
# For every customer we simulate a stream of price quotes from signup date
# to today. Quote volume scales with the segment's typical shipping
# frequency and the customer's tenure. Each quote gets shipment
# characteristics (weight, distance, mode, service level), a market
# reference price (`competitor_price`), our quoted price, and whether the
# customer accepted it -- generated from the ground-truth response
# function in `simulate.py`.

# %%
years_active = ((TODAY - customers["signup_date"]).dt.days / 365.25).clip(lower=0.05)
volume_lambda = customers["true_segment"].map({s: SEGMENT_PARAMS[s]["volume_lambda"] for s in SEGMENTS})
expected_quotes = (volume_lambda * years_active).clip(lower=1)
n_quotes = rng.poisson(expected_quotes).clip(min=1)

print(f"Total quotes to generate: {n_quotes.sum():,}")

rep_idx = np.repeat(customers.index.values, n_quotes)
q_customer_id = customers["customer_id"].values[rep_idx]
q_segment = customers["true_segment"].values[rep_idx]
q_beta = customers["true_beta"].values[rep_idx]
q_contract = customers["contract_type"].values[rep_idx]
q_signup = customers["signup_date"].values[rep_idx]

n_rows = len(rep_idx)
# quote date: uniform between signup and today
q_signup_dt = pd.to_datetime(q_signup)
signup_days_before_today = np.clip((TODAY - q_signup_dt).days.to_numpy().astype(float), 1, None)
rand_offset = rng.random(n_rows) * signup_days_before_today
quote_date = q_signup_dt + pd.to_timedelta(rand_offset.astype(int), unit="D")

mode = rng.choice(["Ground", "Air", "Ocean"], size=n_rows, p=[0.60, 0.25, 0.15])
service_level = np.where(
    mode == "Ocean", "Standard",
    np.where(rng.random(n_rows) < 0.28, "Express", "Standard"),
)

weight_kg = rng.lognormal(mean=3.6, sigma=1.1, size=n_rows).clip(1, 20000)
distance_km = rng.lognormal(mean=6.3, sigma=0.9, size=n_rows).clip(20, 15000)

base_cost = sample_base_cost(rng, weight_kg, distance_km, mode)
competitor_price = sample_competitor_price(rng, base_cost)
price_ratio = sample_price_ratio(rng, q_segment, service_level, q_contract)
quoted_price = competitor_price * price_ratio

accept_prob = true_accept_prob(q_segment, price_ratio, service_level, q_beta, q_contract)
accepted = (rng.random(n_rows) < accept_prob).astype(int)

quotes = pd.DataFrame({
    "quote_id": np.arange(1, n_rows + 1),
    "customer_id": q_customer_id,
    "segment": q_segment,
    "quote_date": quote_date,
    "mode": mode,
    "service_level": service_level,
    "weight_kg": weight_kg.round(1),
    "distance_km": distance_km.round(1),
    "base_cost": base_cost.round(2),
    "competitor_price": competitor_price.round(2),
    "price_ratio": price_ratio.round(4),
    "quoted_price": quoted_price.round(2),
    "accepted": accepted,
})
quotes = quotes.sort_values(["customer_id", "quote_date"]).reset_index(drop=True)
quotes["quote_id"] = np.arange(1, len(quotes) + 1)
quotes.head()

# %% [markdown]
# ## 3. Churn labels (leakage-safe)
#
# We pick a **feature cutoff** 12 months before "today" and a **label
# window** of the 6 months right after it. Features are computed only
# from quotes strictly before the cutoff; the label looks strictly after
# it. A customer is only eligible for labeling if they were demonstrably
# active in the 12 months before the cutoff and had been a customer for
# at least 90 days by then -- this avoids labeling brand-new signups as
# "churned" just because they hadn't ordered yet.

# %%
FEATURE_CUTOFF = TODAY - pd.DateOffset(months=12)
LABEL_END = TODAY - pd.DateOffset(months=6)
PRIOR_START = FEATURE_CUTOFF - pd.DateOffset(months=12)

print(f"Prior window (features): {PRIOR_START.date()} -> {FEATURE_CUTOFF.date()}")
print(f"Label window (outcome):  {FEATURE_CUTOFF.date()} -> {LABEL_END.date()}")

prior = quotes[(quotes["quote_date"] >= PRIOR_START) & (quotes["quote_date"] < FEATURE_CUTOFF)].copy()
label_period = quotes[(quotes["quote_date"] >= FEATURE_CUTOFF) & (quotes["quote_date"] < LABEL_END)].copy()

# first/last 3 months of the prior window, for a price-gap trend feature
prior_early = prior[prior["quote_date"] < PRIOR_START + pd.DateOffset(months=3)]
prior_late = prior[prior["quote_date"] >= FEATURE_CUTOFF - pd.DateOffset(months=3)]

agg = prior.groupby("customer_id").agg(
    n_quotes_12m=("quote_id", "count"),
    n_accepted_12m=("accepted", "sum"),
    avg_price_ratio_12m=("price_ratio", "mean"),
    revenue_12m=("quoted_price", lambda s: (s * prior.loc[s.index, "accepted"]).sum()),
    express_share_12m=("service_level", lambda s: (s == "Express").mean()),
)
agg["acceptance_rate_12m"] = agg["n_accepted_12m"] / agg["n_quotes_12m"]

early_ratio = prior_early.groupby("customer_id")["price_ratio"].mean()
late_ratio = prior_late.groupby("customer_id")["price_ratio"].mean()
agg["price_ratio_trend"] = (late_ratio - early_ratio).reindex(agg.index)

had_activity_in_label = label_period.groupby("customer_id")["accepted"].sum()

churn = customers.copy()
churn["tenure_days_at_cutoff"] = (FEATURE_CUTOFF - churn["signup_date"]).dt.days
churn = churn.set_index("customer_id").join(agg).reset_index()

eligible = (
    (churn["tenure_days_at_cutoff"] >= 90)
    & (churn["n_accepted_12m"].fillna(0) >= 1)
)
churn = churn[eligible].copy()
churn["accepted_in_label_window"] = churn["customer_id"].map(had_activity_in_label).fillna(0)
churn["churned"] = (churn["accepted_in_label_window"] == 0).astype(int)

churn_labels = churn.drop(columns=["accepted_in_label_window"])
print(f"Eligible customers for churn modeling: {len(churn_labels):,}  |  churn rate: {churn_labels['churned'].mean():.1%}")
churn_labels.head()

# %% [markdown]
# ## 4. Monthly revenue panel (for forecasting)
#
# Accepted quotes rolled up to calendar month x segment, plus an
# overall total row's worth of columns. Used as the target series in the
# forecasting notebook.

# %%
accepted_quotes = quotes[quotes["accepted"] == 1].copy()
accepted_quotes["month"] = accepted_quotes["quote_date"].values.astype("datetime64[M]")

monthly = (
    accepted_quotes.groupby(["month", "segment"])
    .agg(revenue=("quoted_price", "sum"), shipments=("quote_id", "count"))
    .reset_index()
)
# keep a clean, mostly-ramped-up window
monthly = monthly[monthly["month"] >= "2021-06-01"].sort_values(["segment", "month"]).reset_index(drop=True)
monthly.head()

# %% [markdown]
# ## 5. Save

# %%
os.makedirs(DATA_DIR, exist_ok=True)
customers.to_csv(os.path.join(DATA_DIR, "customers.csv"), index=False)
quotes.to_csv(os.path.join(DATA_DIR, "quotes.csv"), index=False)
churn_labels.to_csv(os.path.join(DATA_DIR, "churn_labels.csv"), index=False)
monthly.to_csv(os.path.join(DATA_DIR, "monthly_revenue.csv"), index=False)

print("Saved:")
for f in ["customers.csv", "quotes.csv", "churn_labels.csv", "monthly_revenue.csv"]:
    path = os.path.join(DATA_DIR, f)
    print(f"  {f:22s} {os.path.getsize(path)/1024:8.1f} KB")

print("\nShapes:")
print(f"  customers:      {customers.shape}")
print(f"  quotes:         {quotes.shape}")
print(f"  churn_labels:   {churn_labels.shape}")
print(f"  monthly_revenue:{monthly.shape}")
print(f"\nOverall acceptance rate: {quotes['accepted'].mean():.1%}")
print(f"Total historical revenue (accepted quotes): ${accepted_quotes['quoted_price'].sum():,.0f}")
