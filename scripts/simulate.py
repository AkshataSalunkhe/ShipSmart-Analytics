"""
Shared "ground-truth" market-response simulator.

This module is the single source of truth for how a (fictional) shipper
responds to a quoted price. It is used both to generate the historical
quotes dataset and to generate the A/B pricing-experiment dataset, so the
two are consistent with the same underlying customer behavior -- exactly
like a real company's booking/pricing engine would be the shared
mechanism behind both historical data and a live experiment.

Nothing here is "the model" -- it is the simulated world. The WTP model,
churn model, etc. only ever see the *outcomes* (accept/reject, revenue),
never these parameters, just like a data scientist would in real life.
"""
import numpy as np

SEGMENTS = ["Enterprise", "Mid-Market", "SMB", "Occasional"]

SEGMENT_PARAMS = {
    # beta: sensitivity of accept-probability to price vs. competitor.
    #   higher beta -> more price sensitive (more elastic demand)
    # accept_base: baseline log-odds of acceptance when priced at parity
    #   with the competitor (price_ratio == 1)
    # target_markup: the price we *tend* to quote, as a ratio of the
    #   competitor's price, before random case-by-case variation
    # volume_lambda: average number of quotes per active year
    "Enterprise":  dict(beta_mean=2.6, beta_sd=0.5, accept_base=3.1, target_markup=1.05, volume_lambda=16, weight=0.08),
    "Mid-Market":  dict(beta_mean=4.6, beta_sd=0.9, accept_base=2.5, target_markup=0.99, volume_lambda=10, weight=0.35),
    "SMB":         dict(beta_mean=6.6, beta_sd=1.2, accept_base=2.2, target_markup=0.97, volume_lambda=5,  weight=0.40),
    "Occasional":  dict(beta_mean=8.8, beta_sd=1.6, accept_base=1.9, target_markup=0.95, volume_lambda=2,  weight=0.17),
}

MODE_BASE_RATE = {"Ground": 8.0, "Air": 22.0, "Ocean": 5.0}
INDUSTRIES = ["Retail", "Manufacturing", "Electronics", "Food & Beverage", "Pharma", "Apparel", "Automotive", "Industrial"]
REGIONS = ["Northeast", "Southeast", "Midwest", "West", "International"]


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def sample_segment(rng, n):
    weights = np.array([SEGMENT_PARAMS[s]["weight"] for s in SEGMENTS])
    weights = weights / weights.sum()
    return rng.choice(SEGMENTS, size=n, p=weights)


def sample_customer_beta(rng, segments):
    """Each customer gets their own latent price sensitivity, drawn around
    their segment's mean. This is what a segmentation model is trying to
    (imperfectly) recover from behavior."""
    betas = np.empty(len(segments), dtype=float)
    for seg in SEGMENTS:
        mask = segments == seg
        p = SEGMENT_PARAMS[seg]
        betas[mask] = rng.normal(p["beta_mean"], p["beta_sd"], size=mask.sum()).clip(0.5, None)
    return betas


def sample_base_cost(rng, weight_kg, distance_km, mode):
    rate = np.array([MODE_BASE_RATE[m] for m in mode])
    noise = rng.normal(1.0, 0.07, size=len(weight_kg)).clip(0.75, 1.3)
    cost = rate * (weight_kg ** 0.62) * (1 + distance_km / 2200.0) * noise
    return cost.clip(5, None)


def sample_competitor_price(rng, base_cost):
    margin = rng.normal(1.13, 0.05, size=len(base_cost)).clip(1.02, 1.35)
    return base_cost * margin


def sample_price_ratio(rng, segments, service_level, contract_type):
    """Ratio of OUR quoted price to the competitor's price for the same
    shipment. Centered on the segment's usual commercial strategy, plus
    real-world case-by-case variation (sales negotiation, urgency,
    capacity) that gives the historical data enough exogenous price
    variation to estimate elasticity from."""
    target = np.array([SEGMENT_PARAMS[s]["target_markup"] for s in segments])
    noise = rng.normal(0.0, 0.07, size=len(segments))
    express_bump = np.where(service_level == "Express", 0.03, 0.0)
    contract_discount = np.where(contract_type == "Contract", -0.02, 0.0)
    ratio = target + noise + express_bump + contract_discount
    return ratio.clip(0.65, 1.55)


def true_accept_prob(segments, price_ratio, service_level, customer_beta, contract_type):
    accept_base = np.array([SEGMENT_PARAMS[s]["accept_base"] for s in segments])
    # Express shipments are booked with more urgency -> less price sensitive.
    beta_eff = np.where(service_level == "Express", customer_beta * 0.65, customer_beta)
    # Contract customers have already negotiated -> a bit stickier.
    beta_eff = np.where(contract_type == "Contract", beta_eff * 0.85, beta_eff)
    logit = accept_base - beta_eff * (price_ratio - 1.0)
    return sigmoid(logit)
