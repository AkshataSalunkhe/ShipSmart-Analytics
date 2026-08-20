"""Build the self-contained dashboard.html from pre-computed results + figures.

Run this after the six scripts in scripts/ have populated figures/ and
data/ (see the top-level README's "How to run it yourself"). It reads the
PNGs those scripts saved and inlines them as base64 data URIs so
dashboard.html has zero external file dependencies.
"""
import base64, os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

_IMG_PATHS = {
    "seg_crosstab": "figures/02_segmentation/02_cluster_vs_tier.png",
    "seg_pca": "figures/02_segmentation/04_pca_scatter.png",
    "seg_revenue": "figures/02_segmentation/03_cluster_revenue.png",
    "wtp_elasticity": "figures/03_wtp/04_elasticity_curves.png",
    "wtp_revenue": "figures/03_wtp/05_revenue_curves.png",
    "wtp_roc": "figures/03_wtp/01_roc_curve.png",
    "churn_lift": "figures/04_churn/02_decile_lift.png",
    "churn_importance": "figures/04_churn/03_feature_importance.png",
    "churn_segment": "figures/04_churn/05_churn_by_segment.png",
    "ab_primary": "figures/05_ab/01_primary_results.png",
    "ab_segment": "figures/05_ab/02_segment_effects.png",
    "fc_segments": "figures/06_forecast/01_forecast_vs_actual.png",
    "fc_total": "figures/06_forecast/02_total_forecast.png",
}
IMG = {}
for key, rel_path in _IMG_PATHS.items():
    with open(os.path.join(ROOT, rel_path), "rb") as f:
        IMG[key] = base64.b64encode(f.read()).decode("ascii")

CSS = """
:root {
  --surface-1: #fcfcfb; --page: #f4f4f1; --text-primary: #0b0b0b; --text-secondary: #52514e;
  --muted: #898781; --grid: #e1e0d9; --baseline: #c3c2b7; --border: rgba(11,11,11,0.10);
  --blue: #2a78d6; --orange: #eb6834; --aqua: #1baf7a; --yellow: #eda100;
  --good: #0ca30c; --warning: #fab219; --serious: #ec835a; --critical: #d03b3b;
}
* { box-sizing: border-box; }
body {
  margin: 0; font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
  background: var(--page); color: var(--text-primary);
}
header {
  background: var(--surface-1); border-bottom: 1px solid var(--border);
  padding: 20px 32px; display: flex; align-items: baseline; justify-content: space-between; flex-wrap: wrap; gap: 8px;
}
header h1 { font-size: 19px; margin: 0; font-weight: 650; }
header p { margin: 2px 0 0; color: var(--text-secondary); font-size: 13px; }
nav {
  display: flex; gap: 4px; padding: 0 32px; background: var(--surface-1);
  border-bottom: 1px solid var(--border); overflow-x: auto;
}
nav button {
  border: none; background: transparent; padding: 12px 14px; font-size: 13.5px; font-weight: 550;
  color: var(--text-secondary); cursor: pointer; border-bottom: 2px solid transparent; white-space: nowrap;
}
nav button.active { color: var(--blue); border-bottom-color: var(--blue); }
nav button:hover { color: var(--text-primary); }
main { max-width: 1180px; margin: 0 auto; padding: 28px 32px 64px; }
section { display: none; }
section.active { display: block; }
.kpi-row { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr)); gap: 12px; margin-bottom: 24px; }
.kpi {
  background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; padding: 14px 16px;
}
.kpi .label { font-size: 11.5px; color: var(--muted); text-transform: uppercase; letter-spacing: .03em; }
.kpi .value { font-size: 22px; font-weight: 650; margin-top: 4px; }
.kpi .sub { font-size: 12px; color: var(--text-secondary); margin-top: 2px; }
h2 { font-size: 17px; margin: 28px 0 4px; }
h2:first-child { margin-top: 0; }
.lede { color: var(--text-secondary); font-size: 14px; max-width: 780px; margin: 0 0 18px; }
.card {
  background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px;
  padding: 18px 20px; margin-bottom: 18px;
}
.card h3 { margin: 0 0 4px; font-size: 14px; }
.card p { color: var(--text-secondary); font-size: 13px; margin: 4px 0 12px; }
.chart-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.chart-grid img { width: 100%; border-radius: 6px; border: 1px solid var(--grid); display: block; }
.chart-grid.single { grid-template-columns: 1fr; }
@media (max-width: 820px) { .chart-grid { grid-template-columns: 1fr; } }
.callout {
  border-left: 3px solid var(--blue); background: #eef4fc; border-radius: 0 8px 8px 0;
  padding: 12px 16px; font-size: 13.5px; color: var(--text-primary); margin: 14px 0;
}
.callout.warn { border-left-color: var(--warning); background: #fff8e8; }
.callout.critical { border-left-color: var(--critical); background: #fdeeee; }
.callout.good { border-left-color: var(--good); background: #ecf8ec; }
.callout b { font-weight: 650; }
table { border-collapse: collapse; width: 100%; font-size: 12.5px; margin: 10px 0; }
th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid var(--grid); }
th { color: var(--muted); font-weight: 600; text-transform: uppercase; font-size: 10.5px; letter-spacing: .03em; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
.badge { display: inline-block; padding: 2px 9px; border-radius: 999px; font-size: 11px; font-weight: 650; }
.badge.good { background: #ecf8ec; color: #006300; }
.badge.critical { background: #fdeeee; color: #a02222; }
.badge.warn { background: #fff3d6; color: #93650a; }
.module-list { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px,1fr)); gap: 14px; margin-top: 18px; }
.module-list .card h3 { color: var(--blue); }
footer { text-align: center; color: var(--muted); font-size: 12px; padding: 20px; }
"""

JS = """
function showTab(id) {
  document.querySelectorAll('section').forEach(s => s.classList.toggle('active', s.id === id));
  document.querySelectorAll('nav button').forEach(b => b.classList.toggle('active', b.dataset.tab === id));
  window.scrollTo({top:0, behavior:'instant'});
}
"""

html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ShipSmart Logistics -- Customer &amp; Pricing Analytics</title>
<style>{CSS}</style>
</head>
<body>

<header>
  <div>
    <h1>ShipSmart Logistics -- Customer &amp; Pricing Analytics</h1>
    <p>A synthetic-data walkthrough of price sensitivity, account grouping, retention risk, a live price test, and a revenue outlook</p>
  </div>
  <p style="align-self:center;">3,000 customers &middot; 92,530 quotes &middot; 5 notebooks</p>
</header>

<nav>
  <button class="active" data-tab="overview" onclick="showTab('overview')">Overview</button>
  <button data-tab="segmentation" onclick="showTab('segmentation')">Account Groups</button>
  <button data-tab="wtp" onclick="showTab('wtp')">Price Sensitivity</button>
  <button data-tab="churn" onclick="showTab('churn')">Retention Risk</button>
  <button data-tab="ab" onclick="showTab('ab')">Price Test</button>
  <button data-tab="forecast" onclick="showTab('forecast')">Revenue Outlook</button>
</nav>

<main>

<section id="overview" class="active">
  <h2>Project overview</h2>
  <p class="lede">ShipSmart is a fictional B2B freight/parcel shipper. This project builds out about
  5.5 years of quoting history for a realistic customer base, then walks through five connected
  analyses: grouping accounts by how they actually behave, estimating what each type of customer will
  pay before they walk away, catching accounts that are quietly slipping away, running a proper
  randomized test on a proposed price change before betting on it, and projecting monthly revenue by
  customer group. All data is synthetic (see <code>simulate.py</code>); every model, metric and chart
  on this page is a real output from the notebooks in this repo, not a mocked-up number.</p>

  <div class="kpi-row">
    <div class="kpi"><div class="label">Customers</div><div class="value">3,000</div><div class="sub">4 commercial tiers</div></div>
    <div class="kpi"><div class="label">Historical quotes</div><div class="value">92,530</div><div class="sub">2019 -- 2026</div></div>
    <div class="kpi"><div class="label">Acceptance rate</div><div class="value">92.2%</div><div class="sub">all-time average</div></div>
    <div class="kpi"><div class="label">Accepted revenue</div><div class="value">$16.95M</div><div class="sub">trailing history</div></div>
    <div class="kpi"><div class="label">12-mo drop-off rate</div><div class="value">10.6%</div><div class="sub">2,754 eligible accounts</div></div>
    <div class="kpi"><div class="label">Test-arm revenue gain</div><div class="value" style="color:var(--good)">+9.5%</div><div class="sub">p=0.025, guardrail tripped</div></div>
  </div>

  <h2>A quick tour of each notebook</h2>
  <div class="module-list">
    <div class="card"><h3>1. Account Groups</h3><p>RFM + K-Means clustering checked against the tiers sales already assigns (ARI 0.44); surfaces ~26% of accounts worth a second look.</p></div>
    <div class="card"><h3>2. Price Sensitivity</h3><p>Gradient-boosted accept/reject model &rarr; a demand curve per account group &rarr; a suggested price, capped so it never recommends something never actually tested.</p></div>
    <div class="card"><h3>3. Retention Risk</h3><p>Cutoff-based label design so nothing leaks; AUC 0.83, and the riskiest fifth of accounts covers 62% of the ones that actually go quiet.</p></div>
    <div class="card"><h3>4. Price Test</h3><p>Randomized rollout of the suggested price: revenue climbs, but so does the drop-off rate -- a guardrail catches a change that looks good on one number alone.</p></div>
    <div class="card"><h3>5. Revenue Outlook</h3><p>Lag-feature gradient boosting vs. a trend/seasonal regression vs. a plain seasonal repeat -- with a candid call on which one actually earns its keep here.</p></div>
  </div>
</section>

<section id="segmentation">
  <h2>Grouping Accounts by Behavior</h2>
  <p class="lede">Sales ops already assigns a commercial tier at signup. This checks whether clustering on
  actual behavior (order frequency, spend, acceptance rate, price ratio -- K-Means, k=4) lands on the same
  groups -- and where it doesn't.</p>
  <div class="kpi-row">
    <div class="kpi"><div class="label">Clusters (k)</div><div class="value">4</div></div>
    <div class="kpi"><div class="label">Adjusted Rand Index</div><div class="value">0.44</div><div class="sub">vs. assigned tier</div></div>
    <div class="kpi"><div class="label">Tier disagreement</div><div class="value">26.2%</div><div class="sub">of accounts</div></div>
  </div>
  <div class="chart-grid">
    <img src="data:image/png;base64,{IMG['seg_crosstab']}" alt="Cluster vs tier crosstab">
    <img src="data:image/png;base64,{IMG['seg_pca']}" alt="PCA scatter of segments">
  </div>
  <div class="callout">
    <b>Finding:</b> the "High-Value Core" cluster absorbs almost all Enterprise accounts plus most Mid-Market
    accounts, while "Steady Mid-Value" is mostly SMB with a Mid-Market tail -- i.e. tier boundaries blur in
    actual behavior. Accounts sitting in a cluster dominated by a different tier are good candidates for a
    manual account-tier review (upgrade candidates, mostly).
  </div>
</section>

<section id="wtp">
  <h2>Estimating Price Sensitivity</h2>
  <p class="lede">A gradient-boosted accept/reject model (AUC 0.605 held-out, time-based split) turned into a
  demand curve per account group by sweeping the quoted price at prediction time -- essentially reading off
  how much each group is willing to pay before they walk. Suggested prices are capped at the edge of price
  variation we've actually observed, instead of letting a tree model extrapolate into territory it's never seen.</p>
  <div class="chart-grid">
    <img src="data:image/png;base64,{IMG['wtp_elasticity']}" alt="Elasticity curves by segment">
    <img src="data:image/png;base64,{IMG['wtp_revenue']}" alt="Expected revenue curves">
  </div>
  <div class="card">
    <h3>Price recommendation by segment</h3>
    <table>
      <tr><th>Segment</th><th class="num">Current price ratio</th><th class="num">Recommended</th><th class="num">Implied move</th><th>Note</th></tr>
      <tr><td>Enterprise</td><td class="num">1.04</td><td class="num">1.17</td><td class="num">+12.5%</td><td><span class="badge warn">still rising at data edge</span></td></tr>
      <tr><td>Mid-Market</td><td class="num">0.99</td><td class="num">1.12</td><td class="num">+13.5%</td><td><span class="badge warn">still rising at data edge</span></td></tr>
      <tr><td>SMB</td><td class="num">0.97</td><td class="num">1.07</td><td class="num">+10.0%</td><td><span class="badge good">interior optimum found</span></td></tr>
      <tr><td>Occasional</td><td class="num">0.96</td><td class="num">1.07</td><td class="num">+11.8%</td><td><span class="badge good">interior optimum found</span></td></tr>
    </table>
  </div>
  <div class="callout warn">
    <b>Caveat carried into the A/B test:</b> for Enterprise and Mid-Market, expected revenue is still rising at
    the edge of the price range we've ever actually charged -- the model can't see past that, so "raise the
    price further" is not a safe conclusion without deliberately testing higher price points.
  </div>
</section>

<section id="churn">
  <h2>Flagging Accounts at Risk of Going Quiet</h2>
  <p class="lede">Predicts, from a fixed cutoff date, which active accounts stop booking shipments over the
  next 6 months. Every feature is computed strictly from data before that cutoff; the outcome it's predicting
  looks strictly after it -- so the model never gets to peek at the answer.</p>
  <div class="kpi-row">
    <div class="kpi"><div class="label">ROC-AUC</div><div class="value">0.83</div></div>
    <div class="kpi"><div class="label">PR-AUC</div><div class="value">0.32</div><div class="sub">base rate 10.6%</div></div>
    <div class="kpi"><div class="label">Top-20% capture</div><div class="value">62%</div><div class="sub">of accounts that actually went quiet</div></div>
  </div>
  <div class="chart-grid">
    <img src="data:image/png;base64,{IMG['churn_lift']}" alt="Decile lift chart">
    <img src="data:image/png;base64,{IMG['churn_importance']}" alt="Feature importance">
  </div>
  <div class="callout">
    <b>Action:</b> calling just the riskiest fifth of the book (20% of accounts) reaches 62% of the ones that
    actually go dark within 6 months -- a realistic outreach list for an account team, not "call everyone."
  </div>
</section>

<section id="ab">
  <h2>Testing a Price Change the Right Way</h2>
  <p class="lede">Randomized (by customer, stratified by account group) rollout of the suggested price vs.
  business-as-usual, simulated over a 90-day window with a guardrail agreed on before looking at results:
  don't ship if acceptance rate drops more than 3 points in any group.</p>
  <div class="kpi-row">
    <div class="kpi"><div class="label">Acceptance: control</div><div class="value">92.1%</div></div>
    <div class="kpi"><div class="label">Acceptance: treatment</div><div class="value" style="color:var(--critical)">87.4%</div><div class="sub">p&lt;0.001</div></div>
    <div class="kpi"><div class="label">Revenue/quote lift</div><div class="value" style="color:var(--good)">+9.5%</div><div class="sub">p=0.025</div></div>
    <div class="kpi"><div class="label">Guardrail</div><div class="value"><span class="badge critical">Breached</span></div><div class="sub">3 of 4 segments</div></div>
  </div>
  <div class="chart-grid">
    <img src="data:image/png;base64,{IMG['ab_primary']}" alt="Primary A/B results">
    <img src="data:image/png;base64,{IMG['ab_segment']}" alt="Segment-level A/B effects">
  </div>
  <div class="callout critical">
    <b>Recommendation:</b> revenue is up meaningfully, but the acceptance-rate guardrail trips in
    Mid-Market, SMB and Occasional -- and the retention model earlier in this dashboard shows acceptance
    rate is one of the strongest signals of an account going quiet, so today's guardrail trip is a plausible
    retention problem two or three quarters out. Ship the full move for <b>Enterprise only</b> (no guardrail
    issue there); re-test the other three groups at half the proposed move before rolling out further.
    Projected annual impact if the tested policy went company-wide is <b>+$355,822</b> -- but that number
    ignores the guardrail problem, which is exactly why it isn't rolled out as-is.
  </div>
</section>

<section id="forecast">
  <h2>Where Revenue Is Headed, by Account Group</h2>
  <p class="lede">Lag-feature gradient boosting (recursive, walk-forward) vs. a trend-plus-seasonality linear
  regression (no recursion needed) vs. simply repeating last year's number, backtested on the trailing 6
  months per account group.</p>
  <div class="kpi-row">
    <div class="kpi"><div class="label">Simple-repeat error</div><div class="value">8.7%</div><div class="sub">MAPE, avg across groups -- the bar to clear</div></div>
    <div class="kpi"><div class="label">Best fitted-model error</div><div class="value">16.2%</div><div class="sub">MAPE, better of the two models per group</div></div>
    <div class="kpi"><div class="label">History available</div><div class="value">~4 yrs</div><div class="sub">per group, monthly</div></div>
  </div>
  <div class="chart-grid single">
    <img src="data:image/png;base64,{IMG['fc_segments']}" alt="Forecast vs actual by segment">
  </div>
  <div class="chart-grid single">
    <img src="data:image/png;base64,{IMG['fc_total']}" alt="Company-level forecast">
  </div>
  <div class="callout warn">
    <b>Honest verdict:</b> neither fitted model reliably beats just repeating last year's number, with only
    ~4 years of monthly history to learn from -- the trend/seasonal regression is actively unstable on the
    smallest group (Occasional, ~198% error). The straightforward call is to treat the simple-repeat number
    as the one to trust today, keep both fitted models running alongside it for comparison, and revisit once
    12+ more months of history land. That's a data-volume problem, not a reason to reach for a fancier model.
  </div>
</section>

</main>

<footer>Synthetic-data project &middot; built end-to-end with pandas / scikit-learn / matplotlib &middot; see the notebooks/ folder for full methodology</footer>

<script>{JS}</script>
</body>
</html>
"""

out_path = os.path.join(HERE, "dashboard.html")
with open(out_path, "w") as f:
    f.write(html)
print(f"Wrote {out_path}  ({os.path.getsize(out_path)/1024:.0f} KB)")
