# Core inventory planning

The app uses daily periodic review. Review interval R is one calendar day;
protection period P is product lead time L plus R. Inventory position equals
usable on-hand inventory. Outstanding purchase orders and backorders are not
modeled. All stock is assumed usable unless the snapshot explicitly indicates it
has already expired. Future expiration does not adjust the core policy in this phase.

For expected cumulative demand D(t), daily demand standard deviation sigma,
and normal service-level quantile z:

- Lead-time safety stock = z * sigma * sqrt(L).
- Reorder point = D(L) + lead-time safety stock.
- Planning safety stock = z * sigma * sqrt(P).
- Order-up-to target = D(P) + planning safety stock.
- Below target: Increase by ceil(target - inventory position).
- Within less than one whole unit of target, with no pre-arrival shortage: Maintain,
  quantity zero. Confirmed zero stock with a positive target still requires Increase.
- Above target: Reduce replenishment / run down excess by floor(position - target).
- Potential pre-arrival shortage = max(0, D(L) - usable on-hand).

The target is not rounded before comparison. Sub-unit differences are treated as operationally at target under the rule above. Reduction means defer replenishment and sell/use stock
normally, not disposal. A pre-arrival service gap may require earlier supply;
ordinary replenishment may arrive too late. Order quantities do not automatically
adjust for lost sales. The normal approximation is a planning assumption, not a
measured stockout probability or guaranteed service level.

## Input and calendar conventions

Missing, blank, and unparseable on-hand inventory stays unknown. Explicit zero
means confirmed zero stock. Negative/infinite on-hand stock is invalid. Unknown
or invalid stock has no action, quantity, excess or shortage, while supported
demand, safety-stock and target metrics remain available. Earlier stock snapshots
are never substituted for a missing latest snapshot. Missing lead time uses the
configured default and is marked as such; invalid nonpositive lead time blocks planning.

Demand rows are period totals, not rates or cumulative counters. A product must
represent one inventory pool. Duplicate dates, multiple locations, irregular
dates, business-day calendars, and subdaily/timezone-dependent observations block
inventory recommendations until the calendar or aggregation is explicitly resolved.

Daily and fixed multi-day/weekly labels denote periods ending on the labeled
calendar date. Monthly start/end labels denote the entire corresponding calendar
month. The inventory snapshot is assumed to be at the close of the last demand
period; `planning_as_of` is the start of the next period. Historical uploads are
therefore retrospective planning scenarios, not automatically current-day plans.
Expiration features retain the legacy observation-date convention in this phase.

Expected demand is uniform within each forecast period. Calendar overlap prorates
weekly/monthly totals using actual period lengths. The assumptions are explicit;
no missing days are silently filled with zero demand.

Historical variability assumes independent daily increments and a common daily
mean: mean_daily = sum(period demand) / sum(period days). Daily variance is
sum((period demand - mean_daily * period_days)^2 / period_days) / (n - 1).
For fixed k-day periods this equals sample_variance(period demand) / k. This avoids
inventing daily observations or treating monthly volume as daily volume. Trends,
seasonality and correlated errors can make this approximation inaccurate;
forecast-error calibration is outside this phase.

## Forecasting and outputs

Model selection uses exactly the requested display horizon for its existing
holdout scoring. An optional output horizon extends only the winning model's
forecast to cover P. Candidate families, eligibility, fitting, scores and ranking
are unchanged. The monthly PeriodIndex representation accepts month-start/end
date offsets. `future_forecast` and `forecast_output.csv` retain the displayed
horizon; `planning_forecast` and `planning_forecast.csv` provide additional coverage.
Changing display horizon can still change the selected model through the existing
holdout rule, but does not directly redefine the inventory protection period.

Unsupported calendars, negative/nonfinite forecasts, gaps or insufficient coverage
produce `planning_status=unavailable`, a `planning_issue`, and null actions and
quantities. No flat-forecast extrapolation substitutes for missing model coverage.

Core metrics retain full precision in CSV and AI context. `safety_stock` is a
compatibility alias for `lead_time_safety_stock`; `inventory_action` mirrors the
single canonical `recommended_action`. Priority is a simple display ordering:
3 for positive pre-arrival shortage, 2 for other Increase actions, 1 otherwise,
and null when planning is unavailable. It is not a risk probability.

Legacy spoilage/promotion diagnostics remain in a separate module. Their scores
do not set core actions, quantities or priority; stock-based spoilage outputs are
unavailable when inventory is unknown/invalid or the calendar is not daily.
The old relative `risk_level` remains labeled as legacy demand volatility.
Weighted spoilage scores, promotion decisions, FEFO/batch depletion, and incoming
shelf-life constraints are deferred to the separate perishability redesign.

The AI receives saved run settings, core calculation components, nullable
recommendations and planning issues. Starting a successful new analysis clears
the previous chat. Prompt instructions explain these fields; response validation
is outside this implementation.

## Checks

Run `MPLCONFIGDIR=/tmp/supply-chain-mpl .venv/bin/python -B -m unittest discover -s tests -v`.
Integration checks write forecasts and charts into temporary directories.

Dashboard, AI and accuracy CSV use selected-model holdout MAE/MAPE. Overall errors
are unweighted means across products, explicitly labeled; fitted historical errors
are not used for the displayed accuracy metric. Model selection itself is unchanged.
