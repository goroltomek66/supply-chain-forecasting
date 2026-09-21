"""Local explanations of app terminology, not product-specific planning answers.

Sources: forecasting_models (evaluation), inventory_planning (core policy),
app (chart range and labels), data_preprocessing and legacy_perishability.
Keep definitions beside aliases so changes in the application are easy to audit.
"""
import re


# Title, alternative names, beginner-facing explanation.
GLOSSARY = [
    ("Holdout MAPE", ("MAPE", "average holdout MAPE"),
     "Average forecast error expressed as a percentage of actual demand; lower is better. The app tests forecasts against recent observations withheld from fitting, then averages absolute error divided by actual demand × 100. Zero-demand observations are excluded from percentage error; if none remain, the code returns 0%, which does not prove a perfect forecast. Overall MAPE is an unweighted mean of product holdout MAPEs."),
    ("Holdout MAE", ("MAE",),
     "Average forecast error in demand units; lower is better. The app averages the absolute differences between holdout demand and its forecasts. Unlike MAPE, MAE is not a percentage and includes zero-demand observations."),
    ("AIC", ("Akaike information criterion",),
     "A model-comparison score that balances fit with model complexity; lower is preferred. The app uses AIC from the holdout training fit only after MAE and MAPE tie. It is not a percentage or a direct measure of forecast accuracy."),
    ("Holdout period", ("holdout", "holdout window", "validation points", "validation window", "holdout performance"),
     "Recent historical observations set aside to test a model on data it did not fit. The app chooses a window using the requested forecast horizon, roughly a quarter of the history, and available training history. The winner is then fitted on all history for the final forecast."),
    ("Forecast horizon", ("displayed forecast horizon",),
     "How many future observation periods appear in the forecast, not necessarily how many days. A weekly dataset uses weekly periods. The app separately extends the planning forecast when needed to cover lead time plus one review day; the displayed horizon is not the inventory target period."),
    ("Service level", (),
     "A planning setting that controls the size of the safety-stock buffer. The app converts it to a standard-normal z-score; increasing it increases safety stock when demand variability is positive. It does not guarantee that the same percentage of customer orders will be fulfilled."),
    ("Z-score", ("z score",),
     "The multiplier used to turn demand variability into safety stock. The app obtains it from the standard-normal distribution at the selected service level; it is not a product risk score."),
    ("Lead time", ("resolved lead time", "product-specific lead time", "default lead time", "lead time days"),
     "The assumed wait for replenishment, measured in calendar days. The app uses the product's latest lead time from uploaded data; the sidebar default fills missing lead times. It is a planning input, not evidence that a shipment is late."),
    ("Safety stock", ("lead-time safety stock",),
     "Extra inventory used as a buffer for demand variability. In this app, lead-time safety stock = z-score × daily demand standard deviation × square root of lead time in calendar days. It is added to expected lead-time demand to calculate the reorder point."),
    ("Planning safety stock", (),
     "The buffer for the full period the daily-review plan needs to cover. It equals z-score × daily demand standard deviation × square root of (lead time + one review day). This buffer is used in target inventory, rather than the shorter lead-time safety stock."),
    ("Reorder point", (),
     "An inventory reference level covering expected demand while replenishment is on its way, plus a buffer. Reorder point = expected demand during lead time + lead-time safety stock. The app's daily-review order quantity is based on target inventory, not just this reorder point."),
    ("Target inventory", ("target inventory level", "inventory target", "order-up-to target", "target stock"),
     "The inventory level the current plan aims toward. Target inventory = expected demand during the protection period + planning safety stock. The protection period is lead time plus one day. This is the current daily-review target, not a separately calculated monthly target."),
    ("Pre-arrival service gap", ("service gap", "pre-arrival shortage",),
     "A potential shortfall before ordinary replenishment can arrive. The app calculates the larger of zero and expected demand during lead time minus usable inventory on hand. It is a planning warning, not a guaranteed stockout or an exact stockout date."),
    ("Priority score", ("planning priority", "prioritization",),
     "A planning order of attention, not a stockout probability. The app assigns 3 when there is a pre-arrival service gap, 2 for an Increase action without that gap, and 1 for other ready plans. Assistant prioritization sorts by score first, then larger service gap, then product name for ties. Unavailable plans have no score."),
    ("Increase", (),
     "Build inventory toward the calculated target. The recommended quantity is the positive target-minus-usable-inventory gap rounded up to whole units, except for the app's Maintain tolerance. This does not assume that an order can be expedited or prevent a pre-arrival service gap."),
    ("Run down excess", ("reduce replenishment", "reduce replenishment run down excess", "reduce",),
     "Reduce or defer replenishment so existing stock can move toward the calculated target. Excess is usable inventory minus target, rounded down to whole units, subject to the Maintain tolerance. It does not mean dispose of inventory. Recommended adjustment and excess quantity describe the same excess and must not be subtracted twice."),
    ("Maintain", (),
     "No inventory adjustment is recommended. The app uses zero units when stock matches the target, or differs by less than one unit with no service gap. Confirmed zero stock with a positive target still requires an Increase."),
    ("Awaiting data", ("plans awaiting data", "unavailable", "planning issue",),
     "The app cannot safely produce an inventory recommendation from the available inputs. Missing or invalid stock, unsupported date frequency, multiple locations, or insufficient forecast coverage can cause this. Forecasts or some targets may still be available; missing inventory is not treated as zero."),
    ("Inventory status", ("unknown inventory", "invalid inventory", "known inventory",),
     "Whether the latest stock snapshot can be used. Numeric zero is confirmed zero stock; missing or unparseable stock is unknown, and negative or infinite stock is invalid. Unknown or invalid inventory prevents a quantity recommendation."),
    ("Planning status", ("ready",),
     "Whether the inputs support a complete recommendation. Ready means the app could calculate an action and quantity; unavailable means it withheld them and records the reason. Ready does not mean the product has no planning concern."),
    ("ARIMA", ("ARIMA order",),
     "A forecasting method that uses patterns in past demand and past prediction errors. Its order (p,d,q) specifies past-value terms, differencing, and past-error terms. The app evaluates p and q from 0 to 2 and d from 0 to 1 when history permits, using the same holdout ranking as other methods."),
    ("Holt Trend", ("Holt",),
     "A smoothing method that tracks both a demand level and a trend. The app estimates those components from historical demand and compares its holdout errors with the other candidates; it is not automatically selected just because demand rises."),
    ("Holt-Winters Seasonal", ("Holt Winters",),
     "A smoothing method for a level, trend, and repeating pattern. The app uses additive trend and seasonality with a four-observation seasonal cycle when enough history exists. Four observations do not necessarily mean four calendar days."),
    ("Simple Exponential Smoothing", ("SES",),
     "A forecasting method that updates a demand level, giving more recent observations greater weight. The app estimates its smoothing setting and compares holdout performance with other candidates. It has no explicit trend or seasonal component."),
    ("Model selection", ("selected model", "selected method", "candidate models evaluated", "candidate count",),
     "Choosing a forecast method separately for each product. The app ranks successful holdout evaluations by lowest MAE, then MAPE, then training-fit AIC; exact ties retain candidate order. It fits the best candidate on full history, trying the next ranked candidate if that fit fails. Candidate counts describe successful holdout evaluations, summed across products in the dashboard."),
    ("Runner-up model", ("runner up", "runner-up ARIMA order", "top candidates",),
     "The next holdout-ranked alternative after the selected model. The details table shows up to three leading holdout candidates. Losing candidates are not fitted on full history merely for display, so a good holdout score does not guarantee a successful full-history fit."),
    ("Forecast", ("forecasted demand", "future forecast",),
     "An estimate of future demand from the selected product model, not a guaranteed outcome. Final forecasts use a model fitted on all available product history. The inventory engine uses forecast demand over the relevant calendar-day planning windows."),
    ("Estimated range", ("forecast range", "forecast estimated range", "estimated planning range", "shaded band", "confidence interval", "forecast lower bound", "forecast upper bound"),
     "The shaded chart band illustrates uncertainty around the forecast. Its margin is 1.96 times the standard deviation of historical fitted errors, with historical demand variability as a fallback; the lower bound is clipped at zero. It is an approximate planning range, not a calibrated 95% forecast confidence interval or a stockout prediction."),
    ("Forecast quality", ("high confidence", "moderate confidence", "low confidence",),
     "A simple label based on holdout MAPE: below 10% is High confidence, 10% to below 20% is Moderate confidence, and 20% or more is Low confidence. These are dashboard labels, not statistical confidence levels or guarantees."),
    ("Review interval", ("daily review", "daily periodic review",),
     "How often the planning policy assumes inventory is reviewed. The app uses one calendar day and plans to cover lead time plus that day. It does not place orders or automatically run a new analysis each day."),
    ("Protection period", ("planning period",),
     "The window covered by target inventory: product lead time plus one calendar-day review interval. It is a planning horizon, not a predicted time until stock runs out."),
    ("Lead-time demand", ("expected demand during lead time",),
     "Forecast demand accumulated over the product's lead time. For weekly or monthly totals, the app apportions demand uniformly within each period using calendar-day overlap. It does not multiply a weekly or monthly observation directly by lead-time days."),
    ("Protection-period demand", ("expected demand during protection period",),
     "Forecast demand accumulated over lead time plus one review day. The app adds planning safety stock to this demand to calculate target inventory."),
    ("On-hand inventory", ("on hand", "current inventory", "inventory on hand",),
     "The latest recorded stock quantity for a product. Zero means confirmed zero stock; missing data means unknown. The quantity used for planning is usable stock, which can differ if the latest stock is already expired."),
    ("Usable inventory", ("usable on-hand inventory", "inventory position",),
     "Stock the app treats as available for planning. It uses the latest on-hand snapshot, treating already expired stock as unusable. Inventory position currently equals this usable stock; outstanding purchase orders and backorders are not modeled, and future expiry is not deducted from the core target calculation."),
    ("Recommended quantity", ("recommended qty", "recommended adjustment", "excess quantity",),
     "The whole-unit amount associated with the action, not a separate target. Increase rounds target minus usable stock up; Run down excess rounds usable stock minus target down; Maintain is zero. Unknown or invalid stock has no recommendation. The action determines the direction."),
    ("Daily average demand", ("daily mean demand",),
     "Historical demand per calendar day. The app divides total observed demand by the total days represented by those observations. This historical average is distinct from forecast demand during lead time."),
    ("Daily demand standard deviation", ("daily demand std", "daily demand std dev",),
     "An estimate of how much demand varies from day to day, used for safety stock. Daily data uses sample standard deviation. Longer observation periods are converted to a daily variability estimate assuming independent daily increments, with actual calendar lengths for months."),
    ("Demand volatility", ("risk level", "demand standard deviation", "demand std dev",),
     "How variable historical demand is. The legacy High/Medium/Low volatility filter compares product demand standard deviations with the dataset's 33rd and 67th percentiles. These relative labels are separate from service-gap priority and are not stockout probabilities."),
    ("Demand trend", ("trend detected", "seasonality detected",),
     "A description of change or repetition in historical demand. The overview compares the first and last total demand: above 1% is Rising, below -1% is Softening, otherwise Stable. Model details separately detect trend using slope and seasonality using lag-four correlation; these flags do not select the winner."),
    ("Forecast frequency", ("data frequency",),
     "The spacing of demand observations, such as daily, weekly, or monthly. Planning converts supported regular period totals into calendar-day windows; unsupported or irregular calendars cause inventory quantities to be withheld."),
    ("Spoilage risk", ("spoilage risk score", "perishable flag", "expiring before expected sale",),
     "A legacy diagnostic about possible expiry, not a confirmed loss. Its score combines expiry pressure, stock coverage relative to expiry, and projected surplus. These diagnostics do not set the core replenishment action or quantity and should not be interpreted as a probability."),
    ("Promotion", ("promo flag", "promo lift", "promo lift pct", "promo effect detected",),
     "An uploaded indicator that an observation involved a promotion. Legacy diagnostics compare average demand on flagged and unflagged observations; a lift above 5% with promotion observations is flagged. This is an association, not proof of causation, and does not adjust the core replenishment recommendation."),
    ("Cleaned data", ("standardized data", "standardized cleaned preview", "column detection",),
     "The uploaded data after the app maps recognized column names, parses dates and numbers, sorts observations, and fills missing demand within each product using interpolation and fallbacks. Missing inventory stays unknown, while missing lead times use the sidebar default."),
    ("Historical demand", ("actual demand", "raw uploaded data",),
     "Demand recorded in the uploaded history, as distinct from a model's predictions. Raw Uploaded Data shows the input; the standardized preview shows the cleaned values used by the pipeline. Historical fitted forecasts are not the holdout forecasts used for model selection."),
    ("Days to expire", ("expiration date", "shelf life", "shelf life days",),
     "The time remaining before the recorded expiry. The app subtracts the observation date from the expiration date, using shelf-life days when expiry is missing. This is relative to the uploaded observation, not today's wall-clock date. The latest snapshot with zero or fewer days remaining is treated as unusable stock."),
    ("Average forecast demand", ("avg forecast demand", "total forecast demand",),
     "A summary of the displayed future forecast: the average is per observation period, and the total adds all displayed periods. These summaries are distinct from the calendar-day lead-time and protection-period demand used for core inventory decisions."),
    ("Product", ("SKU", "products modeled", "products reviewed",),
     "An item identified by the uploaded product field. Forecasts and inventory plans are calculated separately for each product. Products Modeled and Products Reviewed count the products represented in the corresponding results."),
    ("Location", ("category",),
     "Descriptive fields from the upload: location identifies the stock location, and category groups products for filtering. The current core planner requires a single inventory pool per product and withholds recommendations when its history contains multiple locations; it does not optimize transfers."),
    ("Planning as of", (),
     "The start of the next planning period after the latest historical period ends. Planning windows are anchored to the uploaded history, not automatically advanced to today's date."),
]


def normalize(text):
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def terminology_answer(prompt):
    """Answer explicit, self-contained definitions; leave planning requests alone.

    Require the whole question to consist of known terms and definition wording.
    Product names, requests for values, and extra instructions stay in the existing
    router rather than being swallowed by a matching metric keyword.
    """
    text = normalize(prompt)
    if not re.match(r"^(?:(?:what|how) (?:is|are|does|do)\b|explain\b|define\b|why is (?:this |it )?called\b|(?:what is the )?difference between\b|compare\b)", text):
        return None
    matches = []
    aliases = sorted(((normalize(alias), index) for index, (title, names, _) in enumerate(GLOSSARY)
                      for alias in (title, *names)), key=lambda item: -len(item[0]))
    remaining = text
    for alias, index in aliases:
        pattern = r"\b" + re.escape(alias) + r"\b"
        for found in re.finditer(pattern, remaining):
            matches.append((found.start(), index))
        remaining = re.sub(pattern, lambda match: " " * len(match.group()), remaining)
    allowed = set("what is are does do explain define why this it called the a an mean means meaning of to me please in simple plain english terms words between difference different from and versus vs compare for beginners work how calculated computed used app application".split())
    if not matches or set(remaining.split()) - allowed:
        return None
    indices = list(dict.fromkeys(index for _, index in sorted(matches)))
    return "\n\n".join(f"**{GLOSSARY[index][0]}** — {GLOSSARY[index][2]}" for index in indices)
