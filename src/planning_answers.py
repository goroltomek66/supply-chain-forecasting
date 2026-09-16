"""Deterministic display text for calculated planning facts. No model calls."""
from __future__ import annotations


LABELS = {
    "reorder_point": "reorder point",
    "on_hand_inventory": "current inventory",
    "expected_demand_lead_time": "expected demand during lead time",
    "lead_time_safety_stock": "lead-time safety stock",
    "order_up_to_target": "target inventory",
    "expected_demand_protection_period": "expected demand during the protection period",
    "planning_safety_stock": "planning safety stock",
    "product_mape": "holdout MAPE", "product_mae": "holdout MAE",
    "pre_arrival_shortage": "pre-arrival service gap",
}


def number(value, decimals=0):
    """Rounded presentation only; small nonzero differences retain useful precision."""
    if value is None:
        return "unavailable"
    if decimals == 0 and 0 < abs(value) < 1:
        return f"{value:,.2f}"
    return f"{value:,.{decimals}f}"


def action_text(row):
    action, quantity = row.get("recommended_action"), row.get("recommended_quantity_adjustment")
    if action is None or quantity is None:
        return "Recommendation unavailable; the current analysis lacks valid planning inputs."
    if action == "Increase":
        return f"Increase inventory by {number(quantity)} units."
    if action == "Reduce replenishment / run down excess":
        return f"Run down {number(quantity)} units of excess by reducing or deferring replenishment."
    if action == "Maintain":
        return "Maintain current inventory; no adjustment is recommended."
    return "Recommendation unavailable; the calculated action is not recognized."


def inventory_answer(name, row, explain=False):
    stock = row.get("on_hand_inventory", row.get("usable_on_hand_inventory"))
    target = row.get("order_up_to_target")
    text = f"{name} currently has {number(stock)} units on hand. " if stock is not None else f"{name}'s current inventory is unknown. "
    usable = row.get("usable_on_hand_inventory")
    if stock is not None and usable is not None and usable != stock:
        text += f"Of these, {number(usable)} units are usable. "
    if target is not None:
        text += f"The calculated target is about {number(target)} units. "
    else:
        text += "The calculated target is unavailable. "
    text += action_text(row)
    period = row.get("protection_period_days")
    if period is not None:
        text += f" This target covers {period:g} calendar days under daily review (lead time plus one day)."
    gap = row.get("pre_arrival_shortage")
    if gap is not None and gap > 0:
        text += f" Potential pre-arrival service gap: {number(gap)} units; ordinary replenishment may arrive too late."
    if explain:
        score = row.get("priority_score")
        if score is not None:
            text += f" Calculated priority score: {number(score)} (higher scores take priority)."
        if row.get("reorder_point") is not None:
            text += (f" Reorder point: {number(row['reorder_point'])} units, composed of "
                     f"{number(row.get('expected_demand_lead_time'))} units of expected demand during lead time "
                     f"and {number(row.get('lead_time_safety_stock'))} units of lead-time safety stock.")
    return text


def comparison_answer(payload):
    comparison = payload["comparison"]
    a, b = comparison["first_product"], comparison["second_product"]
    differences = comparison["component_differences"]
    total, *components = differences
    rows = payload["products_by_name"]
    lines = []
    for name in [a, b]:
        row = rows[name]
        lines.append(
            f"{name}: {LABELS[total]} is about {number(row[total])} units = "
            f"{number(row[components[0]])} units of {LABELS[components[0]]} + "
            f"{number(row[components[1]])} units of {LABELS[components[1]]}."
        )
    difference = differences[total]
    if difference == 0:
        lines.append(f"The calculated {LABELS[total]} values are equal.")
    else:
        higher, lower = (a, b) if difference > 0 else (b, a)
        lines.append(f"{higher}'s {LABELS[total]} is {number(abs(difference))} units higher than {lower}'s.")
    for field in components:
        delta = differences[field]
        if delta == 0:
            lines.append(f"Their {LABELS[field]} values are equal, so this component contributes no difference.")
        else:
            higher, lower = (a, b) if delta > 0 else (b, a)
            lines.append(f"{higher} has {number(abs(delta))} units more {LABELS[field]} than {lower}; this raises its {LABELS[total]} relative to the other product.")
    lines.append("Component values are rounded for display; the comparison uses the unrounded calculations.")
    return "\n\n".join(lines)


def model_answer(name, row):
    model = row.get("selected_method") or "unavailable"
    order = row.get("selected_arima_order")
    if model == "ARIMA" and order:
        model += " " + order
    runner = row.get("runner_up_model") or "unavailable"
    if runner == "ARIMA" and row.get("runner_up_arima_order"):
        runner += " " + row["runner_up_arima_order"]
    mape = row.get("product_mape")
    accuracy = number(mape, 2) + "%" if mape is not None else "unavailable"
    return f"{name} — {model}. Holdout MAPE: {accuracy}; holdout MAE: {number(row.get('product_mae'), 2)}. Runner-up: {runner}."


def assessment_answer(payload):
    """Summarize all calculated rows; prioritize service gaps without inventing causes."""
    rows, counts = payload["products_by_name"], payload["counts"]
    ranked = payload["priority_order"]
    concerns = [name for name in ranked if (rows[name].get("pre_arrival_shortage") or 0) > 0
                or rows[name].get("recommended_action") == "Increase"]
    parts = []
    if concerns:
        first = concerns[0]
        parts.append(f"{first} is the first planning concern. Priority follows calculated priority score, then pre-arrival service gap; remaining ties use product name.")
        for name in concerns[:3]:
            row = rows[name]
            text = f"{name}: " + action_text(row) + f" Priority score: {number(row.get('priority_score'))}."
            gap = row.get("pre_arrival_shortage")
            if gap is not None and gap > 0:
                text += f" Pre-arrival service gap: {number(gap)} units; ordinary replenishment may arrive too late."
            parts.append(text)
        parts.append("Focus first on " + ", then ".join(concerns[:3]) + ".")
    elif counts["increase_actions"] or counts["pre_arrival_service_gaps"]:
        parts.append("Increases or service gaps are calculated, but priority inputs are missing; a reliable attention order is unavailable.")
    else:
        parts.append("No inventory increases or pre-arrival service gaps are identified in the available calculations.")
    excess = [name for name, row in rows.items() if row.get("recommended_action") == "Reduce replenishment / run down excess"]
    for name in excess[:3]:
        parts.append(f"{name}: " + action_text(rows[name]))
    parts.append(f"Across {counts['products']} products: {counts['increase_actions']} increase recommendations, "
                 f"{counts['reduce_actions']} run-down recommendations, {counts['maintain_actions']} maintain recommendations; "
                 f"{counts['pre_arrival_service_gaps']} potential pre-arrival service gaps.")
    # Reuse the app's existing forecast-quality classification; no new threshold.
    weak = sorted((name for name, row in rows.items() if row.get("forecast_quality") == "Low confidence"
                   and row.get("product_mape") is not None), key=lambda name: (-rows[name]["product_mape"], name))
    if weak:
        parts.append("Review forecast accuracy: " + "; ".join(f"{name} has holdout MAPE {number(rows[name]['product_mape'], 2)}%"
                                                             for name in weak[:3]) + ". These carry the app's low-confidence label.")
    if counts["unavailable_plans"]:
        parts.append(f"{counts['unavailable_plans']} products lack valid recommendations. Resolve their missing or invalid inventory, lead-time, or forecast-coverage inputs before assessing them.")
    if ranked and any(row.get("priority_score") is None for row in rows.values()):
        parts.append("Some products lack calculated priority scores; the attention order covers only products with those scores.")
    if not ranked:
        parts.append("Calculated priorities are unavailable; valid priority and service-gap inputs are needed to choose what to address first.")
    return "\n\n".join(parts)


def deterministic_planning_answer(payload):
    category, rows = payload["question_category"], payload["products_by_name"]
    if category in {"inventory_action", "product_explanation"}:
        answer = "\n\n".join(inventory_answer(name, row, category == "product_explanation") for name, row in rows.items())
    elif category == "comparison":
        answer = comparison_answer(payload)
    elif category == "ranking" or (category == "model_accuracy" and "ranking" in payload):
        ranking = payload["ranking"]
        field = ranking["field"]
        value = number(ranking["value"], 2 if field in {"product_mae", "product_mape"} else 0)
        unit = "%" if field == "product_mape" else " units"
        answer = f"{', '.join(rows)} — {ranking['direction']} {LABELS[field]}: {value}{unit}, across all products with this calculated metric."
        if len(rows) > 1:
            answer += " These products are tied."
    elif category == "model_accuracy":
        if rows:
            answer = "\n\n".join(model_answer(name, row) for name, row in rows.items())
        else:
            mape = (payload.get("overall_accuracy") or {}).get("overall_mape")
            answer = (f"Average holdout MAPE is {number(mape, 2)}%, the unweighted mean across products in Forecast Models."
                      if mape is not None else "Holdout forecast accuracy is unavailable.")
    elif category == "prioritization":
        order = payload["priority_order"]
        if not order:
            return "No calculated priorities are available. Resolve missing planning inputs before ranking products."
        name = order[0]
        row = rows[name]
        answer = (f"{name} needs attention first: calculated priority score {number(row.get('priority_score'))}, "
                  f"with a potential pre-arrival service gap of {number(row.get('pre_arrival_shortage'))} units. "
                  + action_text(row) + " Products are ranked by priority score (highest first), then service gap; remaining ties use product name.")
        if (row.get("pre_arrival_shortage") or 0) > 0:
            answer += " Ordinary replenishment may arrive too late to cover that gap."
    elif category == "summary":
        answer = assessment_answer(payload)
    else:
        return "This question is outside the supported planning summaries."
    if payload.get("time_scope"):
        answer += " No separate target for next month or another future date was calculated; the target above is for the current daily-review planning period."
    return answer
