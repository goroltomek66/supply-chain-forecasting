"""Visualization helpers for inventory-focused charts."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd


RISK_COLORS = {
    "High": "#c0392b",
    "Medium": "#f39c12",
    "Low": "#27ae60",
}


def _build_bar_chart(
    inventory_metrics: pd.DataFrame,
    value_column: str,
    output_path: Path,
    title: str,
    y_label: str,
) -> Path:
    """Create a product-level bar chart colored by risk level."""
    chart_data = inventory_metrics.dropna(subset=[value_column]).sort_values("product").copy()
    colors = chart_data["risk_level"].map(RISK_COLORS).fillna("#7f8c8d")

    fig, axis = plt.subplots(figsize=(12, 6))
    axis.bar(chart_data["product"], chart_data[value_column], color=colors, edgecolor="#2c3e50")
    axis.set_title(title)
    axis.set_xlabel("Product")
    axis.set_ylabel(y_label)
    if chart_data.empty:
        axis.text(0.5, 0.5, "Metric unavailable: inspect planning issues", transform=axis.transAxes, ha="center")
    axis.grid(axis="y", alpha=0.3)
    plt.setp(axis.get_xticklabels(), rotation=45, ha="right")

    legend_handles = [
        Patch(facecolor=color, edgecolor="#2c3e50", label=label)
        for label, color in RISK_COLORS.items()
    ]
    axis.legend(handles=legend_handles, title="Legacy demand volatility")

    for index, value in enumerate(chart_data[value_column]):
        axis.text(index, value, f"{value:.2f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def create_inventory_charts(inventory_metrics: pd.DataFrame, output_dir: Path) -> list[Path]:
    """Create inventory-focused charts for safety stock and reorder point."""
    safety_stock_chart = _build_bar_chart(
        inventory_metrics=inventory_metrics,
        value_column="planning_safety_stock",
        output_path=output_dir / "safety_stock_chart.png",
        title="Planning Safety Stock (Lead Time + Daily Review)",
        y_label="Planning Safety Stock (units)",
    )
    reorder_point_chart = _build_bar_chart(
        inventory_metrics=inventory_metrics,
        value_column="reorder_point",
        output_path=output_dir / "reorder_point_chart.png",
        title="Reorder Point by Product",
        y_label="Reorder Point",
    )
    target_chart = _build_bar_chart(
        inventory_metrics, "order_up_to_target", output_dir / "order_up_to_target_chart.png",
        "Order-Up-To Target by Product", "Target Inventory Position (units)",
    )
    return [safety_stock_chart, reorder_point_chart, target_chart]
