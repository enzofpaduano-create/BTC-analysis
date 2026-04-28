"""Equity curve + drawdown plot using plotly (HTML output).

Plotly is already a vectorbt dependency, so no extra install.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from backtest.engine import BacktestResult


def plot_equity_and_drawdown(result: BacktestResult, *, title: str = "Backtest") -> go.Figure:
    """Two-panel figure: equity (top) + drawdown (bottom)."""
    eq = result.equity
    running_max = eq.cummax()
    dd = eq / running_max - 1.0

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        row_heights=[0.7, 0.3],
        vertical_spacing=0.05,
        subplot_titles=("Equity curve", "Drawdown"),
    )
    fig.add_trace(go.Scatter(x=eq.index, y=eq.to_numpy(), name="equity"), row=1, col=1)
    fig.add_trace(
        go.Scatter(
            x=dd.index,
            y=dd.to_numpy(),
            name="drawdown",
            fill="tozeroy",
            line={"color": "crimson"},
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="Equity", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown", tickformat=".0%", row=2, col=1)
    fig.update_layout(title=title, showlegend=False, height=700)
    return fig


def save_html_report(result: BacktestResult, *, path: Path, title: str = "Backtest report") -> None:
    """Persist a self-contained HTML report (equity + drawdown + metrics table)."""
    fig = plot_equity_and_drawdown(result, title=title)
    metrics_df = pd.DataFrame(result.metrics.to_dict().items(), columns=["metric", "value"])
    metrics_html = metrics_df.to_html(index=False, float_format=lambda v: f"{v:.4f}")
    plot_html = fig.to_html(full_html=False, include_plotlyjs="cdn")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"""<!doctype html>
<html><head><meta charset='utf-8'><title>{title}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif;
       padding: 24px; max-width: 1100px; margin: 0 auto; }}
table {{ border-collapse: collapse; margin-top: 16px; }}
th, td {{ padding: 6px 12px; border: 1px solid #ddd; }}
th {{ background: #f5f5f5; text-align: left; }}
</style>
</head><body>
<h1>{title}</h1>
{plot_html}
<h2>Metrics</h2>
{metrics_html}
</body></html>
""",
        encoding="utf-8",
    )
