"""
viz.py
======
Visualizations for the AMS meter failure analysis. All plots are matplotlib
so they save cleanly to PNG and drop into PowerPoint without fuss.

The plots are intentionally divided into two tiers:

Tier 1 - boss-deck plots (designed for a non-technical audience):
    plot_failure_anatomy_timeline()    Per-meter event timeline. Pick a few
                                        non-com meters and a healthy comparison;
                                        the visual difference does the talking.
    plot_pre_failure_signature()        Aggregate event rate vs days-before
                                        non-com. The "something is happening"
                                        chart.
    plot_hardware_model_bar()           Sorted bar of non-com rate per 1,000
                                        meters by hardware model.
    plot_county_choropleth()            Texas county map shaded by non-com rate
                                        per 1,000 meters.

Tier 2 - investigation plots (for your own analysis):
    plot_calendar_heatmap()             Day-of-year heatmap of non-com counts.
    plot_prior3_composition_bar()       Stacked bar of actual vs control
                                        prior-3 type composition.

Every function:
  - takes a long event_df with at minimum [meter_id, event_date, event_type]
  - takes optional output_path (str or Path) to save to PNG
  - returns the matplotlib Figure for further customization

Conventions
-----------
EVENT_COLORS gives each event type a stable color across all plots so a viewer
glancing between charts maintains visual continuity. Override with a custom
dict if your event type names differ.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# ---------------------------------------------------------------------------
# Color conventions
# ---------------------------------------------------------------------------

EVENT_COLORS: dict[str, str] = {
    "non_com":          "#d62728",   # red - the failure
    "meter_change":     "#9467bd",   # purple
    "outage":           "#ff7f0e",   # orange
    "weather_extreme":  "#1f77b4",   # blue
    "firmware_update":  "#7f7f7f",   # gray
    "hail":             "#17becf",   # cyan
    "flash_flood":      "#1f77b4",   # blue
    "thunderstorm":     "#2ca02c",   # green
    "tornado":          "#8c564b",   # brown
    "high_wind":        "#bcbd22",   # olive
    "tamper":           "#e377c2",   # pink
}

DEFAULT_FALLBACK_COLOR = "#cccccc"


def _color_for(event_type: str, palette: Optional[dict] = None) -> str:
    palette = palette or EVENT_COLORS
    return palette.get(event_type, DEFAULT_FALLBACK_COLOR)


def _save_or_return(fig, output_path):
    if output_path is not None:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=150, bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# 1. Anatomy of a failure timeline
# ---------------------------------------------------------------------------

def plot_failure_anatomy_timeline(
    events_df: pd.DataFrame,
    failed_meter_ids: list,
    healthy_meter_ids: Optional[list] = None,
    anchor_event_type: str = "non_com",
    window_days: int = 90,
    meter_col: str = "meter_id",
    date_col: str = "event_date",
    type_col: str = "event_type",
    palette: Optional[dict] = None,
    output_path: Optional[str] = None,
    figsize_per_meter: float = 0.9,
):
    """One row per meter: horizontal timeline of all events in the
    ``window_days`` leading up to (and including) the meter's non-com event.

    Pass a few failed meters and a few healthy meters in the same period to
    create a stark side-by-side. This is the "look at the difference" slide.
    """
    events_df = events_df.copy()
    events_df[date_col] = pd.to_datetime(events_df[date_col])

    rows = []
    # For failed meters, anchor on their non-com date
    for mid in failed_meter_ids:
        mevents = events_df[events_df[meter_col] == mid]
        anchor = mevents[mevents[type_col] == anchor_event_type]
        if anchor.empty:
            continue
        anchor_date = anchor[date_col].iloc[0]
        win = mevents[
            (mevents[date_col] >= anchor_date - pd.Timedelta(days=window_days))
            & (mevents[date_col] <= anchor_date)
        ]
        rows.append(("FAILED",  mid, anchor_date, win))

    # For healthy meters, use the median non-com date as the comparison anchor
    # so we're looking at the same time window.
    if healthy_meter_ids:
        # Use the median non-com date in the failed set as the comparison anchor
        ref_dates = []
        for mid in failed_meter_ids:
            anch = events_df[(events_df[meter_col] == mid)
                              & (events_df[type_col] == anchor_event_type)]
            if not anch.empty:
                ref_dates.append(anch[date_col].iloc[0])
        ref_date = pd.Series(ref_dates).median() if ref_dates else events_df[date_col].max()
        for mid in healthy_meter_ids:
            mevents = events_df[events_df[meter_col] == mid]
            win = mevents[
                (mevents[date_col] >= ref_date - pd.Timedelta(days=window_days))
                & (mevents[date_col] <= ref_date)
            ]
            rows.append(("HEALTHY", mid, ref_date, win))

    n = len(rows)
    fig, ax = plt.subplots(figsize=(11, max(2.5, figsize_per_meter * n + 1.5)))

    seen_types = set()
    for i, (status, mid, anchor_date, win) in enumerate(rows):
        y = n - i - 1   # plot top-down
        # Background band
        ax.axhspan(y - 0.4, y + 0.4,
                   color=("#fdecea" if status == "FAILED" else "#eafaf1"),
                   alpha=0.5, zorder=0)
        # Plot each event as a colored marker
        for _, ev in win.iterrows():
            et = ev[type_col]
            seen_types.add(et)
            marker = "X" if et == anchor_event_type else "o"
            size = 220 if et == anchor_event_type else 90
            ax.scatter(ev[date_col], y, s=size, c=_color_for(et, palette),
                       marker=marker, edgecolor="black", linewidth=0.6, zorder=3)
        # Label
        ax.text(anchor_date - pd.Timedelta(days=window_days + 2), y,
                f"{status}\n{mid}", ha="right", va="center",
                fontsize=9, fontweight="bold")

    ax.set_yticks([])
    ax.set_xlabel("Date")
    ax.set_title(f"Anatomy of failure: {window_days}-day event history per meter",
                 fontsize=13, fontweight="bold", pad=14)
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.grid(True, axis="x", alpha=0.3)
    ax.spines[["top", "right", "left"]].set_visible(False)

    legend_items = [
        Line2D([0], [0], marker="X", color="w", markerfacecolor=_color_for(anchor_event_type, palette),
               markeredgecolor="black", markersize=14, label=anchor_event_type),
    ]
    for et in sorted(seen_types - {anchor_event_type}):
        legend_items.append(
            Line2D([0], [0], marker="o", color="w", markerfacecolor=_color_for(et, palette),
                   markeredgecolor="black", markersize=10, label=et)
        )
    ax.legend(handles=legend_items, loc="upper center",
              bbox_to_anchor=(0.5, -0.12), ncol=min(len(legend_items), 5),
              frameon=False)

    fig.tight_layout()
    return _save_or_return(fig, output_path)


# ---------------------------------------------------------------------------
# 2. Pre-failure signature (aggregate event rate vs days-before-non-com)
# ---------------------------------------------------------------------------

def plot_pre_failure_signature(
    events_df: pd.DataFrame,
    anchor_event_type: str = "non_com",
    window_days: int = 90,
    bin_days: int = 7,
    meter_col: str = "meter_id",
    date_col: str = "event_date",
    type_col: str = "event_type",
    event_types_to_show: Optional[list[str]] = None,
    palette: Optional[dict] = None,
    output_path: Optional[str] = None,
):
    """For each event type, plot rate of occurrence per non-com vs days-before
    the non-com. If a type spikes near day 0, that's your precursor."""
    events_df = events_df.copy()
    events_df[date_col] = pd.to_datetime(events_df[date_col])

    anchors = events_df[events_df[type_col] == anchor_event_type][[meter_col, date_col]].copy()
    anchors = anchors.rename(columns={date_col: "_anchor_date"})

    # Join every event for an anchor's meter
    joined = anchors.merge(events_df, on=meter_col)
    joined["days_before"] = (joined["_anchor_date"] - joined[date_col]).dt.days
    joined = joined[(joined["days_before"] > 0) & (joined["days_before"] <= window_days)]
    joined = joined[joined[type_col] != anchor_event_type]   # don't plot the anchor on its own chart

    n_anchors = len(anchors)
    if n_anchors == 0:
        raise ValueError(f"No {anchor_event_type} events found in events_df")

    if event_types_to_show is None:
        event_types_to_show = sorted(joined[type_col].unique())

    bins = np.arange(0, window_days + bin_days, bin_days)
    bin_centers = bins[:-1] + bin_days / 2

    fig, ax = plt.subplots(figsize=(10, 5.5))
    for et in event_types_to_show:
        sub = joined[joined[type_col] == et]
        counts, _ = np.histogram(sub["days_before"].values, bins=bins)
        rate = counts / n_anchors / bin_days   # events per anchor per day
        ax.plot(bin_centers, rate, marker="o", linewidth=2.2,
                color=_color_for(et, palette), label=et)

    ax.invert_xaxis()   # so day 0 (the failure) is on the right
    ax.set_xlabel(f"Days before {anchor_event_type} event")
    ax.set_ylabel(f"Events per {anchor_event_type} per day")
    ax.set_title(
        f"Pre-failure signature: event rate in the {window_days} days before {anchor_event_type}\n"
        f"(N = {n_anchors:,} {anchor_event_type} events)",
        fontsize=12, fontweight="bold", pad=12,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left", frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _save_or_return(fig, output_path)


# ---------------------------------------------------------------------------
# 3. Hardware model bar chart (non-com rate per 1,000 meters)
# ---------------------------------------------------------------------------

def plot_hardware_model_bar(
    meters_df: pd.DataFrame,
    events_df: pd.DataFrame,
    anchor_event_type: str = "non_com",
    meter_col: str = "meter_id",
    type_col: str = "event_type",
    hardware_col: str = "hardware_model",
    min_meters_per_model: int = 50,
    output_path: Optional[str] = None,
):
    """Sorted bar of non-com rate per 1,000 meters, by hardware model.

    Annotates each bar with the sample size so you can tell apart "Model X is
    really bad" from "Model X has 4 meters and 1 failure."
    """
    fail_meters = events_df.loc[
        events_df[type_col] == anchor_event_type, meter_col
    ].unique()

    grouped = (
        meters_df.groupby(hardware_col)[meter_col]
        .agg(["nunique"])
        .rename(columns={"nunique": "n_meters"})
    )
    grouped["n_failed"] = (
        meters_df[meters_df[meter_col].isin(fail_meters)]
        .groupby(hardware_col)[meter_col].nunique()
    )
    grouped["n_failed"] = grouped["n_failed"].fillna(0).astype(int)
    grouped = grouped[grouped["n_meters"] >= min_meters_per_model]
    grouped["rate_per_1k"] = (grouped["n_failed"] / grouped["n_meters"] * 1000).round(1)
    grouped = grouped.sort_values("rate_per_1k", ascending=True)

    fig, ax = plt.subplots(figsize=(9, max(3, 0.45 * len(grouped) + 2)))
    bars = ax.barh(
        grouped.index.astype(str), grouped["rate_per_1k"],
        color=plt.cm.Reds(np.linspace(0.3, 0.85, len(grouped))),
        edgecolor="black", linewidth=0.5,
    )
    for bar, (_, row) in zip(bars, grouped.iterrows()):
        ax.text(
            bar.get_width() + max(grouped["rate_per_1k"].max() * 0.01, 0.5),
            bar.get_y() + bar.get_height() / 2,
            f"{row['rate_per_1k']:.1f}/1k   (n={row['n_meters']:,}, failed={row['n_failed']:,})",
            va="center", fontsize=9,
        )
    ax.set_xlabel(f"{anchor_event_type} rate per 1,000 meters")
    ax.set_title(
        f"{anchor_event_type.capitalize()} rate by hardware model "
        f"(models with >= {min_meters_per_model} meters)",
        fontsize=12, fontweight="bold", pad=10,
    )
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(True, axis="x", alpha=0.3)
    ax.set_xlim(0, grouped["rate_per_1k"].max() * 1.35)
    fig.tight_layout()
    return _save_or_return(fig, output_path)


# ---------------------------------------------------------------------------
# 4. Texas county choropleth
# ---------------------------------------------------------------------------

def plot_county_choropleth(
    meters_df: pd.DataFrame,
    events_df: pd.DataFrame,
    anchor_event_type: str = "non_com",
    meter_col: str = "meter_id",
    type_col: str = "event_type",
    county_fips_col: str = "county_fips",
    cache_dir: str = "./weather_cache",
    state_fips: str = "48",
    output_path: Optional[str] = None,
):
    """Choropleth of non-com rate per 1,000 meters by Texas county.

    Reuses the TIGER county shapefile cached by weather_pull.assign_county_fips
    (so you only download it once across the whole project).
    """
    try:
        import geopandas as gpd
    except ImportError as e:
        raise RuntimeError(
            "plot_county_choropleth requires geopandas. Install with:\n"
            "    pip install geopandas shapely pyproj"
        ) from e

    counties_cache = Path(cache_dir) / f"counties_state{state_fips}.parquet"
    if counties_cache.exists():
        counties = gpd.read_parquet(counties_cache)
    else:
        # Fall back to fetching it directly. Keep this in sync with weather_pull.
        from weather_pull import TIGER_COUNTIES_URL
        counties = gpd.read_file(TIGER_COUNTIES_URL)
        counties = counties[counties["STATEFP"] == state_fips].copy()
        counties = counties[["GEOID", "NAME", "geometry"]].rename(
            columns={"GEOID": "county_fips", "NAME": "county_name"}
        )
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
        counties.to_parquet(counties_cache)

    fail_meters = events_df.loc[
        events_df[type_col] == anchor_event_type, meter_col
    ].unique()

    rate_by_county = (
        meters_df.groupby(county_fips_col)
        .agg(
            n_meters=(meter_col, "nunique"),
            n_failed=(meter_col, lambda s: s.isin(fail_meters).sum()),
        )
        .reset_index()
    )
    rate_by_county["rate_per_1k"] = (
        rate_by_county["n_failed"] / rate_by_county["n_meters"] * 1000
    )
    rate_by_county[county_fips_col] = rate_by_county[county_fips_col].astype(str)
    counties["county_fips"] = counties["county_fips"].astype(str)
    merged = counties.merge(
        rate_by_county, left_on="county_fips", right_on=county_fips_col, how="left"
    )

    fig, ax = plt.subplots(figsize=(11, 9))
    merged.plot(
        column="rate_per_1k", cmap="Reds", linewidth=0.3,
        edgecolor="white", legend=True, ax=ax,
        missing_kwds={"color": "#eeeeee", "edgecolor": "white",
                      "label": "no meters"},
        legend_kwds={"label": f"{anchor_event_type} per 1,000 meters",
                      "shrink": 0.6},
    )
    ax.set_title(
        f"{anchor_event_type.capitalize()} rate by Texas county",
        fontsize=14, fontweight="bold",
    )
    ax.axis("off")
    fig.tight_layout()
    return _save_or_return(fig, output_path)


# ---------------------------------------------------------------------------
# 5. Calendar heatmap (day-of-year x year)
# ---------------------------------------------------------------------------

def plot_calendar_heatmap(
    events_df: pd.DataFrame,
    anchor_event_type: str = "non_com",
    date_col: str = "event_date",
    type_col: str = "event_type",
    output_path: Optional[str] = None,
):
    """Calendar heatmap of daily ``anchor_event_type`` counts. Rows are years,
    columns are day-of-year. Reveals seasonality and one-off event days.
    """
    df = events_df[events_df[type_col] == anchor_event_type].copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df["year"] = df[date_col].dt.year
    df["doy"] = df[date_col].dt.dayofyear
    counts = (
        df.groupby(["year", "doy"]).size().rename("n").reset_index()
    )
    grid = counts.pivot(index="year", columns="doy", values="n").fillna(0)
    grid = grid.reindex(columns=range(1, 367), fill_value=0)

    fig, ax = plt.subplots(figsize=(13, max(2, 0.5 * len(grid) + 1.5)))
    im = ax.imshow(grid.values, aspect="auto", cmap="Reds",
                   interpolation="nearest")
    ax.set_yticks(range(len(grid.index)))
    ax.set_yticklabels(grid.index)
    # Month tick locations
    month_starts = [pd.Timestamp(f"2024-{m:02d}-01").dayofyear - 1 for m in range(1, 13)]
    ax.set_xticks(month_starts)
    ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                         "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"])
    ax.set_xlabel("Day of year")
    ax.set_ylabel("Year")
    ax.set_title(f"Daily {anchor_event_type} counts (calendar view)",
                 fontsize=12, fontweight="bold", pad=10)
    fig.colorbar(im, ax=ax, label=f"# {anchor_event_type} per day", shrink=0.7)
    fig.tight_layout()
    return _save_or_return(fig, output_path)


# ---------------------------------------------------------------------------
# 6. Prior-3 composition: actual vs control stacked bar
# ---------------------------------------------------------------------------

def plot_prior3_composition_bar(
    actual_summary: pd.DataFrame,
    control_summary: pd.DataFrame,
    actual_label: str = "non_com",
    control_label: str = "control",
    palette: Optional[dict] = None,
    output_path: Optional[str] = None,
):
    """Side-by-side stacked bar: actual prior-3 type composition vs control.

    Takes the outputs of summarize_prior_composition() for both groups.
    """
    types = sorted(set(actual_summary.index) | set(control_summary.index))
    actual_pct = [actual_summary.loc[t, "pct_of_filled_prior_slots"]
                  if t in actual_summary.index else 0 for t in types]
    control_pct = [control_summary.loc[t, "pct_of_filled_prior_slots"]
                   if t in control_summary.index else 0 for t in types]

    fig, ax = plt.subplots(figsize=(7, 6))
    bottoms_actual = 0
    bottoms_control = 0
    bar_w = 0.45
    for t, a, c in zip(types, actual_pct, control_pct):
        ax.bar(0, a, width=bar_w, bottom=bottoms_actual,
               color=_color_for(t, palette), edgecolor="white", label=t)
        ax.bar(1, c, width=bar_w, bottom=bottoms_control,
               color=_color_for(t, palette), edgecolor="white")
        if a > 4:
            ax.text(0, bottoms_actual + a / 2, f"{t}\n{a:.0f}%",
                    ha="center", va="center", fontsize=9, color="white",
                    fontweight="bold")
        if c > 4:
            ax.text(1, bottoms_control + c / 2, f"{t}\n{c:.0f}%",
                    ha="center", va="center", fontsize=9, color="white",
                    fontweight="bold")
        bottoms_actual += a
        bottoms_control += c

    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"Before {actual_label}\nevents",
                         f"Random matched\n{control_label} windows"])
    ax.set_ylabel("Share of prior-3 event slots (%)")
    ax.set_title(
        f"What fills the prior-3 event window?\n"
        f"{actual_label} vs matched-meter random {control_label}",
        fontsize=12, fontweight="bold", pad=10,
    )
    ax.set_ylim(0, max(sum(actual_pct), sum(control_pct)) * 1.05)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return _save_or_return(fig, output_path)
