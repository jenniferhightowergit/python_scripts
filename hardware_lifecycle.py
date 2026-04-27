"""
hardware_lifecycle.py
=====================
Breakouts of meter failure rate by hardware model, install age, collector, and
transformer. The "where does the failure live" view of the data.

Why these matter
----------------
Before claiming weather or outages are causing failures, you need to rule out
that a few specific hardware models, age cohorts, or upstream devices account
for most of the non-coms. If Hardware Model X has 5x the failure rate of Model
Y, "weather caused the spike" might really be "weather hit a region with mostly
Model X meters."

Functions
---------
non_com_rate_by_hardware_model(meters_df, events_df, ...)
non_com_rate_by_collector(meters_df, events_df, ...)
non_com_rate_by_transformer(meters_df, events_df, ...)
non_com_rate_by_install_age(meters_df, events_df, ...)   # bathtub-curve view
hardware_model_x_age(meters_df, events_df, ...)           # 2D pivot
time_to_first_failure(meters_df, events_df, ...)          # distribution

All functions take:
    meters_df : DataFrame with one row per meter. Must include meter_id and the
                grouping column being analyzed (hardware_model, collector,
                transformer, install_date as appropriate).
    events_df : Long event table with [meter_id, event_date, event_type].

All return a tidy DataFrame ready to print, save, or feed into viz.py.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Generic helper: failure rate per group
# ---------------------------------------------------------------------------

def _rate_by_group(
    meters_df: pd.DataFrame,
    events_df: pd.DataFrame,
    group_col: str,
    anchor_event_type: str = "non_com",
    meter_col: str = "meter_id",
    type_col: str = "event_type",
    min_meters_per_group: int = 1,
) -> pd.DataFrame:
    """Failure rate per ``group_col`` with sample sizes."""
    failed_meter_ids = events_df.loc[
        events_df[type_col] == anchor_event_type, meter_col
    ].unique()

    counts = (
        meters_df.groupby(group_col)
        .agg(
            n_meters=(meter_col, "nunique"),
            n_failed=(meter_col, lambda s: s.isin(failed_meter_ids).sum()),
        )
    )
    counts["rate_per_1k"] = (counts["n_failed"] / counts["n_meters"] * 1000).round(2)
    counts = counts[counts["n_meters"] >= min_meters_per_group]
    counts["pct_failed"] = (counts["n_failed"] / counts["n_meters"] * 100).round(2)
    return counts.sort_values("rate_per_1k", ascending=False)


# ---------------------------------------------------------------------------
# Hardware model
# ---------------------------------------------------------------------------

def non_com_rate_by_hardware_model(
    meters_df: pd.DataFrame,
    events_df: pd.DataFrame,
    hardware_col: str = "hardware_model",
    anchor_event_type: str = "non_com",
    meter_col: str = "meter_id",
    type_col: str = "event_type",
    min_meters_per_model: int = 50,
) -> pd.DataFrame:
    """Non-com rate per 1,000 meters, by hardware model. Filter out models with
    too few meters to produce a stable rate."""
    return _rate_by_group(
        meters_df, events_df, hardware_col,
        anchor_event_type=anchor_event_type,
        meter_col=meter_col, type_col=type_col,
        min_meters_per_group=min_meters_per_model,
    )


# ---------------------------------------------------------------------------
# Collector / transformer (network-equipment outliers)
# ---------------------------------------------------------------------------

def non_com_rate_by_collector(
    meters_df: pd.DataFrame,
    events_df: pd.DataFrame,
    collector_col: str = "collector",
    anchor_event_type: str = "non_com",
    meter_col: str = "meter_id",
    type_col: str = "event_type",
    min_meters_per_collector: int = 20,
    top_n: Optional[int] = 25,
) -> pd.DataFrame:
    """Non-com rate per collector. A bad collector can manufacture phantom
    non-coms, so this is an essential sanity check before claiming weather or
    outage causality.
    """
    out = _rate_by_group(
        meters_df, events_df, collector_col,
        anchor_event_type=anchor_event_type,
        meter_col=meter_col, type_col=type_col,
        min_meters_per_group=min_meters_per_collector,
    )
    if top_n is not None:
        out = out.head(top_n)
    return out


def non_com_rate_by_transformer(
    meters_df: pd.DataFrame,
    events_df: pd.DataFrame,
    transformer_col: str = "transformer",
    anchor_event_type: str = "non_com",
    meter_col: str = "meter_id",
    type_col: str = "event_type",
    min_meters_per_transformer: int = 5,
    top_n: Optional[int] = 25,
) -> pd.DataFrame:
    """Non-com rate per transformer. Transformers serve a small number of
    meters, so set ``min_meters_per_transformer`` low. A failing transformer
    can fry every meter downstream.
    """
    out = _rate_by_group(
        meters_df, events_df, transformer_col,
        anchor_event_type=anchor_event_type,
        meter_col=meter_col, type_col=type_col,
        min_meters_per_group=min_meters_per_transformer,
    )
    if top_n is not None:
        out = out.head(top_n)
    return out


# ---------------------------------------------------------------------------
# Install age (bathtub curve)
# ---------------------------------------------------------------------------

def non_com_rate_by_install_age(
    meters_df: pd.DataFrame,
    events_df: pd.DataFrame,
    install_date_col: str = "install_date",
    age_buckets_years: tuple = (0, 1, 2, 5, 10, 15, 20, 25),
    anchor_event_type: str = "non_com",
    meter_col: str = "meter_id",
    date_col: str = "event_date",
    type_col: str = "event_type",
    as_of_date: Optional[str] = None,
) -> pd.DataFrame:
    """Non-com rate per 1,000 meters by install-date age bucket.

    The expected pattern is a "bathtub curve": elevated rates in the first
    year (manufacturing defects / install issues), low in the middle, rising
    again at end-of-life. Departures from this -- e.g., an unexpected bump in
    years 5-7 -- often point at a specific bad batch.
    """
    meters_df = meters_df.copy()
    meters_df[install_date_col] = pd.to_datetime(meters_df[install_date_col])

    if as_of_date is None:
        as_of = pd.Timestamp.today().normalize()
    else:
        as_of = pd.Timestamp(as_of_date)

    meters_df["_age_years"] = ((as_of - meters_df[install_date_col]).dt.days / 365.25)
    bins = list(age_buckets_years) + [np.inf]
    labels = [f"{a}-{b}y" for a, b in zip(bins[:-1], bins[1:])]
    meters_df["_age_bucket"] = pd.cut(
        meters_df["_age_years"], bins=bins, labels=labels, right=False
    )

    failed = events_df.loc[
        events_df[type_col] == anchor_event_type, meter_col
    ].unique()

    grouped = (
        meters_df.groupby("_age_bucket", observed=True)
        .agg(
            n_meters=(meter_col, "nunique"),
            n_failed=(meter_col, lambda s: s.isin(failed).sum()),
        )
    )
    grouped["rate_per_1k"] = (grouped["n_failed"] / grouped["n_meters"] * 1000).round(2)
    grouped["pct_failed"] = (grouped["n_failed"] / grouped["n_meters"] * 100).round(2)
    return grouped


# ---------------------------------------------------------------------------
# 2D: hardware model x install age
# ---------------------------------------------------------------------------

def hardware_model_x_age(
    meters_df: pd.DataFrame,
    events_df: pd.DataFrame,
    hardware_col: str = "hardware_model",
    install_date_col: str = "install_date",
    age_buckets_years: tuple = (0, 2, 5, 10, 20),
    anchor_event_type: str = "non_com",
    meter_col: str = "meter_id",
    type_col: str = "event_type",
    min_cell_meters: int = 30,
    as_of_date: Optional[str] = None,
    metric: str = "rate_per_1k",
) -> pd.DataFrame:
    """2D pivot: hardware model x install-age bucket -> failure rate.

    A clear signal here ("Model X is fine until year 5, then rate spikes") is
    the kind of finding that drives a preemptive replacement program.
    Cells with fewer than ``min_cell_meters`` meters are blanked to NaN since
    small samples produce wild rates.
    """
    meters_df = meters_df.copy()
    meters_df[install_date_col] = pd.to_datetime(meters_df[install_date_col])
    as_of = (pd.Timestamp.today().normalize() if as_of_date is None
             else pd.Timestamp(as_of_date))
    meters_df["_age_years"] = ((as_of - meters_df[install_date_col]).dt.days / 365.25)
    bins = list(age_buckets_years) + [np.inf]
    labels = [f"{a}-{b}y" for a, b in zip(bins[:-1], bins[1:])]
    meters_df["_age_bucket"] = pd.cut(
        meters_df["_age_years"], bins=bins, labels=labels, right=False
    )

    failed = set(events_df.loc[
        events_df[type_col] == anchor_event_type, meter_col
    ].unique())
    meters_df["_failed"] = meters_df[meter_col].isin(failed).astype(int)

    pivot_meters = meters_df.pivot_table(
        index=hardware_col, columns="_age_bucket", values=meter_col,
        aggfunc="nunique", fill_value=0, observed=False,
    )
    pivot_failed = meters_df.pivot_table(
        index=hardware_col, columns="_age_bucket", values="_failed",
        aggfunc="sum", fill_value=0, observed=False,
    )
    pivot_rate = (pivot_failed / pivot_meters * 1000).round(2)

    if metric == "rate_per_1k":
        out = pivot_rate.where(pivot_meters >= min_cell_meters)
    elif metric == "n_failed":
        out = pivot_failed.where(pivot_meters >= min_cell_meters)
    elif metric == "n_meters":
        out = pivot_meters
    else:
        raise ValueError(f"Unknown metric: {metric}")
    return out


# ---------------------------------------------------------------------------
# Time-to-first-failure distribution
# ---------------------------------------------------------------------------

def time_to_first_failure(
    meters_df: pd.DataFrame,
    events_df: pd.DataFrame,
    install_date_col: str = "install_date",
    anchor_event_type: str = "non_com",
    meter_col: str = "meter_id",
    date_col: str = "event_date",
    type_col: str = "event_type",
) -> pd.DataFrame:
    """For each failed meter, time from install to first non-com event.

    Returns a per-meter DataFrame [meter_id, install_date, first_failure_date,
    days_to_failure, years_to_failure]. Useful for histograms and survival
    curves.
    """
    meters_df = meters_df.copy()
    meters_df[install_date_col] = pd.to_datetime(meters_df[install_date_col])
    events_df = events_df.copy()
    events_df[date_col] = pd.to_datetime(events_df[date_col])

    first_fail = (
        events_df[events_df[type_col] == anchor_event_type]
        .groupby(meter_col)[date_col].min()
        .rename("first_failure_date")
        .reset_index()
    )
    out = meters_df[[meter_col, install_date_col]].merge(
        first_fail, on=meter_col, how="inner"
    )
    out["days_to_failure"] = (out["first_failure_date"] - out[install_date_col]).dt.days
    out["years_to_failure"] = (out["days_to_failure"] / 365.25).round(2)
    return out.sort_values("days_to_failure")
