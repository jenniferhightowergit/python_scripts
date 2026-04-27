"""
prior_n_events.py
=================
For each anchor event (typically non-com), fetch the N events that immediately
preceded it on the same meter, summarize what's in those prior events, and
compare against a matched random-timestamp control so any pattern you find is
actually a signal rather than a base-rate artifact.

The module assumes a long event table shaped like:

    meter_id | hardware_model | collector | transformer | event_date | event_type

Column names are all parameterized -- pass whatever yours are called.

Top-level entry points
----------------------
attach_prior_n_events(events_df, ...)
    Returns the anchor rows with prior_1_*, prior_2_*, ... prior_N_* columns
    (type, date, days_before_anchor) wide-joined onto each anchor.

summarize_prior_composition(anchors_with_priors, ...)
    Per event type, what % of anchors have at least 1 / 2 / N occurrences in
    their prior-N window, plus the share of all prior slots that event type
    fills. This is the table you'd put on a slide.

build_random_control_anchors(events_df, ...)
    For each non-com anchor, pick a random non-anchor event on the SAME meter.
    Same-meter matching controls for hardware model, location, age, etc.

compare_prior_compositions(actual_summary, control_summary)
    Side-by-side actual vs control with a lift column. Lift > 1 means the
    event type is overrepresented in non-com prior-Ns relative to baseline.

cohort_followup(events_df, ...)
    The "if-then" funnel stat. Example:
        Of meters with 3+ outages in any 30-day window, what % had a non-com
        in the next 60 days? Compare to baseline (meters with <3 outages).

Typical pipeline
----------------
    from prior_n_events import (
        attach_prior_n_events, summarize_prior_composition,
        build_random_control_anchors, compare_prior_compositions,
        cohort_followup,
    )

    # 1) Attach prior 3 events to each non-com
    anchors = attach_prior_n_events(events, n=3, anchor_event_type='non_com')

    # 2) Summarize what types appear in those prior windows
    actual = summarize_prior_composition(anchors, n=3)

    # 3) Build a same-meter random-date control and summarize it the same way
    controls = build_random_control_anchors(events, anchor_event_type='non_com',
                                            seed=42)
    controls = attach_prior_n_events(
        pd.concat([events, controls], ignore_index=True),
        n=3, anchor_event_type='_control',
    )
    control_summary = summarize_prior_composition(controls, n=3)

    # 4) Compare side-by-side
    lift_table = compare_prior_compositions(actual, control_summary)
    print(lift_table)

    # 5) Funnel: 3+ outages in 30 days -> non-com in next 60 days?
    funnel = cohort_followup(
        events,
        lookback_window_days=30, lookback_event_type='outage',
        lookback_count_threshold=3,
        followup_window_days=60, followup_event_type='non_com',
    )
    print(funnel)
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core: attach prior N events to each anchor
# ---------------------------------------------------------------------------

def attach_prior_n_events(
    events_df: pd.DataFrame,
    n: int = 3,
    anchor_event_type: str = "non_com",
    meter_col: str = "meter_id",
    date_col: str = "event_date",
    type_col: str = "event_type",
    max_lookback_days: Optional[int] = None,
    exclude_same_day: bool = True,
    include_anchor_type_in_priors: bool = True,
) -> pd.DataFrame:
    """For each event of ``anchor_event_type``, attach the N most recent prior
    events on the same meter as wide columns.

    Parameters
    ----------
    events_df :
        Long event table. Must have meter_col, date_col, type_col.
    n :
        How many prior events to fetch per anchor.
    anchor_event_type :
        The event type that defines an "anchor" (e.g., 'non_com').
    max_lookback_days :
        If set, prior events older than this are dropped (their prior_*_* cells
        become NaN). None means no time limit.
    exclude_same_day :
        If True, events on the same calendar date as the anchor are not counted
        as priors. Defaults True because same-day events are coincident, not
        antecedent.
    include_anchor_type_in_priors :
        If True, prior anchor-type events count (a meter that's gone non-com
        before is meaningful signal). If False, we filter them out.

    Returns
    -------
    DataFrame
        One row per anchor event, with the original anchor columns PLUS:
            prior_1_type, prior_1_date, prior_1_days_before
            prior_2_type, prior_2_date, prior_2_days_before
            ...
            prior_N_type, prior_N_date, prior_N_days_before
        Slot 1 is the most-recent prior event.
    """
    df = events_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    # Optionally drop same-day priors before doing the shift logic. We'll
    # handle this via filtering after the shift is easier than pre-filtering.

    # Build a sortable order key. Sort by (meter, date) globally, then group.
    df = df.sort_values([meter_col, date_col]).reset_index(drop=True)

    # We need to be able to identify priors that are excluded by filters
    # (e.g., same day, or anchor-type when include_anchor_type_in_priors=False)
    # *before* we take the shift. The cleanest way: build a "candidate priors"
    # frame that only contains events eligible to be a prior, then do groupby
    # shift on that frame, then merge anchors against it by meter and the
    # candidate's running rank just-below the anchor's date.

    # For simplicity and clarity, do the per-meter walk: it's O(events) and
    # vectorized within each group via shift on a filtered candidate frame.

    # Step 1: candidate priors = all events that could serve as a prior.
    cand = df.copy()
    if not include_anchor_type_in_priors:
        cand = cand[cand[type_col] != anchor_event_type]

    cand = cand.sort_values([meter_col, date_col]).reset_index(drop=True)
    cand["_cand_rank"] = cand.groupby(meter_col).cumcount()

    # Step 2: anchors
    anchors = df[df[type_col] == anchor_event_type].copy().reset_index(drop=True)
    anchors["_anchor_idx"] = anchors.index

    # Step 3: For each anchor, find the candidate row that is the LATEST one
    # strictly before the anchor (with optional same-day exclusion). We do this
    # with merge_asof, which is the vectorized way.
    cand_sorted = cand.sort_values(date_col)
    anchors_sorted = anchors.sort_values(date_col)

    if exclude_same_day:
        # Shift anchor date back by 1 day so merge_asof gives us strictly
        # before-the-anchor-date events (handles times within the same day too).
        anchors_sorted["_lookup_date"] = (
            anchors_sorted[date_col].dt.normalize() - pd.Timedelta(days=1)
        ) + pd.Timedelta(hours=23, minutes=59, seconds=59)
    else:
        anchors_sorted["_lookup_date"] = anchors_sorted[date_col]

    matched = pd.merge_asof(
        anchors_sorted,
        cand_sorted[[meter_col, date_col, "_cand_rank"]].rename(
            columns={date_col: "_cand_date"}
        ),
        left_on="_lookup_date",
        right_on="_cand_date",
        by=meter_col,
        direction="backward",
        allow_exact_matches=True,
    )

    # _cand_rank is the rank of the most recent eligible prior. The Nth prior
    # back is at rank (_cand_rank - (k-1)).
    cand_lookup = cand.set_index([meter_col, "_cand_rank"])

    out = anchors.set_index("_anchor_idx").copy()
    matched_indexed = matched.set_index("_anchor_idx")

    for k in range(1, n + 1):
        target_ranks = matched_indexed["_cand_rank"] - (k - 1)
        keys = list(zip(matched_indexed[meter_col], target_ranks))

        prior_types = []
        prior_dates = []
        for key in keys:
            mid, rk = key
            if pd.isna(rk) or rk < 0:
                prior_types.append(np.nan)
                prior_dates.append(pd.NaT)
                continue
            try:
                row = cand_lookup.loc[(mid, int(rk))]
                # Could return Series (one match) or DataFrame (rare dup)
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                prior_types.append(row[type_col])
                prior_dates.append(row[date_col])
            except KeyError:
                prior_types.append(np.nan)
                prior_dates.append(pd.NaT)

        out[f"prior_{k}_type"] = prior_types
        out[f"prior_{k}_date"] = prior_dates
        days_before = (out[date_col] - out[f"prior_{k}_date"]).dt.days
        if max_lookback_days is not None:
            too_old = days_before > max_lookback_days
            out.loc[too_old, f"prior_{k}_type"] = np.nan
            out.loc[too_old, f"prior_{k}_date"] = pd.NaT
            days_before = days_before.where(~too_old)
        out[f"prior_{k}_days_before"] = days_before

    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Summarize composition of prior events across all anchors
# ---------------------------------------------------------------------------

def summarize_prior_composition(
    anchors_with_priors: pd.DataFrame,
    n: int = 3,
    event_types: Optional[list[str]] = None,
) -> pd.DataFrame:
    """For each event type, how often does it appear in the prior-N window?

    Returns a DataFrame indexed by event_type with columns:
        n_anchors                    - total anchors evaluated
        n_with_at_least_1            - anchors with >=1 occurrence in prior N
        pct_with_at_least_1          - same as above, as percentage
        n_with_at_least_2            - anchors with >=2 occurrences in prior N
        pct_with_at_least_2
        n_total_occurrences          - total times this event type appears
                                       across all prior slots
        pct_of_filled_prior_slots    - share of all (non-NaN) prior slots that
                                       were this event type
    """
    n_anchors = len(anchors_with_priors)
    type_cols = [f"prior_{k}_type" for k in range(1, n + 1)]

    long = (
        anchors_with_priors[type_cols]
        .stack(future_stack=True)
        .dropna()
        .reset_index()
    )
    long.columns = ["anchor_idx", "slot", "event_type"]

    if event_types is None:
        event_types = sorted(long["event_type"].dropna().unique())

    rows = []
    for et in event_types:
        per_anchor_count = (
            long[long["event_type"] == et]
            .groupby("anchor_idx").size()
            .reindex(range(n_anchors), fill_value=0)
        )
        rows.append({
            "event_type": et,
            "n_anchors": n_anchors,
            "n_with_at_least_1": int((per_anchor_count >= 1).sum()),
            "pct_with_at_least_1": round(100 * (per_anchor_count >= 1).mean(), 1),
            "n_with_at_least_2": int((per_anchor_count >= 2).sum()),
            "pct_with_at_least_2": round(100 * (per_anchor_count >= 2).mean(), 1),
            "n_total_occurrences": int((long["event_type"] == et).sum()),
            "pct_of_filled_prior_slots": round(
                100 * (long["event_type"] == et).mean(), 1
            ) if len(long) else 0.0,
        })

    return pd.DataFrame(rows).set_index("event_type").sort_values(
        "pct_with_at_least_1", ascending=False
    )


# ---------------------------------------------------------------------------
# Random-timestamp control: same meter, non-anchor date
# ---------------------------------------------------------------------------

def build_random_control_anchors(
    events_df: pd.DataFrame,
    anchor_event_type: str = "non_com",
    meter_col: str = "meter_id",
    date_col: str = "event_date",
    type_col: str = "event_type",
    n_controls_per_anchor: int = 1,
    seed: int = 42,
    control_label: str = "_control",
) -> pd.DataFrame:
    """For each anchor, pick N random NON-anchor events on the same meter and
    relabel them as control "anchors" of type ``control_label``.

    Returns a frame with the same columns as events_df where matched rows have
    their type_col replaced by ``control_label``. Concatenate this with the
    original events table and run attach_prior_n_events with
    anchor_event_type=control_label to compute the control prior-Ns.
    """
    rng = np.random.default_rng(seed)
    df = events_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    anchors = df[df[type_col] == anchor_event_type]
    anchor_meters = anchors[meter_col].value_counts()

    control_rows = []
    non_anchor_pool = df[df[type_col] != anchor_event_type]
    by_meter = dict(tuple(non_anchor_pool.groupby(meter_col)))

    for meter, n_anchors_for_meter in anchor_meters.items():
        if meter not in by_meter:
            continue
        candidate_pool = by_meter[meter]
        if len(candidate_pool) == 0:
            continue
        n_to_draw = n_anchors_for_meter * n_controls_per_anchor
        # Sample with replacement if needed (small candidate pools).
        replace = n_to_draw > len(candidate_pool)
        idx = rng.choice(
            candidate_pool.index.values, size=n_to_draw, replace=replace
        )
        sampled = candidate_pool.loc[idx].copy()
        sampled[type_col] = control_label
        control_rows.append(sampled)

    if not control_rows:
        return df.iloc[0:0].copy()
    return pd.concat(control_rows, ignore_index=True)


# ---------------------------------------------------------------------------
# Side-by-side comparison with lift
# ---------------------------------------------------------------------------

def compare_prior_compositions(
    actual_summary: pd.DataFrame,
    control_summary: pd.DataFrame,
    actual_label: str = "non_com",
    control_label: str = "control",
) -> pd.DataFrame:
    """Join an actual and a control summary side-by-side with a lift column.

    Lift > 1 means the event type is overrepresented in actual prior-Ns
    relative to baseline. A lift of 3.0 means 3x as common before non-coms as
    in random matched windows -- that's a real signal worth investigating.
    """
    a = actual_summary[["pct_with_at_least_1", "pct_of_filled_prior_slots",
                        "n_anchors"]].rename(columns={
        "pct_with_at_least_1": f"{actual_label}_pct_with_1",
        "pct_of_filled_prior_slots": f"{actual_label}_pct_of_slots",
        "n_anchors": f"{actual_label}_n",
    })
    c = control_summary[["pct_with_at_least_1", "pct_of_filled_prior_slots",
                         "n_anchors"]].rename(columns={
        "pct_with_at_least_1": f"{control_label}_pct_with_1",
        "pct_of_filled_prior_slots": f"{control_label}_pct_of_slots",
        "n_anchors": f"{control_label}_n",
    })
    out = a.join(c, how="outer").fillna({
        f"{actual_label}_pct_with_1": 0,
        f"{control_label}_pct_with_1": 0,
        f"{actual_label}_pct_of_slots": 0,
        f"{control_label}_pct_of_slots": 0,
    })
    # Lift on "% of anchors with at least one of this type in prior N"
    out["lift_pct_with_1"] = (
        out[f"{actual_label}_pct_with_1"]
        / out[f"{control_label}_pct_with_1"].replace(0, np.nan)
    ).round(2)
    out["lift_pct_of_slots"] = (
        out[f"{actual_label}_pct_of_slots"]
        / out[f"{control_label}_pct_of_slots"].replace(0, np.nan)
    ).round(2)
    return out.sort_values("lift_pct_with_1", ascending=False)


# ---------------------------------------------------------------------------
# Funnel: cohort followup probability
# ---------------------------------------------------------------------------

def cohort_followup(
    events_df: pd.DataFrame,
    lookback_window_days: int = 30,
    lookback_event_type: str = "outage",
    lookback_count_threshold: int = 3,
    followup_window_days: int = 60,
    followup_event_type: str = "non_com",
    meter_col: str = "meter_id",
    date_col: str = "event_date",
    type_col: str = "event_type",
) -> dict:
    """The "if-then" funnel stat for the boss deck.

    Walks every (meter, day) eligible position. Counts how many meter-days had
    >= ``lookback_count_threshold`` events of ``lookback_event_type`` in the
    prior ``lookback_window_days``. For those, what fraction had a
    ``followup_event_type`` event in the next ``followup_window_days``?

    Returns a dict you can put directly on a slide:
        {
          "exposed_meter_days": ...,
          "exposed_with_followup": ...,
          "exposed_followup_rate": "12.4%",
          "unexposed_meter_days": ...,
          "unexposed_with_followup": ...,
          "unexposed_followup_rate": "1.8%",
          "lift": "6.9x",
          "headline": "Meters with 3+ outages in the prior 30 days are 6.9x
                       more likely to go non-com in the next 60 days."
        }
    """
    df = events_df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([meter_col, date_col]).reset_index(drop=True)

    # For each meter event, compute (a) prior-window count of lookback type,
    # (b) whether there's a followup-type event in the next followup window.
    out_rows = []
    for meter, mdf in df.groupby(meter_col):
        mdf = mdf.sort_values(date_col)
        dates = mdf[date_col].values
        types = mdf[type_col].values
        for i, (d, t) in enumerate(zip(dates, types)):
            d = pd.Timestamp(d)
            # Prior lookback window: events strictly before d, within lookback days
            prior_mask = (dates < np.datetime64(d)) & (
                dates >= np.datetime64(d - pd.Timedelta(days=lookback_window_days))
            )
            n_prior = int(((types == lookback_event_type) & prior_mask).sum())
            exposed = n_prior >= lookback_count_threshold

            # Followup window: events strictly after d, within followup days
            follow_mask = (dates > np.datetime64(d)) & (
                dates <= np.datetime64(d + pd.Timedelta(days=followup_window_days))
            )
            had_followup = bool(((types == followup_event_type) & follow_mask).any())

            out_rows.append({
                "meter": meter, "anchor_date": d,
                "exposed": exposed, "had_followup": had_followup,
            })

    fdf = pd.DataFrame(out_rows)

    exposed = fdf[fdf["exposed"]]
    unexposed = fdf[~fdf["exposed"]]

    exp_rate = exposed["had_followup"].mean() if len(exposed) else 0.0
    unexp_rate = unexposed["had_followup"].mean() if len(unexposed) else 0.0
    lift = (exp_rate / unexp_rate) if unexp_rate > 0 else float("inf")

    return {
        "lookback": f"{lookback_count_threshold}+ {lookback_event_type} in {lookback_window_days}d",
        "followup": f"{followup_event_type} in next {followup_window_days}d",
        "exposed_meter_days": len(exposed),
        "exposed_with_followup": int(exposed["had_followup"].sum()),
        "exposed_followup_rate": f"{exp_rate*100:.1f}%",
        "unexposed_meter_days": len(unexposed),
        "unexposed_with_followup": int(unexposed["had_followup"].sum()),
        "unexposed_followup_rate": f"{unexp_rate*100:.1f}%",
        "lift": f"{lift:.1f}x" if np.isfinite(lift) else "inf",
        "headline": (
            f"Meters with {lookback_count_threshold}+ {lookback_event_type} events "
            f"in the prior {lookback_window_days} days are {lift:.1f}x more likely "
            f"to have a {followup_event_type} event in the next {followup_window_days} days."
        ),
    }


# ---------------------------------------------------------------------------
# Convenience: run actual + control + comparison in one call
# ---------------------------------------------------------------------------

def run_full_prior_n_analysis(
    events_df: pd.DataFrame,
    n: int = 3,
    anchor_event_type: str = "non_com",
    meter_col: str = "meter_id",
    date_col: str = "event_date",
    type_col: str = "event_type",
    max_lookback_days: Optional[int] = None,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """One call to run the full pipeline. Returns a dict of intermediate +
    final outputs so you can dig in or just print the lift table."""
    if verbose:
        print(f"Attaching prior {n} events to each {anchor_event_type} ...")
    actual_anchors = attach_prior_n_events(
        events_df, n=n, anchor_event_type=anchor_event_type,
        meter_col=meter_col, date_col=date_col, type_col=type_col,
        max_lookback_days=max_lookback_days,
    )
    actual_summary = summarize_prior_composition(actual_anchors, n=n)

    if verbose:
        print(f"Building random-timestamp control on same meters ...")
    controls = build_random_control_anchors(
        events_df, anchor_event_type=anchor_event_type,
        meter_col=meter_col, date_col=date_col, type_col=type_col,
        seed=seed,
    )
    combined = pd.concat([events_df, controls], ignore_index=True)
    control_anchors = attach_prior_n_events(
        combined, n=n, anchor_event_type="_control",
        meter_col=meter_col, date_col=date_col, type_col=type_col,
        max_lookback_days=max_lookback_days,
    )
    control_summary = summarize_prior_composition(control_anchors, n=n)

    lift = compare_prior_compositions(
        actual_summary, control_summary,
        actual_label=anchor_event_type, control_label="control",
    )

    return {
        "actual_anchors": actual_anchors,
        "actual_summary": actual_summary,
        "control_anchors": control_anchors,
        "control_summary": control_summary,
        "lift_table": lift,
    }
