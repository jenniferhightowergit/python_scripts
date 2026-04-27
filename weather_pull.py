"""
weather_pull.py
===============
Pull daily weather + severe-storm events for AMS meter failure analysis.

Two data sources:
  - Open-Meteo Archive API (free, no key) for daily tmax/tmin/precip.
    Backed by ECMWF ERA5 reanalysis. Queried by lat/lon.
  - NOAA Storm Events Database (NCEI) for severe weather events
    (hail, flash flood, thunderstorm wind, tornado, high wind, etc.).
    Reported at the county-FIPS level.

Design notes for tens-of-thousands of meters and 5+ years:
  * Meter lat/lon is binned to a 0.1° grid (~7 mi) so we only call the
    weather API once per cell, not once per meter. Open-Meteo's underlying
    ERA5 grid is ~25 km, so 0.1° binning loses no real signal.
  * Both sources cache to parquet under cache_dir, so re-runs are free.
  * Storm events are joined to meters by county FIPS (NWS issues warnings
    at the county/zone level, so this is the natural granularity).

Top-level entry points:
    build_weather_features(meters_df, ...)   -> (weather_daily, storm_events)
    assign_county_fips(meters_df, ...)       -> meters_df + county_fips column
    attach_30day_lookback(events_df, ...)    -> events_df + weather features

Typical pipeline:
    import pandas as pd
    from weather_pull import (
        build_weather_features, assign_county_fips, attach_30day_lookback
    )

    meters = pd.read_csv('meters.csv')   # has lat, lon
    meters = assign_county_fips(meters, lat_col='lat', lon_col='lon')

    weather, storms = build_weather_features(
        meters, lat_col='lat', lon_col='lon',
        start_date='2021-01-01', end_date='2026-04-26',
        cache_dir='./weather_cache',
    )

    noncom = pd.read_csv('noncom_events.csv')  # has lat, lon, event_date
    noncom = assign_county_fips(noncom, lat_col='lat', lon_col='lon')
    features = attach_30day_lookback(
        noncom, weather, storms,
        event_date_col='event_date', lat_col='lat', lon_col='lon',
    )

Dependencies:
    pip install pandas numpy requests pyarrow geopandas shapely
"""

from __future__ import annotations

import io
import re
import time
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
import pandas as pd
import requests


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"
STORM_EVENTS_INDEX = "https://www.ncei.noaa.gov/pub/data/swdi/stormevents/csvfiles/"
TIGER_COUNTIES_URL = (
    "https://www2.census.gov/geo/tiger/TIGER2023/COUNTY/tl_2023_us_county.zip"
)

TEXAS_STATE_FIPS = "48"
DEFAULT_GRID_RESOLUTION = 0.1     # degrees, ~7 miles
OPEN_METEO_BATCH = 50             # locations per Open-Meteo request
DEFAULT_USER_AGENT = "OncorAMSFailureAnalysis/1.0 (research; contact: jennifer.susan.hightower@gmail.com)"


# ---------------------------------------------------------------------------
# Lat/lon binning
# ---------------------------------------------------------------------------

def bin_locations(
    df: pd.DataFrame,
    lat_col: str = "lat",
    lon_col: str = "lon",
    resolution: float = DEFAULT_GRID_RESOLUTION,
) -> pd.DataFrame:
    """Add ``bin_lat``/``bin_lon`` columns that snap each point to a coarse grid.

    Two meters that round to the same cell will share one weather pull, which
    is what makes tens-of-thousands of points tractable.
    """
    out = df.copy()
    out["bin_lat"] = (out[lat_col] / resolution).round().astype(int) * resolution
    out["bin_lon"] = (out[lon_col] / resolution).round().astype(int) * resolution
    out["bin_lat"] = out["bin_lat"].round(2)
    out["bin_lon"] = out["bin_lon"].round(2)
    return out


# ---------------------------------------------------------------------------
# Open-Meteo daily weather
# ---------------------------------------------------------------------------

def _open_meteo_request(
    lats: list[float],
    lons: list[float],
    start_date: str,
    end_date: str,
    retries: int = 3,
) -> list[dict]:
    """One Open-Meteo Archive call for a batch of points. Returns list-of-dicts."""
    params = {
        "latitude": ",".join(f"{x:.4f}" for x in lats),
        "longitude": ",".join(f"{x:.4f}" for x in lons),
        "start_date": start_date,
        "end_date": end_date,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum",
        "temperature_unit": "fahrenheit",
        "precipitation_unit": "inch",
        "timezone": "America/Chicago",
    }
    last_exc: Optional[Exception] = None
    for attempt in range(retries):
        try:
            r = requests.get(OPEN_METEO_URL, params=params, timeout=120,
                              headers={"User-Agent": DEFAULT_USER_AGENT})
            r.raise_for_status()
            data = r.json()
            # Single location returns dict; multiple locations return a list.
            if isinstance(data, dict):
                data = [data]
            return data
        except Exception as e:
            last_exc = e
            time.sleep(2 ** attempt)
    raise RuntimeError(
        f"Open-Meteo request failed after {retries} retries: {last_exc}"
    )


def _fetch_year_for_cells(
    cells: pd.DataFrame,
    year: int,
    overall_start: pd.Timestamp,
    overall_end: pd.Timestamp,
    lat_col: str,
    lon_col: str,
    verbose: bool,
) -> pd.DataFrame:
    """Pull one calendar year of weather for a set of unique cells, batched."""
    year_start = max(pd.Timestamp(f"{year}-01-01"), overall_start).strftime("%Y-%m-%d")
    year_end = min(pd.Timestamp(f"{year}-12-31"), overall_end).strftime("%Y-%m-%d")

    rows: list[dict] = []
    n_batches = (len(cells) + OPEN_METEO_BATCH - 1) // OPEN_METEO_BATCH
    for b_idx, i in enumerate(range(0, len(cells), OPEN_METEO_BATCH), start=1):
        batch = cells.iloc[i : i + OPEN_METEO_BATCH]
        if verbose:
            print(f"  {year}: batch {b_idx}/{n_batches} ({len(batch)} cells)")
        results = _open_meteo_request(
            batch[lat_col].tolist(),
            batch[lon_col].tolist(),
            year_start,
            year_end,
        )
        for cell_row, result in zip(batch.itertuples(index=False), results):
            daily = result.get("daily", {}) or {}
            dates = pd.to_datetime(daily.get("time", []))
            tmax = daily.get("temperature_2m_max", []) or []
            tmin = daily.get("temperature_2m_min", []) or []
            precip = daily.get("precipitation_sum", []) or []
            for d, tx, tn, pr in zip(dates, tmax, tmin, precip):
                rows.append({
                    lat_col: getattr(cell_row, lat_col),
                    lon_col: getattr(cell_row, lon_col),
                    "date": d,
                    "tmax_f": tx,
                    "tmin_f": tn,
                    "precip_in": pr,
                })
        time.sleep(0.5)   # be polite to a free API
    return pd.DataFrame(rows)


def fetch_daily_weather(
    unique_cells_df: pd.DataFrame,
    start_date: str,
    end_date: str,
    cache_dir: str = "./weather_cache",
    lat_col: str = "bin_lat",
    lon_col: str = "bin_lon",
    verbose: bool = True,
) -> pd.DataFrame:
    """Fetch daily tmax/tmin/precip for a deduped set of cells, with parquet cache.

    Cache is keyed by year. If a year file exists but is missing some cells,
    only those missing cells are pulled and merged in.

    Returns a long-form DataFrame:
        [bin_lat, bin_lon, date, tmax_f, tmin_f, precip_in]
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    years = list(range(start.year, end.year + 1))

    cells = unique_cells_df[[lat_col, lon_col]].drop_duplicates().reset_index(drop=True)
    if verbose:
        print(f"Pulling weather for {len(cells):,} unique cells x {len(years)} years")

    all_frames: list[pd.DataFrame] = []
    for year in years:
        cache_file = cache / f"weather_daily_{year}.parquet"
        if cache_file.exists():
            cached = pd.read_parquet(cache_file)
            cached_keys = set(zip(cached[lat_col].round(2), cached[lon_col].round(2)))
            needed_keys = set(zip(cells[lat_col].round(2), cells[lon_col].round(2)))
            missing = needed_keys - cached_keys
            if not missing:
                if verbose:
                    print(f"  {year}: cache hit ({len(cached):,} rows)")
                all_frames.append(cached)
                continue
            if verbose:
                print(f"  {year}: cache missing {len(missing)} cells, fetching")
            missing_df = pd.DataFrame(list(missing), columns=[lat_col, lon_col])
            new_data = _fetch_year_for_cells(
                missing_df, year, start, end, lat_col, lon_col, verbose
            )
            combined = pd.concat([cached, new_data], ignore_index=True)
            combined.to_parquet(cache_file, index=False)
            all_frames.append(combined)
        else:
            year_data = _fetch_year_for_cells(
                cells, year, start, end, lat_col, lon_col, verbose
            )
            year_data.to_parquet(cache_file, index=False)
            all_frames.append(year_data)

    out = pd.concat(all_frames, ignore_index=True)
    out["date"] = pd.to_datetime(out["date"])
    out = out[(out["date"] >= start) & (out["date"] <= end)].reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# NOAA Storm Events Database
# ---------------------------------------------------------------------------

def _list_storm_events_files() -> dict[int, str]:
    """Scrape NCEI's directory listing for the latest details CSV per year."""
    r = requests.get(STORM_EVENTS_INDEX, timeout=60,
                     headers={"User-Agent": DEFAULT_USER_AGENT})
    r.raise_for_status()
    pattern = re.compile(
        r'href="(StormEvents_details-ftp_v1\.0_d(\d{4})_c\d+\.csv\.gz)"'
    )
    files: dict[int, str] = {}
    for m in pattern.finditer(r.text):
        fname, year = m.group(1), int(m.group(2))
        # The "c" suffix is a publish timestamp; keep the most recent per year.
        if year not in files or fname > files[year]:
            files[year] = fname
    return files


def fetch_storm_events(
    start_date: str,
    end_date: str,
    state: str = "TEXAS",
    cache_dir: str = "./weather_cache",
    verbose: bool = True,
) -> pd.DataFrame:
    """Download + filter NOAA Storm Events for a state and date range.

    Returns a DataFrame with one row per event, including:
        begin_date, end_date, state, county_name, county_fips,
        event_type, magnitude, magnitude_type, narrative
    """
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)

    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    years = list(range(start.year, end.year + 1))

    cache_file = cache / f"storm_events_{state}_{years[0]}_{years[-1]}.parquet"
    if cache_file.exists():
        if verbose:
            print("Loading storm events from cache")
        df = pd.read_parquet(cache_file)
    else:
        if verbose:
            print("Listing NCEI storm events directory...")
        files_by_year = _list_storm_events_files()
        frames: list[pd.DataFrame] = []
        for year in years:
            if year not in files_by_year:
                if verbose:
                    print(f"  {year}: no file available, skipping")
                continue
            fname = files_by_year[year]
            url = STORM_EVENTS_INDEX + fname
            if verbose:
                print(f"  {year}: downloading {fname}")
            r = requests.get(url, timeout=300,
                             headers={"User-Agent": DEFAULT_USER_AGENT})
            r.raise_for_status()
            df_year = pd.read_csv(
                io.BytesIO(r.content),
                compression="gzip",
                low_memory=False,
            )
            df_year = df_year[
                df_year["STATE"].astype(str).str.upper() == state.upper()
            ].copy()
            frames.append(df_year)
        df = pd.concat(frames, ignore_index=True)
        df.to_parquet(cache_file, index=False)

    # Date columns in the file look like '01-JAN-23 00:00:00'
    df["begin_date"] = pd.to_datetime(
        df["BEGIN_DATE_TIME"], format="%d-%b-%y %H:%M:%S", errors="coerce"
    )
    df["end_date"] = pd.to_datetime(
        df["END_DATE_TIME"], format="%d-%b-%y %H:%M:%S", errors="coerce"
    )
    df = df[(df["begin_date"] >= start) & (df["begin_date"] <= end)].copy()

    # county_fips = state(2) + county/zone(3)
    df["county_fips"] = (
        df["STATE_FIPS"].astype(int).astype(str).str.zfill(2)
        + df["CZ_FIPS"].astype(int).astype(str).str.zfill(3)
    )

    keep_cols = [
        "begin_date", "end_date", "STATE", "CZ_NAME", "county_fips",
        "EVENT_TYPE", "MAGNITUDE", "MAGNITUDE_TYPE",
        "TOR_F_SCALE", "INJURIES_DIRECT", "DEATHS_DIRECT",
        "EVENT_NARRATIVE",
    ]
    keep_cols = [c for c in keep_cols if c in df.columns]
    return df[keep_cols].rename(columns={
        "STATE": "state",
        "CZ_NAME": "county_name",
        "EVENT_TYPE": "event_type",
        "MAGNITUDE": "magnitude",
        "MAGNITUDE_TYPE": "magnitude_type",
        "TOR_F_SCALE": "tor_f_scale",
        "INJURIES_DIRECT": "injuries_direct",
        "DEATHS_DIRECT": "deaths_direct",
        "EVENT_NARRATIVE": "narrative",
    }).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Lat/lon -> County FIPS via TIGER spatial join
# ---------------------------------------------------------------------------

def assign_county_fips(
    df: pd.DataFrame,
    lat_col: str = "lat",
    lon_col: str = "lon",
    cache_dir: str = "./weather_cache",
    state_fips: str = TEXAS_STATE_FIPS,
) -> pd.DataFrame:
    """Add ``county_fips`` (and ``county_name``) to ``df`` via spatial join.

    Requires geopandas. Caches the (Texas-only) county geometries after the
    first run, so subsequent calls are instant.
    """
    try:
        import geopandas as gpd  # noqa: F401
    except ImportError as e:
        raise RuntimeError(
            "assign_county_fips requires geopandas. Install with:\n"
            "    pip install geopandas shapely pyproj"
        ) from e
    import geopandas as gpd

    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    counties_cache = cache / f"counties_state{state_fips}.parquet"

    if counties_cache.exists():
        counties = gpd.read_parquet(counties_cache)
    else:
        print("Downloading TIGER county shapefile (one-time, ~80 MB)...")
        counties = gpd.read_file(TIGER_COUNTIES_URL)
        counties = counties[counties["STATEFP"] == state_fips].copy()
        counties = counties[["GEOID", "NAME", "geometry"]].rename(
            columns={"GEOID": "county_fips", "NAME": "county_name"}
        )
        counties.to_parquet(counties_cache)

    points = gpd.GeoDataFrame(
        df.copy(),
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs="EPSG:4326",
    )
    counties = counties.to_crs("EPSG:4326")
    joined = gpd.sjoin(
        points,
        counties[["county_fips", "county_name", "geometry"]],
        how="left",
        predicate="within",
    )
    return pd.DataFrame(joined.drop(columns=["geometry", "index_right"]))


# ---------------------------------------------------------------------------
# 30-day lookback aggregation
# ---------------------------------------------------------------------------

def attach_30day_lookback(
    events_df: pd.DataFrame,
    weather_df: pd.DataFrame,
    storm_df: pd.DataFrame,
    event_date_col: str = "event_date",
    lat_col: str = "lat",
    lon_col: str = "lon",
    county_fips_col: str = "county_fips",
    window_days: int = 30,
    grid_resolution: float = DEFAULT_GRID_RESOLUTION,
) -> pd.DataFrame:
    """For each event, compute weather + storm features over the prior N days.

    Adds these columns:
        max_temp_<N>d, min_temp_<N>d, total_precip_<N>d,
        days_above_100_<N>d, days_below_32_<N>d, n_weather_days_<N>d,
        n_storm_events_<N>d,
        n_hail_<N>d, n_flash_flood_<N>d, n_thunderstorm_<N>d,
        n_tornado_<N>d, n_high_wind_<N>d
    """
    suffix = f"_{window_days}d"

    events = events_df.copy()
    events[event_date_col] = pd.to_datetime(events[event_date_col])
    events = bin_locations(events, lat_col=lat_col, lon_col=lon_col,
                           resolution=grid_resolution)

    # ---- Weather features (per cell, slice the prior window) --------------
    weather = weather_df.copy()
    weather["date"] = pd.to_datetime(weather["date"])

    weather_features: list[dict] = []
    weather_by_cell = dict(tuple(weather.groupby(["bin_lat", "bin_lon"])))
    for idx, ev in events.iterrows():
        key = (ev["bin_lat"], ev["bin_lon"])
        win_end = ev[event_date_col]
        win_start = win_end - pd.Timedelta(days=window_days)
        if key in weather_by_cell:
            w = weather_by_cell[key]
            w = w[(w["date"] >= win_start) & (w["date"] <= win_end)]
        else:
            w = weather.iloc[0:0]
        weather_features.append({
            "_idx": idx,
            f"max_temp{suffix}":       w["tmax_f"].max() if len(w) else np.nan,
            f"min_temp{suffix}":       w["tmin_f"].min() if len(w) else np.nan,
            f"total_precip{suffix}":   w["precip_in"].sum() if len(w) else np.nan,
            f"days_above_100{suffix}": int((w["tmax_f"] > 100).sum()) if len(w) else 0,
            f"days_below_32{suffix}":  int((w["tmin_f"] < 32).sum()) if len(w) else 0,
            f"n_weather_days{suffix}": len(w),
        })
    weather_features_df = pd.DataFrame(weather_features).set_index("_idx")

    # ---- Storm features (per county, slice the prior window) --------------
    storms = storm_df.copy()
    storms["begin_date"] = pd.to_datetime(storms["begin_date"])
    storms["event_type_lc"] = storms["event_type"].astype(str).str.lower().str.strip()
    storm_by_county = dict(tuple(storms.groupby("county_fips")))

    storm_features: list[dict] = []
    for idx, ev in events.iterrows():
        cf = ev.get(county_fips_col)
        win_end = ev[event_date_col]
        win_start = win_end - pd.Timedelta(days=window_days)
        if pd.isna(cf) or cf not in storm_by_county:
            s_win = storms.iloc[0:0]
        else:
            s = storm_by_county[cf]
            s_win = s[(s["begin_date"] >= win_start) & (s["begin_date"] <= win_end)]
        types = s_win["event_type_lc"] if len(s_win) else pd.Series([], dtype=str)
        storm_features.append({
            "_idx": idx,
            f"n_storm_events{suffix}": len(s_win),
            f"n_hail{suffix}":         int(types.str.contains("hail").sum()),
            f"n_flash_flood{suffix}":  int(types.str.contains("flash flood").sum()),
            f"n_thunderstorm{suffix}": int(types.str.contains("thunderstorm").sum()),
            f"n_tornado{suffix}":      int(types.str.contains("tornado").sum()),
            f"n_high_wind{suffix}":    int(types.str.contains("high wind|strong wind", regex=True).sum()),
        })
    storm_features_df = pd.DataFrame(storm_features).set_index("_idx")

    return events.join(weather_features_df).join(storm_features_df)


# ---------------------------------------------------------------------------
# One-call convenience
# ---------------------------------------------------------------------------

def build_weather_features(
    meters_df: pd.DataFrame,
    lat_col: str = "lat",
    lon_col: str = "lon",
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    cache_dir: str = "./weather_cache",
    grid_resolution: float = DEFAULT_GRID_RESOLUTION,
    state: str = "TEXAS",
    verbose: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Pull both weather sources for a meter set and date range.

    Returns
    -------
    weather_daily : DataFrame [bin_lat, bin_lon, date, tmax_f, tmin_f, precip_in]
    storm_events  : DataFrame [begin_date, end_date, state, county_name,
                               county_fips, event_type, magnitude, narrative, ...]
    """
    if start_date is None or end_date is None:
        raise ValueError("start_date and end_date are required (YYYY-MM-DD).")

    meters_binned = bin_locations(
        meters_df, lat_col=lat_col, lon_col=lon_col, resolution=grid_resolution
    )
    unique_cells = meters_binned[["bin_lat", "bin_lon"]].drop_duplicates()
    if verbose:
        print(
            f"{len(meters_df):,} meters -> {len(unique_cells):,} unique weather cells"
        )

    weather = fetch_daily_weather(
        unique_cells, start_date, end_date,
        cache_dir=cache_dir, verbose=verbose,
    )
    storms = fetch_storm_events(
        start_date, end_date, state=state,
        cache_dir=cache_dir, verbose=verbose,
    )
    return weather, storms
