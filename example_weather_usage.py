"""
example_weather_usage.py
========================
Minimal walkthrough of weather_pull.py against a few synthetic meters.

Replace the ``meters`` and ``noncom_events`` DataFrames with your real ones
(your meter table with lat/lon and your non-com event table with lat/lon
plus an event_date column) and the rest will work the same.
"""

import pandas as pd

from weather_pull import (
    assign_county_fips,
    attach_30day_lookback,
    build_weather_features,
)


# ---------------------------------------------------------------------------
# 1. Stand in a tiny meter table. Real call: replace with your own DataFrame.
# ---------------------------------------------------------------------------
meters = pd.DataFrame({
    "meter_id": ["M001", "M002", "M003", "M004"],
    "lat": [32.7767, 32.7555, 31.7619, 29.4241],   # Dallas, FW, Austin-ish, San Antonio
    "lon": [-96.7970, -97.3308, -106.4850, -98.4936],
})

# Add county_fips so storm events can join later. One-time download (~80 MB)
# of TIGER county geometries the first time, then cached.
meters = assign_county_fips(meters, lat_col="lat", lon_col="lon")
print(meters)


# ---------------------------------------------------------------------------
# 2. Pull both weather sources for the date range you care about.
#    First run hits the APIs; subsequent runs read from ./weather_cache/*.
# ---------------------------------------------------------------------------
START = "2024-01-01"
END   = "2024-12-31"

weather, storms = build_weather_features(
    meters,
    lat_col="lat", lon_col="lon",
    start_date=START, end_date=END,
    cache_dir="./weather_cache",
)

print("\nWeather daily (head):")
print(weather.head())
print(f"  {len(weather):,} rows total")

print("\nStorm events (head):")
print(storms.head())
print(f"  {len(storms):,} TX events in {START} -> {END}")


# ---------------------------------------------------------------------------
# 3. Stand in a tiny non-com event table (replace with your real one).
# ---------------------------------------------------------------------------
noncom_events = pd.DataFrame({
    "meter_id":   ["M001", "M002", "M003", "M004"],
    "event_date": pd.to_datetime(["2024-07-15", "2024-08-22", "2024-05-10", "2024-09-03"]),
    "lat":        [32.7767, 32.7555, 31.7619, 29.4241],
    "lon":        [-96.7970, -97.3308, -106.4850, -98.4936],
})
noncom_events = assign_county_fips(noncom_events, lat_col="lat", lon_col="lon")


# ---------------------------------------------------------------------------
# 4. Compute 30-day lookback features for each non-com event.
# ---------------------------------------------------------------------------
features = attach_30day_lookback(
    noncom_events, weather, storms,
    event_date_col="event_date",
    lat_col="lat", lon_col="lon",
    window_days=30,
)

print("\nNon-com events with 30-day weather lookback features:")
print(features.to_string())
