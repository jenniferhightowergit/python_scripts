"""
shared_drive_ncm.py
===================
Find the latest file whose name contains "ncm" on a shared (or local) drive,
retrieve its last-modified date, and stamp that date onto any DataFrame as a
new column.

Works with:
  - UNC / Windows network paths  (\\\\server\\share\\folder)
  - Linux / macOS mount points   (/mnt/shared/folder)
  - Google Drive via PyDrive2    (set use_google_drive=True)

Quick start
-----------
    from shared_drive_ncm import find_latest_ncm_file, stamp_ncm_date

    path, mtime = find_latest_ncm_file(r"\\server\share\data")
    df = stamp_ncm_date(df, mtime)
    print(df["ncm_file_date"].iloc[0])   # e.g. 2024-03-15
"""

from __future__ import annotations

import os
import glob
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Core: find the latest NCM file on a local / network share
# ---------------------------------------------------------------------------

def find_latest_ncm_file(
    directory: str | os.PathLike,
    pattern: str = "*ncm*",
    recursive: bool = False,
) -> Tuple[Path, datetime]:
    """Return the most recently modified file matching *pattern* under *directory*.

    Parameters
    ----------
    directory : str | Path
        Root folder to search (e.g. ``r"\\\\server\\share\\data"`` or
        ``"/mnt/gdrive/data"``).
    pattern : str
        Glob pattern used to filter filenames.  Defaults to ``"*ncm*"``.
    recursive : bool
        If True, search subdirectories as well.

    Returns
    -------
    (Path, datetime)
        Absolute path to the latest matching file and its last-modified time
        (UTC-aware).

    Raises
    ------
    FileNotFoundError
        If *directory* does not exist or no matching files are found.
    """
    root = Path(directory)
    if not root.exists():
        raise FileNotFoundError(f"Directory not found: {root}")

    glob_pattern = f"**/{pattern}" if recursive else pattern
    matches = list(root.glob(glob_pattern))

    if not matches:
        raise FileNotFoundError(
            f"No files matching '{pattern}' found in {root}"
            + (" (recursive)" if recursive else "")
        )

    # Sort by modification time descending; take the newest
    latest = max(matches, key=lambda p: p.stat().st_mtime)
    mtime_utc = datetime.fromtimestamp(latest.stat().st_mtime, tz=timezone.utc)
    return latest, mtime_utc


# ---------------------------------------------------------------------------
# Core: find the latest NCM file via Google Drive API (PyDrive2)
# ---------------------------------------------------------------------------

def find_latest_ncm_file_gdrive(
    folder_id: str,
    name_contains: str = "ncm",
    credentials_file: Optional[str] = "mycreds.txt",
    client_secrets_file: str = "client_secrets.json",
) -> Tuple[str, str, datetime]:
    """Return the most recently modified Google Drive file whose name contains
    *name_contains*.

    Requires ``pydrive2`` (``pip install pydrive2``).

    Parameters
    ----------
    folder_id : str
        Google Drive folder ID (the long string in the Drive URL).
    name_contains : str
        Substring that must appear in the filename (case-insensitive).
    credentials_file : str | None
        Path where cached OAuth tokens are stored.  Created automatically on
        first run if it does not exist.
    client_secrets_file : str
        Path to the OAuth 2.0 client secrets JSON downloaded from Google Cloud
        Console.

    Returns
    -------
    (file_id, file_name, modified_datetime_utc)
    """
    try:
        from pydrive2.auth import GoogleAuth
        from pydrive2.drive import GoogleDrive
    except ImportError as exc:
        raise ImportError(
            "pydrive2 is required for Google Drive access.  "
            "Install it with:  pip install pydrive2"
        ) from exc

    gauth = GoogleAuth()
    gauth.settings["client_config_file"] = client_secrets_file
    if credentials_file and os.path.exists(credentials_file):
        gauth.LoadCredentialsFile(credentials_file)
        if gauth.credentials is None or gauth.access_token_expired:
            gauth.Refresh()
    else:
        gauth.LocalWebserverAuth()
    if credentials_file:
        gauth.SaveCredentialsFile(credentials_file)

    drive = GoogleDrive(gauth)

    query = (
        f"'{folder_id}' in parents and trashed=false "
        f"and mimeType != 'application/vnd.google-apps.folder'"
    )
    file_list = drive.ListFile({"q": query}).GetList()

    ncm_files = [
        f for f in file_list
        if name_contains.lower() in f["title"].lower()
    ]
    if not ncm_files:
        raise FileNotFoundError(
            f"No files containing '{name_contains}' found in Drive folder {folder_id}"
        )

    latest = max(
        ncm_files,
        key=lambda f: f["modifiedDate"],  # ISO 8601 string; lexicographic sort works
    )
    mtime_utc = datetime.fromisoformat(
        latest["modifiedDate"].replace("Z", "+00:00")
    )
    return latest["id"], latest["title"], mtime_utc


# ---------------------------------------------------------------------------
# Stamp: add the file date as a column to any DataFrame
# ---------------------------------------------------------------------------

def stamp_ncm_date(
    df: pd.DataFrame,
    file_modified: datetime,
    col_name: str = "ncm_file_date",
    date_only: bool = True,
) -> pd.DataFrame:
    """Add *col_name* to *df* containing the NCM file's last-modified date.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to annotate.
    file_modified : datetime
        Last-modified timestamp returned by ``find_latest_ncm_file`` or
        ``find_latest_ncm_file_gdrive``.
    col_name : str
        Name of the new column.
    date_only : bool
        If True (default), store only the date portion (``datetime.date``).
        If False, store the full UTC-aware ``datetime``.

    Returns
    -------
    pd.DataFrame
        A copy of *df* with the new column added.
    """
    df = df.copy()
    value = file_modified.date() if date_only else file_modified
    df[col_name] = value
    return df


# ---------------------------------------------------------------------------
# Convenience wrapper: one call does everything for a local/network share
# ---------------------------------------------------------------------------

def load_ncm_with_date(
    directory: str | os.PathLike,
    df: pd.DataFrame,
    pattern: str = "*ncm*",
    recursive: bool = False,
    col_name: str = "ncm_file_date",
    date_only: bool = True,
) -> Tuple[pd.DataFrame, Path, datetime]:
    """Find the latest NCM file, read its modified date, and stamp *df*.

    Returns
    -------
    (stamped_df, file_path, file_modified_utc)

    Example
    -------
    ::

        df, ncm_path, ncm_date = load_ncm_with_date(
            r"\\\\fileserver\\AMI\\reports",
            meters_df,
        )
        print(f"NCM file: {ncm_path.name}  updated: {ncm_date.date()}")
        print(df[["meter_id", "ncm_file_date"]].head())
    """
    file_path, mtime = find_latest_ncm_file(directory, pattern=pattern, recursive=recursive)
    stamped = stamp_ncm_date(df, mtime, col_name=col_name, date_only=date_only)
    return stamped, file_path, mtime
