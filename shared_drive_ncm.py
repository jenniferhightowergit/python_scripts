"""
shared_drive_ncm.py
===================
Find the latest file whose name contains "ncm" on a Windows shared drive,
retrieve its last-modified date, and stamp that date onto any DataFrame as a
new column.

Two access modes
----------------
1. **Mounted share** – the Windows share is already mapped or mounted as a
   drive letter (Windows) or CIFS mount (Linux).  Use
   ``find_latest_ncm_file(path)``.

2. **Direct SMB** – connect over the network without mounting first.  Requires
   ``pip install smbprotocol``.  Use ``find_latest_ncm_file_smb(...)``.

Quick start — mounted share (simplest)
---------------------------------------
    from shared_drive_ncm import load_ncm_with_date

    # Windows drive letter
    df, path, mtime = load_ncm_with_date(r"Z:\\AMI\\reports", meters_df)

    # Linux CIFS mount (sudo mount -t cifs //server/share /mnt/share ...)
    df, path, mtime = load_ncm_with_date("/mnt/share/AMI/reports", meters_df)

    print(f"NCM file : {path.name}")
    print(f"Updated  : {mtime.date()}")
    print(df[["meter_id", "ncm_file_date"]].head())

Quick start — direct SMB (no mount needed)
-------------------------------------------
    from shared_drive_ncm import find_latest_ncm_file_smb, stamp_ncm_date

    fname, mtime = find_latest_ncm_file_smb(
        server="fileserver",
        share="AMI",
        remote_dir="reports",
        username="DOMAIN\\\\user",
        password="secret",
    )
    df = stamp_ncm_date(df, mtime)
    print(f"NCM file : {fname}  updated: {mtime.date()}")
    print(df[["meter_id", "ncm_file_date"]].head())
"""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd


# ---------------------------------------------------------------------------
# Mode 1: mounted share (drive letter or CIFS mount point)
# ---------------------------------------------------------------------------

def find_latest_ncm_file(
    directory: str | os.PathLike,
    pattern: str = "*ncm*",
    recursive: bool = False,
) -> Tuple[Path, datetime]:
    """Return the most recently modified file matching *pattern* under *directory*.

    Works with Windows drive letters (``Z:\\folder``), UNC paths that are
    already accessible as a filesystem path, or Linux CIFS mount points
    (``/mnt/share/folder``).

    Parameters
    ----------
    directory : str | Path
        Root folder to search.
    pattern : str
        Glob pattern for filenames.  Default ``"*ncm*"`` matches any file
        whose name contains "ncm" (case-sensitive on Linux).
    recursive : bool
        If True, descend into subdirectories.

    Returns
    -------
    (Path, datetime)
        Absolute path of the newest matching file and its last-modified time
        (UTC-aware datetime).

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

    latest = max(matches, key=lambda p: p.stat().st_mtime)
    mtime_utc = datetime.fromtimestamp(latest.stat().st_mtime, tz=timezone.utc)
    return latest, mtime_utc


# ---------------------------------------------------------------------------
# Mode 2: direct SMB connection (no mount required)
# ---------------------------------------------------------------------------

def find_latest_ncm_file_smb(
    server: str,
    share: str,
    remote_dir: str = "",
    username: Optional[str] = None,
    password: Optional[str] = None,
    name_contains: str = "ncm",
    port: int = 445,
) -> Tuple[str, datetime]:
    """Return the most recently modified file whose name contains *name_contains*
    on a Windows SMB share, without requiring the share to be mounted first.

    Requires ``smbprotocol`` (``pip install smbprotocol``).

    Parameters
    ----------
    server : str
        Hostname or IP address of the file server (e.g. ``"fileserver"`` or
        ``"192.168.1.10"``).
    share : str
        Share name (e.g. ``"AMI"`` for ``\\\\fileserver\\AMI``).
    remote_dir : str
        Path inside the share to search (e.g. ``"reports\\ncm_exports"``).
        Use forward or back slashes; leave empty for the share root.
    username : str | None
        Domain or local account, e.g. ``"DOMAIN\\\\user"`` or just ``"user"``.
        If None, attempts anonymous / current-user auth (Windows only).
    password : str | None
        Password for *username*.
    name_contains : str
        Substring that must appear in the filename (case-insensitive).
    port : int
        SMB port.  Default 445.

    Returns
    -------
    (filename, modified_datetime_utc)
        The bare filename of the newest matching file and its last-modified
        time as a UTC-aware datetime.

    Raises
    ------
    ImportError
        If ``smbprotocol`` is not installed.
    FileNotFoundError
        If no matching files are found.
    """
    try:
        import smbclient
        import smbclient.path as smbpath
    except ImportError as exc:
        raise ImportError(
            "smbprotocol is required for direct SMB access.  "
            "Install it with:  pip install smbprotocol"
        ) from exc

    smbclient.register_session(
        server, username=username, password=password, port=port
    )

    remote_path = f"\\\\{server}\\{share}"
    if remote_dir:
        remote_path = remote_path + "\\" + remote_dir.replace("/", "\\")

    try:
        entries = list(smbclient.scandir(remote_path))
    except Exception as exc:
        raise FileNotFoundError(
            f"Could not list {remote_path}: {exc}"
        ) from exc

    matches = [
        e for e in entries
        if not e.is_dir() and name_contains.lower() in e.name.lower()
    ]
    if not matches:
        raise FileNotFoundError(
            f"No files containing '{name_contains}' found in {remote_path}"
        )

    latest = max(matches, key=lambda e: e.stat().st_mtime)
    mtime_utc = datetime.fromtimestamp(latest.stat().st_mtime, tz=timezone.utc)
    return latest.name, mtime_utc


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
        Last-modified timestamp from ``find_latest_ncm_file`` or
        ``find_latest_ncm_file_smb``.
    col_name : str
        Name of the new column.  Default ``"ncm_file_date"``.
    date_only : bool
        If True (default), store the date portion only (``datetime.date``).
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
# Convenience wrapper for mounted shares
# ---------------------------------------------------------------------------

def load_ncm_with_date(
    directory: str | os.PathLike,
    df: pd.DataFrame,
    pattern: str = "*ncm*",
    recursive: bool = False,
    col_name: str = "ncm_file_date",
    date_only: bool = True,
) -> Tuple[pd.DataFrame, Path, datetime]:
    """Find the latest NCM file on a mounted share, get its date, and stamp *df*.

    Returns
    -------
    (stamped_df, file_path, file_modified_utc)

    Example
    -------
    ::

        df, ncm_path, ncm_date = load_ncm_with_date(
            r"Z:\\AMI\\reports",   # or "/mnt/share/AMI/reports" on Linux
            meters_df,
        )
        print(f"NCM file: {ncm_path.name}  updated: {ncm_date.date()}")
        print(df[["meter_id", "ncm_file_date"]].head())
    """
    file_path, mtime = find_latest_ncm_file(
        directory, pattern=pattern, recursive=recursive
    )
    stamped = stamp_ncm_date(df, mtime, col_name=col_name, date_only=date_only)
    return stamped, file_path, mtime
