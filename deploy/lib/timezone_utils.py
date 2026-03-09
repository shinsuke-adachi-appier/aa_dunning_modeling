"""
Timezone and local-time processing for inference (aligned with txn_pipeline.add_timezone_features).
Adds timezone per row from country (and optional zip), then localized_time and local_* columns.
"""
from __future__ import annotations

import pandas as pd

from .country_timezones import COUNTRY_TZ_MAP

_tf = None
_nomi_cache = {}


def _get_timezonefinder():
    global _tf
    if _tf is None:
        try:
            from timezonefinder import TimezoneFinder
            _tf = TimezoneFinder()
        except ImportError:
            _tf = False
    return _tf if _tf is not False else None


def _zip_to_timezone(country: str, postal: str) -> str | None:
    if pd.isna(country) or pd.isna(postal):
        return None
    country = str(country).upper()
    try:
        import pgeocode
        if country not in _nomi_cache:
            _nomi_cache[country] = pgeocode.Nominatim(country)
        loc = _nomi_cache[country].query_postal_code(str(postal).strip())
        if loc is None:
            return None
        lat = getattr(loc, "latitude", None) or (loc.get("latitude") if hasattr(loc, "get") else None)
        lng = getattr(loc, "longitude", None) or (loc.get("longitude") if hasattr(loc, "get") else None)
        if lat is None or lng is None or (hasattr(pd, "isna") and (pd.isna(lat) or pd.isna(lng))):
            return None
        tf = _get_timezonefinder()
        if tf is None:
            return None
        return tf.timezone_at(lat=float(lat), lng=float(lng))
    except Exception:
        return None


def add_timezone_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add timezone and localized timestamp columns (aligned with txn_pipeline).
    - Ensures updated_at is UTC.
    - timezone from billing_country (COUNTRY_TZ_MAP); override with zip when fill_zip_code == 'Zip Filled'.
    - localized_time = updated_at converted to that timezone (naive).
    - local_day_of_month, local_hour, local_day_of_week for downstream use.
    """
    df = df.copy()

    if "updated_at" not in df.columns:
        return df
    df["updated_at"] = pd.to_datetime(df["updated_at"], utc=True)

    # Country-based timezone (normalize to uppercase)
    country_col = "billing_country" if "billing_country" in df.columns else None
    if country_col is None:
        df["timezone"] = "UTC"
    else:
        country_norm = df[country_col].fillna("").astype(str).str.upper()
        df["timezone"] = country_norm.map(COUNTRY_TZ_MAP).fillna("UTC")

    # Zip-based override where available
    if "fill_zip_code" in df.columns and "billing_zip" in df.columns:
        mask = df["fill_zip_code"] == "Zip Filled"
        if mask.any():
            unique_pairs = df.loc[mask, ["billing_country", "billing_zip"]].drop_duplicates()
            tz_by_pair = {}
            for _, r in unique_pairs.iterrows():
                c, z = r["billing_country"], r["billing_zip"]
                key = (str(c).upper(), str(z).strip() if pd.notna(z) else "")
                if key not in tz_by_pair:
                    tz_by_pair[key] = _zip_to_timezone(c, z)
            unique_pairs = unique_pairs.copy()
            unique_pairs["timezone_zip"] = [
                tz_by_pair.get((str(c).upper(), str(z).strip() if pd.notna(z) else ""))
                for c, z in zip(unique_pairs["billing_country"], unique_pairs["billing_zip"])
            ]
            df = df.merge(unique_pairs, on=["billing_country", "billing_zip"], how="left")
            df["timezone"] = df["timezone_zip"].fillna(df["timezone"])
            df = df.drop(columns=["timezone_zip"], errors="ignore")

    # Localized timestamp (naive, in local tz)
    df["localized_time"] = pd.NaT
    for tz, idx in df.groupby("timezone", dropna=False).groups.items():
        if tz is None or (isinstance(tz, float) and pd.isna(tz)):
            tz = "UTC"
        try:
            tz_str = str(tz).strip() or "UTC"
            df.loc[idx, "localized_time"] = (
                df.loc[idx, "updated_at"]
                .dt.tz_convert(tz_str)
                .dt.tz_localize(None)
            )
        except Exception:
            df.loc[idx, "localized_time"] = df.loc[idx, "updated_at"].dt.tz_localize(None)

    df["localized_time"] = pd.to_datetime(df["localized_time"])
    df["local_day_of_month"] = df["localized_time"].dt.day
    df["local_hour"] = df["localized_time"].dt.hour
    df["local_day_of_week"] = df["localized_time"].dt.dayofweek  # 0=Monday for sin/cos

    return df
