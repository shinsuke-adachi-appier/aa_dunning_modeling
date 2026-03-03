import os
import pandas as pd
import pydata_google_auth
from google.cloud import bigquery

from timezonefinder import TimezoneFinder
import pgeocode

from country_timezones import COUNTRY_TZ_MAP


# =========================
# AUTH & BIGQUERY
# =========================

def get_bq_client(project: str, location: str):
    # Clear session overrides
    if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
        del os.environ["GOOGLE_APPLICATION_CREDENTIALS"]

    scopes = [
        "https://www.googleapis.com/auth/cloud-platform",
        "https://www.googleapis.com/auth/drive",
    ]

    credentials = pydata_google_auth.get_user_credentials(
        scopes,
        use_local_webserver=False
    )

    return bigquery.Client(
        project=project,
        credentials=credentials,
        location=location
    )


def load_bigquery_table(client, query: str) -> pd.DataFrame:
    return client.query(query).to_dataframe()


# =========================
# TIMEZONE LOOKUPS
# =========================

tf = TimezoneFinder()
nomi_cache = {}


def zip_to_timezone(country, postal):
    if pd.isna(country) or pd.isna(postal):
        return None

    country = country.upper()

    if country not in nomi_cache:
        nomi_cache[country] = pgeocode.Nominatim(country)

    loc = nomi_cache[country].query_postal_code(postal)

    if pd.isna(loc.latitude) or pd.isna(loc.longitude):
        return None

    return tf.timezone_at(lat=loc.latitude, lng=loc.longitude)


def safe_zip_to_timezone(country, zip_code):
    try:
        return zip_to_timezone(country, zip_code)
    except Exception:
        return None


# =========================
# FEATURE ENGINEERING
# =========================

def add_timezone_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Ensure UTC
    df["updated_at"] = pd.to_datetime(df["updated_at"], utc=True)

    # Base timezone from country (normalize to uppercase for consistent lookup)
    country_norm = df["billing_country"].str.upper()
    df["timezone"] = country_norm.map(COUNTRY_TZ_MAP).fillna("UTC")

    # Zip-based override: resolve timezone only for unique (country, zip) pairs
    mask = df["fill_zip_code"] == "Zip Filled"
    unique_pairs = df.loc[mask, ["billing_country", "billing_zip"]].drop_duplicates()

    tz_by_pair = {}
    for country, zip_code in unique_pairs.itertuples(index=False):
        key = (country, zip_code)
        if key not in tz_by_pair:
            tz_by_pair[key] = safe_zip_to_timezone(country, zip_code)

    unique_pairs = unique_pairs.copy()
    unique_pairs["timezone_zip"] = [
        tz_by_pair[(c, z)] for c, z in zip(unique_pairs["billing_country"], unique_pairs["billing_zip"])
    ]

    df = df.merge(
        unique_pairs,
        on=["billing_country", "billing_zip"],
        how="left"
    )

    df["timezone"] = df["timezone_zip"].combine_first(df["timezone"])

    # Localized timestamp
    df["localized_time"] = pd.NaT

    for tz, idx in df.groupby("timezone").groups.items():
        try:
            df.loc[idx, "localized_time"] = (
                df.loc[idx, "updated_at"]
                  .dt.tz_convert(tz)
                  .dt.tz_localize(None)
            )
        except Exception:
            continue

    df["localized_time"] = pd.to_datetime(df["localized_time"])

    # Date parts
    df["local_day_of_month"] = df["localized_time"].dt.day
    df["local_hour"] = df["localized_time"].dt.hour
    df["local_day_of_week"] = df["localized_time"].dt.day_name()

    # Time buckets
    df["local_time_period"] = df["local_hour"].apply(get_time_period)
    df["local_time_period_detailed"] = df["local_hour"].apply(get_time_period_detailed)

    return df


# =========================
# TIME BUCKET HELPERS
# =========================

def get_time_period(hour):
    if 6 <= hour < 12:
        return "6-12"
    elif 12 <= hour < 18:
        return "12-18"
    elif hour >= 18:
        return "18-24"
    elif hour < 6:
        return "0-6"
    return None


def get_time_period_detailed(hour):
    if hour < 3:
        return "0-3"
    elif 3 <= hour < 6:
        return "3-6"
    elif 6 <= hour < 9:
        return "6-9"
    elif 9 <= hour < 12:
        return "9-12"
    elif 12 <= hour < 15:
        return "12-15"
    elif 15 <= hour < 18:
        return "15-18"
    elif 18 <= hour < 21:
        return "18-21"
    elif hour >= 21:
        return "21-24"
    return None


# =========================
# MAIN PIPELINE
# =========================

def run_pipeline(end_date=None):
    client = get_bq_client(
        project="aa-datamart",
        location="europe-west1"
    )

    query = f"""
        SELECT *
        FROM `aa-datamart.billing_dm.MISc_vw_txn_enriched_subID_fallback`
    """
    if end_date:
        query += f"\nWHERE calendar_date <= '{end_date}'"

    df = load_bigquery_table(client, query)

    df = add_timezone_features(df)

    return df


if __name__ == "__main__":
    df = run_pipeline()
    # df.to_parquet("txn_enriched_with_local_time.parquet", index=False)
