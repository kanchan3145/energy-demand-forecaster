"""Download and dataset caching module for energy demand forecaster.

Handles fetching Spain energy + weather, PJM energy, and optional bonus datasets.
Logs source URLs, dates, and metadata to data/raw/dataset_sources.json.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import zipfile
from pathlib import Path
from typing import Any

import requests
import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/config.yaml") -> dict[str, Any]:
    """Load yaml configuration."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def check_kaggle_credentials() -> bool:
    """Check if Kaggle API credentials exist via file or environment variables."""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    if kaggle_json.exists():
        return True
    if os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY"):
        return True
    return False


def record_dataset_source(
    raw_dir: Path,
    dataset_name: str,
    source_url: str,
    files: list[str],
    status: str,
    details: str | None = None,
) -> None:
    """Record dataset provenance metadata in JSON."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    manifest_file = raw_dir / "dataset_sources.json"
    metadata: dict[str, Any] = {}
    if manifest_file.exists():
        try:
            with open(manifest_file, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception as e:
            logger.warning(f"Failed to read existing manifest: {e}")

    metadata[dataset_name] = {
        "source_url": source_url,
        "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        "files": files,
        "status": status,
        "details": details or "",
    }

    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Recorded metadata for {dataset_name} in {manifest_file}")


def download_spain_dataset(raw_dir: Path, force: bool = False) -> bool:
    """Download Spain energy & weather dataset from Kaggle or verify local presence."""
    slug = "nicholasjhana/energy-consumption-generation-prices-and-weather"
    url = f"https://www.kaggle.com/datasets/{slug}"
    energy_csv = raw_dir / "energy_dataset.csv"
    weather_csv = raw_dir / "weather_features.csv"

    if energy_csv.exists() and weather_csv.exists() and not force:
        logger.info("Spain dataset files already present in data/raw/.")
        record_dataset_source(
            raw_dir,
            "spain_energy_weather",
            url,
            [energy_csv.name, weather_csv.name],
            status="available_cached",
            details="Files verified locally in data/raw/",
        )
        return True

    if not check_kaggle_credentials():
        logger.error(
            "Kaggle credentials (~/.kaggle/kaggle.json or KAGGLE_USERNAME/KAGGLE_KEY env vars) missing!\n"
            "Please either configure Kaggle credentials or manually place:\n"
            f"  - {energy_csv.name}\n"
            f"  - {weather_csv.name}\n"
            f"into directory: {raw_dir.resolve()}"
        )
        record_dataset_source(
            raw_dir,
            "spain_energy_weather",
            url,
            [energy_csv.name, weather_csv.name],
            status="missing_credentials",
            details="Credentials missing; awaiting manual file placement",
        )
        return False

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        logger.info(f"Downloading Kaggle dataset: {slug} ...")
        api.dataset_download_files(slug, path=str(raw_dir), unzip=True)
        if energy_csv.exists() and weather_csv.exists():
            record_dataset_source(
                raw_dir,
                "spain_energy_weather",
                url,
                [energy_csv.name, weather_csv.name],
                status="downloaded_kaggle_api",
            )
            return True
        else:
            logger.error("Download completed but expected files were not found.")
            return False
    except Exception as e:
        logger.error(f"Failed to download Spain dataset from Kaggle: {e}")
        return False


def download_pjm_dataset(raw_dir: Path, force: bool = False) -> bool:
    """Download PJM Hourly Energy Consumption dataset from Kaggle or verify local presence."""
    slug = "robikscube/hourly-energy-consumption"
    url = f"https://www.kaggle.com/datasets/{slug}"
    pjm_csv = raw_dir / "PJME_hourly.csv"

    if pjm_csv.exists() and not force:
        logger.info("PJM dataset file already present in data/raw/.")
        record_dataset_source(
            raw_dir,
            "pjm_hourly_energy",
            url,
            [pjm_csv.name],
            status="available_cached",
            details="Files verified locally in data/raw/",
        )
        return True

    if not check_kaggle_credentials():
        logger.warning(
            "Kaggle credentials missing. If generalization test is required, place PJME_hourly.csv in data/raw/."
        )
        record_dataset_source(
            raw_dir,
            "pjm_hourly_energy",
            url,
            [pjm_csv.name],
            status="missing_credentials",
        )
        return False

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        logger.info(f"Downloading Kaggle dataset: {slug} ...")
        api.dataset_download_file(slug, file_name="PJME_hourly.csv", path=str(raw_dir))
        # Sometimes file is zipped
        zip_path = raw_dir / "PJME_hourly.csv.zip"
        if zip_path.exists():
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(raw_dir)
            zip_path.unlink()
        record_dataset_source(
            raw_dir,
            "pjm_hourly_energy",
            url,
            [pjm_csv.name],
            status="downloaded_kaggle_api",
        )
        return True
    except Exception as e:
        logger.warning(f"Failed to download PJM dataset: {e}")
        return False


def fetch_open_meteo_delhi(raw_dir: Path) -> bool:
    """Fetch Delhi historical weather data from Open-Meteo free archive API."""
    url = (
        "https://archive-api.open-meteo.com/v1/archive?"
        "latitude=28.6139&longitude=77.2090&"
        "start_date=2022-01-01&end_date=2022-12-31&"
        "hourly=temperature_2m,relative_humidity_2m,surface_pressure,wind_speed_10m"
    )
    dest_file = raw_dir / "delhi_weather_openmeteo_2022.json"
    try:
        logger.info("Fetching Open-Meteo Delhi historical weather data...")
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        with open(dest_file, "w", encoding="utf-8") as f:
            f.write(resp.text)
        record_dataset_source(
            raw_dir,
            "open_meteo_delhi",
            url,
            [dest_file.name],
            status="downloaded_api",
            details="Open-Meteo free archive API for Delhi lat=28.61, lon=77.21 (2022)",
        )
        return True
    except Exception as e:
        logger.warning(f"Failed to fetch Open-Meteo data: {e}")
        return False


def fetch_yfinance_nsei(raw_dir: Path) -> bool:
    """Download NIFTY 50 index data for bonus stock experiment."""
    dest_file = raw_dir / "nsei_daily.csv"
    try:
        import yfinance as yf

        logger.info("Fetching NSEI daily data via yfinance...")
        df = yf.download("^NSEI", start="2015-01-01", progress=False)
        if not df.empty:
            df.to_csv(dest_file)
            record_dataset_source(
                raw_dir,
                "yfinance_nsei",
                "yfinance:^NSEI",
                [dest_file.name],
                status="downloaded_yfinance",
                details="NIFTY 50 daily index prices from 2015-01-01 onwards",
            )
            return True
        return False
    except Exception as e:
        logger.warning(f"Failed to fetch yfinance data: {e}")
        return False


def main() -> None:
    """Run data download routine."""
    cfg = load_config()
    raw_dir = Path(cfg["paths"]["raw_data_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Starting Dataset Verification & Download ===")
    spain_ok = download_spain_dataset(raw_dir)
    pjm_ok = download_pjm_dataset(raw_dir)
    delhi_ok = fetch_open_meteo_delhi(raw_dir)
    nsei_ok = fetch_yfinance_nsei(raw_dir)

    print("\n" + "=" * 50)
    print("DATASET ACQUISITION SUMMARY:")
    print(f"  Spain Energy & Weather (PRIMARY) : {'READY' if spain_ok else 'MISSING / NEEDS MANUAL PLACEMENT'}")
    print(f"  PJM Energy (SECONDARY)           : {'READY' if pjm_ok else 'MISSING'}")
    print(f"  Open-Meteo Delhi Weather (BONUS) : {'READY' if delhi_ok else 'FAILED'}")
    print(f"  NIFTY 50 Stock Returns (BONUS)   : {'READY' if nsei_ok else 'FAILED'}")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
