"""Download the official STATS19 files used by AccidentIQ v1."""

from __future__ import annotations

import argparse
from pathlib import Path

import requests
import pandas as pd


BASE_URL = "https://data.dft.gov.uk/road-accidents-safety-data"
FILE_TYPES = ("collision", "vehicle", "casualty")
ARCHIVE_TEMPLATE = "dft-road-casualty-statistics-{file_type}-1979-latest-published-year.csv"


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    with requests.get(url, stream=True, timeout=(30, 600), headers={"Accept-Encoding": "identity"}) as response:
        response.raise_for_status()
        expected_size = int(response.headers["Content-Length"]) if "Content-Length" in response.headers else None
        if destination.exists():
            if expected_size is None or destination.stat().st_size != expected_size:
                raise ValueError(f"Cannot verify existing file completeness: {destination}. Original file preserved.")
            print(f"Verified size: {destination}")
            return
        print(f"Downloading: {url}")
        size = 0
        with partial.open("wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output.write(chunk)
                    size += len(chunk)
        if size == 0 or (expected_size is not None and size != expected_size):
            raise ValueError(f"Incomplete download: {size} bytes, expected {expected_size}. Partial file: {partial}")
        partial.replace(destination)


def extract_years_from_cumulative(
    source: Path, output_dir: Path, file_type: str, years: range
) -> None:
    year_column = "collision_year"
    for year in years:
        destination = output_dir / f"dft-road-casualty-statistics-{file_type}-{year}.csv"
        partial = destination.with_suffix(destination.suffix + ".part")
        existing = destination.exists()
        first_chunk = True
        rows_written = 0
        columns = None
        for chunk in pd.read_csv(source, encoding="utf-8-sig", low_memory=False, chunksize=100_000,
                                 dtype={"collision_index": "string", "collision_ref_no": "string"}):
            columns = list(chunk.columns)
            selected = chunk[chunk[year_column] == year]
            if selected.empty:
                continue
            if not existing:
                selected.to_csv(partial, mode="w" if first_chunk else "a", index=False, header=first_chunk)
            first_chunk = False
            rows_written += len(selected)
        if rows_written == 0:
            raise ValueError(f"No rows for {year} in {source}; no final file created")
        if existing:
            current = pd.read_csv(destination, encoding="utf-8-sig", low_memory=False)
            if list(current.columns) != columns or len(current) != rows_written or current[year_column].ne(year).any():
                raise ValueError(f"Existing extracted file disagrees with archive: {destination}. Original preserved.")
            print(f"Verified archive row count: {destination}")
            continue
        partial.replace(destination)
        print(f"Extracted {rows_written:,} rows: {destination}")


def download_historical_cumulative(output_dir: Path, start_year: int, end_year: int) -> None:
    archive_dir = output_dir / "archive"
    years = range(start_year, end_year + 1)
    for file_type in FILE_TYPES:
        filename = ARCHIVE_TEMPLATE.format(file_type=file_type)
        archive_path = archive_dir / filename
        url = f"{BASE_URL}/{filename}"
        download_file(url, archive_path)
        extract_years_from_cumulative(archive_path, output_dir, file_type, years)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-year", type=int, default=2019)
    parser.add_argument("--end-year", type=int, default=2021)
    parser.add_argument(
        "--use-cumulative-archive",
        action="store_true",
        help="Download official 1979-latest cumulative files and extract requested years.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "data" / "raw" / "stats19",
    )
    args = parser.parse_args()

    if args.start_year > args.end_year:
        raise ValueError("start-year must be less than or equal to end-year")

    if args.use_cumulative_archive and args.start_year <= 2020:
        download_historical_cumulative(args.output_dir, args.start_year, min(args.end_year, 2020))
    elif args.start_year <= 2020:
        raise RuntimeError(
            "DfT no longer exposes 2019-2020 as individual CSV endpoints. "
            "Use the official 1979-latest cumulative file or obtain the archived "
            "yearly files before continuing. No unofficial mirror is used automatically."
        )

    for year in range(max(args.start_year, 2021), args.end_year + 1):
        for file_type in FILE_TYPES:
            filename = f"dft-road-casualty-statistics-{file_type}-{year}.csv"
            url = f"{BASE_URL}/{filename}"
            download_file(url, args.output_dir / filename)


if __name__ == "__main__":
    main()
