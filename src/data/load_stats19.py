from __future__ import annotations

from pathlib import Path

import pandas as pd


YEARS = (2019, 2020, 2021)
SEVERITY_LABELS = {1: "Fatal", 2: "Serious", 3: "Slight"}

LABEL_COLUMNS = (
    "collision_severity",
    "road_type",
    "junction_detail",
    "junction_detail_historic",
    "junction_control",
    "light_conditions",
    "weather_conditions",
    "road_surface_conditions",
    "urban_or_rural_area",
)


def read_codebook(path: Path) -> dict[str, dict[int, str]]:
    guide = pd.read_excel(path, sheet_name="2024_code_list")
    codes: dict[str, dict[int, str]] = {}
    for _, row in guide.loc[guide["table"] == "collision"].iterrows():
        if pd.isna(row["label"]):
            continue
        try:
            code = int(row["code/format"])
        except (ValueError, TypeError):
            continue
        codes.setdefault(row["field name"], {})[code] = str(row["label"])
    return codes


def read_table(data_dir: Path, table: str, year: int) -> pd.DataFrame:
    path = data_dir / f"dft-road-casualty-statistics-{table}-{year}.csv"
    frame = pd.read_csv(path, encoding="utf-8-sig", low_memory=False,
                        dtype={"collision_index": "string", "collision_ref_no": "string"})
    required = {"collision_index", "collision_year"}
    if table == "collision":
        required |= {"collision_severity", "date", "time", "day_of_week"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path.name}: missing required columns {sorted(missing)}")
    if frame["collision_year"].ne(year).any():
        raise ValueError(f"{path.name}: contains rows outside requested year {year}")
    return frame


def add_temporal_columns(frame: pd.DataFrame) -> pd.DataFrame:
    collisions = frame.copy()
    collisions["date"] = pd.to_datetime(collisions["date"], format="%d/%m/%Y", errors="coerce")
    collisions["hour"] = pd.to_datetime(collisions["time"], format="%H:%M", errors="coerce").dt.hour
    collisions["month"] = collisions["date"].dt.month
    collisions["weekday"] = collisions["date"].dt.dayofweek
    collisions["season"] = collisions["month"].map(
        {12: "Winter", 1: "Winter", 2: "Winter", 3: "Spring", 4: "Spring", 5: "Spring",
         6: "Summer", 7: "Summer", 8: "Summer", 9: "Autumn", 10: "Autumn", 11: "Autumn"}
    )
    return collisions


def load_collision_data(data_dir: Path, years: tuple[int, ...] = YEARS,
                        guide_path: Path | None = None) -> pd.DataFrame:
    data_dir = Path(data_dir).resolve()
    guide_path = guide_path or data_dir.parents[2] / "config" / "stats19_data_guide_2025.xlsx"
    codebook = read_codebook(guide_path)
    frames = [read_table(data_dir, "collision", year) for year in years]
    if any(list(frame.columns) != list(frames[0].columns) for frame in frames[1:]):
        raise ValueError("Collision schemas differ between years")
    collisions = add_temporal_columns(pd.concat(frames, ignore_index=True))
    for column in LABEL_COLUMNS:
        if column not in collisions or column not in codebook:
            raise ValueError(f"Required EDA column/codebook definition missing: {column}")
        collisions[f"{column}_label"] = collisions[column].map(codebook[column]).fillna("Unmapped code")
    return collisions
