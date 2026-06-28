"""Shared constants and helpers for the mode-choice processing branch."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


LOGGER = logging.getLogger(__name__)
TARGET_COLUMN = "TRPTRANS"
HOUSEHOLD_ID = "HOUSEID"
PERSON_ID = "PERSONID"
DEFAULT_RAW_ROOT = Path("data/raw")
DEFAULT_INTERIM_DIR = Path("data/interim/mode_choice_branch")
DEFAULT_OUTPUT_DIR = Path("outputs/mode_choice_branch")
NEGATIVE_MISSING_CODES: tuple[object, ...] = ("-1", "-2", "-7", "-8", "-9", -1, -2, -7, -8, -9)
INVALID_TARGET_CODES: tuple[str, ...] = ("", "-1", "-2", "-7", "-8", "-9", "97", "98", "99")


@dataclass(frozen=True)
class TableSpec:
    """Source and output metadata for one NHTS table."""

    name: str
    relative_2017_path: Path
    relative_2022_path: Path
    output_2017_name: str
    output_2022_name: str


TABLE_SPECS: dict[str, TableSpec] = {
    "hh": TableSpec("hh", Path("nhts_2017/csv/hhpub.csv"), Path("nhts_2022/csv/hhv2pub.csv"), "hh_2017.csv", "hh_2022.csv"),
    "per": TableSpec("per", Path("nhts_2017/csv/perpub.csv"), Path("nhts_2022/csv/perv2pub.csv"), "per_2017.csv", "per_2022.csv"),
    "trip": TableSpec("trip", Path("nhts_2017/csv/trippub.csv"), Path("nhts_2022/csv/tripv2pub.csv"), "trip_2017.csv", "trip_2022.csv"),
    "veh": TableSpec("veh", Path("nhts_2017/csv/vehpub.csv"), Path("nhts_2022/csv/vehv2pub.csv"), "veh_2017.csv", "veh_2022.csv"),
}


COMMON_FIELDS: dict[str, tuple[str, ...]] = {
    "hh": (
        "CDIVMSAR", "CENSUS_D", "CENSUS_R", "DRVRCNT", "HBHTNRNT", "HBHUR",
        "HBPPOPDN", "HBRESDN", "HHFAMINC", "HHSIZE", "HHVEHCNT", "HH_HISP",
        "HH_RACE", "HOMEOWN", "HOUSEID", "HTEEMPDN", "HTHTNRNT", "HTPPOPDN",
        "HTRESDN", "LIF_CYC", "MSACAT", "MSASIZE", "NUMADLT", "RAIL",
        "TDAYDATE", "TRAVDAY", "URBAN", "URBANSIZE", "URBRUR", "WRKCOUNT",
    ),
    "per": (
        "CDIVMSAR", "CENSUS_D", "CENSUS_R", "CNTTDTR", "CONDNIGH", "CONDPUB",
        "CONDRIDE", "CONDRIVE", "CONDSPEC", "CONDTRAV", "DELIVER", "DRIVER",
        "DRVRCNT", "EDUC", "GCDWORK", "HBHTNRNT", "HBHUR", "HBPPOPDN",
        "HBRESDN", "HHFAMINC", "HHSIZE", "HHVEHCNT", "HH_HISP", "HH_RACE",
        "HOMEOWN", "HOUSEID", "HTEEMPDN", "HTHTNRNT", "HTPPOPDN", "HTRESDN",
        "LIF_CYC", "MEDCOND", "MEDCOND6", "MSACAT", "MSASIZE", "NUMADLT",
        "OUTOFTWN", "PAYPROF", "PERSONID", "PRMACT", "PROXY", "PTUSED",
        "RAIL", "R_AGE", "R_HISP", "R_RACE", "R_RELAT", "R_SEX", "R_SEX_IMP",
        "SAMEPLC", "SCHTRN1", "SCHTYP", "TDAYDATE", "TRAVDAY", "URBAN",
        "URBANSIZE", "URBRUR", "USEPUBTR", "WHOPROXY", "WORKER", "WRKCOUNT",
        "WRKTRANS", "WTPERFIN", "W_CANE", "W_CHAIR", "W_NONE",
    ),
    "trip": (
        "CDIVMSAR", "CENSUS_D", "CENSUS_R", "DBHTNRNT", "DBHUR", "DBPPOPDN",
        "DBRESDN", "DRIVER", "DRVRCNT", "DRVR_FLG", "DTEEMPDN", "DTHTNRNT",
        "DTPPOPDN", "DTRESDN", "DWELTIME", "EDUC", "ENDTIME", "GASPRICE",
        "HHFAMINC", "HHMEMDRV", "HHSIZE", "HHVEHCNT", "HH_HISP", "HH_RACE",
        "HOMEOWN", "HOUSEID", "LIF_CYC", "LOOP_TRIP", "MSACAT", "MSASIZE",
        "NONHHCNT", "NUMADLT", "NUMONTRP", "OBHTNRNT", "OBHUR", "OBPPOPDN",
        "OBRESDN", "ONTD_P1", "ONTD_P10", "ONTD_P2", "ONTD_P3", "ONTD_P4",
        "ONTD_P5", "ONTD_P6", "ONTD_P7", "ONTD_P8", "ONTD_P9", "OTEEMPDN",
        "OTHTNRNT", "OTPPOPDN", "OTRESDN", "PERSONID", "PRMACT", "PROXY",
        "PSGR_FLG", "PUBTRANS", "RAIL", "R_AGE", "R_SEX", "R_SEX_IMP",
        "STRTTIME", "TDAYDATE", "TDCASEID", "TDWKND", "TRAVDAY", "TRIPPURP",
        "TRPHHVEH", "TRPMILES", "TRPTRANS", "TRVLCMIN", "URBAN", "URBANSIZE",
        "URBRUR", "VEHID", "VEHTYPE", "VMT_MILE", "WHODROVE", "WHYFROM",
        "WHYTO", "WHYTRP1S", "WHYTRP90", "WORKER", "WRKCOUNT", "WTTRDFIN",
    ),
    "veh": (
        "ANNMILES", "CDIVMSAR", "CENSUS_D", "CENSUS_R", "DRVRCNT", "HBHTNRNT",
        "HBHUR", "HBPPOPDN", "HBRESDN", "HHFAMINC", "HHSIZE", "HHVEHCNT",
        "HH_HISP", "HH_RACE", "HOMEOWN", "HOUSEID", "HTEEMPDN", "HTHTNRNT",
        "HTPPOPDN", "HTRESDN", "HYBRID", "LIF_CYC", "MAKE", "MSACAT",
        "MSASIZE", "NUMADLT", "RAIL", "TDAYDATE", "TRAVDAY", "URBAN",
        "URBANSIZE", "URBRUR", "VEHAGE", "VEHID", "VEHOWNED", "VEHOWNMO",
        "VEHTYPE", "VEHYEAR", "WHOMAIN", "WRKCOUNT", "WTHHFIN",
    ),
}


NUMERIC_FIELDS: tuple[str, ...] = (
    "DRVRCNT", "HBPPOPDN", "HBRESDN", "HHSIZE", "HHVEHCNT", "HTEEMPDN",
    "HTPPOPDN", "HTRESDN", "MSASIZE", "NUMADLT", "URBANSIZE", "WRKCOUNT",
    "CNTTDTR", "GCDWORK", "R_AGE", "R_SEX_IMP", "DBPPOPDN", "DBRESDN",
    "DTEEMPDN", "DTPPOPDN", "DTRESDN", "GASPRICE", "NONHHCNT", "NUMONTRP",
    "OBPPOPDN", "OBRESDN", "OTEEMPDN", "OTPPOPDN", "OTRESDN", "TRPMILES",
    "TRVLCMIN", "VMT_MILE", "WTTRDFIN", "ANNMILES", "VEHAGE", "VEHOWNMO",
    "VEHYEAR", "WTHHFIN", "WTPERFIN",
)


CATEGORICAL_FIELDS: tuple[str, ...] = (
    "CDIVMSAR", "CENSUS_D", "CENSUS_R", "HBHTNRNT", "HBHUR", "HHFAMINC",
    "HH_HISP", "HH_RACE", "HOMEOWN", "HOUSEID", "HTHTNRNT", "LIF_CYC",
    "MEDCOND", "MEDCOND6", "MSACAT", "OUTOFTWN", "PAYPROF", "PERSONID",
    "PRMACT", "PROXY", "PTUSED", "RAIL", "R_HISP", "R_RACE", "R_RELAT",
    "R_SEX", "SAMEPLC", "SCHTRN1", "SCHTYP", "TDAYDATE", "TRAVDAY",
    "URBAN", "URBRUR", "USEPUBTR", "WHOPROXY", "WORKER", "WRKTRANS",
    "W_CANE", "W_CHAIR", "W_NONE", "CONDNIGH", "CONDPUB", "CONDRIDE",
    "CONDRIVE", "CONDSPEC", "CONDTRAV", "DELIVER", "DRIVER", "EDUC",
    "DBHTNRNT", "DBHUR", "DRVR_FLG", "DTHTNRNT", "DWELTIME", "ENDTIME",
    "HHMEMDRV", "LOOP_TRIP", "OBHTNRNT", "OBHUR", "ONTD_P1", "ONTD_P10",
    "ONTD_P2", "ONTD_P3", "ONTD_P4", "ONTD_P5", "ONTD_P6", "ONTD_P7",
    "ONTD_P8", "ONTD_P9", "OTHTNRNT", "PSGR_FLG", "PUBTRANS", "STRTTIME",
    "TDCASEID", "TDWKND", "TRIPPURP", "TRPHHVEH", "TRPTRANS", "VEHID",
    "VEHTYPE", "WHODROVE", "WHYFROM", "WHYTO", "WHYTRP1S", "WHYTRP90",
    "HYBRID", "MAKE", "VEHOWNED", "WHOMAIN",
)


MODE_CHOICE_FEATURES: tuple[str, ...] = (
    "HHVEHCNT", "DRVRCNT", "RAIL", "DRIVER",
    "GASPRICE", "HHFAMINC",
    "HBHUR", "URBAN", "MSASIZE", "HBPPOPDN", "HTHTNRNT", "URBRUR",
    "TRPMILES", "TRVLCMIN", "WHYTRP90", "LOOP_TRIP", "NUMONTRP",
    "R_AGE", "R_SEX", "EDUC", "WORKER", "PAYPROF", "PRMACT",
    "HHSIZE", "HOMEOWN", "WRKCOUNT", "LIF_CYC",
)


FEATURE_GROUPS: dict[str, tuple[str, ...]] = {
    "transport_access": ("HHVEHCNT", "DRVRCNT", "RAIL", "DRIVER"),
    "travel_cost": ("GASPRICE", "HHFAMINC"),
    "built_environment": ("HBHUR", "URBAN", "MSASIZE", "HBPPOPDN", "HTHTNRNT", "URBRUR"),
    "trip_attributes": ("TRPMILES", "TRVLCMIN", "WHYTRP90", "LOOP_TRIP", "NUMONTRP"),
    "person_attributes": ("R_AGE", "R_SEX", "EDUC", "WORKER", "PAYPROF", "PRMACT"),
    "household_attributes": ("HHSIZE", "HOMEOWN", "WRKCOUNT", "LIF_CYC"),
}


TRPTRANS_LABELS: dict[int, str] = {
    1: "bus or streetcar",
    2: "taxi or ridehail",
    3: "private vehicle driver",
    4: "walk",
    5: "carpool passenger",
    6: "company vehicle",
    7: "motorcycle or scooter",
    8: "subway or light rail",
    9: "commuter rail",
    10: "train or intercity bus",
    11: "ferry",
    12: "other",
    13: "streetcar",
    14: "airport shuttle",
    15: "campus shuttle",
    16: "bikeshare",
    17: "shared e-scooter",
    18: "private vehicle passenger",
    19: "special shuttle",
    20: "bicycle",
    21: "freight truck",
    97: "refused",
    98: "do not know",
    99: "not ascertained",
}


def configure_logging() -> None:
    """Configure command-line logging."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")


def source_path(raw_root: Path, table_name: str, year: int) -> Path:
    """Resolve the source CSV path for a table and year."""
    spec = TABLE_SPECS[table_name]
    if year == 2017:
        return raw_root / spec.relative_2017_path
    if year == 2022:
        return raw_root / spec.relative_2022_path
    raise ValueError(f"Unsupported NHTS year for this branch: {year}")


def preprocessed_path(interim_dir: Path, table_name: str, year: int) -> Path:
    """Resolve the preprocessed CSV path for a table and year."""
    spec = TABLE_SPECS[table_name]
    if year == 2017:
        return interim_dir / spec.output_2017_name
    if year == 2022:
        return interim_dir / spec.output_2022_name
    raise ValueError(f"Unsupported NHTS year for this branch: {year}")


def parse_table_names(raw_value: str) -> tuple[str, ...]:
    """Parse a comma-separated table list."""
    names = tuple(name.strip().lower() for name in raw_value.split(",") if name.strip())
    unknown = sorted(set(names) - set(TABLE_SPECS))
    if unknown:
        raise ValueError(f"Unknown table name(s): {unknown}. Expected one of {sorted(TABLE_SPECS)}.")
    if not names:
        raise ValueError("At least one table name is required.")
    return names


def read_csv_selected(path: Path, columns: Iterable[str], **read_csv_kwargs: object) -> pd.DataFrame:
    """Read selected columns case-insensitively and return uppercase column names."""
    if not path.exists():
        raise FileNotFoundError(f"Missing source CSV: {path}")

    header = pd.read_csv(path, nrows=0)
    column_map = {str(column).upper(): column for column in header.columns}
    requested = tuple(str(column).upper() for column in columns)
    missing = sorted(set(requested) - set(column_map))
    if missing:
        raise ValueError(f"{path} is missing required column(s): {missing}")

    selected = [column_map[column] for column in requested]
    frame = pd.read_csv(path, usecols=selected, **read_csv_kwargs)
    frame.columns = [str(column).upper() for column in frame.columns]
    return frame.loc[:, list(requested)]
