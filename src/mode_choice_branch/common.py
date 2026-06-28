"""Shared constants and helpers for the mode-choice processing branch."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


LOGGER = logging.getLogger(__name__)
TARGET_COLUMN = "TRPTRANS"
MODE_GROUP_COLUMN = "MODE_GROUP"
HOUSEHOLD_ID = "HOUSEID"
PERSON_ID = "PERSONID"
DEFAULT_RAW_ROOT = Path("data/raw")
DEFAULT_INTERIM_DIR = Path("data/interim/mode_choice_branch")
DEFAULT_OUTPUT_DIR = Path("outputs/mode_choice_branch")
NEGATIVE_MISSING_CODES: tuple[object, ...] = ("-1", "-2", "-7", "-8", "-9", -1, -2, -7, -8, -9)
INVALID_TARGET_CODES: tuple[str, ...] = ("", "-1", "-2", "-7", "-8", "-9", "98", "99")


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


TRPTRANS_LABELS_BY_YEAR: dict[int, dict[int, str]] = {
    2017: {
        1: "walk",
        2: "bicycle",
        3: "car",
        4: "suv",
        5: "van",
        6: "pickup truck",
        7: "golf cart or segway",
        8: "motorcycle or moped",
        9: "rv or atv",
        10: "school bus",
        11: "public or commuter bus",
        12: "paratransit",
        13: "private charter or shuttle bus",
        14: "city-to-city bus",
        15: "amtrak or commuter rail",
        16: "subway or light rail",
        17: "taxi limo or ridehail",
        18: "rental car or carshare",
        19: "airplane",
        20: "boat ferry or water taxi",
        97: "other",
    },
    2022: {
        1: "car",
        2: "van",
        3: "suv or crossover",
        4: "pickup truck",
        6: "recreational vehicle",
        7: "motorcycle",
        8: "public or commuter bus",
        9: "school bus",
        10: "streetcar or trolley",
        11: "subway or elevated rail",
        12: "commuter rail",
        13: "amtrak",
        14: "airplane",
        15: "taxicab or limo",
        16: "ride-sharing service",
        17: "paratransit",
        18: "bicycle or bikeshare",
        19: "e-scooter",
        20: "walk",
        21: "other",
    },
}

MODE_GROUP_LABELS: dict[str, str] = {
    "private_vehicle": "Private vehicle",
    "walk": "Walk",
    "bike_micromobility": "Bike or micromobility",
    "bus_paratransit": "Bus, shuttle, or paratransit",
    "rail_transit": "Rail transit",
    "taxi_ridehail": "Taxi or ride-hail",
    "air_water_other": "Air, water, or other",
}

MODE_GROUP_ORDER: tuple[str, ...] = (
    "private_vehicle",
    "walk",
    "bike_micromobility",
    "bus_paratransit",
    "rail_transit",
    "taxi_ridehail",
    "air_water_other",
)

MODE_GROUP_MAPPINGS: dict[int, dict[int, str]] = {
    2017: {
        1: "walk",
        2: "bike_micromobility",
        3: "private_vehicle",
        4: "private_vehicle",
        5: "private_vehicle",
        6: "private_vehicle",
        7: "bike_micromobility",
        8: "private_vehicle",
        9: "private_vehicle",
        10: "bus_paratransit",
        11: "bus_paratransit",
        12: "bus_paratransit",
        13: "bus_paratransit",
        14: "bus_paratransit",
        15: "rail_transit",
        16: "rail_transit",
        17: "taxi_ridehail",
        18: "private_vehicle",
        19: "air_water_other",
        20: "air_water_other",
        97: "air_water_other",
    },
    2022: {
        1: "private_vehicle",
        2: "private_vehicle",
        3: "private_vehicle",
        4: "private_vehicle",
        6: "private_vehicle",
        7: "private_vehicle",
        8: "bus_paratransit",
        9: "bus_paratransit",
        10: "rail_transit",
        11: "rail_transit",
        12: "rail_transit",
        13: "rail_transit",
        14: "air_water_other",
        15: "taxi_ridehail",
        16: "taxi_ridehail",
        17: "bus_paratransit",
        18: "bike_micromobility",
        19: "bike_micromobility",
        20: "walk",
        21: "air_water_other",
    },
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


def map_trptrans_to_mode_group(series: pd.Series, year: int) -> pd.Series:
    """Map year-specific raw TRPTRANS codes into comparable mode groups."""
    if year not in MODE_GROUP_MAPPINGS:
        raise ValueError(f"Unsupported NHTS year for mode harmonization: {year}")
    numeric = pd.to_numeric(series, errors="coerce").astype("Int64")
    mapped = numeric.map(MODE_GROUP_MAPPINGS[year]).astype("string")
    return mapped.mask(mapped.isna())


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
