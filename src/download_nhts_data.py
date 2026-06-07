"""Download and extract public NHTS data packages."""

from __future__ import annotations

import argparse
import logging
import zipfile
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen


LOGGER = logging.getLogger(__name__)
BUFFER_SIZE = 1024 * 1024


@dataclass(frozen=True)
class DataPackage:
    """NHTS public-use data package metadata."""

    year: str
    label: str
    url: str
    expected_bytes: int

    @property
    def archive_name(self) -> str:
        return f"{self.year}_{self.label}.zip"


DATA_PACKAGES: tuple[DataPackage, ...] = (
    DataPackage(
        year="2022",
        label="csv",
        url="https://nhts.ornl.gov/media/2022/download/csv.zip",
        expected_bytes=4_533_528,
    ),
    DataPackage(
        year="2017",
        label="csv",
        url="https://nhts.ornl.gov/media/2016/download/csv.zip",
        expected_bytes=83_726_358,
    ),
    DataPackage(
        year="2009",
        label="xpt",
        url="https://nhts.ornl.gov/media/2009/download/Xpt.zip",
        expected_bytes=100_169_615,
    ),
    DataPackage(
        year="2009",
        label="ascii",
        url="https://nhts.ornl.gov/media/2009/download/Ascii.zip",
        expected_bytes=94_810_130,
    ),
    DataPackage(
        year="2001",
        label="xpt",
        url="https://nhts.ornl.gov/media/2001/download/Xpt.zip",
        expected_bytes=66_288_986,
    ),
    DataPackage(
        year="2001",
        label="ascii",
        url="https://nhts.ornl.gov/media/2001/download/Ascii.zip",
        expected_bytes=63_933_715,
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Directory where raw NHTS data will be stored.",
    )
    parser.add_argument(
        "--archive-dir",
        type=Path,
        default=Path("data/raw/_archives"),
        help="Directory where downloaded zip archives will be stored.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Redownload archives even when a size-matched file already exists.",
    )
    return parser.parse_args()


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )


def should_download(path: Path, expected_bytes: int, overwrite: bool) -> bool:
    if overwrite:
        return True
    if not path.exists():
        return True
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        LOGGER.warning(
            "Archive size mismatch for %s: expected %s bytes, found %s bytes.",
            path,
            expected_bytes,
            actual_bytes,
        )
        return True
    return False


def download_package(package: DataPackage, archive_dir: Path, overwrite: bool) -> Path:
    archive_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archive_dir / package.archive_name
    if not should_download(archive_path, package.expected_bytes, overwrite):
        LOGGER.info("Skipping existing archive: %s", archive_path)
        return archive_path

    tmp_path = archive_path.with_suffix(".zip.part")
    LOGGER.info("Downloading %s %s from %s", package.year, package.label, package.url)
    request = Request(package.url, headers={"User-Agent": "NHTS-data-preparation/1.0"})

    try:
        with urlopen(request, timeout=60) as response, tmp_path.open("wb") as output:
            while True:
                chunk = response.read(BUFFER_SIZE)
                if not chunk:
                    break
                output.write(chunk)
    except URLError as error:
        if tmp_path.exists():
            tmp_path.unlink()
        raise RuntimeError(f"Failed to download {package.url}: {error}") from error

    actual_bytes = tmp_path.stat().st_size
    if actual_bytes != package.expected_bytes:
        tmp_path.unlink()
        raise RuntimeError(
            f"Downloaded size mismatch for {package.url}: "
            f"expected {package.expected_bytes}, found {actual_bytes}."
        )

    tmp_path.replace(archive_path)
    LOGGER.info("Saved archive: %s", archive_path)
    return archive_path


def safe_extract(archive_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    root = output_dir.resolve()
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            target = (output_dir / member.filename).resolve()
            if root not in target.parents and target != root:
                raise RuntimeError(f"Unsafe archive path: {member.filename}")
        archive.extractall(output_dir)


def extract_package(package: DataPackage, archive_path: Path, data_dir: Path) -> None:
    output_dir = data_dir / f"nhts_{package.year}" / package.label
    LOGGER.info("Extracting %s to %s", archive_path.name, output_dir)
    safe_extract(archive_path, output_dir)


def main() -> None:
    args = parse_args()
    configure_logging()

    for package in DATA_PACKAGES:
        archive_path = download_package(package, args.archive_dir, args.overwrite)
        extract_package(package, archive_path, args.data_dir)

    LOGGER.info("NHTS data preparation completed.")


if __name__ == "__main__":
    main()
