# Notes: NHTS Data Preparation

## Sources

### NHTS Official Downloads
- URL: https://nhts.ornl.gov/downloads
- Key points:
  - Public-use downloads are available for 2022, 2017, 2009, and 2001.
  - CSV is the preferred local processing format when available.
  - Recommended citation should be preserved in downstream reports.

### Resolved Official Package URLs
- 2022 CSV: https://nhts.ornl.gov/media/2022/download/csv.zip
- 2017 CSV: https://nhts.ornl.gov/media/2016/download/csv.zip
- 2009 XPT: https://nhts.ornl.gov/media/2009/download/Xpt.zip
- 2009 ASCII/CSV: https://nhts.ornl.gov/media/2009/download/Ascii.zip
- 2001 XPT: https://nhts.ornl.gov/media/2001/download/Xpt.zip
- 2001 ASCII/CSV: https://nhts.ornl.gov/media/2001/download/Ascii.zip

## Synthesized Findings

### Initial Local State
- `data/raw/nhts_2001`, `data/raw/nhts_2009`, `data/raw/nhts_2017`, and `data/raw/nhts_2022` exist.
- No source data files are currently present in those directories.

### Prepared Local State
- Downloaded archives are stored in `data/raw/_archives/`.
- Extracted CSV-readable data are stored under:
  - `data/raw/nhts_2001/ascii/`
  - `data/raw/nhts_2009/ascii/`
  - `data/raw/nhts_2017/csv/`
  - `data/raw/nhts_2022/csv/`
- XPT archives for 2001/2009 were also downloaded and extracted, but pandas reports they are SAS CPORT files and cannot inspect them directly.
- Table inventory report: `outputs/inventory/inventory_summary.md`
- Column inventory report: `outputs/inventory/column_inventory.csv`
- Variable overlap report: `outputs/inventory/variable_overlap_summary.md`
- Initial research directions: `plan/research_directions.md`
