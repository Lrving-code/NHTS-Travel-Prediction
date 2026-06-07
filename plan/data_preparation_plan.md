# Task Plan: NHTS Data Preparation

## Goal
Prepare the public NHTS 2001, 2009, 2017, and 2022 data locally and produce an initial inventory for downstream research design.

## Phases
- [x] Phase 1: Inspect current repository state
- [x] Phase 2: Confirm official NHTS download links and file formats
- [x] Phase 3: Implement reusable download, extraction, and inventory scripts
- [x] Phase 4: Run data preparation and generate inventory reports
- [x] Phase 5: Summarize feasible research directions

## Key Questions
1. Which NHTS versions and public-use formats are available for direct download?
2. Do all target years expose household, person, vehicle, and trip files?
3. Which variables are stable enough for cross-year prediction experiments?

## Decisions Made
- Store project planning files in `plan/` to follow the project AGENTS.md convention.
- Prefer CSV public-use files when available because they are portable and directly usable in Python.
- Use official CSV packages for 2022 and 2017.
- Use official XPT packages for 2009 and 2001 because their primary public downloads do not include a main CSV package.
- Add official ASCII packages for 2009 and 2001 because the downloaded XPT files are SAS CPORT files that pandas cannot inspect directly.
- Keep raw data under `data/raw/`, generated inventory reports under `outputs/inventory/`, and research direction notes under `plan/`.
- Add `.gitignore` so raw data, archives, temp files, and Python caches are not accidentally committed.

## Errors Encountered
- Git status check failed earlier because the project directory is not a Git repository.
- Pandas cannot read the 2001/2009 XPT files because they are SAS CPORT files; ASCII packages are being added as the Python-readable fallback.

## Status
**Completed** - Data packages are downloaded, extracted, inventoried, and summarized for research planning.
