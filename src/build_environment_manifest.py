"""Export a lightweight environment and GPU manifest for reproducibility."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "submission_readiness"
KEY_PACKAGES = [
    "catboost",
    "lightgbm",
    "matplotlib",
    "numpy",
    "pandas",
    "python-pptx",
    "scikit-learn",
    "scipy",
    "seaborn",
    "torch",
    "xgboost",
]


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def all_packages() -> list[str]:
    rows = []
    for dist in sorted(metadata.distributions(), key=lambda item: item.metadata["Name"].lower()):
        name = dist.metadata["Name"]
        rows.append(f"{name}=={dist.version}")
    return rows


def run_command(command: list[str]) -> str:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return f"unavailable: {exc}"
    output = (completed.stdout or completed.stderr).strip()
    return output if output else f"exit_code={completed.returncode}"


def build_manifest() -> dict[str, object]:
    key_versions = {name: package_version(name) for name in KEY_PACKAGES}
    return {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "key_packages": key_versions,
        "nvidia_smi": run_command(
            [
                "nvidia-smi",
                "--query-gpu=name,driver_version,memory.total",
                "--format=csv,noheader",
            ]
        ),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = OUTPUT_DIR / "environment_manifest.json"
    freeze_path = OUTPUT_DIR / "environment_freeze.txt"
    manifest_path.write_text(json.dumps(build_manifest(), indent=2), encoding="utf-8")
    freeze_path.write_text("\n".join(all_packages()) + "\n", encoding="utf-8")
    print(f"Wrote {manifest_path}")
    print(f"Wrote {freeze_path}")


if __name__ == "__main__":
    main()
