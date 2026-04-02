"""Download BioSR datasets used in ResMatching.

Usage:
    uv run python download_data.py               # download all subsets
    uv run python download_data.py --data-dir /my/path
    uv run python download_data.py --subset ccp --subset er
"""

import zipfile
from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

import pooch
import typer

BASE_URL = "https://download.fht.org/jug/Anirban/ISBI2026/"

# Each entry: subset key -> (filename on server, sha256 or None to skip check)
# Update hashes once the files are public.
DATASETS = {
    "ccp":      ("CCPs_SuperRes.zip",              None),
    "er":       ("ER_SuperRes.zip",                None),
    "factin":   ("F-actin_SuperRes.zip",           None),
    "mt":       ("Microtubules_SuperRes.zip",      None),
    "mt_noisy": ("MicrotubulesNoisy_SuperRes.zip", None),
}

DESCRIPTIONS = {
    "ccp":      "Clathrin-Coated Pits",
    "er":       "Endoplasmic Reticulum",
    "factin":   "F-actin",
    "mt":       "Microtubules",
    "mt_noisy": "Microtubules (noisy input)",
}

Subset = Enum("Subset", {k: k for k in DATASETS})

app = typer.Typer()


def _download_subset(key: str, data_dir: Path) -> None:
    filename, known_hash = DATASETS[key]
    typer.echo(f"Downloading {DESCRIPTIONS[key]} ({filename}) ...")

    path = pooch.retrieve(
        url=BASE_URL + filename,
        known_hash=known_hash,
        fname=filename,
        path=data_dir,
        progressbar=True,
    )

    typer.echo(f"  Extracting to {data_dir} ...")
    with zipfile.ZipFile(path, "r") as zf:
        zf.extractall(data_dir)
    Path(path).unlink()
    typer.echo(f"  Done -> {data_dir / filename.replace('.zip', '')}")


@app.command()
def main(
    data_dir: Annotated[Path, typer.Option(help="Directory to download data into.")] = Path("data"),
    subset: Annotated[Optional[list[Subset]], typer.Option(help="Subset(s) to download. Repeat to select multiple. Default: all.")] = None,
):
    keys = [s.value for s in subset] if subset else list(DATASETS)
    data_dir.mkdir(parents=True, exist_ok=True)

    for key in keys:
        _download_subset(key, data_dir)

    typer.echo("\nAll done. Data layout:")
    for key in keys:
        filename, _ = DATASETS[key]
        typer.echo(f"  {data_dir / filename.replace('.zip', '')}/")


if __name__ == "__main__":
    app()
