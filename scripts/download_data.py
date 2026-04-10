"""Download BioSR datasets used in ResMatching.

Usage:
    uv run python scripts/download_data.py               # download all subsets
    uv run python scripts/download_data.py --data-dir /my/path
    uv run python scripts/download_data.py --subset ccp --subset er
"""

import zipfile
from enum import Enum
from pathlib import Path
from typing import Annotated, Optional

import pooch
import typer

BASE_URL = "https://download.fht.org/jug/resmatching/data/"


DATASETS = {
    "ccp": (
        "ccp.zip",
        "56c486df8673591b8e64c4fccdf11b02549d18ce592b0d0d23a148ddffce1d9a",
    ),
    "er": (
        "er.zip",
        "6f3cb5aebc38c9a1400867dc814ce4cb9c50ae93f6acb1e184537a531edbb715",
    ),
    "factin": (
        "factin.zip",
        "28a9d6351dd9d5681212141954a302397986b50a652ae05bcc538393c0fa5c10",
    ),
    "mt": (
        "mt.zip",
        "8bc6fe5c1810add25fa1adab2776e613c0bbc223b1686790fb36f947b4d98eaf",
    ),
    "mt_noisy": (
        "mt_noisy.zip",
        "323fca416ab08edeacd2dc14d3c486065e43f7a400e0ed604a0cb4feb4efe04d",
    ),
}

DESCRIPTIONS = {
    "ccp": "Clathrin-Coated Pits",
    "er": "Endoplasmic Reticulum",
    "factin": "F-actin",
    "mt": "Microtubules",
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
    data_dir: Annotated[
        Path, typer.Option(help="Directory to download data into.")
    ] = Path("data"),
    subset: Annotated[
        Optional[list[Subset]],
        typer.Option(
            help="Subset(s) to download. Repeat to select multiple. Default: all."
        ),
    ] = None,
):
    keys = [s.value for s in subset] if subset else list(DATASETS)
    data_dir.mkdir(parents=True, exist_ok=True)

    for key in keys:
        _download_subset(key, data_dir)

    typer.echo("\nAll done. Data layout:")
    for key in keys:
        typer.echo(f"  {data_dir / key}/")


if __name__ == "__main__":
    app()
