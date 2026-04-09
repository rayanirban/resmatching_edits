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

BASE_URL = "https://download.fht.org/jug/resmatching/data/"


DATASETS = {
    "ccp": (
        "CCPs_SuperRes.zip",
        "4cf2d5a4f529c2ae6d016e51842fd5a2c87252fdee7da1c9926319017596e72b",
    ),
    "er": (
        "ER_SuperRes.zip",
        "bbeabf63a18bb2234aa76426536d080165a2f36bc947e5dff0d310dac9d149bf",
    ),
    "factin": (
        "F-actin_SuperRes.zip",
        "291592810013ecc1cf1cad4f89fc687977e4918a31031f8c0e0a0446b34f30d6",
    ),
    "mt": (
        "Microtubules_SuperRes.zip",
        "8d19b4b26e6d24a15224df23f48c6a9e28331f09590bc2d84d2c5ebe1dc9f3a7",
    ),
    "mt_noisy": (
        "MicrotubulesNoisy_SuperRes.zip",
        "3f2796c6fa9ecca91043928dc3bb0da794bea3773659f206aab9246f5b9c597d",
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
